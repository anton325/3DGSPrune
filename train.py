#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import io
from PIL import Image, ImageDraw
import pathlib
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene
from scene.gaussian_sphere_model import GaussianSphereModel as GaussianSphereModel
from scene.gaussian_model import GaussianModel as GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import json
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import datetime
from utils.evaluator import Evaluator
import time
import utils.plotter as plotter
import numpy as np
from torchvision.utils import save_image
import imageio.v2 as imageio

TAKE_OPTIMIZATION_VIDEO = True
video_rate = 17 # want 2 rotations -> 600 frames, 10.200/600 = 17


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# activate or deactivate logging
TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,**kwargs):
    optimization_video_frames = []
    optimization_sphere_cam = 0
    pure_training_time = time.time()
    total_time = time.time()
    # clean kwargs testing iterations
    kwargs['test_lpips_ssim_iterations'] = [test_it for test_it in kwargs['test_lpips_ssim_iterations'] if test_it <= kwargs['iterations']]
    opt.iterations = kwargs['iterations']
    opt.position_lr_max_steps = opt.iterations
    # opt.densify_from_iter = 500
    if "densify_until_iter" in kwargs.keys() and kwargs.get('densify_until_iter',None) != None:
        opt.densify_until_iter = kwargs['densify_until_iter']    
    else:
        opt.densify_until_iter = int(opt.iterations/2)
    # opt.opacity_reset_interval = 7000
    # opt.iterations = 5000
    first_iter = 0
    saving_iterations = [opt.iterations]
    # saving_iterations.extend([10,50,100,200,300,500,1000,2000,5000])
    kwargs['test_lpips_ssim_iterations'].append(opt.iterations)
    testing_iterations = kwargs['test_lpips_ssim_iterations']
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    if kwargs['model_selector'] == 'sphere':
        gaussians = GaussianSphereModel(dataset.sh_degree)
    elif kwargs['model_selector'] == 'gaussian':
        # print(dataset.sh_degree)
        gaussians = GaussianModel(dataset.sh_degree)
    else:
        raise NotImplementedError(f"Model Selector {kwargs['model_selector']} not implemented")
    scene = Scene(dataset, gaussians,**kwargs)
    train_eval_cameras = [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]
    if kwargs['additional_metrics']:
        evaluator = Evaluator(scene.model_path,train_eval_cameras,scene.getValCameras(),scene.getTestCameras(),opt.iterations,kwargs['test_lpips_ssim_iterations'],kwargs['train_metrics'])
    else:
        evaluator = None
    
    gaussians.training_setup(opt,**kwargs)
    if kwargs['init_method'] == "reconstruct_from_depth_map":
        scene.convert_gt_depth_map(**kwargs)
    # print(f"shape gaussian rots {gaussians.get_rotation.shape}")
    if checkpoint:
        print("Restore from checkpoint")
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    gaussians.training_setup(opt,**kwargs)

    # print("White Background: ",dataset.white_background)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    num_gaussians_per_it = []
    xyz_lrs = []
    its_per_second = []
    val_losses = []
    val_psnrs = []
    if kwargs['prune_down_scheduled']:
        kwargs['last_value_prune_down_scheduled'] = None
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    if kwargs['progress_bar']:
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):  
        start_time_iteration = time.time()
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer,**kwargs)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        xyz_lrs.append(gaussians.get_xyz_lr)
        num_gaussians_per_it.append(gaussians.get_xyz.shape[0])

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg,**kwargs)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            if TAKE_OPTIMIZATION_VIDEO:
                when_display = 1
            else:
                when_display = 100
            if kwargs['progress_bar']:
                # Progress bar
                if iteration % when_display == 0:
                    ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                else:
                    ema_loss_for_log = 0.4 * loss + 0.6 * ema_loss_for_log
                if iteration % when_display == 0:
                    progress_bar.set_postfix({"Loss": {ema_loss_for_log}})
                    progress_bar.update(when_display)
                if iteration == opt.iterations:
                    progress_bar.close()

            # Log and save (but only sometimes)
            eval_time = time.time()
            if iteration % when_display == 0:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene,train_eval_cameras, render, (pipe, background),evaluator,val_losses,val_psnrs,**kwargs)

            if iteration % 1000 == 0:
                plotter.plot_num_gaussians(num_gaussians_per_it, time.time() - pure_training_time, scene.model_path)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            pure_training_time += time.time() - eval_time # die testzeiten addieren, dadurch kommt startzeit näher ran
            # Densification # at some point stop densifying
            if kwargs['densify_and_prune']:
                # print(f"before prune {gaussians.get_xyz.shape}")
                if iteration < opt.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # Densify and prune starting at iteration densify_from_iter and then every densification_interval iterations
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        # print(opt.densify_grad_threshold)
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold,**kwargs)
                        # print(f"after prune {gaussians.get_xyz.shape}")
                    
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
                    
                if kwargs['prune_down_scheduled'] and iteration != opt.iterations and iteration in kwargs['prune_down_scheduled_iterations'] and not iteration % opt.opacity_reset_interval == 0:
                    if kwargs['enforce_hard_limit'] and iteration+500 > opt.densify_until_iter:
                        pass
                    else:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        prune_to = kwargs['prune_down_scheduled_to'][kwargs['prune_down_scheduled_iterations'].index(iteration)]
                        print(f"prune in iteration {iteration} from {gaussians.get_xyz.shape[0]} points down to prune target: {prune_to}")
                        kwargs['last_value_prune_down_scheduled'] = prune_to
                        gaussians.prune_until_number_of_points(0.005,scene.cameras_extent,size_threshold,scene.model_path,opt.iterations,prune_to)
                        print(f"Pruned to {gaussians.get_xyz.shape[0]} points")

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            its_per_second.append(1/(time.time()-start_time_iteration))

            if TAKE_OPTIMIZATION_VIDEO:
                if iteration % video_rate == 0:
                    circle_cams = scene.getCircleCameras()
                    render_pkg = render(circle_cams[optimization_sphere_cam % len(circle_cams)], gaussians, pipe, bg,**kwargs)
                    optimization_sphere_cam += 1
                    sphere_image = render_pkg["render"]
                    buf = io.BytesIO()
                    save_image(sphere_image, buf, format='png')
                    buf.seek(0)
                    img = imageio.imread(buf)
                    img_pil = Image.fromarray(img)
                    I1 = ImageDraw.Draw(img_pil)
                    I1.text((28, 36), f"Iteration {iteration}", fill=(0, 0, 0))
                    I1.text((28, 46), f"Loss {round(ema_loss_for_log,5)}", fill=(0, 0, 0))
                    I1.text((28, 56), f"Points {gaussians.get_xyz.shape[0]}", fill=(0, 0, 0))
                    optimization_video_frames.append(np.array(img_pil))
                    buf.close()
        
    pure_training_time = time.time() - pure_training_time
    with open(os.path.join(scene.model_path,"training_time.json"), 'w') as f:
        json.dump({"pure_training_time" : pure_training_time,
                "total_time" : time.time()-total_time}, f,indent=4)
        
    if TAKE_OPTIMIZATION_VIDEO:
        imageio.mimwrite(os.path.join(scene.model_path,"opt_video.mp4"), optimization_video_frames, fps=60)  # Adjust fps as needed

def prepare_output_and_logger(args):    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
        print("SummaryWriter successfully created, using Tensorboard")
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene:Scene, train_eval_cameras, renderFunc, renderArgs,evaluator: Evaluator,val_losses,val_psnrs,**kwargs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        if iteration == max(testing_iterations):
            validation_configs = ({'name': 'val', 'cameras' : scene.getValCameras()},
                                  {'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                  {'name': 'train', 'cameras' : train_eval_cameras})
        else:
            validation_configs = ({'name': 'val', 'cameras' : scene.getValCameras()}, 
                                  {'name': 'train', 'cameras' : train_eval_cameras})

        for i,config in enumerate(validation_configs):
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                if not kwargs['train_metrics']:
                    if config['name'] == "train":
                        continue
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs,**kwargs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if kwargs['init_method'] == "reconstruct_from_depth_map":
                        gt_depth_map = viewpoint.gt_depth_map.to("cuda") #  no need for clamping, das wurde schon vorher getan beim umrechnen der blender depth zu den world units
                    else:
                        gt_depth_map = None
                    try:
                        depth_map = render_pkg['depth']
                    except:
                        depth_map = None
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    this_l1_loss = l1_loss(image, gt_image).mean().double()
                    l1_test += this_l1_loss
                    this_psnr = psnr(image, gt_image).mean().double()
                    psnr_test += this_psnr
                    if kwargs['additional_metrics'] and iteration in kwargs['test_lpips_ssim_iterations']:
                        evaluator.verbose_eval(iteration,this_psnr,this_l1_loss,viewpoint.image_name,gt_image,image,config['name'],depth_map,gt_depth_map)
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                if config['name'] == "val":
                    val_losses.append(l1_test)
                    val_psnrs.append(psnr_test)
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        torch.cuda.empty_cache()
        if kwargs['additional_metrics'] and iteration in kwargs['test_lpips_ssim_iterations'] and kwargs['init_method'] == "reconstruct_from_depth_map":
            # geht grad nur für depth maps, hab was kaputt gemacht
            evaluator.summarize()


def start_training(path,output_path,white_background,**kwargs):
    kwargs.update({'prune_down' : False,
                   'prune_down_iteration_frequency' : None,
                   'prune_down_number' : None,
                   'spiral_circle': True,
                   'prune_down_scheduled_hard_upper_limit_factor' : None, # deactivate by setting to None
                   })
    assert not (kwargs['prune_down'] and kwargs['enforce_hard_limit']), "Cannot prune down and enforce hard limit at the same time"
    assert not (kwargs['prune_down'] and kwargs['prune_down_scheduled']), "Cannot prune down and prune down scheduled at the same time"
    # assert not (kwargs['enforce_hard_limit'] and kwargs['prune_down_scheduled']), "Cannot enforce_hard_limit and prune down scheduled at the same time"
    if kwargs['prune_down']:
        assert kwargs['densify_and_prune'], "Cant prune down if densify & prune deactivated"
    if kwargs['enforce_hard_limit']:
        assert kwargs['densify_and_prune'], "Cant enforce hard limit when densify & prune deactivated"
    if kwargs['prune_down_scheduled']:
        assert kwargs['densify_and_prune'], "Cant prune down scheduled if densify & prune deactivated"

    start_time = time.time()
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[ 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--category", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    args.port = np.random.randint(1000,7000)
    worked = False
    while not worked:
        try:
            network_gui.init(args.ip, args.port)
            worked = True
        except:
            args.port += 1
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    lp = lp.extract(args)
    lp.source_path = path

    lp.white_background = white_background
    print(f"Training on {lp.source_path}")
    lp.model_path = output_path

    opt = op.extract(args)
    for it in kwargs['test_lpips_ssim_iterations']:
        if it not in args.test_iterations: #, "Every lpips ssim iteration has to be in test iterations"
            args.test_iterations.append(it)
    pathlib.Path(output_path).mkdir(exist_ok=True,parents=True)
    with open(pathlib.Path(output_path,"config.json"), 'w') as config:
        json.dump(kwargs, config,indent=4)
    training(lp, opt , pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,**kwargs)

    # All done
    network_gui.close_socket()
    print("\nTraining complete.")
    print("This took {}s".format(time.time()-start_time))

if __name__ == "__main__":
    prune_down_scheduled_iterations = [500*i for i in range(2,20)]
    CONFIGS = {
            'iterations' : 10_200,
            'n_init_points' : 1_000,
            'additional_metrics' : True,
            'train_metrics' : False,
            'progress_bar' : True,
            'test_lpips_ssim_iterations' : [2000,4000,10000],
            'scheduler' : "exp",
            'densify_until_iter' : 5000, #5000, 15000
            'init_method' : 'reconstruct_from_depth_map' ,# 'adjusted' 'reconstruct_from_depth_map', # "random",
            'model_selector' : 'gaussian', # sphere or gaussian
            'densify_and_prune' : True,
            'rasterizer' : 'normal', # or 'normal' 'depth'
            'depth_information' : {
                'offset' : -1,
                'min' : 0,
                'size' : 0.7,
                },
            'prune_down_scheduled' : True,
            'prune_down_scheduled_iterations' : prune_down_scheduled_iterations,
            'prune_down_scheduled_to' : list(reversed([4000 for i,_ in enumerate(range(len(prune_down_scheduled_iterations)))])),
            'enforce_hard_limit' : False,
            'n_hard_upper_limit' : 8000,
        }
    
    path = "green_airplane" 
    output_path =  os.path.join("./output/", str(datetime.datetime.now()).replace(" ","").replace(":",""))
    start_training(path,output_path,white_background=True,**CONFIGS)