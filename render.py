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
import matplotlib.pyplot as plt

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.gaussian_sphere_model import GaussianSphereModel
from scene.gaussian_model import GaussianModel
import cv2
import imageio.v2 as imageio
import json
import pathlib

def load_config(path):
    with open(pathlib.Path(path,"config.json"), "r") as f:
        return json.load(f)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background,**kwargs):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background,**kwargs)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    # combine to video
    height = views[0].image_height
    width =views[0].image_width
    writer = cv2.VideoWriter(os.path.join(render_path,"..","video.avi"),
                            cv2.VideoWriter_fourcc(*"MJPG"), fps=60,frameSize=(height,width))
    for img in list(sorted(os.listdir(render_path))):
        if not ".png" in img:
            continue
        writer.write(cv2.cvtColor(imageio.imread(os.path.join(render_path,img)), cv2.COLOR_RGB2BGR))
    writer.release()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        print(dataset.model_path)
        print(dataset.source_path)
        
        kwargs = load_config(dataset.model_path)

        if kwargs['model_selector'] == "sphere":
            gaussians = GaussianSphereModel(dataset.sh_degree) # sh_degree is int: spherical harmonics
        elif kwargs['model_selector'] == "gaussian":
            gaussians = GaussianModel(dataset.sh_degree) # sh_degree is int: spherical harmonics
        kwargs['spiral_circle'] = True
        
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False,**kwargs)

        # print(sum(gaussians._rotation)/len(gaussians._rotation))
        print("rendering with white background ",dataset.white_background)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        circle_cameras = scene.getCircleCameras()
        circle_cams_dict = {}
        for i,cam in enumerate(circle_cameras):
            circle_cams_dict[i] = {
                'FoVx' : cam.FoVx,
                'FoVy' : cam.FoVy,
                'world_view_transform' : cam.world_view_transform.cpu().numpy().tolist(),
                'projection_matrix' : cam.projection_matrix.cpu().numpy().tolist(),
            }
            
        with open("circle_cams.json","w") as f:
            json.dump(circle_cams_dict,f)


        render_set(dataset.model_path, "circle", scene.loaded_iter, scene.getCircleCameras(), gaussians, pipeline, background,**kwargs)
        render_set(dataset.model_path, "eval", scene.loaded_iter, scene.getValCameras(), gaussians, pipeline, background,**kwargs)
        render_set(dataset.model_path, "spiral", scene.loaded_iter, scene.getSpiralCameras(), gaussians, pipeline, background,**kwargs)
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background,**kwargs)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background,**kwargs)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)