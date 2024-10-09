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
from argparse import ArgumentParser
import train
import pathlib

skip_training = False
skip_rendering = False
skip_metrics = False
output_path = "./eval"
# deepblending = "db/"

mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["db/drjohnson", "db/playroom"]
shapenet_path = pathlib.Path(pathlib.Path.home(),"Documents","NeRF","NeRF","NeRF_Implementations","renderings_fixed_onefixedspot_ambient")
shapenet_scenes = [x for cat in os.listdir(shapenet_path) for x in os.listdir(pathlib.Path(shapenet_path,cat))]
print(shapenet_scenes)

# parser = ArgumentParser(description="Full evaluation script parameters")
# parser.add_argument("--skip_training", action="store_true")
# parser.add_argument("--skip_rendering", action="store_true")
# parser.add_argument("--skip_metrics", action="store_true")
# parser.add_argument("--output_path", default="./eval")
# args, _ = parser.parse_known_args()

all_scenes = []
# all_scenes.extend(mipnerf360_outdoor_scenes)
# all_scenes.extend(mipnerf360_indoor_scenes)
# all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

# if skip_training or not skip_rendering:
#     # parser.add_argument('--mipnerf360', "-m360", required=True, type=str)
#     # parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
#     parser.add_argument("--deepblending", "-db", required=True, type=str)
#     print("here")
#     args = parser.parse_args()
if not skip_training:
    common_args = " --quiet --eval --test_iterations -1 "
    # for scene in mipnerf360_outdoor_scenes:
    #     source = args.mipnerf360 + "/" + scene
    #     os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args) # -i bezieht sich auf ._images -> images in modelParams, wird aber nur für colmap szenen gebraucht
    # for scene in mipnerf360_indoor_scenes:
    #     source = args.mipnerf360 + "/" + scene
    #     os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args)
    # for scene in tanks_and_temples_scenes:
    #     source = args.tanksandtemples + "/" + scene
    #     output_path = os.path.join(args.output_path,scene)
    #     train.start_training(source,output_path,False)
        # os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    for scene in deep_blending_scenes:
        source = scene
        destination = os.path.join(output_path,scene)
        print("source ",source)
        print("destination ", destination)
        train.start_training(source,destination,False)
        # os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not skip_rendering:

    all_sources = []
    # for scene in mipnerf360_outdoor_scenes:
    #     all_sources.append(args.mipnerf360 + "/" + scene)
    # for scene in mipnerf360_indoor_scenes:
    #     all_sources.append(args.mipnerf360 + "/" + scene)
    # for scene in tanks_and_temples_scenes:
    #     all_sources.append(args.tanksandtemples + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(scene)
    print(all_scenes)
    print(all_sources)

    common_args = " --quiet --eval --skip_train"
    print("rendering")
    for scene, source in zip(all_scenes, all_sources):
        print("-s " ,source)
        print("-m ", output_path + "/" + scene)
        # os.system("python render.py --iteration 7000 -m " + output_path + "/" + scene + common_args)
        # os.system("python render.py --iteration 30000 -m " + output_path + "/" + scene + common_args)
        # os.system("python render.py --iteration 7000 -s " + source + " -m " + output_path + "/" + scene + common_args) # glaub das -s braucht man gar nicht, wo kommt das in render.py vor?
        # os.system("python render.py --iteration 30000 -s " + source + " -m " + output_path + "/" + scene + common_args) 
if not skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + output_path + "/" + scene + "\" "
    print(scenes_string)
    # os.system("python metrics.py -m " + scenes_string)