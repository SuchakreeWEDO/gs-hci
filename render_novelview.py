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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import fb_render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import numpy as np
import time

zfar = 100.0
znear = 0.01

trans = np.array([0.0, 0.0, 0.0])
scale = 1.0

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        makedirs("./fb_rendered/", exist_ok=True)

        # camera constant
        first_view = scene.getTrainCameras()[0]
        FoVx = first_view.FoVx
        FoVy = first_view.FoVy
        image_height = first_view.image_height
        image_width  = first_view.image_width

        all_render_time = []

        for idx, view in enumerate(tqdm(scene.getTrainCameras(), desc="Rendering progress")):
            
            # world_view_transform = view.world_view_transform
            # full_proj_transform  = view.full_proj_transform
            # camera_center = view.camera_center
            # print(view.world_view_transform) # for each camera poses
            # print(view.full_proj_transform)  # for each camera poses
            # print(view.camera_center) # for each camera poses

            # calculate novel camera pose from arbitrary R and T
            # R (3,3)
            # T (3,)

            # tried to randomly created R and T matrix within range (-2,2) => resulting images are strangely distorted
            # R = 0 + 2 * np.random.rand(3,3)
            # T = 0 + 2 * np.random.rand(3,)
            # print(R.shape, T.shape)

            # for now use R and T from original the train camera views
            R = view.R
            T = view.T
            # print(R.shape, T.shape)

            # tried editing the original R and T by adding some noise => got distorted img
            # R = R + np.random.rand(3,3) * 0.5
            # T = T + np.random.rand(3) * 0.5

            start_time = time.time()
            world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            this_view = (FoVx, FoVy, image_height, image_width, world_view_transform, full_proj_transform, camera_center)

            rendering = fb_render(this_view, gaussians, pipeline, background)["render"]
            end_time = time.time()
            delta_time = end_time - start_time
            all_render_time.append(delta_time)

            torchvision.utils.save_image(rendering, f"./fb_rendered/rendered_img{idx}.png")
            print("render done")
            if idx == 10:
                print(all_render_time)
                print("Avg render time per img = ", np.mean(all_render_time))
                break

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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))