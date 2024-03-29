import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from scene.cameras import Camera
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
bg_color =  [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
    makedirs("./fb_render_ref/", exist_ok=True)

    with torch.no_grad():

        # Load Colmap output
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Load First Train Camera & Camera Constant
        first_view = scene.getTrainCameras()[0]
        R = first_view.R
        T = first_view.T
        print("R = ", first_view.R)
        print("T = ", first_view.T)
        print("camera_center = ", first_view.camera_center)
        print("active_sh_degree = ", gaussians.active_sh_degree)
        print("max_sh_degree = ", gaussians.max_sh_degree)
        FoVx = first_view.FoVx
        FoVy = first_view.FoVy
        # image_height = first_view.image_height
        # image_width  = first_view.image_width

        rendering = render(first_view, gaussians, pipeline, background)["render"]

        torchvision.utils.save_image(rendering, f"./fb_render_ref/rendered_first_img.png")
        print("render done")

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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args))

# python fb_render_ref.py -m D:\3d-reconstruction\datasets\scale_figure