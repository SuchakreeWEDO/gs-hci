import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, local_pearson_loss
from gaussian_renderer import network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
import numpy as np
from PIL import Image, ImageOps
from utils.general_utils import PILtoTorch
import tkinter as tk

from datetime import datetime
def test_diff(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, auto_checkpoint):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)



    cameras = scene.getTrainCameras().copy()
    print("total view: ", len(cameras))
    viewpoint_cam = cameras[5] 
    print("method:", dataset.method)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    render_pkg = None
    image = None
    depth_map = None
    d_mode = None

    if(dataset.method == "gsdm"):
        from diff_gaussian_rasterization_depth_dmode import GaussianRasterizationSettings, GaussianRasterizer
        from gaussian_renderer import custom
        render_pkg = custom.render(GaussianRasterizationSettings, GaussianRasterizer, viewpoint_cam, gaussians, pipe, bg)
        image, depth_map, d_mode = render_pkg["render"], render_pkg["depth_map"], render_pkg["d_mode"]
    else:
        from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer
        from gaussian_renderer import render
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, depth_map = render_pkg["render"], render_pkg["depth_map"]

    print("image:",image.shape)
    print("depth:", depth_map.shape)
    if(dataset.method == "gsdm"):
        print("d_mode:", d_mode.shape)
        # d_mode = 1 - ( ( d_mode - torch.min(d_mode) ) / ( torch.max(d_mode) - torch.min(d_mode) ) )
        d_mode = d_mode.unsqueeze(0).repeat(3, 1, 1)
        showImage(d_mode)
    else:
        depth_map = 1 - ( ( depth_map - torch.min(depth_map) ) / ( torch.max(depth_map) - torch.min(depth_map) ) )
        depth_map = depth_map.unsqueeze(0).repeat(3, 1, 1)
        showImage(depth_map)

def showImage(image):
    image_cpu = image.cpu().detach()
    numpy_image = image_cpu.permute(1, 2, 0).numpy()
    numpy_image = (numpy_image * 255).astype('uint8')
    pil_image = Image.fromarray(numpy_image)
    pil_image.show()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 3_000, 5_000, 7_000, 10_000, 13_000, 16_000, 20_000, 25_000, 30_000]) # tensorboard save eval
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 5_000, 7_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--auto_checkpoint', action='store_true', default=True)

    args = parser.parse_args(sys.argv[1:])
    args.model_name = os.path.splitext(os.path.basename(args.source_path))[0]
    print("Optimizing " + args.model_name)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    test_diff(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.auto_checkpoint)