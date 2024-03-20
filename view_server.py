"""COLMAP visualizer

Visualize COLMAP sparse reconstruction outputs. To get demo data, see `./assets/download_colmap_garden.sh`.
"""

import random
import time
import sys
from pathlib import Path

import viser
import viser.transforms as tf
from tqdm.auto import tqdm
from scene.cameras import Camera
import numpy as np
from scene import GaussianModel, Scene
import torch
from gaussian_renderer import render
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w

def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c


def main(dataset, opt, pipe, checkpoint) -> None:
    server = viser.ViserServer()

    gui_reset_up = server.add_gui_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ onp.array(
            [0.0, -1.0, 0.0]
        )

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if opt.random_background else background

    with torch.no_grad():
        while True:
            for client in server.get_clients().values():
                camera = client.camera
                R = qvec2rotmat(camera.wxyz)
                T = camera.position
                W = 1920
                H = int(1920/camera.aspect)
                focal_x = W/2/np.tan(camera.fov/2)
                focal_y = H/2/np.tan(camera.fov/2)
                # view = views[0]
                view = Camera(
                    colmap_id= 1,
                    R= R,
                    T= T,
                    FoVx= 1/camera.fov,
                    FoVy= camera.fov,
                    image= torch.zeros(3, H, W),
                    gt_alpha_mask=None,
                    image_name="test",
                    uid=1
                )
                print(f"fov: {camera.fov}")
                image = render(view, gaussians, pipe, bg)["render"]

                image = torch.flip(image, dims=[2])

                image_nd = image.detach().cpu().numpy().astype(np.float32)
                image_nd = np.transpose(image_nd, (2, 1, 0))
                
                client.set_background_image(image_nd, format="jpeg")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    main(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint)
