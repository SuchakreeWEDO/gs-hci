"""COLMAP visualizer

Visualize COLMAP sparse reconstruction outputs. To get demo data, see `./assets/download_colmap_garden.sh`.
"""

import random
import time
import sys
from pathlib import Path

import copy
import numpy as np
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


def main(pipe, opt, ply_path) -> None:
    server = viser.ViserServer()

    orig_gaus = GaussianModel(sh_degree = 3)
    orig_gaus.training_setup(opt)
    orig_gaus.viser_load_ply(ply_path)

    xmean = torch.mean(orig_gaus._xyz[:,0]).cpu()
    ymean = torch.mean(orig_gaus._xyz[:,1]).cpu()
    zmean = torch.mean(orig_gaus._xyz[:,2]).cpu()
    xmin, xmax = torch.min(orig_gaus._xyz[:,0]).cpu(), torch.max(orig_gaus._xyz[:,0]).cpu()
    ymin, ymax = torch.min(orig_gaus._xyz[:,1]).cpu(), torch.max(orig_gaus._xyz[:,1]).cpu()
    zmin, zmax = torch.min(orig_gaus._xyz[:,2]).cpu(), torch.max(orig_gaus._xyz[:,2]).cpu()

    box_position = server.add_gui_vector3(
            "Box Position",
            initial_value=(xmean, ymean, zmean),
            step=0.25,
        )
    
    box_size = server.add_gui_vector3(
            "Box Size",
            initial_value=(10.0, 10.0, 10.0),
            step=0.25,
        )

    gui_reset_up = server.add_gui_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )

    gui_save_ply = server.add_gui_button(
        "Export .ply",
    )

    # print(orig_gaus._xyz.shape)
    # print(orig_gaus._features_rest.shape)
    # print(orig_gaus._features_rest.shape)
    # print(orig_gaus._opacity.shape)
    # print(orig_gaus._scaling.shape)
    # print(orig_gaus._rotation.shape)

    bg_color = [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        while True:
            for client in server.get_clients().values():
                camera = client.camera
                R = qvec2rotmat(camera.wxyz)
                T = camera.position
                W = 1920
                H = int(1920/camera.aspect)
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

                # BBOX Masking
                new_gaus = copy.copy(orig_gaus)
                sizex, sizey, sizez = box_size.value
                posx, posy, posz = box_position.value
                x_cond = (posx - (sizex/2) >= new_gaus._xyz[:, 0]) | (new_gaus._xyz[:, 0] >= posx + (sizex/2))
                y_cond = (posy - (sizey/2) >= new_gaus._xyz[:, 1]) | (new_gaus._xyz[:, 1] >= posy + (sizey/2))
                z_cond = (posz - (sizez/2) >= new_gaus._xyz[:, 2]) | (new_gaus._xyz[:, 2] >= posz + (sizez/2))
                mask = torch.where( (x_cond|y_cond|z_cond), True, False)
                new_gaus.viser_prune_points(mask) # input = mask to be removed

                image = render(view, new_gaus, pipe, bg)["render"]
                image = torch.flip(image, dims=[2])
                image_nd = image.detach().cpu().numpy().astype(np.float32)
                image_nd = np.transpose(image_nd, (2, 1, 0))
                
                client.set_background_image(image_nd, format="jpeg")

                @gui_save_ply.on_click
                def _(event: viser.GuiEvent) -> None:
                    client = event.client
                    assert client is not None
                    new_gaus.save_ply(f"./saved_point_cloud_{time.time()}.ply")

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--ply", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    main( pp.extract(args),  op.extract(args), args.ply)

# python view_server_crop.py --ply point_cloud.ply
