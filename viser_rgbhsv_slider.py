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
from utils.sh_utils import eval_sh
from gaussian_renderer import render
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scipy.interpolate import interp1d
import matplotlib

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

    xmin, xmax = torch.min(orig_gaus._xyz[:,0]).cpu(), torch.max(orig_gaus._xyz[:,0]).cpu()
    ymin, ymax = torch.min(orig_gaus._xyz[:,1]).cpu(), torch.max(orig_gaus._xyz[:,1]).cpu()
    zmin, zmax = torch.min(orig_gaus._xyz[:,2]).cpu(), torch.max(orig_gaus._xyz[:,2]).cpu()

    x_multi_slider = server.add_gui_multi_slider(
            "X Selection",
            min=xmin,
            max=xmax,
            step=0.25,
            initial_value=(xmin, xmax),
        )
    y_multi_slider = server.add_gui_multi_slider(
           "Y Selection",
            min=ymin,
            max=ymax,
            step=0.25,
            initial_value=(ymin, ymax),
        )
    z_multi_slider = server.add_gui_multi_slider(
            "Z Selection",
            min=zmin,
            max=zmax,
            step=0.25,
            initial_value=(zmin, zmax),
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

    ui_resolution = server.add_gui_slider(
        "resolution",
        min=450,
        max=2560,
        step=20,
        initial_value=920,

    )

    rgb_checkbox_disable = server.add_gui_checkbox(
        "Disable RGB Picker",
        initial_value=False,
    )
    r_multi_slider = server.add_gui_multi_slider(
            "R Selection",
            min=0,
            max=1,
            step=0.05,
            initial_value=(0, 1),
    )
    g_multi_slider = server.add_gui_multi_slider(
            "G Selection",
            min=0,
            max=1,
            step=0.05,
            initial_value=(0, 1),
    )
    b_multi_slider = server.add_gui_multi_slider(
            "B Selection",
            min=0,
            max=1,
            step=0.05,
            initial_value=(0, 1),
    )

    hsv_checkbox_disable = server.add_gui_checkbox(
        "Disable HSV Picker",
        initial_value=False,
    )
    h_multi_slider = server.add_gui_multi_slider(
            "H Selection",
            min=0,
            max=1,
            step=0.05,
            initial_value=(0, 1),
    )
    s_multi_slider = server.add_gui_multi_slider(
            "S Selection",
            min=0,
            max=1,
            step=0.05,
            initial_value=(0, 1),
    )
    v_multi_slider = server.add_gui_multi_slider(
            "V Selection",
            min=0,
            max=1,
            step=0.05,
            initial_value=(0, 1),
    )
    # print(orig_gaus._xyz.shape)
    # print(orig_gaus._features_rest.shape)
    # print(orig_gaus._features_rest.shape)
    # print(orig_gaus._opacity.shape)
    # print(orig_gaus._scaling.shape)
    # print(orig_gaus._rotation.shape)

    bg_color = [0, 0, 0]
    bg = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    max_sh_degree = 3
    active_sh_degree = 3
    prev_pos = {}
    

    with torch.no_grad():
        while True:
            for client in server.get_clients().values():
                r_multi_slider.disabled = rgb_checkbox_disable.value
                g_multi_slider.disabled = rgb_checkbox_disable.value
                b_multi_slider.disabled = rgb_checkbox_disable.value
                h_multi_slider.disabled = hsv_checkbox_disable.value
                s_multi_slider.disabled = hsv_checkbox_disable.value
                v_multi_slider.disabled = hsv_checkbox_disable.value

                camera = client.camera
                id = client.client_id

                if(id not in prev_pos):
                    prev_pos[id] = np.array([-1, -1, -1])
                    camera.position = np.zeros(3)
                    camera.wxyz = np.array([1, 0, 0, 0])

                q = camera.wxyz
                R = np.transpose(qvec2rotmat(np.array([q[0], q[2], q[1], q[3]])))
                T = np.array([-camera.position[1], -camera.position[0], -camera.position[2]])
                W = ui_resolution.value
                H = int(W/camera.aspect)
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
                xslide_min, x_slide_max = x_multi_slider.value
                yslide_min, y_slide_max = y_multi_slider.value
                zslide_min, z_slide_max = z_multi_slider.value

                x_cond = (xslide_min >= new_gaus._xyz[:, 0]) | (new_gaus._xyz[:, 0] >= x_slide_max)
                y_cond = (yslide_min >= new_gaus._xyz[:, 1]) | (new_gaus._xyz[:, 1] >= y_slide_max)
                z_cond = (zslide_min >= new_gaus._xyz[:, 2]) | (new_gaus._xyz[:, 2] >= z_slide_max)

                # Precompute colors from SHs in Python
                if (hsv_checkbox_disable.value == False) or (rgb_checkbox_disable.value == False):
                    # camera position
                    cam_pos = torch.tensor(camera.position).reshape(1,-1).to("cuda")

                    # concat color features 
                    dc_rest = torch.cat((new_gaus._features_dc, new_gaus._features_rest), dim=1)
                    # print("dc_rest.shape", dc_rest.shape) [1580788, 16, 3]

                    shs_view = dc_rest.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
                    # print("shs_view.shape", shs_view.shape) [1580788, 3, 16]

                    dir_pp = (new_gaus._xyz - cam_pos.repeat(dc_rest.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    # print("dir_pp_normalized.shape", dir_pp_normalized.shape) [1580788, 3]

                    sh2rgb = eval_sh(active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                    # print(torch.min(colors_precomp[:,0]), torch.max(colors_precomp[:,0]))
                    # print(torch.min(colors_precomp[:,1]), torch.max(colors_precomp[:,1]))
                    # print(torch.min(colors_precomp[:,2]), torch.max(colors_precomp[:,2]))

                    colors_precomp = torch.clamp(colors_precomp, min=0, max=1)   
                    rgb_precomp = colors_precomp.clone().cpu()      

                    if (hsv_checkbox_disable.value == False):
                        color_hsv = matplotlib.colors.rgb_to_hsv(rgb_precomp)
                        color_hsv = torch.tensor(color_hsv).cuda()

                        h_min, h_max = h_multi_slider.value
                        s_min, s_max = s_multi_slider.value
                        v_min, v_max = v_multi_slider.value
                        h_cond = (h_min >= color_hsv[:, 0]) | (color_hsv[:, 0] >= h_max)
                        s_cond = (s_min >= color_hsv[:, 1]) | (color_hsv[:, 1] >= s_max)
                        v_cond = (v_min >= color_hsv[:, 2]) | (color_hsv[:, 2] >= v_max)

                    if (rgb_checkbox_disable.value == False):
                        color_255 = torch.tensor(rgb_precomp).cuda()

                        r_min, r_max = r_multi_slider.value
                        g_min, g_max = g_multi_slider.value
                        b_min, b_max = b_multi_slider.value
                        r_cond = (r_min >= color_255[:, 0]) | (color_255[:, 0] >= r_max)
                        g_cond = (g_min >= color_255[:, 1]) | (color_255[:, 1] >= g_max)
                        b_cond = (b_min >= color_255[:, 2]) | (color_255[:, 2] >= b_max)


                    if (hsv_checkbox_disable.value == False) and (rgb_checkbox_disable.value == False):
                        # print("RGB on, HSV on")
                        mask = torch.where( (x_cond|y_cond|z_cond|r_cond|g_cond|b_cond|h_cond|s_cond|v_cond), True, False)

                    elif (hsv_checkbox_disable.value == True) and (rgb_checkbox_disable.value == False):
                        # print("RGB on, HSV off")
                        mask = torch.where( (x_cond|y_cond|z_cond|r_cond|g_cond|b_cond), True, False)

                    elif (hsv_checkbox_disable.value == False) and (rgb_checkbox_disable.value == True):
                        # print("RGB off, HSV on")
                        mask = torch.where( (x_cond|y_cond|z_cond|h_cond|s_cond|v_cond), True, False)
                    else: pass

                else: 
                    # print("RGB off, HSV off")
                    mask = torch.where( (x_cond|y_cond|z_cond), True, False)

                new_gaus.viser_prune_points(mask) # input = mask to be removed

                image = render(view, new_gaus, pipe, bg)["render"]
                # image = torch.flip(image, dims=[1])
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

# python viser_rgbhsv_slider.py --ply point_cloud.ply
