"""COLMAP visualizer

Visualize COLMAP sparse reconstruction outputs. To get demo data, see `./assets/download_colmap_garden.sh`.
"""

import random
import time
from pathlib import Path
import os

import imageio.v3 as iio
import numpy as onp
import tyro
import viser
import viser.transforms as tf
from tqdm.auto import tqdm
from viser.extras.colmap import (
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)


# points -> points array
# matrix -> camera transform matrix
def raycast_filter(points, matrix, size=0.9999):

    camera_direction = matrix[:3, 2]
    camera_position = matrix[:3, 3]
    points_start = points - camera_position
    points_norm = points_start/onp.linalg.norm(points_start, axis=-1)[:, onp.newaxis]

    output = onp.dot(points_norm, camera_direction) > size
    sum_filter = onp.all( output == False)
    print(sum_filter, size)

    if sum_filter:
        return raycast_filter(points, matrix, size-0.001)
    else:
        return output
    
def distance_from_camera(points, camera_position):
    distances = onp.linalg.norm(points - camera_position, axis=1)
    nearest_index = onp.argmin(distances)
    nearest_distance = distances[nearest_index]
    nearest_position = points[nearest_index]
    
    return nearest_distance, nearest_position

def main(
    colmap_path: str, #Path = Path(__file__).parent / "..\datasets\scale_figure\sparse/0",
    images_path: str, #Path = Path(__file__).parent / "..\datasets\scale_figure\images",
    downsample_factor: int = 2,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    server = viser.ViserServer()
    server.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load the colmap info.

    # colmap_path = "..\datasets\scale_figure\sparse/0"
    # images_path = "..\datasets\scale_figure\images"

    cameras = read_cameras_binary(os.path.join(colmap_path , "cameras.bin"))
    images = read_images_binary(os.path.join(colmap_path, "images.bin"))
    points3d = read_points3d_binary(os.path.join(colmap_path , "points3D.bin"))
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

    gui_points = server.add_gui_slider(
        "Max points",
        min=1,
        max=len(points3d),
        step=1,
        initial_value=min(len(points3d), 50_000),
    )
    gui_frames = server.add_gui_slider(
        "Max frames",
        min=1,
        max=len(images),
        step=1,
        initial_value=min(len(images), 100),
    )

    prev_button = server.add_gui_button("Prev Frame")
    next_button = server.add_gui_button("Next Frame")
    gui_count = server.add_gui_number("count", initial_value=0, visible=False)

    @prev_button.on_click
    def _(event: viser.GuiEvent) -> None:
        gui_count.value -= 1
        nonlocal need_update
        need_update = True
    
    @next_button.on_click
    def _(event: viser.GuiEvent) -> None:
        gui_count.value += 1
        nonlocal need_update
        need_update = True

    def visualize_colmap() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""
        # Set the point cloud.
        points = onp.array([points3d[p_id].xyz for p_id in points3d])
        colors = onp.array([points3d[p_id].rgb for p_id in points3d])
        # points_selection = onp.random.choice(
        #     points.shape[0], gui_points.value, replace=False
        # )
        # points = points[points_selection]
        # colors = colors[points_selection]

        print(points.shape)

        # Interpret the images and cameras.
        img_ids = [im.id for im in images.values()]
        print("img_ids", img_ids)

        # random.shuffle(img_ids)
        img_ids = sorted(img_ids[: gui_frames.value])
        # img_ids.sort()

        def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position
        
        img_id = img_ids[gui_count.value % len(img_ids)]
        img = images[img_id]
        cam = cameras[img.camera_id]

        print(img, cam)

        # Skip images that don't exist.
        image_filename = os.path.join( images_path , img.name)
        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(img.qvec), img.tvec
        ).inverse()
        matrix = T_world_camera.as_matrix()


        # For pinhole cameras, cam.params will be (fx, fy, cx, cy).
        if cam.model != "PINHOLE":
            print(f"Expected pinhole camera, but got {cam.model}")

        H, W = cam.height, cam.width
        fy = cam.params[1]
        image = iio.imread(image_filename)
        image = image[::downsample_factor, ::downsample_factor]

        print(f"position: {T_world_camera.translation()}, {matrix[:3, 3]}")
        
        

        filter = raycast_filter(points, matrix)
        print(filter)
        points_np = points[filter]
        colors_np = colors[filter]


        camera_direction = matrix[:3, 2]
        camera_position = matrix[:3, 3]
        

        print(points_np.shape, camera_position.shape)
        distance, nearest_point = distance_from_camera(points_np, camera_position)
        print("image_filename", image_filename)
        print("distance: ", distance)
        print("nearest_point: ", nearest_point)

        # display line
        server.add_spline_cubic_bezier(
            f"/Raycasting_line_near",
            (tuple(camera_position), tuple(nearest_point)),
            (tuple(camera_position), tuple(nearest_point)),
            line_width=3.0,
            color=(0, 255, 0),
            segments=100,
        )
        server.add_spline_cubic_bezier(
            f"/Raycasting_line",
            (tuple(nearest_point), tuple(nearest_point + camera_direction * 100)),
            (tuple(nearest_point), tuple(nearest_point + camera_direction * 100)),
            line_width=3.0,
            color=(255, 0, 0),
            segments=100,
        )
        server.add_label("distance", text=distance, position=tuple((camera_position + nearest_point)/2))

        server.add_point_cloud(
            name="/colmap/original",
            points=points,
            colors=onp.full((points.shape[0], 3), [200, 200, 200]),
            point_size=0.01,
        )
        server.add_point_cloud(
            name="/colmap/pcd",
            points=points_np,
            colors=onp.full((points_np.shape[0], 3), [0, 0, 0]),
            point_size=0.03,
        )
        frustum = server.add_camera_frustum(
            f"/colmap/frame_{img_id}/frustum",
            fov=2 * onp.arctan2(H / 2, fy),
            aspect=W / H,
            scale=0.15,
            image=image,
        )
        frame = server.add_frame(
            f"/colmap/frame_{img_id}",
            wxyz=T_world_camera.rotation().wxyz,
            position=T_world_camera.translation(),
            axes_length=0.1,
            axes_radius=0.005,
        )
        server.add_label("distance", text=f"frame_{img_id}", position=tuple(camera_position))
        attach_callback(frustum, frame)

    need_update = True

    @gui_points.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    while True:
        if need_update:
            need_update = False

            server.reset_scene()
            visualize_colmap()

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)

# python viser_direction_distance_visualizer.py --colmap_path D:\3d-reconstruction\datasets\scale_table\scale_table\colmap\sparse\0 --images_path D:\3d-reconstruction\datasets\scale_table\scale_table\colmap\images