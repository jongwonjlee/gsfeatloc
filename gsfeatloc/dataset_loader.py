# =============================================================================
# Project       : GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting
# File          : dataset_loader.py
# Description   : This is the file loading either 
#                 (1) the Blender-formatted dataset or 
#                 (2) the COLMAP-formatted dataset.
# 
# Author        : Jongwon Lee (jongwon5@illinois.edu)
# Year          : 2025
# License       : BSD License
# =============================================================================
import numpy as np
import json
from pathlib import Path
import pycolmap


def load_frames_blender(dataset_path: str) -> tuple[np.ndarray, int, int, dict[str, np.ndarray]]:
    """
    Load all frames from the Blender dataset.

    Args:
        dataset_path (str): The path to the Blender dataset.

    Returns:
        tuple: A tuple containing an array of camera parameters, width, height, and a dictionary with filenames as keys and transformation matrices as values.
    """

    # Load the Blender dataset
    with open(dataset_path, "r") as f:
        test_data = json.load(f)

    # If dataset_path doesn't include the string "blender", raise a warning that calling this function would result in hard-coded camera parameters.
    if "blender" not in str(dataset_path):
        raise Warning("The dataset path does not contain 'blender'. This function is designed for the Blender dataset, and camera parameters are hard-coded here.")

    # In the Blender dataset, camera parameters are hard-coded here.
    # Construct camera parameters for rendering
    w = h = 800
    fx = fy = 0.5 * w / np.tan(0.5 * float(test_data["camera_angle_x"]))
    cx = cy = 0.5 * w

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    frames = {}
    # Iterate through all frames in the dataset
    for index, frame in enumerate(test_data["frames"]):
        # Get the filename of the image
        # Strip both the directory and the file extension if present
        filename = frame["file_path"]
        filename = filename.split("/")[-1]  # Get the last part of the path
        if '.' in filename:
            filename = filename.split(".")[0]   # Remove the file extension

        # Get the pose of the camera in world coordinates
        T_inW_ofC = np.array(frame["transform_matrix"], dtype=np.float32)

        # Convert OpenGL coordinate system to OpenCV by rotating 180 degrees around the x-axis
        T_inW_ofC[0:3, 1:3] *= -1

        # Store the transformation matrix with the filename as the key
        frames[filename] = T_inW_ofC

    return K, w, h, frames


def load_frames_colmap(dataset_path: str, downscale_factor : int = 1) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Load all frames from the COLMAP format.

    Args:
        dataset_path (str): The path to the COLMAP. (This should be the parent path to the sparse directory.)
        downscale_factor (int): The factor by which to downscale the images. Default is 1 (no downscaling).

    Returns:
        tuple: A tuple containing an array of camera parameters, width, height, and a dictionary with filenames as keys and transformation matrices as values.
    """

    # Load the COLMAP reconstruction
    if not dataset_path.exists():
        raise FileNotFoundError(f"No COLMAP reconstruction found in {dataset_path}. Please provide a valid path to the dataset.")
    
    reconstruction = pycolmap.Reconstruction(dataset_path)

    # Load camera parameters
    assert len(reconstruction.cameras) == 1, "There should be only one camera"

    camera = list(reconstruction.cameras.values())[0]

    width = camera.width
    height = camera.height

    fx = fy = camera.params[0]  # Focal length is the same for both x and y by default
    cx, cy = width / 2, height / 2  # Default principal point
    k1, k2, p1, p2 = 0, 0, 0, 0  # Default distortion coefficients

    # Handle different camera models
    if camera.model == pycolmap._core.CameraModelId.SIMPLE_PINHOLE:
        cx, cy = camera.params[1:]
    elif camera.model == pycolmap._core.CameraModelId.PINHOLE:
        fy, cx, cy = camera.params[1:]
    elif camera.model == pycolmap._core.CameraModelId.SIMPLE_RADIAL:
        cx, cy, k1 = camera.params[1:]
    elif camera.model == pycolmap._core.CameraModelId.RADIAL:
        cx, cy , k1, k2 = camera.params[1:]
    elif camera.model == pycolmap._core.CameraModelId.OPENCV:
        fy, cx, cy, k1, k2, p1, p2 = camera.params[1:]
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")
    
    # If downscale_factor > 1, we need to adjust the intrinsic matrix accordingly
    if downscale_factor > 1:
        width //= downscale_factor
        height //= downscale_factor
        fx /= downscale_factor
        fy /= downscale_factor
        cx /= downscale_factor
        cy /= downscale_factor

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    frames = {}
    # Iterate through all frames in the dataset
    # Iterate through the images in the reconstruction
    for image_id, image in reconstruction.images.items():
        # Get the filename of the image
        filename = image.name
        # Strip both the directory and the file extension if present
        filename = filename.split("/")[-1]  # Get the last part of the path
        if '.' in filename:
            filename = filename.split(".")[0]   # Remove the file extension
        
        # Get the pose of the world in camera coordinates
        T_inC_ofW = np.vstack([image.cam_from_world.matrix(), [0., 0., 0., 1.]], dtype=np.float32)

        # Get the pose of the camera in world coordinates
        T_inW_ofC = np.linalg.inv(T_inC_ofW)

        # Store the transformation matrix with the filename as the key
        frames[filename] = T_inW_ofC

    return K, width, height, frames
    

if __name__ == "__main__":
    # Load all frames from the Blender dataset
    blender_path = Path("/home/jongwonlee/datasets/nerfbaselines/blender/lego/transforms_train.json")
    K_blender, w_blender, h_blender, frames_blender = load_frames_blender(blender_path)

    print("\nLoad Blender Dataset:")
    print(f"Total number of frames: {len(frames_blender)}")
    print(f"Intrinsic Matrix (K):\n{K_blender}")
    print(f"Image Size: {w_blender} x {h_blender}")

    # Load all frames from the COLMAP reconstruction
    colmap_path = Path("/home/jongwonlee/models/colmap/blender/lego/sparse/0")
    K_colmap, w_colmap, h_colmap, frames_colmap = load_frames_colmap(colmap_path)

    print("\nLoad COLMAP Reconstruction:")
    print(f"Total number of frames: {len(frames_colmap)}")
    print(f"Intrinsic Matrix (K):\n{K_colmap}")
    print(f"Image Size: {w_colmap} x {h_colmap}")

    assert len(frames_blender) == len(frames_colmap), "The number of frames in Blender and COLMAP should be the same."

    # Let's choose a few random frames to display
    import random

    filenames = random.sample(list(frames_blender.keys()), min(5, len(frames_blender)))  # Choose up to 5 random frames

    print("\nRandomly Selected Frames:")
    for filename in filenames:
        print(f"Frame: {filename}")
        print(f"Blender Transformation Matrix (T_inW_ofC):\n{frames_blender[filename]}")
        print(f"COLMAP Transformation Matrix (T_inW_ofC):\n{frames_colmap[f'train_{filename}']}")
        print()