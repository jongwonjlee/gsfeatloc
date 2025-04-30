# =============================================================================
# Project       : GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting
# File          : gsfeatloc_cli.py
# Description   : This is the command line interface of GSFeatLoc,
#                 used when an initial pose guess is given,
#                 and both the dataset and 3D Gaussian Splatting (gsplat) scene are provided.
# 
# Author        : Jongwon Lee (jongwon5@illinois.edu)
# Year          : 2025
# License       : BSD License
# =============================================================================
import os
import numpy as np
import argparse
import json
from pathlib import Path
import PIL.Image as Image
import time
import pycolmap
import signal
import sys

from scipy.spatial.transform import Rotation as R

# Import the utils module from the gsfeatloc package
from gsfeatloc.utils import get_world_frame_difference, perturb_SE3
from gsfeatloc.visualizer import visualize_matches, visualize_feature_points, visualize_3d_points
from gsfeatloc.feature_matcher import do_feature_matching_SIFT, do_feature_matching_SPSG, do_feature_matching_LoFTR
from gsfeatloc.dataset_loader import load_frames_blender, load_frames_colmap
from gsfeatloc.pose_estimator import get_2d_feature_points, compute_3d_points, estimate_camera_pose

from contextlib import ExitStack
from nerfbaselines import load_checkpoint, new_cameras, camera_model_to_int

# Define the stack as a global variable
stack = None

def main():
    global stack  # Declare stack as global to modify it
    # Set ArgumentParser
    parser = argparse.ArgumentParser(description="iComMa runner for benchmarking visual localization")

    # Add arguments for the script
    parser.add_argument("--scene_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the gsplat model checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save JSON results")
    parser.add_argument("--output_filename", type=str, default="results.json", help="Filename for saving results")
    parser.add_argument("--delta_rot", type=float, default=30, help="Perturbation in initial guess for rotation")
    parser.add_argument("--delta_trs", type=float, default=0.2, help="Perturbation in initial guess for translation")
    parser.add_argument("--axis_mode", type=str, default="random", choices=["random", "x", "y"])
    parser.add_argument("--magnitude_mode", type=str, default="fixed", choices=["fixed", "uniform", "gaussian"])

    # Parse arguments before setting defaults
    args = parser.parse_args()
    
    rng = np.random.default_rng(0)

    # Set output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Start the docker backend and load the checkpoint
    stack = ExitStack().__enter__()
    model, _ = stack.enter_context(load_checkpoint(args.model_path, backend="docker"))
    
    # Determine the source of test data
    transforms_test_path = Path(args.scene_path, "transforms_test.json")
    colmap_sparse_path = Path(args.scene_path, "sparse", "0")
    test_list_path = Path(args.scene_path, "test_list.txt")

    if colmap_sparse_path.exists():
        print(f"Using test data from COLMAP reconstruction in {colmap_sparse_path}")
        
        with open(f"{args.model_path}/nb-info.json", "r") as f:
            metadata = json.load(f)
        K, w, h, frames = load_frames_colmap(colmap_sparse_path, metadata["dataset_metadata"]['downscale_factor'])

        if not test_list_path.exists():
            raise FileNotFoundError(f"test_list.txt not found in {args.scene_path}. Please provide a valid path to the dataset.")

        # Load and filter frames based on the test list
        with open(test_list_path, "r") as f:
            test_list = f.read().splitlines()
            test_list_without_ext = [os.path.splitext(filename)[0] for filename in test_list]
        
        frames = {filename: frames[filename] for filename in test_list_without_ext if filename in frames}
        print(f"Total number of frames: {len(frames)}")
    
    elif transforms_test_path.exists():
        print(f"Using test data from {transforms_test_path}")
        K, w, h, frames = load_frames_blender(transforms_test_path)
        print(f"Total number of frames: {len(frames)}")
    
    else:
        raise FileNotFoundError(f"No test data found in {args.scene_path}. Please provide a valid path to the dataset.")

    results = []
    json_path = os.path.join(args.output_path, args.output_filename)

    for i, (filename, T_inW_ofC) in enumerate(frames.items()):
        # Determine the image path format based on the dataset type
        if "mipnerf360" in args.scene_path.lower():
            image_path = Path(args.scene_path, "images_4", f"{filename}.JPG")
        elif "tanksandtemples" in args.scene_path.lower():
            image_path = Path(args.scene_path, "images_2", f"{filename}.jpg")
        elif "blender" in args.scene_path.lower():
            image_path = Path(args.scene_path, "test", f"{filename}.png")
            
        # Load the frame
        im_query = np.array(Image.open(image_path))

        # Make sure the file is loaded correctly
        assert im_query is not None, "Query image not loaded correctly"

        # Perturb the camera pose
        T_inW_ofC_perturbated = perturb_SE3(T_inW_ofC, 
                                            rotation_magnitude=args.delta_rot, translation_magnitude=args.delta_trs, 
                                            axis_mode=args.axis_mode, magnitude_mode=args.magnitude_mode, 
                                            rng=rng)

        tic = time.time()

        # Create a camera object
        camera = new_cameras(
            poses=T_inW_ofC_perturbated,
            intrinsics=np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float32),
            image_sizes=np.array([w, h], dtype=np.int32),
            camera_models=np.array(camera_model_to_int("pinhole"), dtype=np.int32),
        )
        
        # Render the image
        outputs = model.render(camera=camera, options={"outputs": "depth", "output_type_dtypes": {'color': 'uint8', 'depth': 'float32'}})

        im_reference = outputs["color"]
        depth_reference = outputs["depth"]

        # Make sure the image is rendered correctly
        assert im_reference is not None, "Reference image not rendered correctly"
        assert depth_reference is not None, "Reference depth not rendered correctly"

        # Perform feature matching between the query image and the rendered image
        pts_query_raw_cv, pts_reference_raw_cv, matches = do_feature_matching_SPSG(im_query, im_reference, do_visualize=False)
        pts_query, pts_reference, pts_query_cv, pts_reference_cv = get_2d_feature_points(matches, pts_query_raw_cv, pts_reference_raw_cv)

        # Check all the shapes
        assert pts_query.shape[0] == pts_reference.shape[0], "Number of points do not match"
        assert len(pts_query_cv) == len(pts_reference_cv) == len(matches), "Number of points do not match"

        # Ensure there are enough matches for pose estimation
        if len(matches) >= 4:
            # Compute 3D points in the world and camera frames
            p_inW, p_inC = compute_3d_points(
            pts_reference, depth_reference, K, T_inW_ofC_perturbated, depth_scale=4.0
            )
            
            # Estimate the camera pose using solvePnPRansac
            rvec, tvec, inliers = estimate_camera_pose(
            p_inW, pts_query, K, reprojection_error=5.0, iterations=50
            )

            if inliers is not None and len(inliers) > 0:
                # Construct the estimated transformation matrix
                T_inC_ofW_estimated = np.eye(4, dtype=np.float32)
                T_inC_ofW_estimated[:3, :3] = R.from_rotvec(rvec.flatten()).as_matrix()
                T_inC_ofW_estimated[:3, 3] = tvec.flatten()

                # Compute the inverse to get the camera-to-world transformation
                T_inW_ofC_estimated = np.linalg.inv(T_inC_ofW_estimated)
            else:
                # Fallback to the perturbated pose if pose estimation fails
                print(f"Pose estimation failed for frame {filename}. Using perturbated pose.")
                T_inW_ofC_estimated = T_inW_ofC_perturbated
        else:
            print(f"Not enough matches for frame {filename}. Skipping this frame.")
            T_inW_ofC_estimated = T_inW_ofC_perturbated


        toc = time.time()

        elapsed_time = toc - tic

        T_gt = T_inW_ofC
        T_init = T_inW_ofC_perturbated
        T_pred = T_inW_ofC_estimated

        rot_error, trs_error = get_world_frame_difference(T_gt, T_pred)

        print(f"Frame {filename} - Rotation error: {rot_error:.2f} degrees, Translation error: {trs_error:.2f} meters, Time: {elapsed_time:.2f} seconds")

        results.append({
            "filename": filename,
            "T_gt": T_gt.tolist(),
            "T_init": T_init.tolist(),
            "T_pred": T_pred.tolist(),
            "rotation_error": float(rot_error),
            "translation_error": float(trs_error),
            "seconds_per_frame": float(elapsed_time),
            "trial": j,
        })

        # Save intermediate results to JSON after every ten frames or the last frame
        if i % 10 == 0 or i == len(frames) - 1:
            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Intermediate results saved to {json_path}")
    
    stack.close()
    

def signal_handler(sig, frame):
    print("\nTermination signal received. Cleaning up...")
    if stack is not None:
        stack.close()  # Ensure the ExitStack is closed
    sys.exit(0)


if __name__ == "__main__":
    # Register the signal handler for SIGINT
    signal.signal(signal.SIGINT, signal_handler)
    main()
