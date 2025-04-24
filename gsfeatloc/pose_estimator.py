# =============================================================================
# Project       : GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting
# File          : pose_estimator.py
# Description   : This is the file performs the pose estimation using 2D-3D correspondences.
# 
# Author        : Jongwon Lee (jongwon5@illinois.edu)
# Year          : 2025
# License       : BSD License
# =============================================================================
import numpy as np
import cv2 as cv


def get_2d_feature_points(
        matches : list[cv.DMatch], 
        kp1 : list[cv.KeyPoint], 
        kp2 : list[cv.KeyPoint]
) -> tuple[np.array, np.array, list[cv.KeyPoint], list[cv.KeyPoint]]:
    """
    Get the 2D coordinates of the feature points on the two images.
    
    Args:
        matches (list[cv.DMatch]): List of matches.
        kp1 (list[cv.KeyPoint]): List of feature points from the first image.
        kp2 (list[cv.KeyPoint]): List of feature points from the second image.
    
    Returns:
        tuple: A tuple containing the 2D coordinates of the feature points on the two images.
    """
    pts1_cv = [kp1[m.queryIdx] for m in matches]
    pts1 = np.array([[int(pt_cv.pt[0]), int(pt_cv.pt[1])] for pt_cv in pts1_cv], dtype=np.float32)
    
    pts2_cv = [kp2[m.trainIdx] for m in matches]
    pts2 = np.array([[int(pt_cv.pt[0]), int(pt_cv.pt[1])] for pt_cv in pts2_cv], dtype=np.float32)
    
    return pts1, pts2, pts1_cv, pts2_cv


def compute_3d_points(
        pts2d: np.ndarray, 
        im_depth: np.ndarray, 
        K: np.ndarray, 
        T_inW_ofC: np.ndarray, 
        depth_scale: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 3D coordinates of the feature points in the camera and world coordinates.
    
    Args:
        pts2d (np.ndarray): 2D coordinates of the feature points.
        im_depth (np.ndarray): Depth map of the image.
        K (np.ndarray): Camera intrinsics matrix.
        T_inC_ofW (np.ndarray): Camera pose in world coordinates.
        depth_scale (float): Depth scale.
    
    Returns:
        tuple: A tuple containing the 3D coordinates of the feature points in the world coordinates and the camera coordinates along with the valid mask.
    """
    assert pts2d.shape[1] == 2
    assert im_depth.ndim == 2
    assert K.shape == (3, 3)
    assert T_inW_ofC.shape == (4, 4)

    # Get depth values from the reference depth map
    depths = np.array([im_depth[int(pt[1]), int(pt[0])] for pt in pts2d], dtype=np.float32)

    # Filter out invalid depth values and scale
    valid_mask = depths > 1e-2  # Valid depths only
    depths_filtered = depths * depth_scale
    pts2d_filtered = pts2d
    
    depths_filtered[~valid_mask] = np.nan
    pts2d_filtered[~valid_mask] = np.nan

    # Compute 3D points in the camera frame
    Kinv = np.linalg.inv(K)
    uv1 = np.hstack([pts2d_filtered, np.ones((pts2d_filtered.shape[0], 1), dtype=np.float32)])
    p_inC = (Kinv @ uv1.T).T * depths_filtered[:, None]

    # Convert to world coordinates
    p_inC_homo = np.hstack([p_inC, np.ones((p_inC.shape[0], 1), dtype=np.float32)])
    p_inW = (T_inW_ofC @ p_inC_homo.T).T[:, :3]

    return p_inW, p_inC


def estimate_camera_pose(
        p_inW : np.ndarray, 
        pts2d : np.ndarray, 
        K : np.ndarray, 
        reprojection_error : float = 5.0, 
        iterations : int = 1000
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Estimate the camera pose using solvePnPRansac.
    
    Args:
        p_inW (np.ndarray): 3D coordinates of the feature points in the world coordinates.
        pts2d (np.ndarray): 2D coordinates of the feature points.
        K (np.ndarray): Camera intrinsics matrix.
        reprojection_error (float): Reprojection error threshold.
        iterations (int): Number of iterations.
    
    Returns:
        tuple: A tuple containing the rotation vector, translation vector, and the inliers.
    """
    assert p_inW.shape[1] == 3
    assert pts2d.shape[1] == 2
    assert K.shape == (3, 3)

    # Perform pose estimation using solvePnPRansac
    success, rvec, tvec, inliers = cv.solvePnPRansac(
        objectPoints=p_inW,
        imagePoints=pts2d,
        cameraMatrix=K,
        distCoeffs=None,
        flags=cv.SOLVEPNP_ITERATIVE,
        reprojectionError=reprojection_error,
        iterationsCount=iterations
    )

    # Check if pose estimation was successful
    if not success or inliers is None or len(inliers) == 0:
        print("PnP failed to estimate a valid pose or no inliers were found.")
        return None, None, None

    print(f"Pose estimation successful. Number of inliers: {len(inliers.flatten())} / {len(pts2d)}")
    return rvec, tvec, inliers