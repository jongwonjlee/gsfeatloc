# =============================================================================
# Project       : GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting
# File          : visualizer.py
# Description   : This is the file performing visualization of feature correspondence results.
# 
# Author        : Jongwon Lee (jongwon5@illinois.edu)
# Year          : 2025
# License       : BSD License
# =============================================================================
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def visualize_matches(im1: np.ndarray, im2: np.ndarray, kp1: list[cv.KeyPoint], kp2: list[cv.KeyPoint], matches: list[cv.DMatch]) -> None:
    """
    Visualize the matches between two images.

    Args:
        im1 (np.ndarray): The first input image.
        im2 (np.ndarray): The second input image.
        kp1 (list[cv.KeyPoint]): List of keypoints from the first image.
        kp2 (list[cv.KeyPoint]): List of keypoints from the second image.
        matches (list[cv.DMatch]): List of matches.

    Returns:
        im_matches (np.ndarray): The image with matches drawn.
    """
    # Draw matches
    im_matches = cv.drawMatches(im1, kp1, im2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(im_matches)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches(8, 4)
    plt.tight_layout()
    plt.show()

    return im_matches


def visualize_feature_points(im1, im2, pts1, pts2, inliers):
    """
    Visualize the feature points on two images.
    
    Args:
        im1 (np.array): The first input image.
        im2 (np.array): The second input image.
        pts1 (np.ndarray): 2D coordinates of the feature points on the first image.
        pts2 (np.ndarray): 2D coordinates of the feature points on the second image.
        inliers (list[int]): List of inlier indices.
    
    Returns:
        None
    """
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(im1)
    ax[0].set_title("Feature Points on Query Image")
    ax[0].axis("off")
    
    ax[1].imshow(im2)
    ax[1].set_title("Feature Points on Reference Image")
    ax[1].axis("off")
    
    for i, pt1 in enumerate(pts1):
        color = "red" if i in inliers else "pink"
        scatter_color = "blue" if i in inliers else "skyblue"
        
        ax[0].annotate(str(i), (pt1[0], pt1[1]), color=color)
        ax[1].annotate(str(i), (pts2[i][0], pts2[i][1]), color=color)
        
        ax[0].scatter(pt1[0], pt1[1], color=scatter_color, marker="x")
        ax[1].scatter(pts2[i][0], pts2[i][1], color=scatter_color, marker="x")

    plt.tight_layout()
    plt.show()


def visualize_3d_points(pts3d: np.ndarray) -> None:
    """
    Visualize 3D points.
    
    Args:
        pts3d (np.ndarray): 3D points in camera coordinates.
    Returns:
        None
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2])
    
    for i, pt3d in enumerate(pts3d):
        ax.text(pt3d[0], pt3d[1], pt3d[2], str(i), color="red")
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set the view direction to be x-axis left, y-axis down, z-axis inword
    # ax.view_init(elev=90, azim=90)
    # Flip the y-axis
    # ax.set_xlim(ax.get_xlim()[::-1])

    plt.title("3D Coordinates of Feature Points in Camera Coordinates")
    plt.tight_layout()
    plt.show()
