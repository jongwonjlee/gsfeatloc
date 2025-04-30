# =============================================================================
# Project       : GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting
# File          : feature_matcher.py
# Description   : This is the file performing feature matching between images using SIFT, SuperPoint, SuperGlue, and LoFTR.
# 
# Author        : Jongwon Lee (jongwon5@illinois.edu)
# Year          : 2025
# License       : BSD License
# =============================================================================
import torch
import cv2 as cv
import numpy as np

from superpoint_superglue_deployment import Matcher
from kornia.feature import LoFTR

from .visualizer import visualize_matches

# Initialize the SuperPoint and SuperGlue matcher once, globally
superglue_matcher = Matcher(
    {
        "superpoint": {
            "input_shape": (-1, -1),
            "keypoint_threshold": 0.005,  # Default threshold, can be overridden in the function
        },
        "superglue": {
            "match_threshold": 0.2,  # Default threshold, can be overridden in the function
        },
        "use_gpu": True,
    }
)

# Load the LoFTR model once, globally
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loftr_model = LoFTR(pretrained="outdoor").to(device)


def do_feature_matching_SIFT(
        im1: np.array, 
        im2: np.array, 
        ratio: float = 0.7, 
        do_visualize: bool = False
) -> tuple[list[cv.KeyPoint], list[cv.KeyPoint], list[cv.DMatch]]:
    """
    Perform feature matching using SIFT.

    Args:
        im1 (np.array): The first input image (query image).
        im2 (np.array): The second input image (reference image).
        ratio (float): Lowe's ratio for filtering good matches.
        do_visualize (bool): Whether to visualize the matches.

    Returns:
        tuple: A tuple containing the list of keypoints from the first image, the list of keypoints from the second image, and the list of matches.
    """
    # Convert images to grayscale
    im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
    im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)

    # Detect SIFT features
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1_gray, None)
    kp2, des2 = sift.detectAndCompute(im2_gray, None)

    # Check if descriptors are valid
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print("Not enough SIFT features detected for matching.")
        return list(kp1), list(kp2), []
    
    # Make sure descriptors are float32
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    # Match features using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]

    visualize_matches(im1, im2, kp1, kp2, good_matches) if do_visualize else None

    print(f"Number of total matches (SIFT): {len(matches)}")
    print(f"Number of good matches with Lowe's ratio of {ratio} (SIFT): {len(good_matches)}")

    return list(kp1), list(kp2), good_matches


# Define a function performing feature matching using SuperPoint and SuperGlue
def do_feature_matching_SPSG(
        im1: np.array, 
        im2: np.array,
        superpoint_threshold : float = 0.01, 
        superglue_threshold : float = 0.5, 
        do_visualize: bool = False
) -> tuple[list[cv.KeyPoint], list[cv.KeyPoint], list[cv.DMatch]]:
    """
    Perform feature matching using SuperPoint and SuperGlue.

    Args:
        im1 (np.array): The first input image (query image).
        im2 (np.array): The second input image (reference image).
        superpoint_threshold (float): The keypoint threshold for SuperPoint. Common values are between 0.005 and 0.015. (The high the threshold, the fewer keypoints are detected.)
        superglue_threshold (float): The match threshold for SuperGlue. Common values are between 0.2 and 0.9. (The higher the threshold, the fewer matches are detected.)
        do_visualize (bool): Whether to visualize the matches.

    Returns:
        tuple: A tuple containing the list of keypoints from the first image, the list of keypoints from the second image, and the list of matches.
    """
    # Update thresholds without reinitializing the model
    superglue_matcher._config["superpoint"]["keypoint_threshold"] = superpoint_threshold
    superglue_matcher._config["superglue"]["match_threshold"] = superglue_threshold

    # Convert images to grayscale
    im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
    im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)

    # Perform matching using SuperPoint and SuperGlue
    kp1, kp2, _, _, matches = superglue_matcher.match(im1_gray, im2_gray)

    visualize_matches(im1, im2, kp1, kp2, matches) if do_visualize else None

    print(f"Number of total matches (SuperPoint): {len(matches)}")

    return kp1, kp2, matches


# Define a function performing feature matching using LoFTR
def do_feature_matching_LoFTR(
        im1: np.array, 
        im2: np.array, 
        do_visualize: bool = False
) -> tuple[list[cv.KeyPoint], list[cv.KeyPoint], list[cv.DMatch]]:
    """
    Perform feature matching using LoFTR.

    Args:
        im1 (np.array): The first input image (query image).
        im2 (np.array): The second input image (reference image).
        do_visualize (bool): Whether to visualize the matches.

    Returns:
        tuple: A tuple containing the list of keypoints from the first image, the list of keypoints from the second image, and the list of matches.
    """
    # Convert images to grayscale
    im1_gray = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
    im2_gray = cv.cvtColor(im2, cv.COLOR_RGB2GRAY)

    # Convert images to PyTorch tensors and move to GPU
    img1_tensor = torch.from_numpy(im1_gray).unsqueeze(0).unsqueeze(0).float().to(device) / 255.
    img2_tensor = torch.from_numpy(im2_gray).unsqueeze(0).unsqueeze(0).float().to(device) / 255.

    # Prepare input for the model
    input = {"image0": img1_tensor, "image1": img2_tensor}
    with torch.no_grad():
        output = loftr_model(input)

    # Extract matching results
    keypoints0 = output['keypoints0'].cpu().numpy()  # Keypoints in image1 (N, 2)
    keypoints1 = output['keypoints1'].cpu().numpy()  # Keypoints in image2 (N, 2)
    confidence = output['confidence'].cpu().numpy()  # Match confidence (N,)

    # Create OpenCV keypoints and matches
    matches = [cv.DMatch(i, i, 0) for i in range(len(keypoints0))]
    kp1 = [cv.KeyPoint(x, y, 1) for x, y in keypoints0]
    kp2 = [cv.KeyPoint(x, y, 1) for x, y in keypoints1]

    visualize_matches(im1, im2, kp1, kp2, matches) if do_visualize else None

    print(f"Number of total matches (LoFTR): {len(matches)}")

    return kp1, kp2, matches
