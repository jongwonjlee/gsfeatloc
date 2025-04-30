# =============================================================================
# Project       : GSFeatLoc: Visual Localization Using Feature Correspondence on 3D Gaussian Splatting
# File          : utils.py
# Description   : This is the file with utility functions for SE3 perturbation and error calculation.
# 
# Author        : Jongwon Lee (jongwon5@illinois.edu)
# Year          : 2025
# License       : BSD License
# =============================================================================
import numpy as np

def get_world_frame_difference(T_inW_ofC_gt: np.ndarray, T_inW_ofC_est: np.ndarray) -> tuple[float, float]:
    """
    Get the SE3 difference between two transformations expressed in the world coordinate frame.
    
    Args:
        T_inW_ofC_gt (np.ndarray): Ground Truth SE3 matrix (4x4).
        T_inW_ofC_est (np.ndarray): Estimated SE3 matrix (4x4).
    
    Returns:
        rotation error (float): Rotation error in degrees.
        translation error (float): Translation error.
    """
    if T_inW_ofC_gt.shape != (4, 4) or T_inW_ofC_est.shape != (4, 4):
        raise ValueError("Input matrices must be 4x4 SE3 transformation matrices.")
    
    # Calculate T_diff in the world frame
    T_diff = T_inW_ofC_gt @ np.linalg.inv(T_inW_ofC_est)
    
    # Extract rotation and translation differences
    R_diff = T_diff[:3, :3]
    t_diff = T_inW_ofC_gt[:3, -1] - T_inW_ofC_est[:3, -1] # T_diff[:3, 3]
    
    # Calculate rotation error in degrees
    angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    rotation_error = np.degrees(angle_rad)
    
    # Calculate translation error in the world frame
    translation_error = np.linalg.norm(t_diff)
    
    # print(f"Rotation Error (deg): {rotation_error:.2f}")
    # print(f"Translation Error: {translation_error:.2f}")

    return rotation_error, translation_error


def rotvec_to_matrix(angle, axis_rot):
    """
    Convert a rotation vector (angle * axis) to a rotation matrix.
    Equivalent to: R.from_rotvec(angle * axis).as_matrix()
    
    Args:
        angle (float): Rotation angle in radians.
        axis_rot (np.ndarray): Rotation axis (3D unit vector).
    
    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    axis_rot = axis_rot / np.linalg.norm(axis_rot)  # Ensure it's a unit vector
    K = np.array([
        [0, -axis_rot[2], axis_rot[1]],
        [axis_rot[2], 0, -axis_rot[0]],
        [-axis_rot[1], axis_rot[0], 0]
    ])

    R_mat = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R_mat


def perturb_SE3(T_inW_ofC: np.ndarray, 
                rotation_magnitude: float = 10, translation_magnitude: float = 0.1, 
                axis_mode: str = 'random',         # 'random', 'x', or 'y'
                magnitude_mode: str = 'fixed',      # 'fixed', 'uniform', or 'gaussian'
                rng: np.random.Generator = np.random.default_rng(0), 
                ) -> np.ndarray:
    """
    Perturb a SE3 transformation by translating by a specified magnitude along a random axis,
    then rotating by a specified magnitude along another random axis, both in the camera frame.
    
    Args:
        T_inW_ofC (np.ndarray): Input SE3 transformation matrix (4x4).
        rotation_magnitude (float): Maximum rotation magnitude in degrees.
        translation_magnitude (float): Maximum translation magnitude in meters.
        rng (np.random.Generator): Random number generator for reproducibility.
        axis_mode (str): Mode for selecting rotation/translation axes ('random', 'x', or 'y').
        magnitude_mode (str): Mode for determining perturbation magnitudes ('fixed', 'uniform', or 'gaussian').
    
    Returns:
        np.ndarray: Perturbed SE3 transformation matrix (4x4).
    """
    assert T_inW_ofC.shape == (4, 4)
    assert axis_mode in ['random', 'x', 'y'], "axis_mode must be 'random', 'x', or 'y'."
    assert magnitude_mode in ['fixed', 'uniform', 'gaussian'], "magnitude_mode must be 'fixed', 'uniform', or 'gaussian'."

    if axis_mode == 'random':
        # Generate random rotation axis
        axis_rot = rng.random(3)
        axis_rot /= np.linalg.norm(axis_rot)
        # Generate random translation axis
        axis_trs = rng.random(3)
        axis_trs /= np.linalg.norm(axis_trs)
    elif axis_mode == 'x':
        # Fixed rotation around y-axis (e.g., yaw), representing panning right or left in the x-direction.
        axis_rot = np.array([0, 1, 0])
        # Fixed translation along x-axis (e.g., right from the perspective of the camera)
        axis_trs = np.array([1, 0, 0])
    elif axis_mode == 'y':
        # Fixed rotation around the x-axis (e.g., pitch), representing tilting up or down in the y-direction.
        axis_rot = np.array([1, 0, 0])
        # Fixed translation along y-axis (e.g., down from the perspective of the camera)
        axis_trs = np.array([0, 1, 0])

    if magnitude_mode == 'fixed':
        rotation_magnitude = rotation_magnitude  # Fixed rotation angle in degrees
        translation_magnitude = translation_magnitude
    elif magnitude_mode == 'uniform':
        # Uniformly sample rotation and translation magnitudes
        rotation_magnitude = rng.uniform(-rotation_magnitude, rotation_magnitude)  # Random rotation angle in degrees
        translation_magnitude = rng.uniform(-translation_magnitude, translation_magnitude)  # Random translation distance in meters
    elif magnitude_mode == 'gaussian':
        # Sample from a Gaussian distribution for rotation and translation
        rotation_magnitude = rng.normal(loc=0, scale=rotation_magnitude)
        translation_magnitude = rng.normal(loc=0, scale=translation_magnitude)

    # Create translation perturbation as SE(3) matrix
    T_trs = np.eye(4)
    T_trs[:3, 3] = translation_magnitude * axis_trs

    # Create rotation perturbation as SE(3) matrix
    T_rot = np.eye(4)
    angle = np.deg2rad(rotation_magnitude)
    R_mat = rotvec_to_matrix(angle, axis_rot)
    T_rot[:3, :3] = R_mat

    # Combine perturbation: translation then rotation in camera frame
    T_perturb = T_trs @ T_rot

    # Apply perturbation in camera frame (right-multiply)
    return T_inW_ofC @ T_perturb

if __name__ == "__main__":
    # Example usage
    rng = np.random.default_rng()  # Random number generator

    delta_rot = rng.uniform(10, 30)  # degrees
    delta_trs = rng.uniform(0.1, 0.5)  # meters

    print(f"Perturbation - Rotation: {delta_rot:.2f} degrees, Translation: {delta_trs:.2f} meters")
    
    T_gt = np.eye(4)  # Ground truth transformation (identity for simplicity)

    axis_rot = rng.random(3)
    axis_rot /= np.linalg.norm(axis_rot)
    angle = np.deg2rad(rng.uniform(-np.pi, np.pi))  # Random rotation angle in radians
    R = rotvec_to_matrix(angle, axis_rot)
    
    # Generate random translation 
    t = rng.random(3)

    T_gt[:3, :3] = R  # Set rotation
    T_gt[:3, 3] = t  # Set translation

    T_est = perturb_SE3(T_gt, delta_rot=delta_rot, delta_trs=delta_trs, rng=rng)
    
    rot_error, trs_error = get_world_frame_difference(T_gt, T_est)

    print(f"Rotation Error: {rot_error:.2f} degrees")
    print(f"Translation Error: {trs_error:.2f} meters")