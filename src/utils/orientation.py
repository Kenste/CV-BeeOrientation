import math

import numpy as np


def estimate_orientation_bozek(mask):
    """
    Estimate orientation angle (α) from a predicted segmentation mask.

    Args:
        mask (np.ndarray): 2D array of shape (H, W) with labels:
            0 = background, 1 = head, 2 = tail

    Returns:
        alpha_rad (float): angle in radians, clockwise from vertical up
    """
    head_pixels = np.argwhere(mask == 1)
    tail_pixels = np.argwhere(mask == 2)

    if len(head_pixels) == 0 or len(tail_pixels) == 0:
        return None

    head_center = head_pixels.mean(axis=0)
    tail_center = tail_pixels.mean(axis=0)

    # Vector from tail to head
    dy = head_center[0] - tail_center[0]
    dx = head_center[1] - tail_center[1]

    # Angle from vertical (upward = 0°), clockwise
    angle_rad = math.atan2(dx, -dy)

    return angle_rad


def angular_error_radians(pred_rad, gt_rad):
    """
    Compute the smallest angular difference between prediction and ground truth in radians.

    Args:
        pred_rad (float or np.ndarray): predicted angle(s) in radians.
        gt_rad (float or np.ndarray): ground-truth angle(s) in radians.

    Returns:
        err_rad (float or np.ndarray): angular error(s) in radians ∈ [0, π].
    """
    diff = np.abs(pred_rad - gt_rad) % (2 * np.pi)
    err_rad = np.minimum(diff, 2 * np.pi - diff)
    return err_rad
