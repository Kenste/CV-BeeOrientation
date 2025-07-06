import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src/ to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.orientation import estimate_orientation_bozek, angular_error_radians
from src.utils.plots import plot_orientation_error_distribution


def compute_gt_orientation_errors(gt_csv_path, mask_dir):
    """
    Compute orientation errors between GT masks and ground-truth CSV angles.

    Args:
        gt_csv_path (str): Path to CSV file with columns ["image_filename", "mask_filename", "angle"] (radians).
        mask_dir (str): Path to directory containing GT mask images.

    Returns:
        dict: Results with raw errors and summary statistics:
            - 'errors_deg': numpy.ndarray of errors in degrees
            - 'mean': mean error in degrees
            - 'std': standard deviation of errors
            - 'median': median error
            - 'percentiles': dict of selected percentiles
    """
    df = pd.read_csv(gt_csv_path)

    errors_deg = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing GT masks"):
        mask_path = os.path.join(mask_dir, row["mask_filename"])
        if not os.path.exists(mask_path):
            continue

        gt_angle = float(row["angle"])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

        pred_angle = estimate_orientation_bozek(mask)
        if pred_angle is None:
            continue

        err_deg = np.degrees(angular_error_radians(pred_angle, gt_angle))
        errors_deg.append(err_deg)

    errors_deg = np.array(errors_deg)

    results = {
        "errors_deg": errors_deg,
        "mean": np.mean(errors_deg),
        "std": np.std(errors_deg),
        "median": np.median(errors_deg),
        "percentiles": {
            p: np.percentile(errors_deg, p) for p in [50, 75, 90, 95, 99, 100]
        }
    }

    return results


def save_results_txt(results, save_path):
    """
    Save orientation error summary to a text file.

    Args:
        results (dict): Result dictionary from compute_gt_orientation_errors().
        save_path (str): Path to save the summary text file.
    """
    with open(save_path, "w") as f:
        f.write("GT Mask Orientation Error Report\n")
        f.write("---------------------------------\n")
        f.write(f"Mean Error:    {results['mean']:.2f}°\n")
        f.write(f"Std Dev:       {results['std']:.2f}°\n")
        f.write(f"Median Error:  {results['median']:.2f}°\n")
        f.write("\nPercentiles:\n")
        for p, val in results["percentiles"].items():
            f.write(f"  {p}% ≤ {val:.2f}°\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GT mask orientation error vs. CSV ground-truth."
    )
    parser.add_argument("--csv", required=True, help="Path to CSV file with GT angles.")
    parser.add_argument("--mask-dir", required=True, help="Directory containing GT mask PNGs.")
    parser.add_argument(
        "--out-dir", default="results_gt_error", help="Directory to save results and plots."
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = compute_gt_orientation_errors(args.csv, args.mask_dir)

    txt_path = os.path.join(args.out_dir, "gt_orientation_error.txt")
    save_results_txt(results, txt_path)
    print(f"\nResults saved to: {txt_path}")

    plot_path = os.path.join(args.out_dir, "gt_orientation_error_distribution.png")
    plot_orientation_error_distribution(
        results["errors_deg"],
        model=type("GroundTruthMask", (), {})(),  # dummy class name for title
        save_path=plot_path,
    )
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
