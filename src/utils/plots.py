import os

import matplotlib.pyplot as plt
import numpy as np

from src.utils.orientation import angular_error_radians


def plot_training_curves(train_losses, val_losses, model, save_path=None):
    """
    Plot training and validation loss curves over epochs.

    Args:
        train_losses (list of float): training losses per epoch.
        val_losses (list of float): validation losses per epoch.
        model (torch.nn.Module): trained model instance (to extract class name).
        save_path (str, optional): if provided, save the figure to this path (creates dirs if needed).
    """
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training/Validation Loss - {model.__class__.__name__}")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_predictions(eval_data, model, n=10, save_path=None):
    """
    Plot a grid of input images, ground truth masks, and predicted masks.

    Args:
        eval_data (list of dict): output from collect_evaluation_data.
        model (torch.nn.Module): trained model (for the name).
        n (int): number of examples to display.
        save_path (str, optional): if provided, save the figure to this path (creates dirs if needed).
    """
    n = min(n, len(eval_data))
    samples = eval_data[:n]

    fig, axs = plt.subplots(3, n, figsize=(2.5 * n, 6))
    for i, entry in enumerate(samples):
        image = entry['image']
        gt_mask = entry['gt_mask']
        pred_mask = entry['pred_mask']

        axs[0, i].imshow(image[0], cmap="gray")
        axs[0, i].set_title("Input")

        axs[1, i].imshow(gt_mask, cmap="gray")
        axs[1, i].set_title("Ground Truth")

        axs[2, i].imshow(pred_mask, cmap="gray")
        axs[2, i].set_title("Prediction")

        for j in range(3):
            axs[j, i].axis("off")

    plt.suptitle(f"Predictions - {model.__class__.__name__}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_orientation_error_distribution(errors_deg, model, save_path=None):
    """
    Plot histogram and CDF of orientation errors.

    Args:
        errors_deg (array-like): orientation errors in degrees.
        model (torch.nn.Module): trained model instance (to extract class name).
        save_path (str, optional): if provided, save the figure to this path (creates dirs if needed).
    """
    errors_deg = np.asarray(errors_deg)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axs[0].hist(errors_deg, bins=30, color="skyblue", edgecolor="k", alpha=0.7)
    axs[0].set_xlabel("Orientation Error (degrees)")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Orientation Error Histogram")

    # CDF
    sorted_err = np.sort(errors_deg)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    axs[1].plot(sorted_err, cdf, color="darkorange", lw=2)
    axs[1].set_xlabel("Orientation Error (degrees)")
    axs[1].set_ylabel("Cumulative Probability")
    axs[1].set_title("Orientation Error CDF")
    axs[1].grid(True)

    fig.suptitle(f"Orientation Error Distribution - {model.__class__.__name__}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_worst_orientation_errors(eval_data, model, n=10, save_path=None):
    """
    Plot the n worst masks based on orientation estimation error.

    Args:
        eval_data (list of dict): output from collect_evaluation_data.
        model (torch.nn.Module): trained model (for the name).
        n (int): number of worst examples to display.
        save_path (str, optional): if provided, save the figure to this path (creates dirs if needed).
    """
    all_data = []

    for entry in eval_data:
        pred_angle = entry['pred_angle_rad']
        gt_angle = entry['gt_angle_rad']

        if pred_angle is None:
            continue

        error_rad = angular_error_radians(pred_angle, gt_angle)
        error_deg = np.degrees(error_rad)

        all_data.append((entry['image'], entry['gt_mask'], entry['pred_mask'], error_deg))

    if len(all_data) == 0:
        print("No valid samples found to plot.")
        return

    # Sort by error descending and take top n
    sorted_data = sorted(all_data, key=lambda x: x[3], reverse=True)[:n]

    fig, axs = plt.subplots(4, n, figsize=(2.5 * n, 6))
    for i, (img, gt_mask, pred_mask, err_deg) in enumerate(sorted_data):

        axs[0, i].imshow(img[0], cmap="gray")
        axs[0, i].set_title("Input")

        axs[1, i].imshow(gt_mask, cmap="gray")
        axs[1, i].set_title("Ground Truth")

        axs[2, i].imshow(pred_mask, cmap="gray")
        axs[2, i].set_title("Prediction")

        axs[3, i].text(0.5, 0.5, f"Error: {err_deg:.1f}°", fontsize=12, ha='center', va='center')

        for j in range(4):
            axs[j, i].axis("off")

    plt.suptitle(f"Worst {n} Orientation Errors - {model.__class__.__name__}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
