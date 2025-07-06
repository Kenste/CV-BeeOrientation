import os

import matplotlib.pyplot as plt
import numpy as np
import torch


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


def plot_predictions(model, loader, device, n=10, save_path=None):
    """
    Plot a grid of input images, ground truth masks, and predicted masks.

    Args:
        model (torch.nn.Module): trained model.
        loader (DataLoader): DataLoader to sample from (test or validation).
        device (torch.device): device to run inference on.
        n (int): number of examples to display.
        save_path (str, optional): if provided, save the figure to this path (creates dirs if needed).
    """
    model.eval()
    images, masks, _ = next(iter(loader))
    images, masks = images[:n].to(device), masks[:n]

    with torch.no_grad():
        outputs = model(images).argmax(dim=1).cpu()

    images = images.cpu()

    fig, axs = plt.subplots(3, n, figsize=(2.5 * n, 6))
    for i in range(n):
        axs[0, i].imshow(images[i][0], cmap="gray")
        axs[0, i].set_title("Input")

        axs[1, i].imshow(masks[i], cmap="gray")
        axs[1, i].set_title("Ground Truth")

        axs[2, i].imshow(outputs[i], cmap="gray")
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
