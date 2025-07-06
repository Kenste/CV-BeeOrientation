import numpy as np
import torch
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm

from src.utils.orientation import estimate_orientation_bozek, angular_error_radians


@torch.no_grad()
def evaluate_segmentation(model, loader, criterion, device, num_classes=3):
    """
    Evaluate the model’s segmentation performance on a dataset.

    Computes average loss, per-class IoU, and mean foreground IoU (classes 1 & 2).

    Args:
        model (torch.nn.Module): model to evaluate.
        loader (DataLoader): data loader to iterate over.
        criterion (loss function): loss function to compute.
        device (torch.device): device to run the evaluation on.
        num_classes (int): number of segmentation classes (including background).

    Returns:
        tuple:
            float: average loss over the dataset.
            dict: per-class IoU values {class_index: IoU}.
            float: mean IoU computed over foreground classes (classes 1 and 2).
    """
    model.eval()
    total_loss = 0

    miou_metric = MeanIoU(
        num_classes=num_classes,
        include_background=True,
        per_class=True,
        input_format="index"
    ).to(device)

    for images, masks, _ in tqdm(loader, desc="Evaluating segmentation", leave=True):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        miou_metric.update(preds, masks)

    avg_loss = total_loss / len(loader)
    per_class_iou = miou_metric.compute().cpu().numpy()

    # mean IoU over foreground classes
    fg_ious = per_class_iou[[1, 2]]
    fg_ious_no_nan = fg_ious[~np.isnan(fg_ious)]
    miou_fg = np.mean(fg_ious_no_nan) if len(fg_ious_no_nan) > 0 else float('nan')

    ious_dict = {cls: per_class_iou[cls] for cls in range(num_classes)}

    return avg_loss, ious_dict, miou_fg


@torch.no_grad()
def evaluate_orientation(model, loader, device, gt_csv, percentiles=None):
    """
    Evaluate orientation prediction error against ground-truth angles.

    Compares orientation estimated from predicted masks to GT angles (from CSV).

    Args:
        model (torch.nn.Module): model to evaluate.
        loader (DataLoader): data loader for the dataset.
        device (torch.device): device to run the evaluation on.
        gt_csv (dict): mapping {mask_filename → ground-truth angle in radians}.
        percentiles (list, optional): percentiles to compute and report. Defaults to [50, 75, 90, 95, 99].

    Returns:
        dict: dictionary containing mean error, std deviation, median, and selected percentiles of angular error (degrees).
    """
    if percentiles is None:
        percentiles = [50, 75, 90, 95, 99]

    model.eval()
    pred_angles, gt_angles = [], []

    for images, _, filenames in tqdm(loader, desc="Evaluating orientation", leave=True):
        images = images.to(device)
        outputs = model(images).argmax(dim=1).cpu().numpy()

        for pred_mask, filename in zip(outputs, filenames):
            pred_angle = estimate_orientation_bozek(pred_mask)
            gt_angle = float(gt_csv[filename])

            if pred_angle is None:
                continue

            pred_angles.append(pred_angle)
            gt_angles.append(gt_angle)

    pred_angles = np.array(pred_angles)
    gt_angles = np.array(gt_angles)

    errors_deg = np.degrees(angular_error_radians(pred_angles, gt_angles))
    mean_err = np.mean(errors_deg)
    std_err = np.std(errors_deg)
    median_err = np.median(errors_deg)

    perc_values = {p: np.percentile(errors_deg, p) for p in percentiles}

    return {
        "mean_error_deg": mean_err,
        "std_error_deg": std_err,
        "median_error_deg": median_err,
        "percentiles": perc_values,
        "all_errors_deg": errors_deg
    }


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from a checkpoint file.

    Args:
        model (torch.nn.Module): model instance to load weights into.
        checkpoint_path (str): path to the saved checkpoint (.pth file).
        device (torch.device): device to load the weights onto.

    Returns:
        torch.nn.Module: model with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path} (epoch {checkpoint.get('epoch', '?')})")
    return model


def evaluate_on_test(model, test_loader, checkpoint_path, criterion, device, gt_csv, num_classes=3):
    """
    Run full evaluation on the test set: segmentation + orientation.

    Loads the best checkpoint, evaluates segmentation metrics (loss, per-class IoU, mean foreground IoU)
    and orientation error (mean, std, median, and percentiles).

    Args:
        model (torch.nn.Module): model architecture.
        test_loader (DataLoader): test dataset loader.
        checkpoint_path (str): path to the best checkpoint (.pth).
        criterion (loss function): loss function.
        device (torch.device): device to run on.
        gt_csv (dict): mapping {mask_filename → ground-truth angle in radians}.
        num_classes (int): number of segmentation classes.

    Returns:
        dict: dictionary containing segmentation and orientation evaluation results.
    """
    model = load_checkpoint(model, checkpoint_path, device).to(device)

    seg_loss, ious, miou_fg = evaluate_segmentation(model, test_loader, criterion, device, num_classes)
    orientation_metrics = evaluate_orientation(model, test_loader, device, gt_csv)

    # Report segmentation
    print(f"\nSegmentation Test Loss: {seg_loss:.4f}")
    print(f"Per-class IoUs:")
    for cls, iou in ious.items():
        print(f"  Class {cls}: IoU = {iou:.4f}")
    print(f"Foreground mIoU (head & tail): {miou_fg:.4f}")

    # Report orientation
    print(f"\nOrientation Error:")
    print(f"  Mean Error:   {orientation_metrics['mean_error_deg']:.2f}°")
    print(f"  Std Dev:      {orientation_metrics['std_error_deg']:.2f}°")
    print(f"  Median Error: {orientation_metrics['median_error_deg']:.2f}°")
    for p, val in orientation_metrics["percentiles"].items():
        print(f"  {p}% of samples ≤ {val:.2f}° error")

    return {
        "segmentation": {
            "loss": seg_loss,
            "ious": ious,
            "fg_miou": miou_fg
        },
        "orientation": orientation_metrics
    }
