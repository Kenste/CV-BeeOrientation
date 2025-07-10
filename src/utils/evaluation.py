import numpy as np
import torch
from tqdm import tqdm

from src.utils.orientation import estimate_orientation_bozek, angular_error_radians


@torch.no_grad()
def evaluate_segmentation(eval_data, num_classes=3):
    """
    Evaluate the model’s segmentation performance from collected evaluation data.

    Computes per-class IoU, and mean foreground IoU (classes 1 & 2).

    Args:
        eval_data (list of dict): output from collect_evaluation_data.
        num_classes (int): number of segmentation classes (including background).

    Returns:
        tuple:
            dict: per-class IoU values {class_index: IoU}.
            float: mean IoU computed over foreground classes (classes 1 and 2).
    """
    mious = [0.0] * num_classes
    count = len(eval_data)

    for entry in eval_data:
        for cls in range(num_classes):
            mious[cls] += entry["ious"][cls]

    mious = [iou_sum / count for iou_sum in mious]

    miou_fg = np.mean([mious[1], mious[2]])

    ious_dict = {cls: mious[cls] for cls in range(num_classes)}

    return ious_dict, miou_fg


@torch.no_grad()
def evaluate_orientation(eval_data, percentiles=None):
    """
    Evaluate orientation prediction error from collected evaluation data.

    Args:
        eval_data (list of dict): output from collect_evaluation_data.
        percentiles (list, optional): percentiles to compute and report. Defaults to [50, 75, 90, 95, 99].

    Returns:
        dict: dictionary containing mean error, std deviation, median, and selected percentiles of angular error (degrees).
    """
    if percentiles is None:
        percentiles = [50, 75, 90, 95, 99]

    pred_angles = []
    gt_angles = []

    for entry in eval_data:
        pred_angle = entry['pred_angle_rad']
        gt_angle = entry['gt_angle_rad']

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


def evaluate_on_test(eval_data, avg_loss, num_classes=3):
    """
    Run full evaluation on the test set from collected evaluation data.

    Args:
        eval_data (list of dict): output from collect_evaluation_data.
        avg_loss (float): average loss over the test dataset.
        num_classes (int): number of segmentation classes.

    Returns:
        dict: dictionary containing segmentation and orientation evaluation results.
    """
    ious, miou_fg = evaluate_segmentation(eval_data, num_classes)
    orientation_metrics = evaluate_orientation(eval_data)

    # Report segmentation
    print(f"\nSegmentation Test Loss: {avg_loss:.4f}")
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
            "loss": avg_loss,
            "ious": ious,
            "fg_miou": miou_fg
        },
        "orientation": orientation_metrics
    }


def compute_ious(pred_mask, gt_mask, classes=(0, 1, 2)):
    """
    Compute IoUs over specified classes between a predicted and a ground truth mask.

    Args:
        pred_mask (np.ndarray): predicted mask (H, W), integer labels.
        gt_mask (np.ndarray): ground truth mask (H, W), integer labels.
        classes (tuple): classes to compute IoU over.

    Returns:
        list: IoU values for each requested class.
    """
    ious = []
    for cls in classes:
        pred_cls = (pred_mask == cls)
        gt_cls = (gt_mask == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if union == 0:
            iou = 1.0
        else:
            iou = intersection / union

        ious.append(iou)

    return ious


@torch.no_grad()
def collect_evaluation_data(model, loader, criterion, device, gt_csv):
    """
    Run model inference on loader and create an evaluation dataset.

    Args:
        model (torch.nn.Module): trained model.
        loader (DataLoader): dataset loader yielding (images, masks, filenames).
        criterion (loss function): loss function to compute per-batch loss.
        device (torch.device): device to run inference on.
        gt_csv (dict): mapping {filename → ground-truth angle in radians}.

    Returns:
        tuple:
            list of dict: each dict contains:
                - 'image': np.ndarray, original image (C, H, W)
                - 'gt_mask': np.ndarray, ground truth mask (H, W)
                - 'gt_angle_rad': float, ground truth angle in radians
                - 'pred_mask': np.ndarray, predicted mask (H, W)
                - 'pred_angle_rad': float or None, estimated angle from predicted mask
                - 'ious': list, IoU values for each class
            float: average loss over the dataset.
    """
    model.eval()
    eval_data = []
    total_loss = 0

    for images, masks, filenames in tqdm(loader, desc="Collecting evaluation data", leave=True):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)
        total_loss += loss.item()

        preds = outputs.argmax(dim=1)

        for i, (image, mask, pred_mask, filename) in enumerate(zip(images, masks, preds, filenames)):
            gt_angle = float(gt_csv[filename])
            pred_angle = estimate_orientation_bozek(pred_mask.cpu().numpy())

            ious = compute_ious(pred_mask.cpu().numpy(), mask.cpu().numpy())

            eval_data.append({
                'image': image.cpu().numpy(),
                'gt_mask': mask.cpu().numpy(),
                'gt_angle_rad': gt_angle,
                'pred_mask': pred_mask.cpu().numpy(),
                'pred_angle_rad': pred_angle,
                'ious': ious,
            })

    avg_loss = total_loss / len(loader)
    return eval_data, avg_loss
