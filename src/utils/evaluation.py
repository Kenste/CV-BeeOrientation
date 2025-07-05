import numpy as np
import torch
from torchmetrics.segmentation import MeanIoU
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=3):
    """
    Evaluate the model on a dataset and compute per-class and foreground mIoU.

    Args:
        model (torch.nn.Module): model to evaluate.
        loader (DataLoader): data loader to iterate over.
        criterion (loss function): loss function to compute.
        device (torch.device): device to run on.
        num_classes (int): number of classes in the segmentation task.

    Returns:
        tuple:
            float: average loss over the dataset.
            dict: per-class IoU values {class_index: IoU}.
            float: mean IoU computed over foreground classes (classes 1 and 2).
    """
    model.eval()
    total_loss = 0

    # configure MeanIoU to return per-class IoUs
    miou_metric = MeanIoU(
        num_classes=num_classes,
        include_background=True,
        per_class=True,
        input_format="index"
    ).to(device)

    for images, masks in tqdm(loader, desc="Evaluating", leave=True):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()

        preds = outputs.argmax(dim=1)
        miou_metric.update(preds, masks)

    avg_loss = total_loss / len(loader)

    # get per-class IoUs
    per_class_iou = miou_metric.compute()
    per_class_iou = per_class_iou.cpu().numpy()

    # extract foreground classes (1 & 2)
    fg_classes = [1, 2]
    fg_ious = per_class_iou[fg_classes]

    # mean IoU over foreground classes
    fg_ious_no_nan = fg_ious[~np.isnan(fg_ious)]
    miou_fg = np.mean(fg_ious_no_nan) if len(fg_ious_no_nan) > 0 else float('nan')

    # make a dict of per-class IoUs (including all)
    ious_dict = {cls: per_class_iou[cls] for cls in range(num_classes)}

    return avg_loss, ious_dict, miou_fg


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


def evaluate_on_test(model, test_loader, checkpoint_path, criterion, device, num_classes=3):
    """
    Evaluate a trained model on the test set and report loss, per-class IoU, and foreground mIoU.

    Args:
        model (torch.nn.Module): model architecture to evaluate.
        test_loader (DataLoader): test dataset loader.
        checkpoint_path (str): path to best checkpoint (.pth) to load weights from.
        criterion (loss function): loss function to compute.
        device (torch.device): device to run on.
        num_classes (int): number of classes in the segmentation task.

    Returns:
        tuple:
            float: average test loss.
            dict: per-class IoU values {class_index: IoU}.
            float: mean IoU computed over foreground classes (classes 1 and 2).
    """
    # Load the best weights into the model
    model = load_checkpoint(model, checkpoint_path, device)
    model.to(device)

    test_loss, ious, miou_fg = evaluate(model, test_loader, criterion, device, num_classes)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Per-class IoU:")
    for cls, iou in ious.items():
        print(f"  Class {cls}: IoU = {iou:.4f}")
    print(f"Mean IoU (foreground classes 1 & 2): {miou_fg:.4f}\n")

    return test_loss, ious, miou_fg
