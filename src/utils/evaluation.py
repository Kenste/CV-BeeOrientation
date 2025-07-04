import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on a dataset.

    Args:
        model (torch.nn.Module): model to evaluate.
        loader (DataLoader): data loader to iterate over.
        criterion (loss function): loss function to compute.
        device (torch.device): device to run on.

    Returns:
        float: average loss over the dataset.
    """
    model.eval()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Evaluating", leave=True):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item()
    return total_loss / len(loader)


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


def evaluate_on_test(model, test_loader, checkpoint_path, criterion, device):
    """
    Evaluate a trained model on the test set and report loss.

    Args:
        model (torch.nn.Module): model architecture to evaluate.
        test_loader (DataLoader): test dataset loader.
        checkpoint_path (str): path to best checkpoint (.pth).
        criterion (loss function): loss function to compute.
        device (torch.device): device to run on.

    Returns:
        float: average test loss.
    """
    # Load the best weights into the model
    model = load_checkpoint(model, checkpoint_path, device)
    model.to(device)

    # Run evaluation
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}")
    return test_loss
