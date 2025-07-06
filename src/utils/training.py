import os

import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Run one training epoch.

    Args:
        model (torch.nn.Module): model to train.
        loader (DataLoader): training data loader.
        optimizer (torch.optim.Optimizer): optimizer.
        criterion (loss function): loss function.
        device (torch.device): device to run on.

    Returns:
        float: average training loss over the epoch.
    """
    model.train()
    epoch_loss = 0
    for images, masks, _ in tqdm(loader, desc="Training", leave=True):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate the model on validation data.

    Args:
        model (torch.nn.Module): model to evaluate.
        loader (DataLoader): validation data loader.
        criterion (loss function): loss function.
        device (torch.device): device to run on.

    Returns:
        float: average validation loss.
    """
    model.eval()
    val_loss = 0
    for images, masks, _ in tqdm(loader, desc="Validation", leave=True):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        val_loss += loss.item()
    return val_loss / len(loader)


def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model and optimizer state to disk.

    Args:
        model (torch.nn.Module): model to save.
        optimizer (torch.optim.Optimizer): optimizer to save.
        epoch (int): current epoch.
        path (str): file path to save the checkpoint.
    """
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, path)


def train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        num_epochs=20,
        checkpoint_filename=None,
):
    """
    Train a model with validation, and save the best checkpoint.

    Args:
        model (torch.nn.Module): model to train.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validation data loader.
        optimizer (torch.optim.Optimizer): optimizer.
        criterion (loss function): loss function.
        device (torch.device): device to run on.
        num_epochs (int, optional): number of epochs. Default is 20.
        checkpoint_filename (str, optional): filename (.pth) to save this model's checkpoints. Defaults to the model's name.

    Returns:
        tuple:
            list of float: training losses per epoch.
            list of float: validation losses per epoch.
            str: path to best saved model.
    """
    os.makedirs("results/checkpoints", exist_ok=True)
    if not checkpoint_filename:
        checkpoint_filename = model.__class__.__name__ + ".pth"

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = -1
    best_path = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_path = os.path.join("results/checkpoints", checkpoint_filename)

            save_checkpoint(model, optimizer, best_epoch, best_path)
            print(f"  New best model saved at epoch {best_epoch}")
        print()

    print(f"\nTraining finished. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Best model saved at: {best_path}")
    return train_losses, val_losses, best_path
