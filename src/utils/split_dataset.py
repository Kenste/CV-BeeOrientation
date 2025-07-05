import torch
import random


def split_dataset_two_stage(dataset, train_val_ratio=0.8, train_ratio_within_trainval=0.8, seed=None):
    """
    Split a dataset into train/val/test subsets using two-stage ratios.

    Stage 1: train+val vs test
    Stage 2: train vs val (within train+val)

    Args:
        dataset (torch.utils.data.Dataset): dataset to split.
        train_val_ratio (float): fraction of data for train+val (remaining is test).
        train_ratio_within_trainval (float): fraction of train+val to use for train (remaining is val).
        seed (int, optional): random seed for reproducibility. If None, a random seed is generated.

    Returns:
        tuple:
            train_dataset (Subset)
            val_dataset (Subset)
            test_dataset (Subset)
            seed (int): the seed used for splitting.
    """
    if seed is None:
        seed = random.randint(0, 99999)
        print(f"No seed provided — generated seed: {seed}")
    else:
        print(f"Using provided seed: {seed}")

    generator = torch.Generator().manual_seed(seed)

    total_len = len(dataset)

    # Stage 1: train+val vs test
    train_val_len = int(train_val_ratio * total_len)
    test_len = total_len - train_val_len

    train_val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_val_len, test_len],
        generator=generator
    )

    # Stage 2: train vs val within train_val
    train_len = int(train_ratio_within_trainval * train_val_len)
    val_len = train_val_len - train_len

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset,
        [train_len, val_len],
        generator=generator
    )

    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, seed
