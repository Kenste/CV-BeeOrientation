import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BeeSegmentationDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed bee segmentation image/mask pairs.
    """

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.filenames = sorted([
            fname for fname in os.listdir(self.image_dir)
            if fname.endswith('.png')
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".png", "_mask.png"))

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)

        # Add channel dimension to grayscale image → (1, H, W)
        image = np.expand_dims(image, axis=0)

        return torch.from_numpy(image), torch.from_numpy(mask), img_name.replace(".png", "_mask.png")
