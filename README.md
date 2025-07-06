# 🐝 CV-BeeOrientation

This repository contains a computer vision project focused on **head/tail segmentation and orientation estimation of
honey bees**.
We implement and compare several deep learning models to segment the head and tail regions of each bee, enabling
orientation estimation based on the segmented regions.

## Implemented Models

- **UNet3** – A 3-level U-Net inspired by Bozek et al., [*"Markerless tracking of an entire honey bee
  colony"*](https://www.nature.com/articles/s41467-021-21769-1)
- **ResUNet18** – A U-Net with a ResNet-18 encoder backbone (pretrained on ImageNet) and skip connections

Both models produce segmentation masks that are then used to estimate the bee's body orientation.

---

## Workflow

The entire experiment &ndash; from downloading the dataset to training, evaluation, and visualization &ndash; can now be
run directly from the provided **Jupyter notebook**:
[`Bee Orientation.ipynb`](Bee%20Orientation.ipynb)

The notebook will:

- Download & prepare the dataset
- Split the dataset into train/val/test
- Define and train both UNet3 & ResUNet18
- Evaluate segmentation & orientation metrics
- Save results & plots

You can also run individual scripts if desired (see below).

---

## Dataset Preparation

You can prepare the dataset manually, or let the notebook handle it.

To prepare it manually:

```bash
python scripts/prepare_dataset.py
```

This script will:

- Download two `.tgz` archives with the original frames & annotations
- Extract and crop bee images (160×160)
- Generate segmentation masks:
    - 0 = background
    - 1 = head ellipse
    - 2 = tail ellipse
- Write `data/processed/labels.csv` with:
    - `image_filename`: cropped image file
    - `mask_filename`: corresponding mask file
    - `angle`: ground-truth orientation angle (radians, clockwise from vertical up)

The processed dataset is saved under: `data/processed/`

--- 

## Training & Evaluation

Use the notebook for the full pipeline or call the training script components directly:

- Training & validation losses are plotted
- Segmentation performance is reported as:
    - Loss
    - Per-class IoU
    - Foreground mean IoU
- Orientation error is computed against ground truth angles, reported as:
    - Mean ± std
    - Median
    - Percentile thresholds (50%, 75%, 90%, 95%, 99%)
    - Distribution plots (histogram & CDF)

---

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Notes

- The notebook automatically downloads & prepares the dataset if not already present.
- Results, plots, and checkpoints are saved to the `results/` directory.
- Ground-truth "base error" (between GT mask-derived and GT CSV angles) can optionally be computed by running:

```bash
python scripts/evaluate_gt_error.py --csv data/processed/labels.csv --mask-dir data/processed/masks
```