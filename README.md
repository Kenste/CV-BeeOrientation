# 🐝 CV-BeeOrientation

This repository contains coursework for a computer vision project focused on **head/tail segmentation of honey bees**.
We implement and compare several deep learning models to segment the head and tail regions of each bee, enabling
orientation estimation based on the major axis of the segmented regions.

Implemented models include:

- A 3-level U-Net architecture inspired by Bozek et al.'s paper: [*"Markerless tracking of an entire honey bee
  colony"*](https://www.nature.com/articles/s41467-021-21769-1).
- A ResUNet18, which combines a ResNet-18 encoder (pretrained on ImageNet) with a U-Net–style decoder and skip
  connections.

The segmentation masks produced by the model will be used to estimate each bee’s orientation based on the major axis of
the head/tail regions.


---

## Dataset Preparation

To convert the original annotated bee frames into cropped image/mask pairs suitable for training:

```bash
python scripts/prepare_dataset.py --data-dir /path/to/dataset --out-dir dataset/processed
```

Where:

- `--data-dir` should point to the folder containing frames/ and frames_txt/
- `--out-dir` is where processed crops and masks will be saved

This script will:

- Parse the annotations
- Crop 160×160 grayscale bee images centered on each bee
- Generate corresponding segmentation masks:
    - 0 = background
    - 1 = head ellipse
    - 2 = tail ellipse

--- 

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```