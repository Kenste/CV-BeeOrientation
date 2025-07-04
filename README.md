# 🐝 CV-BeeOrientation

This repository contains coursework for a computer vision project focused on **head/tail segmentation of honey bees**
using a 3-level U-Net architecture inspired by Bozek et al.'s paper: [*"Markerless tracking of an entire honey bee
colony"*](https://www.nature.com/articles/s41467-021-21769-1).

The segmentation masks produced by the model will be used to estimate each bee’s orientation based on the major axis of
the head/tail regions.


---

## Dataset Preparation

To convert the original annotated bee frames into cropped image/mask pairs suitable for training:

```bash
python scripts/prepare_dataset.py --input-dir /path/to/dataset --output-dir dataset/processed
```

Where:

- `--input-dir` should point to the folder containing frames/ and frames_txt/
- `--output-dir` is where processed crops and masks will be saved

This script will:

- Parse the annotations
- Crop 160×160 grayscale bee images centered on each bee
- Generate corresponding segmentation masks:
    - 0 = background
    - 1 = head ellipse
    - 2 = tail ellipse
- Save a `labels.csv` file alongside the data, containing:
    - `image_filename`: file name of the cropped bee image
    - `mask_filename`: file name of the corresponding mask
    - `angle`: the bee’s orientation angle in radians, as given in the dataset: measured clockwise from vertical upward

--- 

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```
