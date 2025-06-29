# 🐝 CV-BeeOrientation

This repository contains coursework for a computer vision project focused on **head/tail segmentation of honey bees**
using a 3-level U-Net architecture inspired by Bozek et al.'s paper: [*"Markerless tracking of an entire honey bee
colony"*](https://www.nature.com/articles/s41467-021-21769-1)


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