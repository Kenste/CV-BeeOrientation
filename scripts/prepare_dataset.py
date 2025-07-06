import csv
import math
import os
import tarfile
import urllib.request

import cv2
import numpy as np
from tqdm import tqdm

# Paths
DATA_ROOT = "data"
RAW_DIR = os.path.join(DATA_ROOT, "raw")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")

# Mask parameters
ELLIPSE_AXES = (55, 20)
CROP_SIZE = 160

# Dataset files to download
DATASET_FILES = [
    {
        "url": "https://beepositions.unit.oist.jp/frame_imgs_30fps.tgz",
        "name": "frame_imgs_30fps.tgz"
    },
    {
        "url": "https://beepositions.unit.oist.jp/frame_annotations_30fps.tgz",
        "name": "frame_annotations_30fps.tgz"
    }
]


def download_and_extract():
    """
    Download and extract dataset archives if not already present.

    Downloads two `.tgz` archives into `data/`, extracts them into `data/raw/`.
    Skips files already downloaded.
    """
    os.makedirs(DATA_ROOT, exist_ok=True)

    for file_info in DATASET_FILES:
        archive_path = os.path.join(DATA_ROOT, file_info["name"])

        if not os.path.exists(archive_path):
            download_with_progress(file_info["url"], archive_path)
        else:
            print(f"{file_info['name']} already downloaded. Skipping.")

        with tarfile.open(archive_path, "r:gz") as tar:
            print(f"Extracting {file_info['name']}...")
            tar.extractall(RAW_DIR)
            print(f"Extracted to: {RAW_DIR}")


def download_with_progress(url, output_path):
    """
    Download a file from a URL with a tqdm progress bar.

    Args:
        url (str): URL to download.
        output_path (str): local file path to save the downloaded file.
    """
    response = urllib.request.urlopen(url)
    total = int(response.info().get("Content-Length", -1))
    block_size = 1024

    with open(output_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=f"Downloading {os.path.basename(output_path)}"
    ) as pbar:
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            f.write(buffer)
            pbar.update(len(buffer))


def create_bee_segmentation_mask(cropped_img, annotation):
    """
    Create a dense segmentation mask for a cropped bee image.

    Args:
        cropped_img (np.ndarray): cropped bee image (H, W) grayscale.
        annotation (dict): dict with key 'angle' (radians).

    Returns:
        np.ndarray: 2D array (H, W) with labels:
            0 = background,
            1 = head half-ellipse,
            2 = tail half-ellipse.
    """
    h, w = cropped_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Ellipse center in mask/image coords: center of cropped image
    ellipse_center = (w // 2, h // 2)

    # Convert angle to degrees + 90 offset for OpenCV ellipse drawing
    angle_rad = annotation["angle"]
    angle_deg = math.degrees(angle_rad + math.pi / 2)

    # Determine head and tail angle ranges (with +90 offset)
    head_angles = (90, 270)
    tail_angles = (270, 450)

    # Draw head half ellipse with label 1
    cv2.ellipse(
        mask,
        ellipse_center,
        ELLIPSE_AXES,
        angle_deg,
        head_angles[0],
        head_angles[1],
        color=1,
        thickness=-1
    )

    # Draw tail half ellipse with label 2
    cv2.ellipse(
        mask,
        ellipse_center,
        ELLIPSE_AXES,
        angle_deg,
        tail_angles[0],
        tail_angles[1],
        color=2,
        thickness=-1
    )

    return mask


def process_images():
    """
    Process raw frames & annotations into cropped bee images, segmentation masks, and a CSV metadata file.

    Saves:
        - Cropped grayscale images → `data/processed/images/`
        - Corresponding masks → `data/processed/masks/`
        - Metadata CSV → `data/processed/labels.csv`
    """
    frames_dir = os.path.join(RAW_DIR, "frames")
    annos_dir = os.path.join(RAW_DIR, "frames_txt")

    os.makedirs(os.path.join(PROCESSED_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(PROCESSED_DIR, "masks"), exist_ok=True)

    csv_path = os.path.join(PROCESSED_DIR, "labels.csv")
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_filename", "mask_filename", "angle"])

        frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
        counter = 0

        for frame_file in tqdm(frame_files, desc="Processing frames"):
            frame_id = os.path.splitext(frame_file)[0]
            img_path = os.path.join(frames_dir, frame_file)
            anno_path = os.path.join(annos_dir, frame_id + ".txt")

            if not os.path.exists(anno_path):
                continue

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            with open(anno_path) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                ox, oy, bee_class, px, py, angle = map(float, parts)
                if int(bee_class) != 1:
                    continue  # skip cell-bees

                cx = int(px + ox)
                cy = int(py + oy)
                half = CROP_SIZE // 2
                x1, y1 = max(cx - half, 0), max(cy - half, 0)
                x2, y2 = x1 + CROP_SIZE, y1 + CROP_SIZE

                if x2 > image.shape[1] or y2 > image.shape[0]:
                    continue  # crop outside bounds

                cropped = image[y1:y2, x1:x2].copy()
                mask = create_bee_segmentation_mask(cropped, {"angle": angle})

                img_name = f"bee_{counter:06d}.png"
                mask_name = f"bee_{counter:06d}_mask.png"

                cv2.imwrite(os.path.join(PROCESSED_DIR, "images", img_name), cropped)
                cv2.imwrite(os.path.join(PROCESSED_DIR, "masks", mask_name), mask)

                writer.writerow([img_name, mask_name, angle])
                counter += 1

    print(f"\nProcessed {counter} crops.")
    print(f"CSV saved to: {csv_path}")


def main():
    download_and_extract()
    if os.path.exists(os.path.join(PROCESSED_DIR, "labels.csv")):
        print("Processed dataset already exists. Skipping processing.")
    else:
        process_images()


if __name__ == "__main__":
    main()
