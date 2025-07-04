import os
import cv2
import math
import argparse
import numpy as np
import csv
from tqdm import tqdm

ELLIPSE_AXES = (55, 20)
CROP_SIZE = 160


def create_bee_segmentation_mask(cropped_img, annotation, ellipse_axes):
    """
    Create a dense segmentation mask for a cropped bee image.

    Args:
        cropped_img (np.array): cropped bee RGB image (H, W, 3)
        annotation (dict): dict with keys 'angle' (radians)
        ellipse_axes (tuple): (major_axis, minor_axis) length for ellipse

    Returns:
        mask (np.array): 2D array (H, W) with labels:
            0 = background
            1 = head half ellipse
            2 = tail half ellipse
    """
    h, w = cropped_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Ellipse center in mask/image coords: center of cropped image
    ellipse_center = (w // 2, h // 2)

    # Convert angle to degrees + 90 offset for OpenCV ellipse drawing
    angle_rad = annotation["angle"]
    angle_deg = math.degrees(angle_rad + math.pi/2)

    # Determine head and tail angle ranges (with +90 offset)
    head_angles = (90, 270)
    tail_angles = (270, 450)

    # Draw head half ellipse with label 1
    cv2.ellipse(
        mask,
        ellipse_center,
        ellipse_axes,
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
        ellipse_axes,
        angle_deg,
        tail_angles[0],
        tail_angles[1],
        color=2,
        thickness=-1
    )

    return mask


def process_image(image_path, annotation_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    with open(annotation_path, 'r') as f:
        lines = f.readlines()

    bee_annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        offset_x, offset_y, bee_class, pos_x, pos_y, angle = map(float, parts)
        if int(bee_class) == 1:
            bee_annotations.append({
                "offset_x": offset_x,
                "offset_y": offset_y,
                "class": int(bee_class),
                "position_x": pos_x,
                "position_y": pos_y,
                "angle": angle
            })

    crops_and_masks = []
    for annotation in bee_annotations:
        cx = int(annotation["position_x"] + annotation["offset_x"])
        cy = int(annotation["position_y"] + annotation["offset_y"])
        half_crop = CROP_SIZE // 2
        x1, y1 = max(cx - half_crop, 0), max(cy - half_crop, 0)
        x2, y2 = min(cx + half_crop, image.shape[1]), min(cy + half_crop, image.shape[0])

        # Skip if crop would be too small
        if (x2 - x1) != CROP_SIZE or (y2 - y1) != CROP_SIZE:
            continue

        cropped_bee = image[y1:y2, x1:x2].copy()
        mask = create_bee_segmentation_mask(cropped_bee, annotation, ELLIPSE_AXES)

        crops_and_masks.append((cropped_bee, mask, annotation["angle"]))
    return crops_and_masks


def main(input_dir, output_dir):
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    frames_dir = os.path.join(input_dir, "frames")
    annos_dir = os.path.join(input_dir, "frames_txt")

    csv_path = os.path.join(output_dir, "labels.csv")
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["image_filename", "mask_filename", "angle"])

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])

    counter = 0
    for frame_file in tqdm(frame_files, desc="Processing frames"):
        frame_id = os.path.splitext(frame_file)[0]
        img_path = os.path.join(frames_dir, frame_file)
        anno_path = os.path.join(annos_dir, frame_id + ".txt")

        if not os.path.exists(anno_path):
            continue

        samples = process_image(img_path, anno_path)
        for cropped_img, mask, angle in samples:
            img_filename = f"bee_{counter:06d}.png"
            mask_filename = f"bee_{counter:06d}_mask.png"
            img_out_path = os.path.join(output_dir, "images", img_filename)
            mask_out_path = os.path.join(output_dir, "masks", mask_filename)

            cv2.imwrite(img_out_path, cropped_img)
            cv2.imwrite(mask_out_path, mask)
            csv_writer.writerow([img_filename, mask_filename, angle])
            counter += 1

    csv_file.close()
    print(f"\nSaved {counter} image-mask-angle records to {output_dir}")
    print(f"CSV file: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare bee segmentation dataset")
    parser.add_argument("--input-dir", required=True, help="Path to directory with original frames and annotations")
    parser.add_argument("--output-dir", required=True, help="Path to save processed crops, masks, and CSV")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
