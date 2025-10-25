import os
from glob import glob

# Set data paths
IMAGE_DIR = "/Users/seheepark/Downloads/cow_behavior_detection.v1i.yolov8 (1)/valid/images"
LABEL_DIR = "/Users/seheepark/Downloads/cow_behavior_detection.v1i.yolov8 (1)/valid/labels"

# Counters for deleted files
deleted_label_count = 0
deleted_image_count = 0

# Check label files
label_files = glob(os.path.join(LABEL_DIR, "*.txt"))

for label_path in label_files:
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Extract only class IDs (first value of each line)
    class_ids = [int(line.split()[0]) for line in lines]

    # Delete files if class ID 1 is not present
    if 1 not in class_ids:
        # Find corresponding image path
        img_path = os.path.join(IMAGE_DIR, os.path.basename(label_path).replace(".txt", ".jpg"))

        # Delete label
        os.remove(label_path)
        deleted_label_count += 1
        print(f"Deleted label: {label_path}")

        # Delete corresponding image if it exists
        if os.path.exists(img_path):
            os.remove(img_path)
            deleted_image_count += 1
            print(f"Deleted image: {img_path}")

print(f"\nTotal deleted: {deleted_label_count} label files and {deleted_image_count} image files (no class 1 found).")