import os
from glob import glob

# Set dataset paths
IMAGE_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images copy/train/images"
LABEL_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images copy/train/labels"

# Option: delete images without corresponding label files
DELETE_IMAGES_WITHOUT_LABELS = True

# 1️⃣ Check if image exists for each label (delete labels with no corresponding image)
label_files = glob(os.path.join(LABEL_DIR, "*.txt"))
deleted_label_count = 0

for label_path in label_files:
    img_path = os.path.join(IMAGE_DIR, os.path.basename(label_path).replace(".txt", ".jpg"))

    if not os.path.exists(img_path):  # delete label if image is missing
        os.remove(label_path)
        deleted_label_count += 1
        print(f"Deleted label (image missing): {label_path}")

print(f"\nTotal {deleted_label_count} label files deleted (image not found).")

# 2️⃣ Check if label exists for each image (optional)
deleted_image_count = 0

if DELETE_IMAGES_WITHOUT_LABELS:
    image_files = glob(os.path.join(IMAGE_DIR, "*.jpg"))

    for img_path in image_files:
        label_path = os.path.join(LABEL_DIR, os.path.basename(img_path).replace(".jpg", ".txt"))

        if not os.path.exists(label_path):  # delete image if label is missing
            os.remove(img_path)
            deleted_image_count += 1
            print(f"Deleted image (label missing): {img_path}")

    print(f"\nTotal {deleted_image_count} image files deleted (label not found).")
    