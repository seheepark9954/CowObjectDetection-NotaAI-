import os
import pandas as pd

# Set paths for label and image folders
image_dir = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images copy/train/images"
label_dir = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images copy/train/labels"

# Get list of label files (.txt for YOLO format)
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
# If labels are in CSV format, use: [f for f in os.listdir(label_dir) if f.endswith('.csv')]

# Find and delete empty label files and their corresponding images
for label_file in label_files:
    label_path = os.path.join(label_dir, label_file)

    # Check if label file is empty
    if os.path.getsize(label_path) == 0:
        print(f"Empty label file found: {label_file} → deleting")

        # Find corresponding image
        image_name = os.path.splitext(label_file)[0] + ".jpg"  # adjust if images use a different extension
        image_path = os.path.join(image_dir, image_name)

        # Delete label file
        os.remove(label_path)

        # Delete corresponding image if it exists
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_name}")

print("All empty label files and corresponding images have been deleted.")
