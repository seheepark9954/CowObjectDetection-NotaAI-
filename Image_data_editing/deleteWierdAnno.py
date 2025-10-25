import os

# Set paths for image and label folders
image_dir = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images copy/train/images"
label_dir = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images copy/train/labels"

# Get list of label files
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

# Find and delete invalid label files and corresponding images
for label_file in label_files:
    label_path = os.path.join(label_dir, label_file)
    image_name = os.path.splitext(label_file)[0] + ".jpg"  # adjust extension if needed
    image_path = os.path.join(image_dir, image_name)

    # Open and check label file
    with open(label_path, "r") as file:
        lines = file.readlines()

    valid = True  # assume label file is valid

    for line in lines:
        values = line.strip().split()

        # Check for invalid format (must contain class ID + 4 values)
        if len(values) != 5:
            valid = False
            break

    if not valid:
        print(f"Invalid label format detected → {label_file}, deleting...")

        # Delete label file
        os.remove(label_path)

        # Delete corresponding image file
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted related image → {image_name}")

print("All invalid label and corresponding image files have been deleted.")
