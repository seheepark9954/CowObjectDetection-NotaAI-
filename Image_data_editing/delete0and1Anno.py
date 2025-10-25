import os

# Set paths for label and image folders
label_dir = "/Users/seheepark/Downloads/cow_behavior_detection.v5i.yolov8 (1)/test/labels"
image_dir = "/Users/seheepark/Downloads/cow_behavior_detection.v5i.yolov8 (1)/test/images"

# Get list of label files
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

# Find and delete labels/images containing specific class IDs
for label_file in label_files:
    label_path = os.path.join(label_dir, label_file)
    image_name = os.path.splitext(label_file)[0] + ".jpg"
    image_path = os.path.join(image_dir, image_name)

    # Open label file and check class IDs
    with open(label_path, "r") as file:
        lines = file.readlines()

    delete_flag = False  # flag to mark files for deletion

    for line in lines:
        values = line.strip().split()

        # Validate YOLO format (integer + 4 floats)
        if len(values) == 5:
            try:
                class_id = int(values[0])
                # Delete files containing class 0 (adjust conditions if needed)
                if class_id == 0:  # or class_id == 2 or class_id == 3 or class_id == 6
                    delete_flag = True
                    break
            except ValueError:
                continue  # ignore lines with invalid format

    # Delete flagged files
    if delete_flag:
        print(f"Deleting: {label_file} (contains class ID 0)")

        # Delete label file
        os.remove(label_path)

        # Delete corresponding image file if it exists
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted corresponding image: {image_name}")

print("All files containing class ID 0 have been deleted.")