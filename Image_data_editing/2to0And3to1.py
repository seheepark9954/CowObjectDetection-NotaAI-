import os

# Label folder path
label_dir = "/Users/seheepark/Downloads/cow_behavior_detection.v5i.yolov8 (1)/valid/labels"  # YOLO label folder

# Get list of label files
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

# Perform class ID remapping
for label_file in label_files:
    label_path = os.path.join(label_dir, label_file)

    with open(label_path, "r") as file:
        lines = file.readlines()

    modified_lines = []  # Store modified label data
    modified = False     # Check if file was modified

    for line in lines:
        values = line.strip().split()

        # Validate YOLO format (integer + 4 floats)
        if len(values) == 5:
            try:
                class_id = int(values[0])  # Extract class ID
                if class_id == 2:
                    values[0] = "0"
                    modified = True
                elif class_id == 1:
                    values[0] = "3"
                    modified = True
                elif class_id == 3:
                    values[0] = "1"
                    modified = True
            except ValueError:
                continue  # Skip invalid lines

        modified_lines.append(" ".join(values))  # Save modified line

    # Overwrite only if modified
    if modified:
        with open(label_path, "w") as file:
            file.write("\n".join(modified_lines) + "\n")
        print(f"Class ID remapping completed → {label_file}")

print("All label files have been successfully remapped.")
