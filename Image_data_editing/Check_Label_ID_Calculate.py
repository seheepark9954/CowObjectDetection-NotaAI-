import os
from collections import Counter

# Path to YOLO label folder
labels_folder = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images/train/labels"

# Dictionary to store class ID counts
class_counts = Counter()

# Read all label files (.txt) in the folder
for filename in os.listdir(labels_folder):
    if filename.endswith(".txt"):  # process only .txt files
        file_path = os.path.join(labels_folder, filename)

        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])  # extract first value (class ID)
                class_counts[class_id] += 1      # count occurrences

# Print results
print("Class occurrences:")
for class_id, count in sorted(class_counts.items()):
    print(f"Class {class_id}: {count} occurrences")
