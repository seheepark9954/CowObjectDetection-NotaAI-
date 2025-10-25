import os

# Directory path containing label files (change if needed)
LABEL_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images copy/train/labels"

# Process all files in the directory
for filename in os.listdir(LABEL_DIR):
    # Process only text files (.txt)
    if filename.endswith(".txt"):
        filepath = os.path.join(LABEL_DIR, filename)
        print(f"Processing: {filepath}")
        with open(filepath, 'r') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue  # skip empty lines
            # Convert the first element (class ID) from float to int, then to string
            try:
                parts[0] = str(int(float(parts[0])))
            except ValueError:
                # If conversion fails, keep the original value
                pass
            # Join elements with spaces and add a newline
            new_line = " ".join(parts) + "\n"
            new_lines.append(new_line)

        # Overwrite file with modified content
        with open(filepath, 'w') as file:
            file.writelines(new_lines)

print("All label files processed successfully!")