import cv2
import os
import matplotlib.pyplot as plt

# Image and label directories
image_dir = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images 복사본/train/images"
label_dir = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images 복사본/train/labels"

# Class names (edit to match your labels)
class_names = ["lying", "standing", "walking", "grazing"]


# Read YOLO-format labels and draw bounding boxes
def draw_labels(image_path, label_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split()
        class_id = int(data[0])
        x_center, y_center, bbox_width, bbox_height = map(float, data[1:])

        # Convert YOLO (normalized cx, cy, w, h) to pixel coordinates
        x1 = int((x_center - bbox_width / 2) * w)
        y1 = int((y_center - bbox_height / 2) * h)
        x2 = int((x_center + bbox_width / 2) * w)
        y2 = int((y_center + bbox_height / 2) * h)

        # Draw box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image


# Collect image files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))])

# Preview multiple images (auto-advance every 0.2s)
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    label_path = os.path.join(label_dir, image_file.replace(".jpg", ".txt").replace(".png", ".txt"))

    if os.path.exists(label_path):
        annotated_image = draw_labels(image_path, label_path)

        plt.imshow(annotated_image)
        plt.axis("off")
        plt.title(image_file)
        plt.show(block=False)  # enable plt.pause()
        plt.pause(0.2)
        plt.close()  # close current and move to next
    else:
        print(f"Label file not found: {label_path}")