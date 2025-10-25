import os
import cv2
import random
import albumentations as A

# ==== Settings ====
# Directories containing original images and labels
IMAGE_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images/train/images"
LABEL_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images/train/labels"

# Directories to save augmented images and labels
AUG_IMAGE_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/augmented_images_15000_each/train/images"
AUG_LABEL_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/augmented_images_15000_each/train/labels"
os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
os.makedirs(AUG_LABEL_DIR, exist_ok=True)

# Target occurrences per class
TARGET = 15000

# ==== Load data ====
# Store image path, label path, and boxes per original image.
# Box format: [class, x_center, y_center, width, height] in YOLO (normalized).
data = []
for filename in os.listdir(IMAGE_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(IMAGE_DIR, filename)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(LABEL_DIR, label_filename)
        if not os.path.exists(label_path):
            continue
        boxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                x_center, y_center, w_box, h_box = map(float, parts[1:])
                boxes.append([cls, x_center, y_center, w_box, h_box])
        data.append({
            "image_path": image_path,
            "label_path": label_path,
            "boxes": boxes
        })

# ==== Compute global counts per class ====
global_counts = {0: 0, 1: 0, 2: 0, 3: 0}
for item in data:
    for box in item["boxes"]:
        cls = box[0]
        global_counts[cls] += 1

print("Original class counts:", global_counts)

# ==== Define Albumentations pipeline ====
# (e.g., flip, brightness/contrast, ShiftScaleRotate, etc.)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.CoarseDropout(
        num_holes_range=(8, 8),
        hole_height_range=(0.01, 0.03),  # tune per image resolution
        hole_width_range=(0.01, 0.03),
        fill=0,
        p=0.3
    ),
    # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
    # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], min_visibility=0.3)
)

# ==== Coordinate conversion helpers ====
# YOLO (normalized center-size) -> Pascal VOC (x_min, y_min, x_max, y_max)
def yolo_to_pascal(box, img_w, img_h):
    # box: [x_center, y_center, width, height] (normalized)
    x_center, y_center, w_box, h_box = box
    x_center *= img_w
    y_center *= img_h
    w_box *= img_w
    h_box *= img_h
    x_min = x_center - w_box / 2
    y_min = y_center - h_box / 2
    x_max = x_center + w_box / 2
    y_max = y_center + h_box / 2
    return [x_min, y_min, x_max, y_max]

# Pascal VOC -> YOLO (normalized center-size)
def pascal_to_yolo(bbox, img_w, img_h):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    w_box = x_max - x_min
    h_box = y_max - y_min
    return [x_center / img_w, y_center / img_h, w_box / img_w, h_box / img_h]

# ==== Augmentation loop ====
# Until global counts reach TARGET,
# randomly pick images containing underrepresented classes and augment.
aug_count = 0
while any(count < TARGET for count in global_counts.values()):
    # Randomly pick a class still below target
    underrepresented = [cls for cls, count in global_counts.items() if count < TARGET]
    if not underrepresented:
        break
    chosen_class = random.choice(underrepresented)

    # Choose candidates that contain the chosen class and no class already at/over TARGET (except the chosen one).
    candidates = []
    for item in data:
        classes_in_image = {box[0] for box in item["boxes"]}
        if (chosen_class in classes_in_image and
                all(cls == chosen_class or global_counts[cls] < TARGET for cls in classes_in_image)):
            candidates.append(item)

    if not candidates:
        print(f"No candidate images found for class {chosen_class}. Trying another class...")
        continue  # try another underrepresented class if no candidates

    chosen_item = random.choice(candidates)

    # Read image
    image = cv2.imread(chosen_item["image_path"])
    if image is None:
        continue
    img_h, img_w = image.shape[:2]

    # Convert boxes to Pascal VOC (Albumentations expects Pascal VOC here)
    pascal_bboxes = []
    class_labels = []
    for box in chosen_item["boxes"]:
        cls, x_center, y_center, w_box, h_box = box
        bbox_pascal = yolo_to_pascal([x_center, y_center, w_box, h_box], img_w, img_h)
        pascal_bboxes.append(bbox_pascal)
        class_labels.append(cls)

    # Apply augmentation (skip on minor out-of-bounds errors)
    try:
        augmented = transform(image=image, bboxes=pascal_bboxes, class_labels=class_labels)
    except ValueError as e:
        print("Skipping augmentation due to bbox error:", e)
        continue

    aug_image = augmented["image"]
    aug_bboxes = augmented["bboxes"]
    aug_class_labels = augmented["class_labels"]

    # Convert augmented boxes back to YOLO (normalized)
    new_img_h, new_img_w = aug_image.shape[:2]
    new_boxes = []
    for bbox, cls in zip(aug_bboxes, aug_class_labels):
        yolo_box = pascal_to_yolo(bbox, new_img_w, new_img_h)
        new_boxes.append([cls] + yolo_box)

    # Save augmented image/labels (e.g., aug_000000.jpg/.txt)
    new_img_name = f"aug_{aug_count:06d}.jpg"
    new_label_name = f"aug_{aug_count:06d}.txt"
    cv2.imwrite(os.path.join(AUG_IMAGE_DIR, new_img_name), aug_image)
    with open(os.path.join(AUG_LABEL_DIR, new_label_name), "w") as f:
        for box in new_boxes:
            cls, x_center, y_center, w_box, h_box = box
            f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}\n")

    # Update global counts with all boxes from the augmented image
    for box in new_boxes:
        cls = box[0]
        global_counts[cls] += 1

    aug_count += 1

    # Periodically report current counts
    if aug_count % 100 == 0:
        print(f"{aug_count} augmented images created. Current class counts: {global_counts}")

print("Augmentation completed. Final class counts:", global_counts)
