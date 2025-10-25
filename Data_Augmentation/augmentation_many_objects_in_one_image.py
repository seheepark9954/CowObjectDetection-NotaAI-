import os
import random
import cv2
from glob import glob
import albumentations as A
from tqdm import tqdm

# ---------------------------------------------------------
# 1) Settings: paths & parameters
# ---------------------------------------------------------
IMAGE_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images/train/images"
LABEL_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images/train/labels"

OUTPUT_IMAGE_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/only augmented images/images"
OUTPUT_LABEL_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/only augmented images/labels"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# Target number of bounding boxes per class (total 15,000)
TARGET_PER_CLASS = 15000

# Current counts per class (given)
current_counts = {
    0: 11675,
    1: 13553,
    2: 1490,
    3: 2002
}

# needed: number of additional bounding boxes required
needed = {}
for c in current_counts.keys():
    short = TARGET_PER_CLASS - current_counts[c]
    needed[c] = max(short, 0)

print("=== Needed BBox per class ===")
for c, val in needed.items():
    print(f"Class {c}: need {val} more.")


# Albumentations augmentation pipeline (example)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=15,
        p=0.5
    ),
    A.Blur(blur_limit=3, p=0.1),
    A.OneOf([
        A.CLAHE(p=0.5),
        A.RandomGamma(p=0.5)
    ], p=0.3)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# ---------------------------------------------------------
# 2) Scan originals: images_info, class_to_imgs
# ---------------------------------------------------------
images_info = {}       # { img_id: {"img_path":..., "label_path":..., "bboxes":[(cls,x,y,w,h), ...]} }
class_to_imgs = {0: [], 1: [], 2: [], 3: []}

label_files = glob(os.path.join(LABEL_DIR, "*.txt"))

for lbl_path in label_files:
    base = os.path.basename(lbl_path)      # e.g., "abc123.txt"
    stem = os.path.splitext(base)[0]       # "abc123"
    img_path = os.path.join(IMAGE_DIR, stem + ".jpg")

    if not os.path.exists(img_path):
        continue  # skip if the image is missing

    with open(lbl_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        continue  # skip empty labels

    # collect bboxes
    bboxes = []
    classes_in_this_image = set()
    for line in lines:
        parts = line.split()
        cls_id = int(parts[0])
        x_c = float(parts[1])
        y_c = float(parts[2])
        w   = float(parts[3])
        h   = float(parts[4])
        bboxes.append((cls_id, x_c, y_c, w, h))
        classes_in_this_image.add(cls_id)

    # register image info
    img_id = stem  # unique ID
    images_info[img_id] = {
        "img_path": img_path,
        "label_path": lbl_path,
        "bboxes": bboxes
    }

    # for each class present in this image, add img_id to the mapping
    for c_ in classes_in_this_image:
        if c_ in class_to_imgs:
            class_to_imgs[c_].append(img_id)


# ---------------------------------------------------------
# 3) Augmentation function: (img_id) -> (aug_image, aug_bboxes)
# ---------------------------------------------------------
def augment_image(img_id):
    """Apply Albumentations to the original identified by img_id and return (aug_image, aug_bboxes, aug_labels)."""
    info = images_info[img_id]
    image = cv2.imread(info["img_path"])
    if image is None:
        return None, [], []

    bboxes = []
    class_labels = []
    for (cls_id, x_c, y_c, w_, h_) in info["bboxes"]:
        bboxes.append([x_c, y_c, w_, h_])
        class_labels.append(cls_id)

    augmented = transform(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels
    )
    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']
    return aug_image, aug_bboxes, aug_labels


# ---------------------------------------------------------
# 4) Augmentation loop: until all needed values drop to 0
# ---------------------------------------------------------
def any_needed(need_dict):
    """Return True if any class still needs at least 1 more bbox."""
    return any(v > 0 for v in need_dict.values())

out_idx = 0  # index for naming augmented outputs
MAX_ITERS = 100000  # safety cap to avoid infinite loops

pbar = tqdm(total=sum(needed.values()), desc="Augmenting BBoxes")

while any_needed(needed) and out_idx < MAX_ITERS:
    # 1) Choose the class with the largest deficit (greedy).
    cls_to_pick = max(needed, key=needed.get)  # class with max needed[c]
    if needed[cls_to_pick] <= 0:
        break

    # 2) Randomly pick an image that contains that class
    candidate_imgs = class_to_imgs[cls_to_pick]
    if not candidate_imgs:
        # If no images contain that class (unlikely), just skip it
        print(f"No images contain class {cls_to_pick}, skipping.")
        needed[cls_to_pick] = 0
        continue

    chosen_img_id = random.choice(candidate_imgs)

    # 3) Perform augmentation
    aug_image, aug_bboxes, aug_labels = augment_image(chosen_img_id)
    if aug_image is None or len(aug_labels) == 0:
        # augmentation failed or resulted in zero bboxes (e.g., extreme crop)
        continue

    # 4) Decrease needed counts per bbox in the augmented result
    for c_ in aug_labels:
        if needed[c_] > 0:
            needed[c_] -= 1
            pbar.update(1)  # progress++

    # 5) Save results
    new_img_name = f"aug_{out_idx:05d}.jpg"
    new_lbl_name = f"aug_{out_idx:05d}.txt"

    out_img_path = os.path.join(OUTPUT_IMAGE_DIR, new_img_name)
    out_lbl_path = os.path.join(OUTPUT_LABEL_DIR, new_lbl_name)

    cv2.imwrite(out_img_path, aug_image)
    with open(out_lbl_path, 'w') as f:
        for bbox, lbl in zip(aug_bboxes, aug_labels):
            # lbl → int, bbox are floats
            x_c, y_c, w_, h_ = bbox
            f.write(f"{int(lbl)} {x_c:.6f} {y_c:.6f} {w_:.6f} {h_:.6f}\n")

    out_idx += 1

pbar.close()

print("\n=== Done! ===")
print("Needed after augmentation:")
for c, val in needed.items():
    print(f"Class {c}: {val} still needed (should be 0 or negative).")

print(f"Total augmented images created: {out_idx}")
