import os
import random
import shutil
import cv2
from glob import glob
from tqdm import tqdm
import albumentations as A


# ----------------------------------------------------
# 1. Settings: paths & parameters
# ----------------------------------------------------
IMAGE_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images/train/images"  # source images
LABEL_DIR = "/Users/seheepark/Desktop/internship/projects/object_detection_cow/images/train/labels"  # source labels

OUTPUT_IMAGE_DIR = "/Users/seheepark/Desktop/internship/projects/dataArgumentation/ArgumentedTo20000/images"  # output images (augmented)
OUTPUT_LABEL_DIR = "/Users/seheepark/Desktop/internship/projects/dataArgumentation/ArgumentedTo20000/labels"  # output labels (augmented)
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# Target counts (balanced total 10,000 per earlier plan; here 20,000)
TARGET_COUNTS = {
    0: 6666,  # lying
    1: 6666,  # standing
    2: 6667   # walking
}

# Current per-class counts (corrected)
CURRENT_COUNTS = {
    0: 1439,  # lying
    1: 1593,  # standing
    2: 914    # walking
}

# Albumentations pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),   # horizontal flip
    A.RandomBrightnessContrast(p=0.3),   # brightness/contrast
    A.ShiftScaleRotate(
        shift_limit=0.1,      # ±10% shift
        scale_limit=0.2,      # ±20% scale
        rotate_limit=15,      # ±15° rotate
        p=0.5
    ),
    A.Blur(blur_limit=3, p=0.1),         # slight blur
    A.OneOf([
        A.CLAHE(p=0.5),
        A.RandomGamma(p=0.5)
    ], p=0.3),  # pick one tone/contrast adjustment
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# ----------------------------------------------------
# 2. Index originals by class
# ----------------------------------------------------
class_to_files = {0: [], 1: [], 2: []}

label_files = sorted(glob(os.path.join(LABEL_DIR, "*.txt")))

for label_path in label_files:
    filename = os.path.basename(label_path)           # e.g., standing_001.txt
    img_name = filename.replace(".txt", ".jpg")
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not os.path.exists(img_path):
        continue  # skip if image missing

    # read label file
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        continue  # skip empty labels

    # assume all boxes share the same class (use first line)
    first_line = lines[0].split()
    cls_id = int(first_line[0])

    # safety: skip if any line has a different class
    all_same_class = True
    for line in lines[1:]:
        c_ = int(line.split()[0])
        if c_ != cls_id:
            all_same_class = False
            break
    if not all_same_class:
        continue

    # add to its class list
    class_to_files[cls_id].append((img_path, label_path))


# ----------------------------------------------------
# 3. Copy originals into the augmented dataset
#    (include originals in the final set)
# ----------------------------------------------------
print("[Copying original data ...]")

for cls_id, file_list in class_to_files.items():
    for img_path, label_path in file_list:
        # image/label filenames
        img_name = os.path.basename(img_path)
        lbl_name = os.path.basename(label_path)

        # destination paths
        dst_img_path = os.path.join(OUTPUT_IMAGE_DIR, img_name)
        dst_lbl_path = os.path.join(OUTPUT_LABEL_DIR, lbl_name)

        # copy if not already copied (avoid overwriting)
        if not os.path.exists(dst_img_path):
            shutil.copy2(img_path, dst_img_path)

        if not os.path.exists(dst_lbl_path):
            shutil.copy2(label_path, dst_lbl_path)

print("Original data copy completed!")


# ----------------------------------------------------
# 4. Compute needed counts per class (toward target)
# ----------------------------------------------------
need_counts = {}
for cls_id in [0, 1, 2]:
    needed = TARGET_COUNTS[cls_id] - CURRENT_COUNTS[cls_id]
    need_counts[cls_id] = max(0, needed)

print("\n=== Required augmentations per class ===")
for cls_id in [0, 1, 2]:
    print(f" Class {cls_id} - Current: {CURRENT_COUNTS[cls_id]}, Target: {TARGET_COUNTS[cls_id]}, Need: {need_counts[cls_id]}")


# ----------------------------------------------------
# 5. Augmentation function (create a new sample)
# ----------------------------------------------------
def augment_and_save(img_path, label_path, out_idx):
    image = cv2.imread(img_path)
    if image is None:
        return False

    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.split()
        cls_ = int(parts[0])
        x_c = float(parts[1])
        y_c = float(parts[2])
        w_ = float(parts[3])
        h_ = float(parts[4])
        bboxes.append([x_c, y_c, w_, h_])
        class_labels.append(cls_)

    # apply Albumentations
    augmented = transform(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels
    )
    aug_image = augmented['image']
    aug_bboxes = augmented['bboxes']
    aug_labels = augmented['class_labels']

    if len(aug_labels) == 0:
        return False

    final_cls_id = int(aug_labels[0])  # single-class assumption → use first bbox
    new_img_name = f"aug_{final_cls_id}_{out_idx:05d}.jpg"
    new_lbl_name = f"aug_{final_cls_id}_{out_idx:05d}.txt"

    cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, new_img_name), aug_image)

    with open(os.path.join(OUTPUT_LABEL_DIR, new_lbl_name), 'w') as f:
        for bbox, lbl in zip(aug_bboxes, aug_labels):
            lbl = int(lbl)  # ensure int class id
            x_c = f"{bbox[0]:.6f}"
            y_c = f"{bbox[1]:.6f}"
            w_ = f"{bbox[2]:.6f}"
            h_ = f"{bbox[3]:.6f}"
            f.write(f"{lbl} {x_c} {y_c} {w_} {h_}\n")

    return True


# ----------------------------------------------------
# 6. Generate augmented samples to fill the gap
# ----------------------------------------------------
for cls_id in [0, 1, 2]:
    needed = need_counts[cls_id]
    if needed <= 0:
        continue  # already at/over target

    print(f"\n[Class {cls_id}] Generating {needed} augmented images ...")
    file_list = class_to_files[cls_id]

    count_generated = 0
    pbar = tqdm(total=needed)
    while count_generated < needed:
        # pick a random original
        img_path, label_path = random.choice(file_list)

        if augment_and_save(img_path, label_path, count_generated):
            count_generated += 1
            pbar.update(1)

    pbar.close()

print("\n=== All augmentation completed! ===")
print(f"Final dataset paths: {OUTPUT_IMAGE_DIR}, {OUTPUT_LABEL_DIR}")
print("Original + augmented data are included.")
