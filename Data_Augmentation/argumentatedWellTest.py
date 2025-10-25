import os
import cv2
import matplotlib.pyplot as plt

def visualize_augment_result(image, bboxes, window_name="Augmentation Check"):
    """
    Simple helper to draw an augmented image with bounding boxes.
    bboxes: YOLO format [x_center, y_center, w, h], normalized to 0–1.
    """
    img_h, img_w = image.shape[:2]

    # Convert BGR -> RGB for matplotlib display
    draw_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    for bbox in bboxes:
        x_c, y_c, w, h = bbox  # YOLO format
        # Convert to pixel coordinates
        x_c_pix = int(x_c * img_w)
        y_c_pix = int(y_c * img_h)
        w_pix = int(w * img_w)
        h_pix = int(h * img_h)

        # Compute top-left and bottom-right corners
        x1 = x_c_pix - w_pix // 2
        y1 = y_c_pix - h_pix // 2
        x2 = x_c_pix + w_pix // 2
        y2 = y_c_pix + h_pix // 2

        # Draw rectangle
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.figure(figsize=(8, 6))
    plt.imshow(draw_img)
    plt.title(window_name)
    plt.axis("off")
    plt.show()



IMG_PATH = "/Users/seheepark/Desktop/internship/projects/dataArgumentation/ArgumentedTo10000/images/aug_0_00009.jpg"  # example
LBL_PATH = "/Users/seheepark/Desktop/internship/projects/dataArgumentation/ArgumentedTo10000/labels/aug_0_00009.txt"

image = cv2.imread(IMG_PATH)
bboxes = []

with open(LBL_PATH, 'r') as f:
    lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        # parts: [class_id, x_center, y_center, w, h]
        cls_id = int(parts[0])  # use if needed
        x_c = float(parts[1])
        y_c = float(parts[2])
        w_ = float(parts[3])
        h_ = float(parts[4])
        bboxes.append([x_c, y_c, w_, h_])

visualize_augment_result(image, bboxes, window_name="Final Augmented Image")
