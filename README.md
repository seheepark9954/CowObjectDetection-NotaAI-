# CowObjectDetection-NotaAI-
Overview

This repository contains scripts for preparing, cleaning, and augmenting image datasets for YOLO-based object detection of cattle behaviors.
It is organized into two main folders:
1) Data_Augmentation/ — Scripts for generating and visualizing augmented datasets.
2) Image_data_editing/ — Scripts for cleaning, correcting, and managing YOLO label files.
------------------------------------------------------------------------------------------------------------------------------------------------------
   1. Data_Augmentation
argumentatedWellTest.py - Draws bounding boxes on an augmented image and visualizes the result using Matplotlib.
Used to verify whether YOLO-format augmentation outputs are correct.

augmentation_if_in_one_pic_one_object.py - Augments images that contain only one class per picture using Albumentations.
Balances dataset classes by generating new samples until each class reaches the target count.

augmentation_many_objects_in_one_image.py - Performs data augmentation for images containing multiple object classes.
Generates new images to balance bounding box counts across all classes.

each15000_images_per_class(good).py - Augments the dataset until each class reaches 15,000 instances.
Converts between YOLO and Pascal VOC formats during augmentation for accurate bounding boxes.
------------------------------------------------------------------------------------------------------------------------------------------------------
2. Image_data_editing
2to0And3to1.py - Remaps YOLO class IDs (e.g., converts class 2 → 0, class 3 → 1) across all label files.
Used to unify inconsistent class numbering before training.

annotation_class_float_to_int.py - Converts class IDs in YOLO label files from float values (e.g., 0.0) to integers.
Prevents parsing errors caused by non-integer class labels.

check_annotation_well.py - Visualizes bounding boxes from YOLO annotations using OpenCV and Matplotlib.
Helps check if bounding boxes are drawn correctly for each class.

Check_Label_ID_Calculate.py - Counts the total number of occurrences per class ID across all YOLO label files.
Useful for verifying class balance or identifying missing annotations.

delete_all_files_unless_it_has_id_1.py - Deletes images and labels that do not contain class ID 1.
Used to filter datasets to include only specific target classes.

delete_image_no_labelfile.py - Deletes label files with missing images, and optionally deletes images without labels.
Ensures that image-label pairs are fully synchronized.

delete0and1Anno.py - Deletes image and label files that contain annotations with specific class IDs (e.g., class 0).
Useful for removing unwanted classes from a dataset.

deleteImagesAndLabelsWithoutAnno.py - Removes empty label files and their corresponding images.
Ensures all remaining files contain at least one valid annotation.

deleteWierdAnno.py - Detects label files with invalid YOLO format (not exactly 5 elements per line).
Deletes those corrupted labels and their corresponding images.
