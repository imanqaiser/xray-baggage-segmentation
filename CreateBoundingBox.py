import cv2
import os
import numpy as np
import json
import time

# Directories for images and masks
image_dir = '/Users/Iman/PycharmProjects/CNNS/BWIMAGES/test/combined'
mask_dir = '/Users/Iman/PycharmProjects/CNNS/BWIMAGES/annotationsCOMBINED'

# Output COCO JSON file
output_json = '/Users/Iman/PycharmProjects/CNNS/BWIMAGES/test/test_annotations.coco.json'

# Initialize COCO JSON skeleton
coco_data = {
    "categories": [
        {"id": 1, "name": "Gun", "supercategory": "TIP"},
        {"id": 2, "name": "Knife", "supercategory": "TIP"}
    ],
    "images": [],
    "annotations": []
}

# Get list of files in both directories
image_files = os.listdir(image_dir)
mask_files = os.listdir(mask_dir)

# Ensure lists are sorted to match each other
image_files.sort()
mask_files.sort()

annotation_id = 0  # Counter for annotation IDs

# Iterate through each image file
for image_file in image_files:
    # Construct full paths to image and corresponding mask
    image_path = os.path.join(image_dir, image_file)
    base_name = os.path.splitext(image_file)[0]
    mask_file = f'{base_name}.png'  # Assuming masks have the same base name but different extension

    mask_path = os.path.join(mask_dir, mask_file)

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image: {image_path}")
        continue

    # Add image entry to coco_data
    image_entry = {
        "id": annotation_id,  # Use the same ID for image and annotations for simplicity
        "license": 1,
        "file_name": image_file,
        "height": image.shape[0],
        "width": image.shape[1],
        "date_captured": "2023-02-07T13:41:12+00:00"
    }
    coco_data["images"].append(image_entry)

    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Error loading mask: {mask_path}")
        continue

    # Normalize the mask to the range [0, 255]
    min_val = np.min(mask)
    max_val = np.max(mask)
    normalized_mask = ((mask - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Resize the mask to match the image size
    resized_mask = cv2.resize(normalized_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply Connected Component Analysis (CCA)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(resized_mask, connectivity=8)

    # Iterate over all labels but skip the background label (0)
    for label in range(1, num_labels):
        # Find contours for the current label
        contour_mask = np.uint8(labels == label) * 255
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the mask image
        mask_contours = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)  # Convert mask to BGR for drawing contours
        cv2.drawContours(mask_contours, contours, -1, (0, 255, 0), 2)

        # Find bounding box coordinates for each contour and draw bounding box on mask
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(mask_contours, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Translate bounding box coordinates to original image and draw
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Display the image with the bounding box and contours
            cv2.imshow('Image with Bounding Box and Contours', image)
            cv2.imshow('Mask with Contours', mask_contours)

            # Wait for 0.5 seconds (500 milliseconds) before closing the image window
            cv2.waitKey(500)

            # Prompt user for category label
            while True:
                category_id = int(input(f"Enter category ID for bbox {annotation_id} (1 for Gun, 2 for Knife): "))
                if category_id in [1, 2,3]:
                    break
                else:
                    print("Invalid category ID. Please enter 1 for Gun or 2 for Knife.")

            # Create annotation entry
            annotation = {
                "id": annotation_id,
                "image_id": annotation_id,  # Use the same ID for image and annotations for simplicity
                "category_id": category_id,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": w * h,
                "segmentation": [],
                "iscrowd": 0
            }
            coco_data["annotations"].append(annotation)
            annotation_id += 1

            # Close all windows to proceed to the next bounding box annotation
            cv2.destroyAllWindows()

    # Close the display windows after processing each image
    cv2.destroyAllWindows()

# Write COCO JSON annotations to file
with open(output_json, 'w') as f:
    json.dump(coco_data, f)

print("Annotations saved to", output_json)
