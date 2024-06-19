import json
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

# Path to your .coco.json file
annotations_file = r'C:\Users\IRAJ\Desktop\COMBINEDCOCO.coco.json'
# Define the output directory for the masks
output_dir = r'C:\Users\IRAJ\Desktop\colorthingymasks'
os.makedirs(output_dir, exist_ok=True)
#annotations_file = r'C:\Users\IRAJ\Desktop\cocopuffs\testcoco\combined\cleaned_annotations.coco.json'

# Load JSON file
with open(annotations_file, 'r') as f:
    annotations = json.load(f)


# Function to segment ROI using morphological operations and contour detection
def segment_roi(roi, category_id):
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply a binary threshold to the image
    _, thresholded = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to remove noise and close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to segment the object
    mask = np.zeros_like(gray)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    # Label the mask according to the category
    labeled_mask = np.where(mask == 255, category_id, 0).astype(np.uint8)
    return labeled_mask


# Function to draw bounding boxes and segment image
def draw_bounding_boxes(image, annotations):
    combined_mask = np.zeros_like(image[:, :, 0])  # Create an empty mask for the whole image
    # Calculate the area for each annotation and sort by area in ascending order
    annotations = sorted(annotations, key=lambda ann: ann['bbox'][2] * ann['bbox'][3])

    for annotation in annotations:
        bbox = annotation['bbox']
        category_id = annotation['category_id']

        # Print intermediate values for debugging
        print(f"Category ID: {category_id}")

        if category_id == 1:
            category = "Gun"
        elif category_id == 2:
            category = "Knife"
        else:
            category = "Safe"
        print (bbox)

        if bbox != [0,0,0,0]:
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            # Extract ROI
            roi = image[y:y + h, x:x + w]
            # Segment the ROI and get the labeled mask
            segmented_roi = segment_roi(roi, category_id)

            # Create a mask for the current bounding box
            current_mask = np.zeros_like(combined_mask)
            current_mask[y:y + h, x:x + w] = segmented_roi

            # Add the current mask to the combined mask
            # Only update the combined mask where it is currently zero
            combined_mask = np.where((current_mask != 0) & (combined_mask == 0), current_mask, combined_mask)
            # Draw bounding box and label on the image
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, category, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image, combined_mask

count = 0
# Loop through each image in the annotations
for img_info in annotations['images']:
    img_id = img_info['id']

    img_path = os.path.join(r'C:\Users\IRAJ\Desktop\full\test', img_info['file_name'])
    print(img_path)

    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        print("Image found")
        count += 1
    else:
        print(f"Image not found: {img_path}")
        continue

    # Get annotations for the image
    img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]

    # Draw bounding boxes and segment the image
    labeled_image, combined_mask = draw_bounding_boxes(image.copy(), img_annotations)

    # Save the combined mask
    mask_output_path = os.path.join(output_dir, img_info['file_name'])
    cv2.imwrite(mask_output_path, combined_mask)
    #label check
    unique, counts = np.unique(combined_mask, return_counts=True)
    pixel_counts = dict(zip(unique, counts))
    for value in range(5):  # Assuming pixel values 0, 1, and 2
        count = pixel_counts.get(value, 0)
        print(f'Value {value}: {count} pixels')

    color_mapping = {
        0: (0, 0, 0),
        1: (255, 0, 0),  # red -> gun
        2: (0, 0, 255),  # knife -> blue
    }
    #colored mask for visualisation
    mask_rgb = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2RGB)
    for label, color in color_mapping.items():
        mask_rgb[combined_mask == label] = color

    # Combine the mask and labeled image (assuming they have the same dimensions)
    combined_image = np.hstack((mask_rgb, cv2.cvtColor(labeled_image, cv2.COLOR_BGR2RGB)))


    # Display the combined image with title and turned-off axes
    plt.imshow(combined_image)
    plt.axis('off')
    plt.title(f"Combined: Mask & Labeled Image (ID: {img_id})")
    plt.show()
    
