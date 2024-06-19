import json
import cv2
import os
import matplotlib.pyplot as plt

# Path to your .coco.json file
annotations_file = '/Users/Iman/PycharmProjects/CNNS/predictionsCOMBINED_coco_format (1).json'

# Load JSON file
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, annotations):
    for annotation in annotations:
        bbox = annotation['bbox']
        category_id = annotation['category_id']

        if category_id == 1:
            category = "Gun"
        elif category_id == 2:
            category = "Knife"
        elif category_id == 3:
            category = "Safe"

        # Draw bounding boxes only if bbox is not empty
        if bbox:
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)

            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, category, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# Function to check if an image is safe based on annotations
def is_image_safe(annotations, image_id):
    # Get annotations for the image
    img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]

    # Check if all bounding boxes for this image ID are empty
    for annotation in img_annotations:
        if annotation['bbox']:
            return False  # Not safe if any bbox is non-empty

    return True  # Safe if all bboxes are empty

# List of actual image files in the directory
image_dir = '/Users/Iman/PycharmProjects/CNNS/FULLCOCO/test'
actual_image_files = set(os.listdir(image_dir))

# List of image files listed in annotations
annotated_image_files = {img_info['file_name'] for img_info in annotations['images']}

# Print actual and annotated image files
print("Actual image files in directory:")
print(actual_image_files)
print("\nImage files listed in annotations:")
print(annotated_image_files)

# Loop through each image in the annotations
count_safe_images = 0
count_found_images = 0
count_notfound_images = 0
loop_count = 0

for img_info in annotations['images']:
    img_id = img_info['id']
    img_file_name = img_info['file_name']
    img_path = os.path.join(image_dir, img_file_name)

    loop_count += 1

    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        count_found_images += 1
        print(f"Image found: {img_file_name}")
    else:
        print(f"Image not found: {img_file_name}")
        count_notfound_images += 1
        continue

    # Check if the image is safe
    if is_image_safe(annotations, img_id):
        count_safe_images += 1
        print(f"Image ID {img_id} ({img_file_name}) is safe.")
    else:
        print(f"Image ID {img_id} ({img_file_name}) is not safe.")
        # Draw bounding boxes
        img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]
        image_with_boxes = draw_bounding_boxes(image.copy(), img_annotations)

        # Display the image
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Image with Bounding Boxes (ID: {img_id})")
        plt.show()

# Find images in the directory that are not in the annotations
unannotated_images = actual_image_files - annotated_image_files

# Print summary
print("\nNumber of safe images:", count_safe_images)
print("Number of found images:", count_found_images)
print("Number of not found images:", count_notfound_images)
print("Loop Count:", loop_count)
print("Images in directory not found in annotations:", unannotated_images)

# Print missing images that are listed in annotations but not found in the directory
missing_images = annotated_image_files - actual_image_files
print("Images listed in annotations but not found in directory:", missing_images)
