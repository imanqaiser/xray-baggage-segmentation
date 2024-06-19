import json
import cv2
import os

# Path to your .coco.json file
annotations_file = '/Users/Iman/PycharmProjects/CNNS/COMBINEDCOCO.coco.json'

# Load JSON file
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Function to check if an image is safe based on annotations
def is_image_safe(annotations, image_id):
    # Get annotations for the image
    img_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]

    # Check if all annotations have category ID of 3 and bbox coordinates [0, 0, 0, 0]
    for annotation in img_annotations:
        if annotation['category_id'] != 3 or annotation['bbox'] != [0, 0, 0, 0]:
            return False  # Not safe if any annotation does not meet the criteria

    return True  # Safe if all annotations meet the criteria

# Output directory path for saving results
output_dir = '/Users/Iman/PycharmProjects/CNNS/output/'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Output file path for saving results
output_file = os.path.join(output_dir, 'safety_results_predicted_colored.txt')

# Open the output file for writing
with open(output_file, 'w') as output_f:
    # Loop through each image in the annotations
    count_safe_images = 0
    for img_info in annotations['images']:
        img_name = img_info['file_name']
        img_id = img_info['id']

        img_path = os.path.join('/Users/Iman/PycharmProjects/CNNS/FULLCOCO/test', img_name)
        print(img_path)

        if os.path.exists(img_path):
            image = cv2.imread(img_path)
            print("Image found")
        else:
            print(f"Image not found: {img_path}")
            continue

        # Check if the image is safe
        if is_image_safe(annotations, img_id):
            safety_label = "safe"
            count_safe_images += 1
        else:
            safety_label = "threat"

        # Write the result to the output file
        output_f.write(f"{img_name},{safety_label}\n")

    print("Number of safe images:", count_safe_images)

print(f"Results saved to {output_file}")
