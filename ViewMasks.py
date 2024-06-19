
import cv2
import os
import numpy as np
# Directory setup
image_dir = r'C:\Users\IRAJ\Desktop\bw\k_train'
mask_dir = r'C:\Users\IRAJ\Desktop\bw\k_mask_train_new'

# Get the list of image and mask files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))], reverse=True)
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))], reverse= True)

# Loop through the image files
for img_file, mask_file in zip(image_files, mask_files):

    # Define colors for visualization
    color_mapping = {
        0: (0, 0, 0),
        1: (0, 0, 255),  # gun red
        2: (255, 0, 0),  # knife blue
    }

    image_path = os.path.join(image_dir, img_file)
    img = cv2.imread(image_path)
    cv2.imshow('img',img)


    # Load the corresponding mask
    mask_path = os.path.join(mask_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    unique, counts = np.unique(img, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    print(f'mask {mask_file}:')
    print(f'img {img_file}:')

    for value in range(5):  # Assuming pixel values 0, 1, and 2
        count = pixel_counts.get(value, 0)
        print(f'Value {value}: {count} pixels')

    # Create an RGB image from the grayscale mask
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Replace pixel values according to the color mapping
    for label, color in color_mapping.items():
        mask_rgb[mask == label] = color

    # Display the colored mask
    cv2.imshow('Mask', mask_rgb)
    cv2.waitKey(0)

