
import cv2
import os
import numpy as np

# Directory setup
image_dir = r'C:\Users\IRAJ\Desktop\bw\k_train'
mask_dir = r'C:\Users\IRAJ\Desktop\bw\k_mask_train'
output_dir = r'C:\Users\IRAJ\Desktop\maskfixed'

condition = True

if condition:

    # Define colors for visualization
    color_mapping = {
        0: (0, 0, 0),
        1: (0, 0, 255),  # red
        2: (255, 0, 0),  # knife blue
        3: (0, 0, 0)  # shuriken red
    }
    # Load the corresponding mask
    mask_path = os.path.join(mask_dir, 'B0008_0037.png')
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


    unique, counts = np.unique(mask, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    for value in range(5):
        count = pixel_counts.get(value, 0)
        print(f'Value {value}: {count} pixels')

    # Draw only the largest three contours on the original image
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # Replace pixel values according to the color mapping
    for label, color in color_mapping.items():
        mask_rgb[mask == label] = color

    # Display the colored mask
    cv2.imshow('old Mask', mask_rgb)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_rgb = cv2.erode(mask_rgb, kernel, iterations=1)
    #cv2.imshow('eroded', mask_rgb)

    mask_rgb = cv2.dilate(mask_rgb, kernel, iterations=1)
    #cv2.imshow('dilated', mask_rgb )

    # Remap colors back to labels
    label_mapping = {
        (0, 0, 0): 0,  # Background
        (255, 0, 0): 2,  # Knife
        (0, 255, 0): 3,  # Shuriken
        (0, 0, 255): 1  # Another object (if any)
    }

    remapped_mask = np.zeros_like(mask)
    for color, label in label_mapping.items():
        remapped_mask[np.all(mask_rgb == color, axis=-1)] = label

    output_mask_path = os.path.join(output_dir, 'B0008_0037.png')
    cv2.imwrite(output_mask_path, remapped_mask)

    # Perform connected component analysis
    num_labels, labels_im = cv2.connectedComponents(remapped_mask)
    print(num_labels)

  # Create a colored mask for visualization
    colored_mask = np.zeros((remapped_mask.shape[0], remapped_mask.shape[1], 3), dtype=np.uint8)
    for label, color in color_mapping.items():
        colored_mask[remapped_mask == label] = color

    # Display the image and the colored mask
    cv2.imshow('Colored new Mask', colored_mask)
    cv2.waitKey()



