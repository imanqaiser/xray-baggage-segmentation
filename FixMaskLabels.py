import cv2
import os
import numpy as np

# Directory setup1
image_dir = r'C:\Users\IRAJ\Desktop\finalgirl\gt_coloredfixed'
mask_dir = r'C:\Users\IRAJ\Desktop\finalgirl\gt_coloredfixed'
output_dir = r'C:\Users\IRAJ\Desktop\knifemaskfixed'

# Ensure output directory exists2
os.makedirs(output_dir, exist_ok=True)

# Get the list of image and mask files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], reverse=True)
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')], reverse=True)

# Loop through the image files
for img_file, mask_file in zip(image_files, mask_files):

    # Define colors for visualization
    color_mapping = {
        0: (0, 0, 0),
        1: (0, 0, 255),  # red
        2: (255, 0, 0),  # knife blue
        3: (0, 255, 0)  # shuriken red
    }

    # Load the corresponding mask
    mask_path = os.path.join(mask_dir, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('Mask', mask * 225)  # Scaling to [0, 255] for visualization
    # Count the number of pixels with values 0, 1, and 2
    unique, counts = np.unique(mask, return_counts=True)
    pixel_counts = dict(zip(unique, counts))

    print(f'Pixel counts for new mask {mask_file}:')
    for value in range(5):  # Assuming pixel values 0, 1, and 2
        count = pixel_counts.get(value, 0)
        print(f'Value {value}: {count} pixels')

    # Create an RGB image from the grayscale mask
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Replace pixel values according to the color mapping
    for label, color in color_mapping.items():
        mask_rgb[mask == label] = color

    # Display the colored mask
    cv2.imshow('old Mask', mask_rgb)
    cv2.waitKey(0)

    ch = int(input("0:do nothing, 1: relabelling, 2: fix boundary problem "))

    if ch == 0:
        print("no relabelling done")

        # Save the cleaned mask
        output_mask_path = os.path.join(output_dir, mask_file)
        cv2.imwrite(output_mask_path, mask)

    elif ch == 1:
        print("relabelling needed")
        # Define a mapping dictionary for labels
        label_mapping = {
            0: 0,  # Background remains 0
            1: 2,  # knife label is remapped to 1
            2: 1,  # gun label is also remapped to 1
            3: 3,  # Other objects are mapped to background (0)
        }

        # Remap labels according to the mapping dictionary
        remapped_mask = np.zeros_like(mask)
        for from_label, to_label in label_mapping.items():
            remapped_mask[mask == from_label] = to_label

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        remapped_mask = cv2.erode(remapped_mask, kernel, iterations=1)

        # Perform connected component analysis
        num_labels, labels_im = cv2.connectedComponents(remapped_mask)
        print('num_label', num_labels)

        # Create a new mask to store the cleaned labels
        cleaned_mask = np.zeros_like(remapped_mask)

        # Relabel the connected components
        for label in range(1, num_labels):  # Skip background label 0
            component_mask = (labels_im == label)
            unique_values, counts = np.unique(remapped_mask[component_mask], return_counts=True)
            print('unique_values', unique_values)
            print('count', counts)
            most_frequent_value = unique_values[np.argmax(counts)]
            cleaned_mask[component_mask] = most_frequent_value

        cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=1)

        # Save the cleaned mask
        output_mask_path = os.path.join(output_dir, mask_file)
        cv2.imwrite(output_mask_path, cleaned_mask)

        # Count the number of pixels with values 0, 1, and 2
        unique, counts = np.unique(cleaned_mask, return_counts=True)
        pixel_counts = dict(zip(unique, counts))

        # Create a colored mask for visualization
        colored_mask = np.zeros((cleaned_mask.shape[0], cleaned_mask.shape[1], 3), dtype=np.uint8)
        for label, color in color_mapping.items():
            colored_mask[cleaned_mask == label] = color

        cv2.imshow('Colored Mask', colored_mask)

        # Print pixel counts
        print(f'Pixel counts for new mask {mask_file}:')
        for value in range(5):  # Assuming pixel values 0, 1, and 2
            count = pixel_counts.get(value, 0)
            print(f'Value {value}: {count} pixels')

        cv2.waitKey(0)

        # Close all OpenCV windows
        cv2.destroyAllWindows()

    else:
        print("boundary fix needed")
        # Define a mapping dictionary for labels
        label_mapping = {
            0: 0,  # Background remains 0
            1: 1,  # knife label is remapped to 1
            2: 2,  # gun label is also remapped to 1
            3: 3,  # Other objects are mapped to background (0)
        }

        # Remap labels according to the mapping dictionary
        remapped_mask = np.zeros_like(mask)
        for from_label, to_label in label_mapping.items():
            remapped_mask[mask == from_label] = to_label

        # Perform connected component analysis
        num_labels, labels_im = cv2.connectedComponents(remapped_mask)

        # Create a new mask to store the cleaned labels
        cleaned_mask = np.zeros_like(remapped_mask)

        # Relabel the connected components
        for label in range(1, num_labels):  # Skip background label 0
            component_mask = (labels_im == label) # where there is a label
            
            #count unique values in that area object + boundry and the umber of pixels
            unique_values, counts = np.unique(remapped_mask[component_mask], return_counts=True)
            
            #pick the most frequent pixel
            most_frequent_value = unique_values[np.argmax(counts)]
            
            #set most frequent pixel
            cleaned_mask[component_mask] = most_frequent_value

        # Save the cleaned mask
        output_mask_path = os.path.join(output_dir, mask_file)
        cv2.imwrite(output_mask_path, cleaned_mask)

        # Count the number of pixels with values 0, 1, and 2
        unique, counts = np.unique(cleaned_mask, return_counts=True)
        pixel_counts = dict(zip(unique, counts))

        # Create a colored mask for visualization
        colored_mask = np.zeros((cleaned_mask.shape[0], cleaned_mask.shape[1], 3), dtype=np.uint8)
        for label, color in color_mapping.items():
            colored_mask[cleaned_mask == label] = color

        # Display the image and the colored mask
        cv2.imshow('Colored Mask', colored_mask)

        # Print pixel counts
        print(f'Pixel counts for new mask {mask_file}:')
        for value in range(5):  # Assuming pixel values 0, 1, and 2
            count = pixel_counts.get(value, 0)
            print(f'Value {value}: {count} pixels')

        cv2.waitKey(0)

        # Close all OpenCV windows
        cv2.destroyAllWindows()







