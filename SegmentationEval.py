import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


# Function to load mask
def load_mask(mask_path):
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


# Function to calculate Dice Coefficient
def calculate_dice_coefficient(predicted_mask, ground_truth_mask):
    tp = np.sum((predicted_mask == 1) & (ground_truth_mask == 1))
    tp = tp + np.sum((predicted_mask == 2) & (ground_truth_mask == 2))
    tp = tp + np.sum((predicted_mask == 0) & (ground_truth_mask == 0))
    # print ("tp", tp)

    fp = np.sum((predicted_mask == 0) & (ground_truth_mask != 0))
    fp = fp + np.sum((predicted_mask == 1) & (ground_truth_mask != 1))
    fp = fp + np.sum((predicted_mask == 2) & (ground_truth_mask != 2))
    # print("fp", fp)

    fn = np.sum((predicted_mask != 0) & (ground_truth_mask == 0))
    fn = fn + np.sum((predicted_mask != 1) & (ground_truth_mask == 1))
    fn = fn + np.sum((predicted_mask != 2) & (ground_truth_mask == 2))

    # Check if tp, fp, fn are all zero
    if tp == 0 and fp == 0 and fn == 0:
        # print("All metrics are zero, returning one Dice Coefficient.")
        return 1.0

    dice_coefficient = (2 * tp) / (2 * tp + fp + fn)
    return dice_coefficient


# Paths to your predicted and ground truth masks
predicted_mask_dir = '/Users/Iman/PycharmProjects/CNNS/finalgirl/predicted_color'
ground_truth_mask_dir = '/Users/Iman/PycharmProjects/CNNS/finalgirl/gt_colored'

# Include both .jpg and .png files
predicted_mask_files = sorted([f for f in os.listdir(predicted_mask_dir) if f.endswith(('.jpg', '.png'))])
ground_truth_mask_files = sorted([f for f in os.listdir(ground_truth_mask_dir) if f.endswith(('.jpg', '.png'))])

# Index to keep track of the current image
index = 0

def display_image(event):
    print (len(predicted_mask_files))
    print(len(ground_truth_mask_files))
    global index
    if event.key == 'right':
        index = (index + 1) % len(predicted_mask_files)
    elif event.key == 'left':
        index = (index - 1) % len(predicted_mask_files)

    predicted_mask_path = os.path.join(predicted_mask_dir, predicted_mask_files[index])
    ground_truth_mask_path = os.path.join(ground_truth_mask_dir, ground_truth_mask_files[index])

    predicted_mask = load_mask(predicted_mask_path)
    ground_truth_mask = load_mask(ground_truth_mask_path)

    # Resize the masks to the same shape using nearest neighbor interpolation
    h, w = ground_truth_mask.shape
    predicted_mask = cv2.resize(predicted_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    dice_coefficient = calculate_dice_coefficient(predicted_mask, ground_truth_mask)

    dice_coefficient = calculate_dice_coefficient(predicted_mask, ground_truth_mask)

    color_mapping = {
        0: (0, 0, 0),
        1: (255, 0, 0),  # gun red
        2: (0, 0, 255),  # knife blue
    }
    # Create an RGB image from the grayscale mask
    predicted_mask_rgb = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2RGB)
    ground_truth_mask_rgb = cv2.cvtColor(ground_truth_mask, cv2.COLOR_GRAY2RGB)

    # Replace pixel values according to the color mapping
    for label, color in color_mapping.items():
        predicted_mask_rgb[predicted_mask == label] = color
        ground_truth_mask_rgb[ground_truth_mask == label] = color

    combined_image = np.hstack((ground_truth_mask_rgb, predicted_mask_rgb))

    plt.imshow(combined_image, cmap='gray')
    plt.axis('off')
    plt.title(f"Dice Coefficient: {dice_coefficient:.4f}")
    plt.draw()


fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', display_image)

# Display the first image
display_image(type('Event', (object,), {'key': 'right'})())

plt.show()