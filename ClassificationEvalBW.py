
import matplotlib.pyplot as plt
import numpy as np


# File paths for ground truth and predicted results
ground_truth_file = '/Users/Iman/PycharmProjects/CNNS/output/safety_results_ground_bw.txt'
predicted_results_file = '/Users/Iman/PycharmProjects/CNNS/output/safety_results_predicted_bw.txt'

# Read ground truth classifications into a dictionary
ground_truth = {}
with open(ground_truth_file, 'r') as f_ground:
    for line in f_ground:
        img_name, safety_label = line.strip().split(',')
        ground_truth[img_name] = safety_label

# Read predicted classifications into a dictionary
predicted_results = {}
with open(predicted_results_file, 'r') as f_predicted:
    for line in f_predicted:
        img_name, safety_label = line.strip().split(',')
        predicted_results[img_name] = safety_label

# Find common image names and initialize counts
common_images = set(ground_truth.keys()).intersection(predicted_results.keys())
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

# Iterate through common images and update counts
for img_name in common_images:
    ground_truth_label = ground_truth[img_name]
    predicted_label = predicted_results[img_name]

    if ground_truth_label == 'safe' and predicted_label == 'safe':
        true_positive += 1
    elif ground_truth_label == 'threat' and predicted_label == 'threat':
        true_negative += 1
    elif ground_truth_label == 'threat' and predicted_label == 'safe':
        false_positive += 1
    elif ground_truth_label == 'safe' and predicted_label == 'threat':
        false_negative += 1

# Print the results
print(f"True Positive: {true_positive}")
print(f"True Negative: {true_negative}")
print(f"False Positive: {false_positive}")
print(f"False Negative: {false_negative}")

# Print the total number of images
total_images = len(common_images)
print(f"\nTotal number of images: {total_images}\n")


# Define the confusion matrix data
conf_matrix = np.array([[true_positive, false_negative],
                        [false_positive, true_negative]])

# Define labels for the matrix
labels = [['True Positive', 'False Negative'],
          ['False Positive', 'True Negative']]

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, ['Predicted Safe', 'Predicted Threat'])
plt.yticks(tick_marks, ['Actual Safe', 'Actual Threat'])

# Add text annotations
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Show plot
plt.show()


# Calculate overall accuracy
accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

# Print the overall accuracy
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

denom = (2 * true_positive + false_positive + false_negative)
if denom == 0:
    dice_coefficient = 1.0
else:
    # Calculate Dice coefficient
    dice_coefficient = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)

# Print the Dice coefficient
print(f"Dice Coefficient: {dice_coefficient:.4f}")

