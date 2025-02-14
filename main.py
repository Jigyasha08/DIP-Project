import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to calculate Intersection over Union (IoU)
def calculate_iou(box1, box2):
    # Extract coordinates of intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate area of intersection
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

# Function to calculate accuracy
def calculate_accuracy(ground_truth, predictions, threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for gt_box in ground_truth:
        gt_box_matched = False
        for pred_box in predictions:
            iou = calculate_iou(gt_box, pred_box)
            if iou >= threshold:
                true_positives += 1
                gt_box_matched = True
                break
        if not gt_box_matched:
            false_negatives += 1

    false_positives = len(predictions) - true_positives

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 1e-9)

    return precision, recall, f1_score

# Example ground truth bounding boxes and predicted bounding boxes
ground_truth_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
predicted_boxes = np.array([[20, 20, 60, 60], [70, 70, 110, 110], [5, 5, 40, 40]])

# Calculate accuracy
precision, recall, f1_score = calculate_accuracy(ground_truth_boxes, predicted_boxes)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# Display accuracy graph
thresholds = np.linspace(0, 1, 100)
precisions = []
recalls = []

for threshold in thresholds:
    precision, recall, _ = calculate_accuracy(ground_truth_boxes, predicted_boxes, threshold)
    precisions.append(precision)
    recalls.append(recall)

plt.figure()
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Value')
plt.title('Precision and Recall vs. Threshold')
plt.legend()
plt.show()
