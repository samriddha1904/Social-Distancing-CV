# Uninstall default OpenCV
#!pip uninstall opencv-python-headless -y

# Install OpenCV with CUDA support
#!pip install --upgrade opencv-python-headless[full]

import cv2
import numpy as np
import os
from PIL import Image
import time
# from google.colab.patches import cv2_imshow

# Load YOLO model with weights and configuration
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Distance threshold for violations
distance_thres = 50

# Load the video file
cap = cv2.VideoCapture('humans.mp4')

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Read the first frame to get video dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read a frame from the video file.")
    exit()

# Set up video writer with dimensions from the first frame
writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame.shape[1], frame.shape[0]), True)

# Distance calculation function
def dist(pt1, pt2):
    try:
        return ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    except:
        return None

# YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
print('Output layers:', output_layers)

# Main processing loop
while True:
    ret, img = cap.read()
    if not ret:
        break

    height, width = img.shape[:2]

    # Prepare input for YOLO model
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    boxes = []

    # Process detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id != 0:  # Only keep 'person' class
                continue
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    persons = []
    person_centres = []
    violate = set()

    # Track positions of detected persons
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            persons.append(boxes[i])
            person_centres.append([x + w // 2, y + h // 2])

    # Check for violations based on distance threshold
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            if dist(person_centres[i], person_centres[j]) <= distance_thres:
                violate.add(tuple(persons[i]))
                violate.add(tuple(persons[j]))

    # Draw bounding boxes and display violations
    v = 0
    for (x, y, w, h) in persons:
        if (x, y, w, h) in violate:
            color = (0, 0, 255)  # Red for violation
            v += 1
        else:
            color = (0, 255, 0)  # Green for safe distance
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.circle(img, (x + w // 2, y + h // 2), 2, (0, 0, 255), 2)

    # Display the number of violations
    cv2.putText(img, 'No of Violations: ' + str(v), (15, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 126, 255), 2)
    writer.write(img)
    cv2.imshow("Image",img)

    # Press 'Esc' to exit
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
writer.release()
cv2.destroyAllWindows()
