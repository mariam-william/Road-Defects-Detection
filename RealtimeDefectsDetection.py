import cv2
import numpy as np
import glob
import random
import darknet as dnet

import time

# Load Yolo
net = cv2.dnn.readNet("yolov3_custom_final.weights", "yolov3_custom.cfg")

# Name custom object
classes = ['Pothole', 'Longitudinal Crack', 'Transverse Crack', 'Alligator Crack']

cap = cv2.VideoCapture("video3.mp4")
# cap = 'test_images/<your_test_image>.jpg'

font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

img_id = 0
starting_time = time.time()

while True:
    _, img = cap.read()
    # img = cv2.imread("test_images/41.jpg")
    height, width, channels = img.shape
    img_id += 1

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()  ########
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # coordiantes of rectangle
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 1, (255, 255, 255), 2)

    elapsed_time = time.time() - starting_time
    fps = img_id / elapsed_time
    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 3)
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
