import cv2
import numpy as np

# Load the pre-trained YOLOv3 model for object detection
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Load the cascade classifier for fire detection
fire_cascade = cv2.CascadeClassifier('fire_detection.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect fires using the cascade classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fires = fire_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in fires:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Apply additional filtering for fire detection (you can adjust these parameters)
        _, thresholded = cv2.threshold(roi_gray, 220, 255, cv2.THRESH_BINARY)
        fire_pixels = cv2.countNonZero(thresholded)

        if fire_pixels / (w * h) > 0.4:  # Adjust the threshold as needed
            print('Fire is detected')
            # You can play a sound alert here using playsound or any other method

        # Draw a bounding box around the detected fire
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Detect objects using YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(outputNames)

    # Process YOLOv3 output and draw bounding boxes around objects
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes around detected objects
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        if label == 'fire' and confidence > 0.5:
            print('Fire is detected')
            # You can play a sound alert here using playsound or any other method
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Fire Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
