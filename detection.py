from imutils.video import VideoStream
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
import imutils
import time
import cv2

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#            "sofa", "train", "tvmonitor"]
# COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the class labels from disk
rows = open("synset_words.txt").read().strip().split("\n")
CLASSES = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

CONFIDENCE = 0.7


# Passes the input frame and extracts detections
def get_detections(network, inputQ, outputQ):
    while True:
        if not inputQ.empty():
            frame = inputQ.get()
            frame = cv2.resize(frame, (224, 224))
            blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), 127.5)

            # input blob to neural net
            network.setInput(blob)
            detections = network.forward()

            # set as new output
            outputQ.put(detections)


# Start a process which continuously runs get_detections
def start_background_detections(network, inputQ, outputQ):
    p = Process(target=get_detections, args=(network, inputQ, outputQ,))
    p.daemon = True
    p.start()


# Start the Picamera video stream
def start_video_stream():
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    return vs


# reads the width and height values of the box from the frame
def read_frame(vs):
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (frHeight, frWidth) = frame.shape[:2]
    return frame, frHeight, frWidth


# Modifies input and output Queues
def check_queues(frame, detections, inputQ, outputQ):
    if inputQ.empty():
        inputQ.put(frame)

    if not outputQ.empty():
        detections = outputQ.get()

    return detections, inputQ, outputQ


# check to see if there are detections in the frame
def check_detections(frame, detections, frHeight, frWidth):
    # Gets highest confidence level detection and displays it on the frame
    idx = np.argsort(detections[0])[::-1][0]
    itemstr = CLASSES[idx]
    text = "Label: {}, {:.2f}%".format(itemstr,
	    detections[0][idx] * 100)
    cv2.putText(frame, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	    0.7, (0, 0, 255), 2)
    
    return itemstr


# Extract the index of the class label from the detections and compute x, y
# co-ordinates
def extract_classes(detections, i, frWidth, frHeight):
    idx = int(detections[0, 0, i, 1])
    dims = np.array([frWidth, frHeight, frWidth, frHeight])
    box = detections[0, 0, i, 3:7] * dims
    (startX, startY, endX, endY) = box.astype("int")
    return idx, startX, startY, endX, endY


# Draw the prediction box on the frame
def draw_frames(idx, frame, startX, startY, endX, endY):
    label = "{}: {:.2f}%".format(CLASSES[idx],
                                 CONFIDENCE * 100)
    cv2.rectangle(frame, (startX, startY), (endX, endY),
                  COLORS[idx], 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
