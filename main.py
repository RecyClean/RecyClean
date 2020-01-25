import detection as dt
# from led import *

import argparse
import cv2
from multiprocessing import Queue


'''CONSTANTS'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
network = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the queues and detections
inputQ = Queue(maxsize=1)
outputQ = Queue(maxsize=1)
detections = None


'''PROGRAM'''
dt.start_background_detections(network, inputQ, outputQ)

vs = dt.start_video_stream()

while True:
    frame, frHeight, frWidth = dt.read_frame(vs)

    detections, inputQ, outputQ = dt.check_queues(
        frame, detections, inputQ, outputQ)

    # TODO
    # If detection is in the category
    if detections is not None:
        dt.check_detections(frame, detections, frHeight, frWidth)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if q key is pressed then exit
    if key == ord('q'):
        break


# cleanup
cv2.destroyAllWindows()
vs.stop()
