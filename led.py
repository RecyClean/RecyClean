from gpiozero import LED, Servo
from time import sleep
from signal import pause
from detection import *

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

RECYCLE = ["bottle", "tvmonitor"]


yesLED = LED(17) #GPIO17, pin 11
noLED  = LED(18) #GPIO18, pin 12
servo  = Servo(26) #GPIO37


def sorter(objName):
    if objName in RECYCLE:
        servo.max()

yesLED.blink()
pause()