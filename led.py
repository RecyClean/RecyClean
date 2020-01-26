from gpiozero import LED, Servo
from time import sleep
from signal import pause
from detection import *

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

RECYCLE = ["bottle", "tvmonitor"]


plasticLED = LED(21) #GPIO17, pin 11
paperLED   = LED(16) #GPIO18, pin 12
trashLED   = LED(26)
#servo  = Servo(17) #GPIO37


def turn_on(material):
    if material == "plastic":
        #yesLED.blink(on_time=2, off_time=1, n=1)
        plasticLED.on()
        #servo.max()
    elif material == "paper":
        paperLED.on()
        #servo.min()
    elif material == "trash":
        trashLED.on()
        #servo.max()

def turn_off():
    plasticLED.off()
    paperLED.off()
    trashLED.off()
     