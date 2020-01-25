from gpiozero import LED, Servo
from time import sleep
from signal import pause
from detection import *

yesLED = LED(17) #GPIO17, pin 11
noLED  = LED(18) #GPIO18, pin 12
servo  = Servo(26) #GPIO37

while True:
    servo.max()
    sleep(2)
    servo.min()
    sleep(2)

yesLED.blink()
pause()