import cv2
import signal

flag = 0

# default
cap = cv2.VideoCapture('normal_blur.gif')

def signalHappyFaceHandler(signum, frame):
    print(signum)
    global flag
    global cap

    cap = cv2.VideoCapture('happy_blur.gif')
    flag = 1

def signalDefaultFaceHandler(signum, frame):
    print(signum)
    global flag
    global cap

    cap = cv2.VideoCapture('normal_blur.gif')
    flag = 0

signal.signal(signal.SIGUSR1, signalHappyFaceHandler)
signal.signal(signal.SIGUSR2, signalDefaultFaceHandler)

import os
print(os.getpid())

while True:
    ret, frame = cap.read()

    if not ret:
        if flag == 0:
            cap = cv2.VideoCapture('normal_blur.gif')
        elif flag == 1:
            cap = cv2.VideoCapture('happy_blur.gif')
        ret, frame = cap.read()

    frame = cv2.resize(frame, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.relase()
cv2.destoryAllWindows()
