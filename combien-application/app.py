# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import time
import gestureCNN as myNN

minValue = 70

x0 = 400
y0 = 200
height = 200
width = 200

img_flag = "CAM"
cap = cv2.VideoCapture(0)

saveImg = False
guessGesture = True
visualize = False

lastgesture = -1

kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Which mask mode to use BinaryMask or SkinMask (True|False)
binaryMode = True
counter = 0

# This parameter controls number of image samples to be taken PER gesture
numOfSamples = 301
gestname = ""
path = ""
mod = 0

def saveROIImg(img):
    global counter, gestname, path, saveImg
    if counter > (numOfSamples - 1):
        # Reset the parameters
        saveImg = False
        gestname = ''
        counter = 0
        return
    
    counter = counter + 1
    name = gestname + str(counter)
    print("Saving img:", name)
    cv2.imwrite(path+name + ".png", img)
    time.sleep(0.04 )

def skinMask(frame, x0, y0, width, height ):
    global guessGesture, visualize, mod, lastgesture, saveImg
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    #Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)
    
    mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    
    #blur
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    
    #bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask)

    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    guess_res = "^______^Skin"

    if saveImg == True:
        saveROIImg(res)

    elif guessGesture == True:
        retgesture, guess_res  = myNN.guessGesture(mod, res)

        if lastgesture != retgesture:
            lastgesture = retgesture
            print myNN.output[lastgesture]
            time.sleep(0.01 )
            #guessGesture = False

    elif visualize == True:
        layer = int(raw_input("Enter which layer to visualize "))
        cv2.waitKey(0)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False
    
    
    return res, guess_res


def binaryMask(frame, x0, y0, width, height):

    global guessGesture, visualize, mod, lastgesture, saveImg
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)
   
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    guess_res = "Bin"

    if saveImg == True:
        saveROIImg(res)

    elif guessGesture == True:
        retgesture, guess_res = myNN.guessGesture(mod, res)

        if lastgesture != retgesture:
            lastgesture = retgesture

    elif visualize == True:
        layer = int(raw_input("Enter which layer to visualize "))
        cv2.waitKey(1)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False

    return res, guess_res

def Main():
    global guessGesture, visualize, mod, binaryMode, x0, y0, width, height, saveImg, gestname, path, img_flag, cap
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.7
    fx = 10
    fy = 355
    fh = 18

    mod = myNN.loadCNN(0)

    import os
    print os.getpid()

    ## Grab camera input
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)  # Set Camera Size

    import signal

    # MAC : SIGPROF 27
    def signalCameraOnHandler(signum, frame):
        print signum
        global img_flag
        global cap

        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        img_flag = "CAM"

    # MAC : SIGINFO 29
    def signalMapHandler(signum, frame):
        print signum
        global img_flag
        global cap

        cap = cv2.VideoCapture('map.gif')
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        img_flag = "MAP"

    # MAC : SIGUSR1 30
    def signalSmileFaceHandler(signum, frame):
        print signum
        global img_flag
        global cap

        cap = cv2.VideoCapture('smile_glow.gif')
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        img_flag = "SMILE"

    # MAC : SIGUSR2 31
    def signalDefaultFaceHandler(signum, frame):
        print signum
        global img_flag
        global cap

        cap = cv2.VideoCapture('normal_glow.gif')
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        img_flag = "NORMAL"

    signal.signal(signal.SIGPROF, signalCameraOnHandler)
    signal.signal(signal.SIGINFO, signalMapHandler)
    signal.signal(signal.SIGUSR1, signalSmileFaceHandler)
    signal.signal(signal.SIGUSR2, signalDefaultFaceHandler)

    thumbs_up_guess_stack = 0

    while True:
        ret, frame = cap.read()

        # for end of gif
        if not ret:
            if img_flag == "NORMAL":
                cap = cv2.VideoCapture('normal_glow.gif')
                cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

            elif img_flag == "SMILE":
                cap = cv2.VideoCapture('smile_glow.gif')
                cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

            elif img_flag == "MAP":
                cap = cv2.VideoCapture('map.gif')
                cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

            ret, frame = cap.read()

        if img_flag == "CAM":

            frame = cv2.flip(frame, 3)

            guess_res = ":)"

            if ret:
                if binaryMode: # on
                    roi, guess_res = binaryMask(frame, x0, y0, width, height)
                else:
                    roi, guess_res = skinMask(frame, x0, y0, width, height)


            if guess_res.split()[0] == "PEACE":
                thumbs_up_guess_stack += 1
            else:
                thumbs_up_guess_stack = 0

            # print GUESS
            cv2.putText(frame, str(thumbs_up_guess_stack) + " " + guess_res, (fx, fy), font, 1, (0, 0, 255), 2, 1)
            cv2.putText(frame, "Show Thumbs-Up on TONY's hat camera!", (fx, fy + fh), font, size, (0, 0, 255), 1, 1)


        elif img_flag == "SMILE":
            cap = cv2.VideoCapture('smile_glow.gif')

        elif img_flag == "NORMAL":
            cap = cv2.VideoCapture('normal_glow.gif')

        elif img_flag == "MAP":
            cap = cv2.VideoCapture('map.gif')


        if thumbs_up_guess_stack >= 7:
            os.system("kill -30 "+str(os.getpid()))
            thumbs_up_guess_stack = 0

        cv2.imshow('Original', frame)

        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break

    #Realse & destroy
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()
