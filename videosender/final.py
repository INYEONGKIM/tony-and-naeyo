#!/usr/bin/env python

import cv2
import signal
import os
import rospy
import vlc
import time
from cv_bridge import  CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_msgs.msg import String
from geometry_msgs.msg import Point

stop_motor_ment = "GO"

# MAC : SIGPROF 27, ubuntu : SIGPROF 27
def signalCameraOnHandler(signum, frame):
    print signum, "CAM"
    global img_flag, cap

    cap = cv2.VideoCapture(4)
    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    img_flag = "CAM"

# MAC : SIGINFO 29, ubuntu : SIGRTMAX 64
def signalMapHandler(signum, frame):
    print signum, "MAP"
    global img_flag, cap

    cap = cv2.VideoCapture('naeyo_map.gif')
    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    img_flag = "MAP"

# MAC : SIGUSR1 30, ubuntu : SIGUSR1 10
def signalSmileFaceHandler(signum, frame):
    print signum, "SMILE"
    global img_flag, cap

    # add sound
    # soundBeep()
    soundPPipPPip()

    cap = cv2.VideoCapture('naeyo_smile_blur.gif')
    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    img_flag = "SMILE"

# MAC : SIGUSR2 31, ubuntu : SIGUSR2 12
def signalDefaultFaceHandler(signum, frame):
    print signum, "NORMAL"
    global img_flag, cap, normal_change_flag

    normal_change_flag = True

    cap = cv2.VideoCapture('naeyo_normal_blur.gif')
    cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
    img_flag = "NORMAL"
    

def soundBeep(): 
    instance = vlc.Instance() 
    player = instance.media_player_new() 
    media = instance.media_new('/home/nvidia/Downloads/car_alarm.wav') 
    player.set_media(media) 
    player.play()
    __import__('time').sleep(1)


def soundPPipPPip(): 
    instance = vlc.Instance() 
    player = instance.media_player_new() 
    media = instance.media_new('/home/nvidia/Downloads/calldrop.wav') 
    player.set_media(media) 
    player.play()
    __import__('time').sleep(1)

# Set Signal
signal.signal(signal.SIGPROF, signalCameraOnHandler)
signal.signal(signal.SIGRTMAX, signalMapHandler)
signal.signal(signal.SIGUSR1, signalSmileFaceHandler)
signal.signal(signal.SIGUSR2, signalDefaultFaceHandler)

##### set init #####

# default page = normal
cap = cv2.VideoCapture('naeyo_normal_blur.gif')
img_flag = "NORMAL"

cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('frame',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
reset_flag = False

normal_change_flag = True

thumbs_up_guess_stack = 0
smile_face_stack = 0
camera_page_stack = 0
map_page_stack = 0


#####
#set png files:
png= cv2.imread('thumbsup2.png')
png = cv2.resize(png,(464,450))
tonyment = cv2.imread('tonyment.png')
tonyment = tonyment[200:280,0:640]
map = cv2.imread('map.png')

map_roi = map[0:1080,0:1080]
map_roi2 = map[1000:1068,1080:1460]

#####

print "pid : ", os.getpid()

# only CAM called
def callback(data):
    global thumbs_up_guess_stack

    if data.data == 1.0:
        thumbs_up_guess_stack += 1
        print "SSIBAL 1.0"
    else:
        thumbs_up_guess_stack = 0
        print "GOT 0.0"

def XYZCallback(data):
    global reset_flag, img_flag, stop_motor_ment

    # if close to target person
    if img_flag=="NORMAL" and (0 < data.z < 1.0):
        # paging to map
        # soundBeep()
        soundPPipPPip()

        stop_motor_ment = "STOP"

        os.system("kill -64 " + str(os.getpid()))
        reset_flag = True

normal_change_cnt = 0


if __name__ == '__main__':
    global cap, img_flag, thumbs_up_guess_stack, smile_face_stack, camera_page_stack, map_page_stack, reset_flag, normal_change_flag
    global normal_change_cnt, stop_motor_ment

    rospy.init_node('Naeyo_node', anonymous=False)

    # Publish
    pub = rospy.Publisher('image_topic', Image, queue_size=10)
    pub_tony_gesture = rospy.Publisher('tony_state', String, queue_size=10)
    pub_to_motor = rospy.Publisher('stop_motor', String, queue_size=10)

    rate = rospy.Rate(30)
    
     
    # Subscribe
    rospy.Subscriber('thumb_pred', Float32, callback)
    rospy.Subscriber('XYZ_topic', Point, XYZCallback)

    # rospy.Subscriber('final_node', String, final_callback)

    bridge = CvBridge()
    # soundBeep()
    while not rospy.is_shutdown():
        # send to motor
        pub_to_motor.publish(stop_motor_ment) # STOP, GO
        print "[stop_motor_ment] ", stop_motor_ment
        
        ret, frame = cap.read()
        
        # for end of gif
        if not ret:            
            if img_flag == "NORMAL":
                cap = cv2.VideoCapture('naeyo_normal_blur.gif')
                cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)

            elif img_flag == "SMILE":
                cap = cv2.VideoCapture('naeyo_smile_blur.gif')
                cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)

            elif img_flag == "MAP":
                cap = cv2.VideoCapture('naeyo_map.gif')
                cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)

            ret, frame = cap.read()

        if img_flag == "CAM":
            if ret:
                camera_page_stack += 1
                
                frame = cv2.flip(frame, 1)
                msg_frame = cv2.resize(frame, (100,100))
                
                frame= cv2.resize(frame,(1920,1080))

                frame_copy = frame[220:300,1210:1850]
                roi= frame[220:300,1210:1850]
                png2gray=cv2.cvtColor(tonyment,cv2.COLOR_BGR2GRAY)
                ret,mask = cv2.threshold(png2gray,180,255,cv2.THRESH_BINARY_INV)
                mask_inv = cv2.bitwise_not(mask)
                frame_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
                png_fg= cv2.bitwise_and(tonyment,tonyment,mask=mask)
                dst=cv2.add(frame_bg,png_fg)
                dst = cv2.addWeighted(frame_copy,0.1,dst,1,0)

                
                frame_copy2 = frame[315:765,1298:1762]
                roi2= frame[315:765,1298:1762]
                png2gray2=cv2.cvtColor(png,cv2.COLOR_BGR2GRAY)
                ret2,mask2 = cv2.threshold(png2gray2,180,255,cv2.THRESH_BINARY)
                mask_inv2 = cv2.bitwise_not(mask2)
                frame_bg2 = cv2.bitwise_and(roi2,roi2,mask=mask_inv2)
                png_fg2= cv2.bitwise_and(png,png,mask=mask2)
                dst2=cv2.add(frame_bg2,png_fg2)
                dst2 = cv2.addWeighted(frame_copy2,0.3,dst2,0.7,0)

                cv2.rectangle(frame,(1298,315),(1762,765),(211,145,18),6)
                cv2.rectangle(frame,(1080,0),(1920,1080),(119,215,131),10)
                frame[315:765,1298:1762]=dst2

                frame[220:300,1210:1850]=dst
                frame[0:1080,0:1080]= map_roi
                frame[1000:1068,1080:1460]=map_roi2
                
                msg = bridge.cv2_to_imgmsg(msg_frame, encoding="passthrough")

                pub.publish(msg)
                rate.sleep()

        elif img_flag == "SMILE":
            smile_face_stack += 1
            cap = cv2.VideoCapture('naeyo_smile_blur.gif')

        elif img_flag == "NORMAL":
            cap = cv2.VideoCapture('naeyo_normal_blur.gif')

        elif img_flag == "MAP":
            map_page_stack += 1
            cap = cv2.VideoCapture('naeyo_map.gif')

        else:
            pass

        ## add ###
        if normal_change_flag and normal_change_cnt < 30:
            normal_change_cnt += 1
            print "Change cnt :", normal_change_cnt

            cv2.imshow('frame', frame)
            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                break

            continue
        else:
            normal_change_flag = False
            normal_change_cnt = 0
        ##########


        # for signal call
        if thumbs_up_guess_stack >= 6:
            # paging to smile
        
            # Combine TONY
            pub_tony_gesture.publish("THUMBS_UP")

            os.system("kill -10 " + str(os.getpid()))
            thumbs_up_guess_stack = 0
            reset_flag = True

        if camera_page_stack >= 300: #150->300
            # paging to normal
            stop_motor_ment = "GO"
            os.system("kill -12 " + str(os.getpid()))
            reset_flag = True

        if smile_face_stack >= 150: #100->150
            # paging to normal 
            stop_motor_ment = "GO"
            os.system("kill -12 " + str(os.getpid()))
            reset_flag = True

        # now threshold == 300
        if map_page_stack >= 100: 
            # paging to cam
            os.system("kill -27 " + str(os.getpid()))
            reset_flag = True

        if reset_flag:
            thumbs_up_guess_stack = 0
            smile_face_stack = 0
            camera_page_stack = 0
            map_page_stack = 0
            reset_flag = False

        
        cv2.imshow('frame', frame)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            break

    rospy.spin()