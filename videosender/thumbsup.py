#!/usr/bin/env python

import rospy
import cv2
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
import math
kernel = np.ones((3, 3), np.uint8)


def callback(img_msg):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough")

    frame = cv2.resize(cv_image,(640,480))
 
    roi = frame[140:340, 395:595]
    roi_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    roi_s = roi[100:140, 80:120]
    roi_s_ycrcb = cv2.cvtColor(roi_s, cv2.COLOR_BGR2YCrCb)
    cv2.rectangle(frame, (420, 140), (570, 340), (0, 0, 255), 3)
    
    y = roi_s_ycrcb[20][20][0]
    cr = roi_s_ycrcb[20][20][1]
    cb = roi_s_ycrcb[20][20][2]
    

    if y > 190:
        if y>229:
            y=230
        lower_skin = np.array([y - 25, cr - 10, cb - 10], dtype=np.uint8)
        upper_skin = np.array([y + 25, cr + 10, cb + 10], dtype=np.uint8)
    
        mask = cv2.inRange(roi_ycrcb, lower_skin, upper_skin)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        _,contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        cv2.drawContours(mask, cnt, -1, (255, 0, 0), 5)
        
        area = cv2.contourArea(cnt)

        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # make convex hull around hand
        hull = cv2.convexHull(cnt)

        # define area of hull and area of hand
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(cnt)

        # find the percentage of area not covered by hand in convex hull
        arearatio = ((areahull - areacnt) / areacnt) * 100

        # find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)

        # l = no. of defects
        l = 0

        # code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            # distance between point and convex hull
            d = (2 * ar) / a

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)

            # draw lines around hand
            cv2.line(roi, start, end, [0, 255, 0], 2)

        l += 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if area/30000>0.4 and area/30000<0.6:
            if l==1:
                if arearatio<12:
                    prediction=0.0
                    pub.publish(prediction)
                    print 12
                elif arearatio < 17.5:
                    prediction = 1.0
                    pub.publish(prediction)
                    print 17
                else:
                    prediction = 1.0
                    pub.publish(prediction)
                    print "else"
            elif l == 2:
                print "l=2"
                prediction = 1.0
                pub.publish(prediction)
            else:
                prediction=0.0
                pub.publish(prediction)
        else:
            prediction=0.0
            pub.publish(prediction)


    else:
        print 'no skin color'
        prediction=0.0
        pub.publish(prediction)
    cv2.rectangle(frame, (125, 100), (275, 300), (0, 0, 255), 0)

if __name__ == '__main__':
    rospy.init_node('Thumb_chk_node', anonymous=False)
    pub = rospy.Publisher('thumb_pred',Float32, queue_size=10)

    rospy.Subscriber('image_topic',Image, callback)    
    rospy.spin()
