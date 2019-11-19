import numpy as np
import cv2
import os

def nothing(x):
    pass

vc = cv2.VideoCapture(0)
cv2.namedWindow("hand")

# 1) Creating trackbar for lower hue value so as to find the desired colored object in frame.
cv2.createTrackbar("hue_lower", "hand", 0, 255, nothing)

# Creating trackbar for upper hue value for same reason as above.
cv2.createTrackbar("hue_upper", "hand", 30, 255, nothing)

# Creating trackbar for lower saturation value for same reason as above.
cv2.createTrackbar("saturation_lower", "hand", 41, 255, nothing)

# Creating trackbar for upper saturation value for same reason as above.
cv2.createTrackbar("saturation_upper", "hand", 152, 255, nothing)

# Creating trackbar for lower value for same reason as above.
cv2.createTrackbar("value_lower", "hand", 69, 255, nothing)

# Creating trackbar for upper value for same reason as above.
cv2.createTrackbar("value_upper", "hand", 220, 255, nothing)


# for remove face
current_file_path = os.path.dirname(os.path.realpath(__file__))
cascade = cv2.CascadeClassifier(cv2.samples.findFile(current_file_path + "/haarcascade_frontalface_alt.xml"))

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def removeFaceAra(img, cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rects = detect(gray, cascade)

    height, width = img.shape[:2]

    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1 - 10, 0), (x2 + 10, height), (0, 0, 0), -1)

    return img


# fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=100)

thumbs_up_cnt = 0

while True:
    ret, frame = vc.read()  # Reading one image frame from webcam.
    frame = cv2.flip(frame, 1)

    # removing background
    # fgmask = fgbg.apply(frame)
    #
    # nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgmask)
    #
    # for index, centroid in enumerate(centroids):
    #     if stats[index][0] == 0 and stats[index][1] == 0:
    #         continue
    #     if np.any(np.isnan(centroid)):
    #         continue
    #
    #     x, y, width, height, area = stats[index]
    #     centerX, centerY = int(centroid[0]), int(centroid[1])
    #
    #     if area > 10:
    #         cv2.circle(frame, (centerX, centerY), 1, (0, 255, 0), 2)
    #         cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255))
    #

    # removing face
    frame = removeFaceAra(frame, cascade=cascade)

    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Converting RGB system to HSV system.
    hl = cv2.getTrackbarPos("hue_lower", "hand")
    hu = cv2.getTrackbarPos("hue_upper", "hand")
    sl = cv2.getTrackbarPos("saturation_lower", "hand")
    su = cv2.getTrackbarPos("saturation_upper", "hand")
    vl = cv2.getTrackbarPos("value_lower", "hand")
    vu = cv2.getTrackbarPos("value_upper", "hand")

    hand_lower = np.array([hl, sl, vl])
    hand_upper = np.array([hu, su, vu])

    mask = cv2.inRange(frame_hsv, hand_lower, hand_upper)
    ret, mask = cv2.threshold(mask, 127, 255, 0)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.bilateralFilter(mask, 5, 75,75)

    # finding the approximate contours of all closed objects in image
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    drawing = np.zeros(frame.shape, np.uint8)
    max = 0
    ci = 0 # maximum contour idx

    for i in range(len(contours)):
        # Finding the contour with maximum size. (hand when kept considerably closer to webcam in comparison to face.
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > max:
            max = area
            ci = i

    cnt = contours[ci]  # cnt is the largest contour
    epsilon = 0.25 * cv2.arcLength(cnt, True)  # Further trying to better approximate the contour by making edges sharper and using lesser number of points to approximate contour cnt.
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    hull = cv2.convexHull(cnt, returnPoints=True)  # Finding the convex hull of largest contour
    cv2.drawContours(frame, [cnt], 0, (255, 0, 0), 3)  # storing the hull points and contours in "frame" image variable(matrix).
    cv2.drawContours(frame, [hull], 0, (0, 255, 0), 3)

    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)  # Finding the defects between cnt contour and convex hull of hand.

    count = 0  # count is keeping track of number of defect points
    for i in range(defects.shape[0]):  # count is keeping track of number of defect points
        s, e, f, d = defects[i, 0]
        if d > 14000 and d < 28000:  # If normal distance between farthest point(defect) and contour is > 14000 and < 28000, it is the desired defect point.
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.circle(frame, far, 5, [0, 0, 255], -1)  # draw a circle/ dot at the defect point.
            count += 1  # count is keeping track of number of defect points

    if 1 <= count <= 3:
        thumbs_up_cnt += 1
    else:
        thumbs_up_cnt = 0

    # cv2.drawContours(frame,[cnt],0,(255,0,0),3)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Outputting "count + 1"in "frame" and displaying the output.
    cv2.putText(frame, str(count + 1), (100, 100), font, 1, (0, 0, 255), 1)
    if thumbs_up_cnt >= 3:
        cv2.putText(frame, "Found Thumbs Up :)", (150, 150), font, 1, (0, 0, 255), thickness=2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(20) == 27:
        break

vc.release()
cv2.destroyAllWindows()