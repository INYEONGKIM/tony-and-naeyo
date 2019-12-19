# -*- coding: utf-8 -*-
#!/usr/bin/python

# Desktop
import socket
import cv2
import numpy as np
from keras.models import load_model

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf


TCP_IP = 'localhost'
TCP_PORT = 5001

kernel = np.ones((3, 3), np.uint8)
model = load_model('thr_final_model2.h5')

print "waiting now..."

while True:
    # for socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((TCP_IP, TCP_PORT))

    sock.listen(True)
    conn, addr = sock.accept()

    length = recvall(conn, 16)
    stringData = recvall(conn, int(length))
    data = np.fromstring(stringData, dtype='uint8')
    sock.close()
    decimg = cv2.imdecode(data, 1)

    # for cv2
    frame = cv2.flip(decimg, 1)
    cv2.namedWindow('SERVER', cv2.WINDOW_NORMAL)

    roi = frame[100:300, 100:300]
    roi_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    roi_s = roi[100:140, 80:120]
    roi_s_ycrcb = cv2.cvtColor(roi_s, cv2.COLOR_BGR2YCrCb)

    cv2.rectangle(frame, (125, 100), (275, 300), (0, 0, 255), 0)

    y = roi_s_ycrcb[20][20][0]
    cr = roi_s_ycrcb[20][20][1]
    cb = roi_s_ycrcb[20][20][2]
    predict = -1
    if y > 180:
        lower_skin = np.array([y - 20, cr - 10, cb - 10], dtype=np.uint8)
        upper_skin = np.array([y + 20, cr + 10, cb + 10], dtype=np.uint8)
        if y > 230:
            lower_skin = np.array([y - 20, cr - 10, cb - 10], dtype=np.uint8)
            upper_skin = np.array([250, cr + 10, cb + 10], dtype=np.uint8)

        mask = cv2.inRange(roi_ycrcb, lower_skin, upper_skin)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]
        cv2.drawContours(mask, cnt, -1, (255, 0, 0), 5)

        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(mask, None, fx=0.25, fy=0.25)
        # img = cv2.resize(mask, None, fx=50 / mask.shape[1], fy=50 / mask.shape[0])

        img = img / 255

        test = np.array(img)
        test = np.expand_dims(test, axis=0)

        predict = model.predict_classes(test)[0]

    cv2.imshow('SERVER', frame)

    conn.send(str(predict))

    key = cv2.waitKey(10) & 0xff
    if key == 27:
        break

cv2.destroyAllWindows()
