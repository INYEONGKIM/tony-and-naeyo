# -*- coding: utf-8 -*-
#!/usr/bin/python

# ROS
import socket
import cv2
import numpy

TCP_IP = 'localhost'
TCP_PORT = 5001

while True:
    try:
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        sock.connect((TCP_IP, TCP_PORT))

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        # encoding to string
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)
        data = numpy.array(imgencode)
        stringData = data.tostring()

        sock.send(str(len(stringData)).ljust(16))
        sock.send(stringData)

        predict = sock.recv(1024)

        print "predict : ", repr(predict)

        sock.close()

        __import__('time').sleep(0.05)

        # decoding
        # decimg = cv2.imdecode(data, 1)
        # cv2.imshow('CLIENT', decimg)

        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break
    except:
        # print "socket die"
        pass

cv2.destroyAllWindows()
