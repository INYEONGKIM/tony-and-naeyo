# -*- coding: utf-8 -*-
#!/usr/bin/python

# Desktop
import socket
import cv2
import numpy

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

print "waiting now..."

while True:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((TCP_IP, TCP_PORT))

    sock.listen(True)
    conn, addr = sock.accept()

    length = recvall(conn, 16)
    stringData = recvall(conn, int(length))
    data = numpy.fromstring(stringData, dtype='uint8')
    sock.close()
    decimg = cv2.imdecode(data, 1)

    cv2.imshow('SERVER', decimg)

    conn.send("Predict!")

    key = cv2.waitKey(10) & 0xff
    if key == 27:
        break

cv2.destroyAllWindows()
