#!/usr/bin/env python2

from ctypes import *
import math
import random
import cv2
import time
import numpy as np
import pyrealsense2 as rs

# for realsense
def getVerticalCoordinate(y,distance):
    # realsense RGB : FOV 60.4 x 42.5 x 77 (H V D)
    # realsense Depth : FOV 73 x 58 x 95 (H V D)
    VFov2 = math.radians(42.5 / 2)
    VSize = math.tan(VFov2) * 2
    Vcenter = (height -1 ) /2 
    VPixel = VSize/(height - 1)
    VRatio = (VCenter - y) * VPixel
    return distance * VRatio

def getHorizontalCoordinate(x, distance):
    # realsense RGB : FOV 60.4 x 42.5 x 77 (H V D)
    # realsense Depth : FOV 73 x 58 x 95 (H V D)
    HFov2 = math.radians(69.4 / 2)
    HSize = math.tan(HFov2) * 2
    Hcenter = (width -1 ) /2 
    HPixel = HSize/(width - 1)
    HRatio = (x - width) * HPixel
    return distance * HRatio   



# for darknet
def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]



#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)


def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c, h, w = arr.shape[0:3]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im, image = array_to_image(image)
    rgbgr_image(im)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh,
                             hier_thresh, None, 0, pnum)
    num = pnum[0]
    if nms: do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        a = dets[j].prob[0:meta.classes]
        if any(a):
            ai = np.array(a).nonzero()[0]
            for i in ai:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i],
                           (b.x, b.y, b.w, b.h)))

    res = sorted(res, key=lambda x: -x[1])
    if isinstance(image, bytes): free_image(im)
    free_detections(dets, num)
    return res


if __name__ == "__main__":

    # for realsense
    pipeline =rs.pipeline()
    config =rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8,30)

    pipeline.start(config)
    # for realsense

    # # load video here
    # cap = cv2.VideoCapture(3)
    # ret, img = cap.read()
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    # net = load_net("cfg/your_config.cfg", "your_weights.weights", 0)
    # net = load_net("./cfg/yolov3.cfg", "/home/nvidia/catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights/yolov3.weights", 0)

    net = load_net("/home/nvidia/catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/cfg/yolov3.cfg", "/home/nvidia/catkin_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights/yolov3.weights", 0)
    meta = load_meta("./cfg/coco.data")

    class detected_person:
        def __init__(self, x_idx, y_idx, depth):
            self.x_idx = x_idx
            self.y_idx = y_idx
            self.depth = depth

    try:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        
        while True:
            # ret, img = cap.read()

            # for realsense
            frames = pipeline.wait_for_frames()            

            # if ret:
            # r = detect(net, meta, img)

            #Depth Matching based RGB CODE
            align_to = rs.stream.color
            align = rs.align(align_to)
            aligned_frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            r = detect(net, meta, np.asanyarray(color_frame.get_data()))

            if not depth_frame or not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            depth_image = img
            # color_image = np.asanyarray(color_frame.get_data())

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # images = np.hstack((color_image, depth_colormap))

            detected_people_list = []

            for i in r:
                if i[0].decode() == 'person':
                    x, y, w, h = i[2][0], i[2][1], i[2][2], i[2][3]
                    xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                    pt1 = (xmin, ymin)
                    pt2 = (xmax, ymax)

                    # # for realsense
                    bbox =  (xmin, ymin, xmax, ymax)

                    # for realsense
                    aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                    color_frame = aligned_frames.get_color_frame()

                    # Validate that both frames are valid
                    if not aligned_depth_frame or not color_frame:
                        continue

                    depth_image = np.asanyarray(aligned_depth_frame.get_data())
                    # color_image = np.asanyarray(color_frame.get_data())

                    # Render images
                    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                    # images = np.hstack((color_image, depth_colormap))
                    
                    # ###measure the Depth
                    # cv2.circle(images,(300,300),5,(0,0,255),-1)
                    # cv2.rectangle(images, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)

                    #measuring Bouding BOX distance
                
                    distance = depth_image[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1].sum()/((bbox[3]-bbox[1])*(bbox[2]-bbox[0]))
                    distance = round(distance/1000,2)
                    # for realsense



                    # # cv2.circle(images, (300,300), 5, (0,0,255), -1)

                    # print depth_frame.get_distance(300,300)

                    # cv2.namedWindow('RS', cv2.WINDOW_AUTOSIZE)
                    # cv2.imshow('RealSense', images)


                    cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
                    # cv2.putText(img, i[0].decode() + " [" + str(round(i[1] * 100, 2)) + "]", (pt1[0], pt1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 4)
                    cv2.putText(img, i[0].decode(), (pt1[0], pt1[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 4)

                    detected_people_list.append(detected_person(x_idx=int((xmax-xmin)/2), y_idx=int((ymax-ymin)/2), depth=distance))
            

            if detected_people_list != []:
                detected_people_list.sort(key=lambda d:d.depth)
                print "x_center_idx =", detected_people_list[0].x_idx, " y_center_idx = ", detected_people_list[0].y_idx, " distance = ", detected_people_list[0].depth
            
            cv2.imshow("img", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally :
        pipeline.stop()
