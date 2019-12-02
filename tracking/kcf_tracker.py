import cv2
import sys

def momentum():
    # TODO
    pass

def Main():
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')

    tracker_type = 'KCF' # KCF, MOSSE, MEDIANFLOW

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        # minor_ver = 4
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()

    video = cv2.VideoCapture(0)
    cv2.namedWindow('Tracking', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('Tracking', 0, 0)

    if not video.isOpened():
        print "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    frame = cv2.flip(frame, 3)

    if not ok:
        print 'Cannot read video file'
        sys.exit()

    # # self bbox setting
    # bbox = (287, 23, 86, 320)

    cv2.namedWindow('Select ROI', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('Select ROI', 0, 400)

    bbox = cv2.selectROI('Select ROI', frame, False)
    before_bbox = bbox

    _, _, originWidth, originHeight = [float(i) for i in bbox]

    # Initialize tracker with first frame and bounding box

    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        frame = cv2.flip(frame, 3)

        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        past_bbox = before_bbox
        before_bbox = bbox

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        left, top, width, height = [int(i) for i in bbox]

        past_vector = (before_bbox[0] - past_bbox[0], before_bbox[1] - before_bbox[1])
        before_vector = (left - before_bbox[0], top - before_bbox[1])

        if abs(before_bbox[0] - left) < 4:
            target_direction = "STOP"
        else:
            target_direction = "LEFT" if left - before_bbox[0] < 0 else "RIGHT"

        print "vector : ", past_vector, before_vector, target_direction, " | bbox : ", bbox[0], bbox[1]

        if ok:
            # Tracking success
            p1 = (left, top)
            p2 = (left + width, top + height)

            # print bbox
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            new_left, new_top = before_bbox[0] + before_vector[0], before_bbox[1] + before_vector[1]
            p1 = (int(new_left), int(new_top))
            p2 = (int(new_left + width), int(new_top + height))
            bbox = (new_left, new_top, width, height)

            print "fail in here, now bbox = ", bbox[0], bbox[1],
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            cv2.putText(frame, "Tracking failure detected", (100, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "Target Direction : " + target_direction, (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        # Display result
        cv2.imshow("Tracking", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

if __name__ == '__main__':
    Main()