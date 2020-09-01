import cv2
import numpy as np

class cvTracker():

    def __init__(self, trackType):
        self.OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
        }
        self.trackType=trackType
        self.trackers = cv2.MultiTracker_create()


    def refreshTrack(self, frame, detection):
        h = frame.shape[0]
        w = frame.shape[1]
        cls = detection[0, 0, :, 1]
        conf = detection[0, 0, :, 2]
        # box, conf, cls = (box.astype(np.int32), conf, cls)
        if(conf[0]<0):
            print('bad detection, continue tracking')
            return

        # initialize OpenCV's special multi-object tracker
        self.trackers = cv2.MultiTracker_create()
        for i in range(len(cls)):
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            print(box)
            tracker = self.OPENCV_OBJECT_TRACKERS[self.trackType]()
            (startX, startY, endX, endY) = box.astype("int")
            box = (startX, startY, endX - startX, endY - startY)
            self.trackers.add(tracker, frame, box)

    def track(self, frame):
        # grab the updated bounding box coordinates (if any) for each
        # object that is being tracked
        (success, boxes) = self.trackers.update(frame)
        return (success, boxes)