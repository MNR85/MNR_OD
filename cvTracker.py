import cv2
import numpy as np
import time


class cvTracker():

    def __init__(self, trackType, logger=None):
        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,  # kcf > mil > boosting
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,  # weak in fast moving
            "mosse": cv2.TrackerMOSSE_create
        }
        self.CLASSES = ('background',
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
        self.trackType = trackType
        self.trackersCount = 0
        self.trackers = cv2.MultiTracker_create()

        self.logger = logger

    def refreshTrack(self, frame, detection, detectionFrameNum):
        h = frame.shape[0]
        w = frame.shape[1]
        cls = detection[0, 0, :, 1]
        conf = detection[0, 0, :, 2]
        classes=[]
        box = (detection[0, 0, :, 3:7] * np.array([w, h, w, h])).astype(np.int32)
        if (conf[0] < 0):
            self.logger.info('bad detection at frame: ' + str(detectionFrameNum) + ', continue tracking')
            return classes
        # initialize OpenCV's special multi-object tracker
        self.trackers = cv2.MultiTracker_create()
        self.trackersCount = 0
        for i in range(len(cls)):
            if (conf[i] > 0.5):
                tracker = self.OPENCV_OBJECT_TRACKERS[self.trackType]()
                (startX, startY, endX, endY) = box[i]
                if (startX < 0):
                    startX = 0
                if (startY < 0):
                    startY = 0
                if (endX > w - 1):
                    endX = w - 1
                if (endY > h - 1):
                    endY = h - 1
                bbox = (startX, startY, endX - startX, endY - startY)
                self.trackers.add(tracker, frame, bbox)
                self.trackersCount = self.trackersCount+1
                classes.append(cls[i])
        return classes

    def track(self, frame):
        # grab the updated bounding box coordinates (if any) for each
        # object that is being tracked
        (success, boxes) = self.trackers.update(frame)
        if (len(boxes)!=self.trackersCount):
            self.logger.error("Fatal error in cvTracker. "+str(len(boxes))+"!="+str(self.trackersCount), True)
        return (success, boxes)
