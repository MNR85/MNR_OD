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
        self.CLASSES = ('background',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')
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
            if(conf[i]>0.5):
                # print("%s:%.2f" % (self.CLASSES[int(cls[i])], conf[i]))
                box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
                # print(box)
                tracker = self.OPENCV_OBJECT_TRACKERS[self.trackType]()
                (startX, startY, endX, endY) = box.astype("int")
                if(startX<0):
                    startX=0
                if(startY<0):
                    startY=0
                if(endX>w-1):
                    endX=w-1
                if(endY>h-1):
                    endY=h-1
                box = (startX, startY, endX - startX, endY - startY)
                # print(box)
                self.trackers.add(tracker, frame, box)
        # print()

    def track(self, frame):
        # grab the updated bounding box coordinates (if any) for each
        # object that is being tracked
        (success, boxes) = self.trackers.update(frame)
        return (success, boxes)