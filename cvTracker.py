import cv2
import numpy as np
import time

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
        # self.totalT=0


    def refreshTrack(self, frame, detection):
        # t1 = time.time()
        h = frame.shape[0]
        w = frame.shape[1]
        cls = detection[0, 0, :, 1]
        conf = detection[0, 0, :, 2]
        # box, conf, cls = (box.astype(np.int32), conf, cls)
        box = (detection[0, 0, :, 3:7] * np.array([w, h, w, h])).astype(np.int32)
        if(conf[0]<0):
            print('bad detection, continue tracking')
            return

        # initialize OpenCV's special multi-object tracker
        self.trackers = cv2.MultiTracker_create()
        t2 = time.time()
        for i in range(len(cls)):
            if(conf[i]>0):
                # print("%s:%.2f" % (self.CLASSES[int(cls[i])], conf[i]))
                # box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
                # # print(box)
                # # tracker = self.OPENCV_OBJECT_TRACKERS[self.trackType]()
                # bbox= box[i]
                # (sX,sY,w,h)=(box[i][0],box[i][1],box[i][2]-box[i][0],box[i][3]-box[i][1])
                # bbox1=(sX, sY, w, h)
                (startX, startY, endX, endY) = box[i]
                if(startX<0):
                    startX=0
                if(startY<0):
                    startY=0
                if(endX>w-1):
                    endX=w-1
                if(endY>h-1):
                    endY=h-1
                bbox = (startX, startY, endX - startX, endY - startY)
                # print(box)
                self.trackers.add(cv2.TrackerMOSSE_create(), frame, bbox) #box)
        # print()
        t3 = time.time()
        # self.totalT = self.totalT + t3-t2
        # print("create track: ",str(t2-t1),str(t3-t2))#,str(self.totalT))

    def track(self, frame):
        # grab the updated bounding box coordinates (if any) for each
        # object that is being tracked
        (success, boxes) = self.trackers.update(frame)
        return (success, boxes)