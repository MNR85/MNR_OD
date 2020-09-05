import os
os.environ['GLOG_minloglevel'] = '2'
# 0 - debug
# 1 - info (still a LOT of outputs)
# 2 - warnings
# 3 - errors
import cv2
from multiprocessing import Process, Value, Queue
from imutils.video import FPS
import cvTracker
import MNR_Net
import time
import math

class Arbiter:
    def __init__(self, prototxt, model, useGpu, trackType):
        self.runThread = Value('b', True)
        self.imageQ = Queue(maxsize=0)
        self.detectorInQ = Queue(maxsize=0)
        self.detectorOutQ = Queue(maxsize=0)
        self.detectorImage = Queue(maxsize=0)
        self.trackerQ = Queue(maxsize=0)
        self.trackerQF = Queue(maxsize=0)  # check if frame is first time or second time for processing
        self.resultQ = Queue(maxsize=0)
        self.cvTracker = cvTracker.cvTracker(trackType)
        self.detector = MNR_Net.Detector(prototxt, model)
        self.useGpu = useGpu
        self.initCNN = Value('b', False)
        self.processingCNN = Value('b', False)
        self.CnnCounter = Value('i', 0)
        self.TrackCounter = Value('i', 0)
        self.resultCounter = Value('i', 0)

        self.detector.setRunMode(self.useGpu)
        self.detector.initNet()
        self.counter = 0
        self.ratioAvg=0
        self.tdAvg=0
        self.ttAvg=0

    def newImage(self, frame):
        if (self.counter%10==0):
            td1=time.time()
            detections = self.detector.serialDetector(frame)
            td2=time.time()-td1
            if(self.tdAvg==0):
                self.tdAvg=td2
            self.tdAvg = (self.tdAvg+td2)/2
            print("detected: ",len(detections['detection_out'][0, 0, :, 1]))
            self.cvTracker.refreshTrack(frame, detections['detection_out'])

        tt1=time.time()
        (success, boxes) = self.cvTracker.track(frame)
        tt2=time.time()-tt1
        if (self.ttAvg == 0):
            self.ttAvg = tt2
        self.ttAvg = (self.ttAvg + tt2) / 2
        if (self.ratioAvg == 0):
            self.ratioAvg = (td2/tt2)
        self.ratioAvg = (self.ratioAvg + (td2/tt2)) / 2
        print("ratio: ",str(td2/tt2), ", track: ", str(tt2), ", detection: ", str(td2), ", fps~ ",str(1000000/(td2*td2/tt2)))
        return self.draw(frame, boxes, detections['detection_out'])

    def draw(self, frame, boxes, detection):
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame


arbiter = Arbiter('ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.prototxt','ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.caffemodel', True, 'mosse') #'mosse')
gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(1, 1920, 1080)
# cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture("test_images/soccer_01.mp4")#"test_images/soccer_01.mp4")#1)
fps = FPS().start()
counter=0
try:
    while(cap.isOpened() and counter < 100):
        ret, frame = cap.read()
        if ret == True:
            fps.update()
            res = arbiter.newImage(frame)
        counter = counter + 1
except str:
    print("Exception")
fps.stop()
cap.release()
# cv2.destroyAllWindows()
print("Avg ratio: ",str(arbiter.ratioAvg),", Avg tt: ", str(arbiter.ttAvg),", Avg td: ", str(arbiter.tdAvg))
print("Input frame:")
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("Frame Input: "+str(counter))
print("Result counter: ", str(arbiter.resultCounter.value),"CnnCounter counter: ", str(arbiter.CnnCounter.value),"TrackCounter counter: ", str(arbiter.TrackCounter.value))
print("Result q: ", str(arbiter.resultQ.qsize()),"CnnQ counter: ", str(arbiter.detectorInQ.qsize()),"TrackQ counter: ", str(arbiter.trackerQ.qsize()))