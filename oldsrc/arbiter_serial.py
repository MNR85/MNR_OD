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
        self.ratioAvg=0 # track/detect ratio
        self.tdAvg=0 # detection time average
        self.trAvg=0 # refresh track " "
        self.ttAvg=0 # track " "
        self.td2 = 0 # time detection 2
        self.det=0 # total detection time
        self.ref=0 # total refresh time
        self.tra=0 # total track time

    def newImage(self, frame):
        if (self.counter%5==0):
            td1=time.time()
            detections = self.detector.serialDetector(frame)
            self.td2=time.time()-td1
            # self.det = self.det + self.td2
            if(self.tdAvg==0):
                self.tdAvg=self.td2
            self.tdAvg = (self.tdAvg+self.td2)/2
            self.det = self.det + self.td2
            tr1=time.time()
            self.cvTracker.refreshTrack(frame, detections['detection_out'])
            tr2=time.time() - tr1
            self.ref = self.ref+tr2
            if (self.trAvg == 0):
                self.trAvg = tr2
            self.trAvg = (self.trAvg + tr2) / 2
            print("Detect in: ",str(self.td2))
            print("Refresh in: ",str(tr2))
        else:
            tt1=time.time()
            (success, boxes) = self.cvTracker.track(frame)
            tt2 = time.time() - tt1
            if (self.ttAvg == 0):
                self.ttAvg = tt2
            self.ttAvg = (self.ttAvg + tt2) / 2
            if (self.ratioAvg == 0):
                self.ratioAvg = (self.td2/tt2)
            self.ratioAvg = (self.ratioAvg + (self.td2/tt2)) / 2
            self.tra = self.tra + time.time() - tt1
            print("Track in: ", str(time.time() - tt1))
        self.counter =self.counter+1
        return frame
        # print("ratio: ",str(td2/tt2), ", track: ", str(tt2), ", detection: ", str(td2), ", fps~ ",str(1000000/(td2*td2/tt2)))
        # return self.draw(frame, boxes, detections['detection_out'])



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
t2=0
try:

    while(cap.isOpened() and counter < 100):
        ret, frame = cap.read()
        if ret == True:
            fps.update()
            t1 = time.time()
            res = arbiter.newImage(cv2.resize(frame, (300, 300)) )
            # print("newImage in: ",str(time.time()-t1))
            t2 = t2 + (time.time()-t1)
        counter = counter + 1
    print("Execution: ",str(t2), ", tra: "+str(arbiter.tra)+", det: "+str(arbiter.det)+", refresh: "+str(arbiter.ref))
except str:
    print("Exception")
fps.stop()
cap.release()
# cv2.destroyAllWindows()
print("Avg ratio: ",str(arbiter.ratioAvg),", Avg tt: ", str(arbiter.ttAvg),", Avg td: ", str(arbiter.tdAvg),", Avg ref: ", str(arbiter.trAvg))
print("Input frame:")
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("Frame Input: "+str(counter))
print("Result counter: ", str(arbiter.resultCounter.value),"CnnCounter counter: ", str(arbiter.CnnCounter.value),"TrackCounter counter: ", str(arbiter.TrackCounter.value))
print("Result q: ", str(arbiter.resultQ.qsize()),"CnnQ counter: ", str(arbiter.detectorInQ.qsize()),"TrackQ counter: ", str(arbiter.trackerQ.qsize()))