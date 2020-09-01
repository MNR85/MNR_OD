import cv2
from multiprocessing import Process, Value, Queue
import cvTracker
import MNR_Net

class Arbiter:
    def __init__(self, prototxt, model, trackType):
        self.runThread = Value('b', True)
        self.detectorInQ = Queue(maxsize=0)
        self.detectorOutQ = Queue(maxsize=0)
        self.trackerQ = Queue(maxsize=0)
        self.cvTracker =  cvTracker.cvTracker(trackType)
        self.detector = MNR_Net.Detector(prototxt, model)
        self.detector.setRunMode(False)
        self.detector.initNet()
        self.processingCNN = Value('b', False)
        self.firstDetect = Value('b', False)
        self.detectorP = Process(name='dnn', target=self.detectorThread)  # , args=[detector.runThread])
        self.detectorP.daemon = True  # background run
        self.trackerP = Process(name='track', target=self.trackerThread)  # , args=[detector.runThread])
        self.trackerP.daemon = True  # background run
        self.detectorP.start()
        self.trackerP.start()
        self.counter=0

    def stop(self):
        self.runThread.value=False
        self.detectorP.join()
        self.trackerP.join()

    def newImage(self, frame):
        self.counter=self.counter+1
        if(not self.processingCNN.value): # and self.counter%60==0):
            print("detect", self.trackerQ.qsize())
            self.detectorInQ.put(frame)
        if(self.trackerQ.qsize()>5):
            self.trackerQ.get() # destroy frame
        self.trackerQ.put(frame)

    def detectorThread(self):
        while (self.runThread.value):
            if (not self.detectorInQ.empty()):
                self.processingCNN.value=True
                frame = self.detectorInQ.get()
                detections = self.detector.serialDetector(frame)
                self.detectorOutQ.put(detections['detection_out'])
                self.processingCNN.value = False
                self.firstDetect.value = True

    def trackerThread(self):
        detection=[0, 0, 0, 0]
        while (self.runThread.value):
            if(not self.detectorOutQ.empty()):
                detection = self.detectorOutQ.get()
                frame = self.trackerQ.get()
                self.cvTracker.refreshTrack(frame,detection)
            # elif(self.firstDetect.value):
            while (not self.trackerQ.empty()):
                frame = self.trackerQ.get()
                (success, boxes) = self.cvTracker.track(frame)
                self.draw(frame, boxes, detection)

    def draw(self, frame, boxes, detection):
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

arbiter = Arbiter('ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.prototxt','ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.caffemodel','mosse')

cap = cv2.VideoCapture(1)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow("Raw", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        arbiter.newImage(frame)

arbiter.stop()