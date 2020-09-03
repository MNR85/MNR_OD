import cv2
from multiprocessing import Process, Value, Queue
from imutils.video import FPS
import cvTracker
import MNR_Net

class Arbiter:
    def __init__(self, prototxt, model, trackType):
        self.runThread = Value('b', True)
        self.imageQ = Queue(maxsize=0)
        self.detectorInQ = Queue(maxsize=0)
        self.detectorOutQ = Queue(maxsize=0)
        self.trackerQ = Queue(maxsize=0)
        self.trackerQF = Queue(maxsize=0) # check if frame is first time or second time for processing
        self.resultQ = Queue(maxsize=0)
        self.cvTracker =  cvTracker.cvTracker(trackType)
        self.detector = MNR_Net.Detector(prototxt, model)
        self.detector.setRunMode(False)
        self.detector.initNet()
        self.processingCNN = Value('b', False)
        # self.firstDetect = Value('b', False)
        self.detectorP = Process(name='dnn', target=self.detectorThread)  # , args=[detector.runThread])
        self.detectorP.daemon = True  # background run
        self.trackerP = Process(name='track', target=self.trackerThread)  # , args=[detector.runThread])
        self.trackerP.daemon = True  # background run
        self.getResultP = Process(name='getR', target=self.getResultThread)  # , args=[detector.runThread])
        self.getResultP.daemon = True  # background run
        self.detectorP.start()
        self.trackerP.start()
        self.getResultP.start()
        self.counter=0

    def stop(self):
        self.runThread.value=False
        self.detectorP.join()
        self.trackerP.join()
        self.getResultP.join()
    def newImage(self, frame):
        self.counter=self.counter+1
        if(not self.processingCNN.value): # and self.counter%60==0):
            print("imageQ ", self.imageQ.qsize(), "detectQ ", self.detectorOutQ.qsize(), "TrackerQ ",self.trackerQ.qsize())
            while(not self.trackerQ.empty()): #empty Q
                self.trackerQ.get()
                self.trackerQF.get()
            if (not self.detectorOutQ.empty()):
                detection = self.detectorOutQ.get()
                self.cvTracker.refreshTrack(frame, detection)
            while(not self.imageQ.empty()): # put all image in tracker (image between current cnn and last cnn)
                self.trackerQ.put(self.imageQ.get())
                self.trackerQF.put(False)
            self.detectorInQ.put(frame)
        # if(self.trackerQ.qsize()>5):  # for bug that stacks images and low memory in long run
        #     self.trackerQ.get() # destroy frame
        self.imageQ.put(frame)
        self.trackerQ.put(frame)
        self.trackerQF.put(True)

    def detectorThread(self):
        while (self.runThread.value):
            if (not self.detectorInQ.empty()):
                self.processingCNN.value=True
                frame = self.detectorInQ.get()
                detections = self.detector.serialDetector(frame)
                self.detectorOutQ.put(detections['detection_out'])
                self.processingCNN.value = False

    def trackerThread(self):
        detection=[0, 0, 0, 0]
        while (self.runThread.value):
            if (not self.trackerQ.empty()):
                frame = self.trackerQ.get()
                firstTime = self.trackerQF.get()
                print("track ", str(firstTime), " len=?len: ",str(self.trackerQ.qsize()),"=?",str(self.trackerQF.qsize()))
                (success, boxes) = self.cvTracker.track(frame)
                self.resultQ.put(self.draw(frame, boxes, detection))

    def getResultThread(self):
        fps = FPS().start()
        while (self.runThread.value):
            if(not self.resultQ.empty()):
                fps.update()
                cv2.imshow("out", self.resultQ.get())
        fps.stop()
        print("Output frame:")
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))



    def draw(self, frame, boxes, detection):
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame


arbiter = Arbiter('ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.prototxt','ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.caffemodel','mosse')

cap = cv2.VideoCapture(1)
fpsIn = FPS().start()
try:
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            fpsIn.update()
            cv2.imshow("Raw", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            arbiter.newImage(frame)
except:
    print("Exception")
fpsIn.stop()
arbiter.stop()
cap.release()
print("Input frame:")
print("[INFO] elasped time: {:.2f}".format(fpsIn.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fpsIn.fps()))
print("Frame processed: ",str(arbiter.counter))