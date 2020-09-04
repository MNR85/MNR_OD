import cv2
from multiprocessing import Process, Value, Queue
from imutils.video import FPS
import cvTracker
import MNR_Net
import time

class Arbiter:
    def __init__(self, prototxt, model, useGpu, trackType):
        self.runThread = Value('b', True)
        self.imageQ = Queue(maxsize=0)
        self.detectorInQ = Queue(maxsize=0)
        self.detectorOutQ = Queue(maxsize=0)
        self.trackerQ = Queue(maxsize=0)
        self.trackerQF = Queue(maxsize=0) # check if frame is first time or second time for processing
        self.resultQ = Queue(maxsize=0)
        self.cvTracker = cvTracker.cvTracker(trackType)
        self.detector = MNR_Net.Detector(prototxt, model)
        self.useGpu = useGpu
        self.initCNN = Value('b', False)
        self.processingCNN = Value('b', False)
        self.CnnCounter = Value('i', 0)
        self.TrackCounter = Value('i', 0)
        self.resultCounter = Value('i', 0)
        self.detectorP = Process(name='dnn', target=self.detectorThread)  # , args=[detector.runThread])
        self.detectorP.daemon = True  # background run
        self.trackerP = Process(name='track', target=self.trackerThread)  # , args=[detector.runThread])
        self.trackerP.daemon = True  # background run
        self.getResultP = Process(name='getR', target=self.getResultThread)  # , args=[detector.runThread])
        self.getResultP.daemon = True  # background run
        self.detectorP.start()
        self.trackerP.start()
        self.getResultP.start()
        print('waiting for init net...')
        while not self.initCNN.value:
            time.sleep(0.1)
        print('Ready')


    def stop(self):
        self.runThread.value=False
        self.detectorP.join()
        self.trackerP.join()
        self.getResultP.join()

    def newImage(self, frame):
        if(not self.processingCNN.value): # and self.counter%60==0):
            # print("imageQ ", self.imageQ.qsize(), "detectQ ", self.detectorOutQ.qsize(), "TrackerQ ",self.trackerQ.qsize())
            while(not self.trackerQ.empty()): #empty Q
                self.trackerQ.get()
                self.trackerQF.get()
            while(not self.imageQ.empty()): # put all image in tracker (image between current cnn and last cnn)
                self.trackerQ.put(self.imageQ.get())
                self.trackerQF.put(False)
            self.detectorInQ.put(frame)
        self.imageQ.put(frame)
        self.trackerQ.put(frame)
        self.trackerQF.put(True)

    def detectorThread(self):
        self.detector.setRunMode(self.useGpu)
        self.detector.initNet()
        self.initCNN.value=True
        while (self.runThread.value):
            if (not self.detectorInQ.empty()):
                self.processingCNN.value=True
                frame = self.detectorInQ.get()
                detections = self.detector.serialDetector(frame)
                self.detectorOutQ.put(detections['detection_out'])
                self.processingCNN.value = False
                self.CnnCounter.value=self.CnnCounter.value+1

    def trackerThread(self):
        detection=[0, 0, 0, 0]
        while (self.runThread.value):
            if (not self.detectorOutQ.empty()):
                detection = self.detectorOutQ.get()
                self.cvTracker.refreshTrack(frame, detection)
            if (not self.trackerQ.empty()):
                frame = self.trackerQ.get()
                firstTime = self.trackerQF.get()
                # print("track ", str(firstTime), " len=?len: ",str(self.trackerQ.qsize()),"=?",str(self.trackerQF.qsize()))
                (success, boxes) = self.cvTracker.track(frame)
                if(firstTime):
                    self.resultQ.put(self.draw(frame, boxes, detection))
                    self.TrackCounter.value=self.TrackCounter.value+1

    def getResultThread(self):
        fps = FPS().start()
        while (self.runThread.value):
            if(not self.resultQ.empty()):
                fps.update()
                im=self.resultQ.get()
                self.resultCounter.value = self.resultCounter.value+1
                # cv2.imwrite("test_images/"+str(counter)+".jpg", im)
                # cv2.imshow("tracker",im)
                # cv2.waitKey(1)
        fps.stop()
        print("Output frame:")
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


    def draw(self, frame, boxes, detection):
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame


arbiter = Arbiter('ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.prototxt','ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.caffemodel', True, 'mosse') #'mosse')

cap = cv2.VideoCapture("test_images/soccer_01.mp4") #1)
fps = FPS().start()
counter=0
try:
    while(cap.isOpened() and counter < 20):
        ret, frame = cap.read()
        if ret == True:
            fps.update()
            # cv2.imshow("Raw", frame)
            counter=counter+1
            # key = cv2.waitKey(1) & 0xFF
            # # if the `q` key was pressed, break from the loop
            # if key == ord("q"):
            #     break
            arbiter.newImage(frame)
except:
    print("Exception")
fps.stop()
arbiter.stop()
cap.release()
# cv2.destroyAllWindows()
print("Input frame:")
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("Frame Input: "+str(counter))
print("Result counter: ", str(arbiter.resultCounter.value),"CnnCounter counter: ", str(arbiter.CnnCounter.value),"TrackCounter counter: ", str(arbiter.TrackCounter.value))