import os
os.environ['GLOG_minloglevel'] = '2'
# 0 - debug
# 1 - info (still a LOT of outputs)
# 2 - warnings
# 3 - errors
import cv2
from multiprocessing import Process, Value, Queue, Event
from imutils.video import FPS
import cvTracker
import MNR_Net
import time
import numpy as np


class Arbiter:
    def __init__(self, prototxt, model, useGpu, serial, trackType, ratio=5):
        self.serialProcessing=serial
        self.detectTrackRatio = ratio
        self.cvTracker = cvTracker.cvTracker(trackType)
        self.detector = MNR_Net.Detector(prototxt, model)
        self.useGpu = useGpu

        # --------- times
        self.trackTime=0
        self.detectTime = 0
        self.refreshTime = 0
        self.boxTime = 0

        # --------- process
        if(self.serialProcessing):
            print("serial mode")
            # --------- counters
            self.detectCount = 0
            self.trackCount = 0

            self.counter=0
            self.detection=[]
            self.detector.setRunMode(self.useGpu)
            self.detector.initNet()
            print("serial mode init fined")
        else:
            self.runThread = Value('b', True)
            self.stopSignal = Event()
            self.imageQ = Queue(maxsize=0)
            self.detectorInQ = Queue(maxsize=0)
            self.detectorOutQ = Queue(maxsize=0)
            self.detectorImage = Queue(maxsize=0)
            self.trackerQ = Queue(maxsize=0)
            self.trackerQF = Queue(maxsize=0)  # check if frame is first time or second time for processing
            self.resultQ = Queue(maxsize=0)
            self.initCNN = Value('b', False)
            self.processingCNN = Value('b', False)
            self.CnnCounter = Value('i', 0)
            self.RefreshCounter = Value('i', 0)
            self.TrackCounter = Value('i', 0)
            self.resultCounter = Value('i', 0)
            self.detectorP = Process(name='dnn', target=self.detectorThread, args=(self.stopSignal, self.detector, self.runThread, self.detectorInQ, self.detectorOutQ , self.processingCNN, self.initCNN, self.CnnCounter))
            self.detectorP.daemon = True  # background run
            self.trackerP = Process(name='track', target=self.trackerThread, args=(self.stopSignal, self.runThread, self.detectorOutQ, self.detectorImage, self.trackerQ, self.trackerQF, self.resultQ, self.TrackCounter,self.RefreshCounter))
            self.trackerP.daemon = True  # background run
            self.getResultP = Process(name='getR', target=self.getResultThread, args=(self.stopSignal, self.runThread, self.resultQ,self.resultCounter))# , args=[detector.runThread])
            self.getResultP.daemon = True  # background run
            self.detectorP.start()
            self.trackerP.start()
            self.getResultP.start()
            self.trackerCounterQ=0
            print('waiting for init net...')
            while not self.initCNN.value:
                time.sleep(0.1)
            print('Ready')

    def stop(self):
        if (self.serialProcessing):
            print("With ration: ",str(self.detectTrackRatio),", Detect time: ",str(self.detectTime), ", refresh time: ",str(self.refreshTime), ", track time: ",str(self.trackTime))
            print("Detect count: ",str(self.detectCount),  ", track count: ",str(self.trackCount))
            # print("Nothing to stop in serial mode")
        else:
            print("signal to stop. "),
            # self.imageQ.close()
            # self.trackerQ.close()
            # self.trackerQF.close()
            # self.detectorInQ.close()
            # self.detectorImage.close()
            self.stopSignal.set()
            while self.trackerQ.qsize()>0 or self.detectorInQ.qsize()>0 or self.resultQ.qsize()>0:
                print("DetectQ: ",str(self.detectorInQ.qsize())," Detect count: ",str(self.CnnCounter),"TrackQ: ",str(self.trackerQ.qsize())," Track count: ",str(self.TrackCounter),"Refresh Q: ",str(self.detectorOutQ.qsize())," Refresh count: ",str(self.RefreshCounter),"Result Q: ",str(self.resultQ.qsize()))
                print("."),
                time.sleep(1)

            # self.runThread.value = False
            # self.detectorP.terminate() # terminate will corrupt queue
            # self.trackerP.terminate()
            self.detectorP.join()
            self.trackerP.join()
            self.getResultP.join()
            print("All process finished")

    def newImage(self, frame):
        if (self.serialProcessing):
            if(self.counter%self.detectTrackRatio==0):
                t1 = time.time()
                self.detections = self.detector.serialDetector(frame)
                t2= time.time()
                self.cvTracker.refreshTrack(frame, self.detections['detection_out'])
                t3 = time.time()
                self.detectTime = self.detectTime + t2 - t1
                self.refreshTime = self.refreshTime + t3 - t2
                self.detectCount = self.detectCount + 1
            else:
                t3 = time.time()
                (success, boxes) = self.cvTracker.track(frame)
                t4=time.time()
                self.draw(frame, boxes, self.detections['detection_out'])
                t5=time.time()
                self.trackTime = self.trackTime + t4 - t3
                self.boxTime = self.boxTime + t5 - t4
                self.trackCount = self.trackCount + 1

            self.counter= self.counter+1
        else:
            if (not self.processingCNN.value):  # and self.counter%60==0):
                print("still have track q: ",str(self.trackerQ.qsize()),", from: ",str(self.trackerCounterQ))
                print("next queue is: ",str(self.imageQ.qsize()))
                self.trackerCounterQ=0
                while (not self.trackerQ.empty()):  # empty Q
                    self.trackerQ.get(False,0.01)
                while (not self.trackerQF.empty()):
                    self.trackerQF.get(False,0.01)
                while (not self.imageQ.empty()):  # put all image in tracker (image between current cnn and last cnn)
                    self.trackerQ.put(self.imageQ.get())
                    self.trackerQF.put(False)
                self.detectorInQ.put(frame)
                self.detectorImage.put(frame)
            self.imageQ.put(frame)
            self.trackerQ.put(frame)
            self.trackerQF.put(True)
            self.trackerCounterQ=self.trackerCounterQ+1

    def detectorThread(self,stopSignal, detector, runThread, detectorInQ, detectorOutQ, processingCNN, initCNN, CnnCounter):
        print("detectorThread id = ", os.getpid())
        detector.setRunMode(self.useGpu)
        detector.initNet()
        initCNN.value = True
        while (not stopSignal.is_set() or not detectorInQ.empty()):#runThread.value):
            if(stopSignal.is_set()):
                print("Cleanup detectorInQ: ",detectorInQ.qsize())
            if (not detectorInQ.empty()):
                if (detectorInQ.qsize() != 1):
                    print("Fatal error, detectorInQ has more than 1 frame!!")
                    # raise Exception("Fatal error, detectorInQ has more than 1 frame!!")
                processingCNN.value = True
                frame = detectorInQ.get()
                detections = detector.serialDetector(frame)
                detectorOutQ.put(detections['detection_out'])
                processingCNN.value = False
                CnnCounter.value = CnnCounter.value + 1
        print("Done Detector")

    def trackerThread(self,stopSignal, runThread, detectorOutQ, detectorImage, trackerQ, trackerQF, resultQ, TrackCounter, RefreshCounter):
        print("trackerThread id = ", os.getpid())
        detection = []
        fps=FPS().start()
        while (not stopSignal.is_set() or not detectorOutQ.empty() or not trackerQ.empty()):#runThread.value == True):
            print("sig: ",str(stopSignal.is_set())," ,dOQ: ",str(detectorOutQ.empty())," ,tQ: ",str(trackerQ.empty()))
            if (stopSignal.is_set()):
                print("Cleanup detectorOutQ: ", detectorOutQ.qsize(), ", trackerQ: ",trackerQ.qsize())
            if (not detectorOutQ.empty()):
                if (detectorOutQ.qsize()>1):
                    print("[Warning] Tracker can not keep up to detector: ",str(detectorOutQ.qsize()))
                detection = detectorOutQ.get()
                frame = detectorImage.get()
                self.cvTracker.refreshTrack(frame, detection)
                RefreshCounter.value=RefreshCounter.value+1
            if (not trackerQ.empty()):
                frame = trackerQ.get()
                firstTime = trackerQF.get()
                (success, boxes) = self.cvTracker.track(frame)
                if (firstTime):
                    # print("In tracker: ",str(TrackCounter.value))
                    resultQ.put(self.draw(frame, boxes, detection))
                    TrackCounter.value = TrackCounter.value + 1
                    fps.update()
        print("Done tracker, ", str(trackerQ.qsize()),str(resultQ.qsize()))
        # print("Tracker frame:")
        # print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        # print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    def getResultThread(self,stopSignal, runThread,resultQ,resultCounter):
        print("getResultThread id = ", os.getpid())
        fps = FPS().start()
        while (not stopSignal.is_set()):#runThread.value):
            if (not resultQ.empty()):
                im = resultQ.get()
                resultCounter.value = resultCounter.value + 1
                fps.update()
                cv2.imwrite("out_images/"+str(resultCounter.value)+".jpg", im)
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
        if(len(detection)!=0):
            h = frame.shape[0]
            w = frame.shape[1]
            box = detection[0, 0, :, 3:7] * np.array([w, h, w, h])
            box = box.astype(np.int32)
            for i in range(len(box)):
                p1 = (box[i][0], box[i][1])
                p2 = (box[i][2], box[i][3])
                cv2.rectangle(frame, p1, p2, (255,0,0), 2)
        return frame