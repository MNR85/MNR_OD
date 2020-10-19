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
    def __init__(self, prototxt, model, useGpu, serial, trackType, logger=None, ratio=5):
        self.serialProcessing=serial
        self.detectTrackRatio = ratio
        self.cvTracker = cvTracker.cvTracker(trackType, logger)
        self.detector = MNR_Net.Detector(prototxt, model)
        self.useGpu = useGpu

        # --------- logger
        # if (logger is None):
        #     self.print = print
        # else:
        #     self.print = logger.info
        self.logger=logger
        # --------- draw box
        self.CLASSES = ('background',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

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
            # self.runThread = Value('b', True)
            # self.stopSignal = Event()
            self.stopSignal="done"
            self.imageQ = Queue(maxsize=0)  # main
            self.detectorInQ = Queue(maxsize=0) # main -> detector
            self.detectorOutQ = Queue(maxsize=0) # detector -> tracker
            self.detectorImage = Queue(maxsize=0) # main -> tracker
            self.trackerQ = Queue(maxsize=0) # main -> tracker
            self.resultQ = Queue(maxsize=0) # tracker -> getresult
            self.initCNN = Value('b', False)
            self.processingCNN = Value('b', False)
            self.CnnCounter = Value('i', 0)
            self.RefreshCounter = Value('i', 0)
            self.TrackCounter = Value('i', 0)
            self.resultCounter = Value('i', 0)
            self.detectorP = Process(name='dnn', target=self.detectorThread, args=(self.detector, self.detectorInQ, self.detectorOutQ , self.processingCNN, self.initCNN, self.CnnCounter))
            self.detectorP.daemon = True  # background run
            self.trackerP = Process(name='track', target=self.trackerThread, args=(self.detectorOutQ, self.detectorImage, self.trackerQ, self.resultQ, self.TrackCounter,self.RefreshCounter))
            self.trackerP.daemon = True  # background run
            self.getResultP = Process(name='getR', target=self.getResultThread, args=(self.resultQ,self.resultCounter))# , args=[detector.runThread])
            self.getResultP.daemon = True  # background run
            self.detectorP.start()
            print('waiting for init net...')
            while not self.initCNN.value:
                time.sleep(0.1)
            self.logger.start()
            self.trackerP.start()
            self.getResultP.start()
            self.trackerCounterQ=0
            print('Ready')

    def stop(self):
        if (self.serialProcessing):
            self.logger.info("With ration: "+str(self.detectTrackRatio)+", Detect time: "+str(self.detectTime)+ ", refresh time: "+str(self.refreshTime)+ ", track time: "+str(self.trackTime))
            self.logger.info("Detect count: "+str(self.detectCount)+  ", track count: "+str(self.trackCount))
        else:
            self.logger.info("signal to stop. ")
            self.detectorInQ.put(self.stopSignal)
            self.trackerQ.put(self.stopSignal)
            while(self.imageQ.qsize()>0):
                self.imageQ.get()
            while self.trackerQ.qsize()>0 or self.detectorInQ.qsize()>0 or self.resultQ.qsize()>0:
                self.logger.info("DetectQ: "+str(self.detectorInQ.qsize())+"TrackQ: "+str(self.trackerQ.qsize())+"Refresh Q: "+str(self.detectorOutQ.qsize())+"Result Q: "+str(self.resultQ.qsize()))
                self.logger.flush()
                time.sleep(1)
            self.detectorP.join()
            self.trackerP.join()
            self.getResultP.join()
            self.logger.info("All process finished")
            self.logger.info("With pipeline")
            self.logger.info("Detect count: "+ str(self.CnnCounter)+ ", track count/2: "+ str(self.TrackCounter))

    def newImage(self, frame, counter):
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
                self.trackerCounterQ=0
                remainTrackQ=self.trackerQ.qsize()
                while (not self.trackerQ.empty()):  # empty Q
                    # print("wait for getting tracker: ",str(self.trackerQ.qsize()))
                    time.sleep(0.1)
                self.logger.info("will add "+str(self.imageQ.qsize())+", remained from last: "+str(remainTrackQ))
                while (not self.imageQ.empty()):  # put all image in tracker (image between current cnn and last cnn)
                    self.trackerQ.put([self.imageQ.get(), False])
                self.detectorInQ.put([frame, counter])
                self.detectorImage.put(frame)
            self.imageQ.put(frame)
            self.trackerQ.put([frame, True ,counter, time.time()]) # frame, isFirstTime, frameCounter, frameInputTime
            self.trackerCounterQ=self.trackerCounterQ+1

    def detectorThread(self, detector, detectorInQ, detectorOutQ, processingCNN, initCNN, CnnCounter):
        self.logger.info("detectorThread id = "+str(os.getpid()))
        detector.setRunMode(self.useGpu)
        detector.initNet()
        initCNN.value = True
        while (True):
            if (not detectorInQ.empty()):
                if (detectorInQ.qsize() != 1):
                    self.logger.error("Fatal error, detectorInQ has more than 1 frame!!")
                    qsize=detectorInQ.qsize()
                    imagesInQ=""
                    for i in range(1,qsize):
                        tmp=detectorInQ.get()
                        imagesInQ = imagesInQ+str(frame[1])+" ,"
                        detectorInQ.put(tmp)
                    self.logger.error("We had "+str(qsize)+" image that were from numbers: "+imagesInQ)
                    # raise Exception("Fatal error, detectorInQ has more than 1 frame!!")
                processingCNN.value = True
                frame = detectorInQ.get()
                if(frame==self.stopSignal):
                    break
                detections = detector.serialDetector(frame[0])
                detectorOutQ.put([detections['detection_out'], frame[1]]) # detection out, frame num
                # frame[0]=self.draw(frame[0], detections['detection_out'])
                # self.logger.imwrite("detect_images/" + str(CnnCounter.value) + ".jpg", frame[0])
                processingCNN.value = False
                CnnCounter.value = CnnCounter.value + 1
        detectorOutQ.put(self.stopSignal)
        self.logger.info("Done Detector: "+str(detectorInQ.qsize()))

    def trackerThread(self,detectorOutQ, detectorImage, trackerQ, resultQ, TrackCounter, RefreshCounter):
        self.logger.info("trackerThread id = "+ str(os.getpid()))
        detection = []
        detectionFrameNum = -1
        detectionTime= "x"
        detectionCount=0
        fps=FPS().start()
        detectorDone=False
        trackerDone=False
        while (True):
            if (detectorDone and trackerDone):
                break
            if (not detectorOutQ.empty()):
                if (detectorOutQ.qsize()>1):
                    self.logger.warning("[Warning] Tracker can not keep up to detector: "+str(detectorOutQ.qsize()))
                detectionOut = detectorOutQ.get()
                if (detectionOut == self.stopSignal):
                    self.logger.info ("see detectorOut done")
                    detectorDone=True
                    continue
                detectionTime = time.time()
                detectionFrameNum =detectionOut[1]
                detection = detectionOut[0]
                detectionCount = len(detection[0, 0, :, 1])
                frame = detectorImage.get()
                self.cvTracker.refreshTrack(frame, detection)
                RefreshCounter.value=RefreshCounter.value+1
            if (not trackerQ.empty()):
                frame = trackerQ.get()
                if (frame == self.stopSignal):
                    self.logger.info ("See trackerQ done")
                    trackerDone=True
                    continue
                (success, boxes) = self.cvTracker.track(frame[0])
                if (frame[1]):
                    # frame frameNum frameInputTime, trackOutTime, detectNum, detectOutTime
                    resultQ.put([self.draw(frame[0], detection, success, boxes), frame[2], frame[3], time.time(), detectionFrameNum, detectionTime, detectionCount])
                    TrackCounter.value = TrackCounter.value + 1
                    fps.update()
        resultQ.put(self.stopSignal)
        fps.stop()
        self.logger.info("Done tracker, "+ str(trackerQ.qsize())+" "+str(resultQ.qsize()))
        self.logger.info("Tracker elasped time: {:.2f}".format(fps.elapsed()))
        self.logger.info("Tracker approx. FPS: {:.2f}".format(fps.fps()))

    def getResultThread(self,resultQ,resultCounter):
        self.logger.info("getResultThread id = "+ str(os.getpid()))
        startTime=-1
        lastInputTime = -1
        lastTrackOutTime = -1
        lastDetectOutTime = -1
        fps = FPS().start()
        while (True):
            if (not resultQ.empty()):
                im = resultQ.get()
                if(im==self.stopSignal):
                    break
                resultCounter.value = resultCounter.value + 1
                fps.update()
                self.logger.imwrite(format(resultCounter.value, '05d')+".jpg", im[0])
                frameNum=im[1]
                if(startTime==-1):
                    startTime=im[2]
                inputTime=im[2]-startTime
                trackOutTime=im[3]-startTime
                frameDetectNum=im[4]
                if (im[5] == "x"):
                    im[5] = im[2]
                detectOutTime=im[5]-startTime
                detectCount=im[6]
                self.logger.csv(str(frameNum)+ ", "+ str(frameDetectNum)+ ", "+ str(detectCount)+", "+str(inputTime)+", "+str(inputTime-lastInputTime)+ ", "+str(trackOutTime)+ ", "+str(trackOutTime-lastTrackOutTime)+ ", "+str(detectOutTime)+ ", "+str(detectOutTime-lastDetectOutTime)+", "+str(trackOutTime-inputTime)+", "+str(trackOutTime-detectOutTime))
                lastInputTime=inputTime
                lastTrackOutTime=trackOutTime
                lastDetectOutTime=detectOutTime
                # cv2.imshow("tracker",im)
                # cv2.waitKey(1)
        self.logger.info("Done Get result: "+ str(resultQ.qsize()))
        fps.stop()
        self.logger.info("[INFO] Get result elasped time: {:.2f}".format(fps.elapsed()))
        self.logger.info("[INFO] Get result approx. FPS: {:.2f}".format(fps.fps()))

    def draw(self, frame, detection, validTrackBoxes=False, trackBoxes=[]):
        for box in trackBoxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            sub_img = frame[y:y + h, x:x + w]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255 if validTrackBoxes else 128

            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

            # Putting the image back to its position
            frame[y:y + h, x:x + w] = res
        if(len(detection)!=0):
            h = frame.shape[0]
            w = frame.shape[1]
            box = detection[0, 0, :, 3:7] * np.array([w, h, w, h])
            cls = detection[0, 0, :, 1]
            conf = detection[0, 0, :, 2]
            box = box.astype(np.int32)
            for i in range(len(box)):
                p1 = (box[i][0], box[i][1])
                p2 = (box[i][2], box[i][3])
                cv2.rectangle(frame, p1, p2, (0,0,255), 2)
                p3 = (max(p1[0], 15), max(p1[1], 15))
                title = "%s:%.2f" % (self.CLASSES[int(cls[i])], conf[i])
                cv2.rectangle(frame, (p3[0],p3[1]-20), (p3[0]+100, p3[1]), (0, 0, 0), -1)
                cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1)
        return frame


