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
        self.detectorP = Process(name='dnn', target=self.detectorThread, args=(self.detector, self.runThread, self.detectorInQ, self.detectorOutQ , self.processingCNN, self.initCNN, self.CnnCounter))
        self.detectorP.daemon = True  # background run
        self.trackerP = Process(name='track', target=self.trackerThread, args=(self.runThread, self.detectorOutQ, self.detectorImage, self.trackerQ, self.trackerQF, self.resultQ, self.TrackCounter))
        self.trackerP.daemon = True  # background run
        self.getResultP = Process(name='getR', target=self.getResultThread, args=(self.runThread,self.resultQ,self.resultCounter))# , args=[detector.runThread])
        self.getResultP.daemon = True  # background run
        self.detectorP.start()
        self.trackerP.start()
        self.getResultP.start()
        print('waiting for init net...')
        while not self.initCNN.value:
            time.sleep(0.1)
        print('Ready')

    def stop(self):
        print("signal to stop. "),
        haltDetect=0
        lastT=0
        lastD=0
        lastR=0
        while self.trackerQ.qsize()>0 or self.detectorInQ.qsize()>0 or self.resultQ.qsize()>0:
            if((self.trackerQ.qsize()!=0 and self.trackerQ.qsize()== lastT)or (self.detectorInQ.qsize()!=0 and self.detectorInQ.qsize()== lastT)or (self.resultQ.qsize()!=0 and self.resultQ.qsize()== lastT)):
                haltDetect = haltDetect+1

            lastT = self.trackerQ.qsize()
            lastD = self.detectorInQ.qsize()
            lastR = self.resultQ.qsize()
            if(haltDetect>10):
                print("Halt detected: ",str(self.trackerQ.qsize()), str(self.detectorInQ.qsize(), str(self.resultQ.qsize())))
                break
            time.sleep(0.5)
        self.runThread.value = False
        print("detectorInQ: ", str(self.detectorInQ.qsize()), ", detectorOutQ", str(self.detectorOutQ.qsize()),
              ", detectorImage", str(self.detectorImage.qsize()), ", trackerQ: ", str(self.trackerQ.qsize()),
              ", resultQ", str(self.resultQ.qsize()))
        # print("terminate Ps, "),
        # terminate processes before join
        self.detectorP.terminate()
        self.trackerP.terminate()
        self.getResultP.terminate()
        # print("join Ps")
        # join process
        self.detectorP.join()
        self.trackerP.join()
        self.getResultP.join()
        print("detectorInQ: ", str(self.detectorInQ.qsize()), ", detectorOutQ", str(self.detectorOutQ.qsize()),
              ", detectorImage", str(self.detectorImage.qsize()), ", trackerQ: ", str(self.trackerQ.qsize()),
              ", resultQ", str(self.resultQ.qsize()))
        print("All process finished")

    def newImage(self, frame):
        # print("detectorInQ: ",str(self.detectorInQ.qsize()), ", detectorOutQ", str(self.detectorOutQ.qsize()),", detectorImage", str(self.detectorImage.qsize()), ", trackerQ: ",str(self.trackerQ.qsize()),", resultQ",str(self.resultQ.qsize()))
        if (not self.processingCNN.value):  # and self.counter%60==0):
            while (not self.trackerQ.empty()):  # empty Q
                self.trackerQ.get()
                self.trackerQF.get()
            while (not self.imageQ.empty()):  # put all image in tracker (image between current cnn and last cnn)
                self.trackerQ.put(self.imageQ.get())
                self.trackerQF.put(False)
            self.detectorInQ.put(frame)
            self.detectorImage.put(frame)
        self.imageQ.put(frame)
        self.trackerQ.put(frame)
        self.trackerQF.put(True)

    def detectorThread(self,detector, runThread, detectorInQ, detectorOutQ, processingCNN, initCNN, CnnCounter):
        print("detectorThread id = ", os.getpid())
        detector.setRunMode(self.useGpu)
        detector.initNet()
        initCNN.value = True
        while (runThread.value):
            # print("zzz", str(runThread.value), str(detectorInQ.qsize()), str(self.trackerQ.qsize()), str(self.resultQ.qsize()))
            if (not detectorInQ.empty()):
                if (detectorInQ.qsize() != 1):
                    raise Exception("Fatal error, detectorInQ has more than 1 frame!!")
                processingCNN.value = True
                frame = detectorInQ.get()
                detections = detector.serialDetector(frame)
                detectorOutQ.put(detections['detection_out'])
                processingCNN.value = False
                CnnCounter.value = CnnCounter.value + 1
        print("Done Detector")

    def trackerThread(self,runThread, detectorOutQ, detectorImage, trackerQ, trackerQF, resultQ, TrackCounter):
        print("trackerThread id = ", os.getpid())
        detection = [0, 0, 0, 0]
        while (runThread.value == True):
            # print("aaaa",str(runThread.value), str(trackerQ.qsize()))
            if (not detectorOutQ.empty()):
                detection = detectorOutQ.get()
                frame = detectorImage.get()
                self.cvTracker.refreshTrack(frame, detection)
            if (not trackerQ.empty()):
                frame = trackerQ.get()
                firstTime = trackerQF.get()
                (success, boxes) = self.cvTracker.track(frame)
                if (firstTime):
                    # resultQ.put(self.draw(frame, boxes, detection))
                    TrackCounter.value = TrackCounter.value + 1
        print("Done tracker, ", str(trackerQ.qsize()),str(resultQ.qsize()))

    def getResultThread(self,runThread,resultQ,resultCounter):
        print("getResultThread id = ", os.getpid())
        fps = FPS().start()
        counter = 0
        while (runThread.value):
            # print("bbbb", str(runThread.value), str(resultQ.qsize()))
            if (not resultQ.empty()):
                fps.update()
                counter = counter + 1
                im = resultQ.get()
                resultCounter.value = resultCounter.value + 1
                # cv2.imwrite("out_images/"+str(counter)+".jpg", im)
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