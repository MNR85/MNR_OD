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
from scipy.optimize import linear_sum_assignment
# from collections import deque
import xml.etree.ElementTree as ET


class Arbiter:
    def __init__(self, prototxt, model, useGpu, serial, trackType, fixedTrackType=True, logger=None, eval=False,
                 evalPath="", detectTrackRatio=5, fixedRatio=False, debugMode=False):
        # --------- logger
        self.logger = logger

        # --------- arbiter objects
        self.serialProcessing = serial
        self.detectTrackRatio = int(detectTrackRatio) if (not detectTrackRatio is None) else 5
        self.cvTracker = cvTracker.cvTracker(trackType, logger)
        self.detector = MNR_Net.Detector(prototxt, model)
        self.useGpu = useGpu
        self.eval = eval
        self.evalPath = evalPath
        self.fixedRatio = fixedRatio
        self.debugMode = debugMode

        # --------- msg
        msgMode = "Serial" if self.serialProcessing else "Pipeline"
        msgGPU = "GPU" if self.useGpu else "CPU"
        msgRatio = ("fixed ratio " + str(100 / self.detectTrackRatio) + "%") if self.fixedRatio else "dynamic ratio"
        msgTracker = ("fixed tracker: " + trackType) if fixedTrackType else "dynamic tracker"
        msgEval = ("with evaluate from " + evalPath) if eval else "with no evaluate"
        self.logger.info(
            msgMode + " processing using " + msgGPU + " and detect-track with " + msgRatio + " using " + msgTracker + " and " + msgEval)

        # --------- draw box
        self.CLASSES = ('background',
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
        # from http://image-net.org/challenges/LSVRC/2015/browse-vid-synsets
        self.imageNetDictionary = {
            'n02691156': 'aeroplane',
            'n02834778': 'bicycle',
            'n01503061': 'bird',
            'n02924116': 'bus',
            'n02958343': 'car',
            'n02402425': 'cow',
            'n02084071': 'dog',
            'n02121808': 'cat',
            'n02374451': 'horse',
            'n03790512': 'motorbike',
            'n04468005': 'train',
        }
        self.logger.start(0)
        # --------- process
        if (self.serialProcessing):
            self.detector.setRunMode(self.useGpu)
            self.detector.initNet()
            self.startTime = time.time()
            self.trackOutTime = time.time() - self.startTime
            self.lastInputTime = time.time() - self.startTime
            self.lastTrackOutTime = time.time() - self.startTime
            self.lastDetectOutTime = time.time() - self.startTime
            self.lastFrameDetectNum = -1
            self.fps = FPS().start()
        else:
            self.stopSignal = "done"
            self.imageQ = Queue(maxsize=0)  # main
            self.detectorInQ = Queue(maxsize=0)  # main -> detector
            self.detectorOutQ = Queue(maxsize=0)  # detector -> tracker
            self.detectorImage = Queue(maxsize=0)  # main -> tracker
            self.trackerQ = Queue(maxsize=0)  # main -> tracker
            self.resultQ = Queue(maxsize=0)  # tracker -> getresult
            self.initCNN = Value('b', False)
            self.processingCNN = Value('b', False)
            self.detectorP = Process(name='dnn', target=self.detectorThread, args=(
                self.detector, self.detectorInQ, self.detectorOutQ, self.processingCNN, self.initCNN))
            self.detectorP.daemon = True  # background run
            self.trackerP = Process(name='track', target=self.trackerThread, args=(
                self.detectorOutQ, self.detectorImage, self.trackerQ, self.resultQ))
            self.trackerP.daemon = True  # background run
            self.getResultP = Process(name='getR', target=self.getResultThread, args=(self.resultQ,))
            self.getResultP.daemon = True  # background run
            self.detectorP.start()
            while not self.initCNN.value:
                time.sleep(0.1)
            self.trackerP.start()
            self.startTime = time.time()
            self.getResultP.start()
            self.trackerCounterQ = 0
        self.cvTracker.logger.startTime = self.logger.startTime = time.time()
        self.logger.info('Finished initilization')
        self.logger.flush()

    def stop(self):
        self.logger.info("Stoping.")
        if (self.serialProcessing):
            self.fps.stop()
            self.logger.info("Done serial: elasped time: {:.2f}".format(self.fps.elapsed()), True)
            self.logger.info("Done serial: approx. FPS: {:.2f}".format(self.fps.fps()), True)
        else:
            self.detectorInQ.put(self.stopSignal)
            self.trackerQ.put(self.stopSignal)
            self.logger.info("Image in imageQ for flush: " + str(self.imageQ.qsize()))
            while (self.imageQ.qsize() > 0):
                self.imageQ.get()
            self.logger.flush()
            watchDog=0
            while self.trackerQ.qsize() > 0 or self.detectorInQ.qsize() > 0 or self.resultQ.qsize() > 0:
                self.logger.info("DetectQ: " + str(self.detectorInQ.qsize()) + ", TrackQ: " + str(
                    self.trackerQ.qsize()) + ", Refresh Q: " + str(self.detectorOutQ.qsize()) + ", Result Q: " + str(
                    self.resultQ.qsize()))
                self.logger.flush()
                time.sleep(1)
                watchDog=watchDog+1
                if (watchDog>5):
                    self.logger.Err("Something went wront. stuck at processes")
                    break
            self.logger.info("Image in detectorImage for flush: " + str(self.detectorImage.qsize()))
            while (self.detectorImage.qsize() > 0):
                self.detectorImage.get()
            if(watchDog>5):
                self.logger.Warn("Starting emergency killing processes")
                self.logger.Warn("Image in trackerQ for flush: " + str(self.trackerQ.qsize()))
                while (self.trackerQ.qsize() > 0):
                    self.trackerQ.get()
                self.logger.Warn("Image in detectorInQ for flush: " + str(self.detectorInQ.qsize()))
                while (self.detectorInQ.qsize() > 0):
                    self.detectorInQ.get()
                self.logger.Warn("Image in detectorOutQ for flush: " + str(self.detectorOutQ.qsize()))
                while (self.detectorOutQ.qsize() > 0):
                    self.detectorOutQ.get()
                self.logger.Warn("Image in resultQ for flush: " + str(self.resultQ.qsize()))
                while (self.resultQ.qsize() > 0):
                    self.resultQ.get()
                self.logger.Warn("Terminating processes")
                self.detectorP.terminate()
                self.trackerP.terminate()
                self.getResultP.terminate()
            else:
                self.logger.Info("Joining processes")
                self.detectorP.join()
                self.trackerP.join()
                self.getResultP.join()
            self.logger.info("All process finished")
            self.logger.info("With pipeline")
        self.logger.flush()

    def newImage(self, frame, frameNum):
        if (self.serialProcessing):  # all pipeline process are mixed here!
            if self.debugMode:
                inputTime = time.time() - self.startTime

            if (frameNum % self.detectTrackRatio == 0):  # detection
                self.detection = self.detector.serialDetector(frame)['detection_out']
                self.frameDetectNum = frameNum
                if self.debugMode:
                    self.detectOutTime = time.time() - self.startTime
                    self.detectCount = len(self.detection[0, 0, :, 1]) if self.detection[0, 0, :, 1][0] != -1 else 0
                (succesTrack, boxesTrack, self.clasTrack) = ([], [], [])
            else:  # tracking
                if (frameNum == self.frameDetectNum + 1):
                    self.clasTrack = self.cvTracker.refreshTrack(frame, self.detection, frameNum)
                (succesTrack, boxesTrack) = self.cvTracker.track(frame)
                if (self.eval and len(boxesTrack) != 0):
                    boxesTrack = np.asarray([boxesTrack[:, 0], boxesTrack[:, 1],
                                             boxesTrack[:, 0] + boxesTrack[:, 2],
                                             boxesTrack[:, 1] + boxesTrack[:,
                                                                3]]).transpose()
                if (self.debugMode):
                    self.trackOutTime = time.time() - self.startTime
            if (self.eval):
                boxesGT, newDetection = self.evaluate(self.detection, frame, frameNum,
                                                      self.frameDetectNum, self.lastFrameDetectNum,
                                                      boxesTrack, self.clasTrack)
                self.logger.csv(str(frameNum) + ", " + str(self.frameDetectNum) + ", " + str(
                    self.detectCount) + ", " + str(
                    inputTime) + ", " + str(inputTime - self.lastInputTime) + ", " + str(
                    self.trackOutTime) + ", " + str(
                    self.trackOutTime - self.lastTrackOutTime) + ", " + str(
                    self.detectOutTime) + ", " + str(
                    self.detectOutTime - self.lastDetectOutTime) + ", " + str(
                    self.trackOutTime - inputTime) + ", " + str(
                    self.trackOutTime - self.detectOutTime))
                self.lastInputTime = inputTime
                self.lastTrackOutTime = self.trackOutTime
                self.lastDetectOutTime = self.detectOutTime
                self.lastFrameDetectNum = self.frameDetectNum
                if (self.debugMode):
                    self.logger.imwrite(format(frameNum, '05d') + ".jpg",
                                        self.draw(frame, self.detection, succesTrack, boxesTrack,
                                                  boxesGT, newDetection))
            self.fps.update()
        else:
            while (self.fixedRatio and (frameNum % self.detectTrackRatio == 0) and self.processingCNN.value):
                time.sleep(0.001)  # this or pass??
            if (not self.processingCNN.value and (
                    not self.fixedRatio or frameNum % self.detectTrackRatio == 0)):  # and counter%60==0):
                if (self.debugMode):
                    remainTrackQ = self.trackerQ.qsize()
                    self.logger.info(
                        "will add " + str(self.imageQ.qsize()) + ", remained from last: " + str(remainTrackQ))
                while (not self.trackerQ.empty()):  # empty Q wait for getting tracker
                    time.sleep(0.001)
                while (not self.imageQ.empty()):  # put all image in tracker (image between current cnn and last cnn)
                    self.trackerQ.put([self.imageQ.get(), False])
                self.detectorInQ.put([frame, frameNum])
                self.detectorImage.put(frame)
                # t1 = time.time()
                # while(not self.processingCNN.value):
                #     pass
                # self.logger.warning("timed passed to fetch detect image: "+str(time.time()-t1), True, True)
            self.imageQ.put(frame)
            self.trackerQ.put([frame, True, frameNum, time.time()])  # frame, isFirstTime, frameCounter, frameInputTime

    def detectorThread(self, detector, detectorInQ, detectorOutQ, processingCNN, initCNN):
        detector.setRunMode(self.useGpu)
        detector.initNet()
        initCNN.value = True
        while (True):
            if (not detectorInQ.empty()):
                processingCNN.value = True
                if (detectorInQ.qsize() != 1):
                    self.logger.Warning("Check in progress... detectorInQ has more than 1 frame!!")
                    imagesInQ = ""
                    unwantedFrameCount = 0
                    myStack = []
                    for i in range(1, detectorInQ.qsize()):
                        tmp = detectorInQ.get()
                        myStack.append(tmp)
                        if (tmp != self.stopSignal):
                            unwantedFrameCount = unwantedFrameCount + 1
                            imagesInQ = imagesInQ + str(tmp[1]) + " ,"
                    for i in range(1, len(myStack)):
                        detectorInQ.put(myStack.pop())
                    if (unwantedFrameCount > 1):
                        self.logger.error("Fatal error. We had " + str(
                            unwantedFrameCount) + " image that were from numbers: " + imagesInQ)
                frame = detectorInQ.get()
                if (frame == self.stopSignal):
                    self.logger.info("see detectorIn done", True)
                    break
                detections = detector.serialDetector(frame[0])
                detectorOutQ.put([detections['detection_out'], frame[1]])  # detection out, frame num
                processingCNN.value = False
        detectorOutQ.put(self.stopSignal)
        self.logger.info("Done Detector: " + str(detectorInQ.qsize()), True)

    def trackerThread(self, detectorOutQ, detectorImage, trackerQ, resultQ):
        detection = []
        priorClass = []
        detectionFrameNum = -1
        detectionTime = "x"
        detectionCount = 0
        fps = FPS().start()
        detectorDone = False
        trackerDone = False
        while (True):
            if (detectorDone and trackerDone):
                break
            if (not detectorOutQ.empty()):
                if (detectorOutQ.qsize() > 1):
                    unwantedDetectorOut = 0
                    imagesInQ = ""
                    myStack = []
                    for i in range(1, detectorOutQ.qsize()):
                        tmp = detectorOutQ.get()
                        myStack.append(tmp)
                        if (tmp != self.stopSignal):
                            unwantedDetectorOut = unwantedDetectorOut + 1
                            imagesInQ = imagesInQ + str(tmp[1]) + " ,"
                    for i in range(1, len(myStack)):
                        detectorOutQ.put(myStack.pop())
                    if (unwantedDetectorOut > 1):
                        self.logger.error("Fatal error. Tracker can not keep up to detector: " + str(
                            detectorOutQ.qsize()) + " image that were from numbers: " + imagesInQ)
                detectionOut = detectorOutQ.get()
                if (detectionOut == self.stopSignal):
                    self.logger.info("see detectorOut done", True)
                    detectorDone = True
                    continue
                detectionTime = time.time()
                detectionFrameNum = detectionOut[1]
                detection = detectionOut[0]
                detectionCount = len(detection[0, 0, :, 1]) if detection[0, 0, :, 1][0] != -1 else 0
                frame = detectorImage.get()
                tmpPriorClass = self.cvTracker.refreshTrack(frame, detection, detectionFrameNum)
                if ( tmpPriorClass != [] ):
                    priorClass = tmpPriorClass
            if (not trackerQ.empty()):
                frame = trackerQ.get()
                if (frame == self.stopSignal):
                    self.logger.info("See trackerQ done", True)
                    trackerDone = True
                    continue
                (success, boxes) = self.cvTracker.track(frame[0])
                if (frame[1]):
                    if (len(success) != len(priorClass)):
                        self.logger.error(
                            "Fatal error. Unmatched track class and box number for image: " + str(frame[2]) + "->" + str(
                                len(success)) + ":" + str(len(priorClass)))
                    # frame frameNum frameInputTime, trackOutTime, detectNum, detectOutTime
                    result = [frame[0], frame[2], frame[3], time.time(), detectionFrameNum, detectionTime,
                              detectionCount, detection, boxes, success, priorClass]
                    resultQ.put(result)
                    fps.update()
        resultQ.put(self.stopSignal)
        fps.stop()
        self.logger.info("Done tracker, " + str(trackerQ.qsize()) + " " + str(resultQ.qsize()), True)
        self.logger.info("Tracker elasped time: {:.2f}".format(fps.elapsed()), True)
        self.logger.info("Tracker approx. FPS: {:.2f}".format(fps.fps()), True)

    def evaluate(self, detection, image, frameNum, frameDetectNum, lastFrameDetectNum, boxesTrack, clasTrack):
        boxesGT = []
        newDetection = False
        if (len(detection) != 0):
            h = image.shape[0]
            w = image.shape[1]
            boxesGT, labelsGT, trackIdsGT = self.readAnnotation(frameNum, w, h)
            if (lastFrameDetectNum != frameDetectNum):  # new detection
                newDetection = True
                boxesDet = detection[0, 0, :, 3:7] * np.array([w, h, w, h])
                clasDet = detection[0, 0, :, 1]
                matchedAtDetect, unmatchedGTAtDetect, unmatchedDetect = self.matchDetections(boxesGT, labelsGT,
                                                                                             boxesDet, clasDet)
            else:
                newDetection = False
                matchedAtDetect = []
                unmatchedGTAtDetect = []
                unmatchedDetect = []
            matchedAtTrack, unmatchedGTAtTrack, unmatchedTrack = self.matchDetections(boxesGT, labelsGT,
                                                                                      boxesTrack, clasTrack)
            if (newDetection):
                totalMatched = len(matchedAtDetect)
                totalMissed = len(unmatchedGTAtDetect)
                totalWrong = len(unmatchedDetect)
            else:
                totalMatched = len(matchedAtTrack)
                totalMissed = len(unmatchedGTAtTrack)
                totalWrong = len(unmatchedTrack)
            self.logger.csvEval(
                str(frameNum) + ", " + str(totalMatched) + ", " + str(totalMissed) + ", " + str(
                    totalWrong) + ", " + str(len(matchedAtTrack)) + ", " + str(len(unmatchedTrack)) + ", " + str(
                    len(unmatchedGTAtTrack)) + ", " + str(len(matchedAtDetect)) + ", " + str(
                    len(unmatchedDetect)) + ", " + str(len(unmatchedGTAtDetect)))
        return boxesGT, newDetection

    def getResultThread(self, resultQ):
        lastInputTime = time.time() - self.startTime
        lastTrackOutTime = time.time() - self.startTime
        lastDetectOutTime = time.time() - self.startTime
        lastFrameDetectNum = time.time() - self.startTime
        fps = FPS().start()
        while (True):
            if (not resultQ.empty()):
                im = resultQ.get()
                if (im == self.stopSignal):
                    self.logger.info("see trackerOut done", True)
                    break
                fps.update()
                if (self.eval):
                    # Decode
                    frameNum = im[1]
                    inputTime = im[2] - self.startTime
                    trackOutTime = im[3] - self.startTime
                    frameDetectNum = im[4]
                    if (im[5] == "x"):
                        im[5] = im[2]
                    detectOutTime = im[5] - self.startTime
                    detectCount = im[6]
                    detection = im[7]
                    boxesTrack = im[8]
                    succesTrack = im[9]
                    clasTrack = im[10]

                    if (len(boxesTrack) != 0):
                        boxesTrack = np.asarray(
                            [boxesTrack[:, 0], boxesTrack[:, 1], boxesTrack[:, 0] + boxesTrack[:, 2],
                             boxesTrack[:, 1] + boxesTrack[:, 3]]).transpose()
                    boxesGT, newDetection = self.evaluate(detection, im[0], frameNum, frameDetectNum,
                                                          lastFrameDetectNum,
                                                          boxesTrack, clasTrack)
                    self.logger.csv(str(frameNum) + ", " + str(frameDetectNum) + ", " + str(detectCount) + ", " + str(
                        inputTime) + ", " + str(inputTime - lastInputTime) + ", " + str(trackOutTime) + ", " + str(
                        trackOutTime - lastTrackOutTime) + ", " + str(detectOutTime) + ", " + str(
                        detectOutTime - lastDetectOutTime) + ", " + str(trackOutTime - inputTime) + ", " + str(
                        trackOutTime - detectOutTime))
                    lastInputTime = inputTime
                    lastTrackOutTime = trackOutTime
                    lastDetectOutTime = detectOutTime
                    lastFrameDetectNum = frameDetectNum
                    if (self.debugMode):
                        self.logger.imwrite(format(frameNum, '05d') + ".jpg",
                                            self.draw(im[0], detection, succesTrack, boxesTrack, boxesGT, newDetection))
        self.logger.info("Done Get result: " + str(resultQ.qsize()), True)
        fps.stop()
        self.logger.info("Get result: elasped time: {:.2f}".format(fps.elapsed()), True)
        self.logger.info("Get result: approx. FPS: {:.2f}".format(fps.fps()), True)

    def readAnnotation(self, counter, w=300, h=300):
        tree = ET.parse(self.evalPath + format(counter, '06d') + ".xml")
        root = tree.getroot()

        boxes = list()
        labels = list()
        trackIds = list()
        ignored = 0
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        for object in root.iter('object'):
            trackId = int(object.find('trackid').text)

            label = object.find('name').text.lower().strip()
            if label not in self.imageNetDictionary:
                self.logger.error("Unrecognized label: " + label)
                continue

            bbox = object.find('bndbox')
            xmin = int(bbox.find('xmin').text) * w / width  # - 1
            ymin = int(bbox.find('ymin').text) * h / height  # - 1
            xmax = int(bbox.find('xmax').text) * w / width  # - 1
            ymax = int(bbox.find('ymax').text) * h / height  # - 1

            if ((xmax - xmin) * (ymax - ymin) > 1500):
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.imageNetDictionary[label])
                trackIds.append(trackId)
            else:
                ignored = ignored + 1

        if (ignored > 0):
            self.logger.warning("GT ignored for image " + str(counter) + "=" + str(ignored))
        return boxes, labels, trackIds

    def matchDetections(self, boxesGT, classesGT, boxesDet, classesDet, iou_thrd=0.5):
        IOU_mat = np.zeros((len(boxesGT), len(boxesDet)), dtype=np.float32)
        for g, gt in enumerate(boxesGT):
            # trk = convert_to_cv2bbox(trk)
            for d, det in enumerate(boxesDet):
                # tmp_tracker = self.tracker_list[t]
                IOU_mat[g, d] = self.box_iou2(gt, det) if (
                        classesGT[g] == self.CLASSES[int(classesDet[d])]) else 0  # MNR

        matched_idx = np.asarray(linear_sum_assignment(-IOU_mat)).transpose()

        unmatchedGT, unmatchedDet = [], []
        for g, gt in enumerate(boxesGT):
            if (g not in matched_idx[:, 0]):
                unmatchedGT.append(g)

        for d, det in enumerate(boxesDet):
            if (d not in matched_idx[:, 1]):
                unmatchedDet.append(d)

        matches = []

        # For creating trackers we consider any detection with an
        # overlap less than iou_thrd to signifiy the existence of
        # an untracked object

        for m in matched_idx:
            if (IOU_mat[m[0], m[1]] < iou_thrd):
                unmatchedGT.append(m[0])
                unmatchedDet.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatchedGT), np.array(unmatchedDet)

    def box_iou2(self, a, b):
        '''
        Helper funciton to calculate the ratio between intersection and the union of
        two boxes a and b
        a[0], a[1], a[2], a[3] <-> left, up, right, bottom
        '''

        a = (a[0], a[1], a[0] + a[2], a[1] + a[3])  # convert x y w h to x1 y1 x2 y2
        b = (b[0], b[1], b[0] + b[2], b[1] + b[3])

        w_intsec = np.maximum(0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
        h_intsec = np.maximum(0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
        s_intsec = w_intsec * h_intsec
        s_a = (a[2] - a[0]) * (a[3] - a[1])
        s_b = (b[2] - b[0]) * (b[3] - b[1])

        return float(s_intsec) / (s_a + s_b - s_intsec)

    def draw(self, frame, detection, validTrackBoxes=False, trackBoxes=[], boxesGT=[], newDetection=False):
        for box in boxesGT:  # GREEN
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[2]), int(box[3]))
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 4)  # BGR
        for box in trackBoxes:  # BLUE
            (x1, y1, x2, y2) = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
            sub_img = frame[y1:y2, x1:x2]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255 if validTrackBoxes else 128
            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
            # Putting the image back to its position
            frame[y1:y2, x1:x2] = res
        if (len(detection) != 0 and newDetection):  # RED
            h = frame.shape[0]
            w = frame.shape[1]
            box = detection[0, 0, :, 3:7] * np.array([w, h, w, h])
            cls = detection[0, 0, :, 1]
            conf = detection[0, 0, :, 2]
            box = box.astype(np.int32)
            for i in range(len(box)):
                p1 = (box[i][0], box[i][1])
                p2 = (box[i][2], box[i][3])
                cv2.rectangle(frame, p1, p2, (0, 0, 255), 2)
                p3 = (max(p1[0], 15), max(p1[1], 15))
                title = "%s:%.2f" % (self.CLASSES[int(cls[i])], conf[i])
                cv2.rectangle(frame, (p3[0], p3[1] - 20), (p3[0] + 100, p3[1]), (0, 0, 0), -1)
                cv2.putText(frame, title, p3, cv2.FONT_ITALIC, 0.6, (255, 255, 255), 1)
        return frame
