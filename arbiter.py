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
    def __init__(self, prototxt, model, useGpu, serial, trackType, logger=None, eval=False,
                 evalPath="", ratio=5, fixedRatio=False):
        self.serialProcessing = serial
        self.detectTrackRatio = ratio
        self.cvTracker = cvTracker.cvTracker(trackType, logger)
        self.detector = MNR_Net.Detector(prototxt, model)
        self.useGpu = useGpu

        # --------- logger
        # if (logger is None):
        #     self.print = print
        # else:
        #     self.print = logger.info
        self.logger = logger
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

        # --------- times
        self.trackTime = 0
        self.detectTime = 0
        self.refreshTime = 0
        self.boxTime = 0

        self.eval = eval
        self.evalPath = evalPath
        self.fixedRatio = fixedRatio

        # --------- process
        if (self.serialProcessing):
            print("serial mode")
            # --------- counters
            self.detectCount = 0
            self.trackCount = 0

            self.counter = 0
            self.detection = []
            self.detector.setRunMode(self.useGpu)
            self.detector.initNet()
            print("serial mode init fined")
        else:
            # self.runThread = Value('b', True)
            # self.stopSignal = Event()
            self.stopSignal = "done"
            self.imageQ = Queue(maxsize=0)  # main
            self.detectorInQ = Queue(maxsize=0)  # main -> detector
            self.detectorOutQ = Queue(maxsize=0)  # detector -> tracker
            self.detectorImage = Queue(maxsize=0)  # main -> tracker
            self.trackerQ = Queue(maxsize=0)  # main -> tracker
            self.resultQ = Queue(maxsize=0)  # tracker -> getresult
            self.initCNN = Value('b', False)
            self.processingCNN = Value('b', False)
            self.getOutFromCNN = Value('b', False)
            self.CnnCounter = Value('i', 0)
            self.RefreshCounter = Value('i', 0)
            self.TrackCounter = Value('i', 0)
            self.resultCounter = Value('i', 0)
            self.detectorP = Process(name='dnn', target=self.detectorThread, args=(
                self.detector, self.detectorInQ, self.detectorOutQ, self.processingCNN, self.initCNN, self.CnnCounter,
                self.getOutFromCNN))
            self.detectorP.daemon = True  # background run
            self.trackerP = Process(name='track', target=self.trackerThread, args=(
                self.detectorOutQ, self.detectorImage, self.trackerQ, self.resultQ, self.TrackCounter,
                self.RefreshCounter))
            self.trackerP.daemon = True  # background run
            self.getResultP = Process(name='getR', target=self.getResultThread,
                                      args=(self.resultQ, self.resultCounter))  # , args=[detector.runThread])
            self.getResultP.daemon = True  # background run
            self.detectorP.start()
            print('waiting for init net...')
            while not self.initCNN.value:
                time.sleep(0.1)
            self.startTime = time.time()
            self.logger.start(self.startTime)
            self.trackerP.start()
            self.getResultP.start()
            self.trackerCounterQ = 0
            print('Ready')

    def stop(self):
        if (self.serialProcessing):
            self.logger.info("With ration: " + str(self.detectTrackRatio) + ", Detect time: " + str(
                self.detectTime) + ", refresh time: " + str(self.refreshTime) + ", track time: " + str(self.trackTime))
            self.logger.info("Detect count: " + str(self.detectCount) + ", track count: " + str(self.trackCount))
        else:
            self.logger.info("signal to stop. ")
            self.getOutFromCNN.value = True
            self.detectorInQ.put(self.stopSignal)
            self.trackerQ.put(self.stopSignal)
            self.logger.info("Image in imageQ for flush: " + str(self.imageQ.qsize()))
            while (self.imageQ.qsize() > 0):
                self.imageQ.get()
            self.logger.flush()
            while self.trackerQ.qsize() > 0 or self.detectorInQ.qsize() > 0 or self.resultQ.qsize() > 0:
                self.getOutFromCNN.value = True
                self.logger.info("DetectQ: " + str(self.detectorInQ.qsize()) + "TrackQ: " + str(
                    self.trackerQ.qsize()) + "Refresh Q: " + str(self.detectorOutQ.qsize()) + "Result Q: " + str(
                    self.resultQ.qsize()))
                self.logger.flush()
                time.sleep(1)
            self.detectorP.join()
            self.trackerP.join()
            self.getResultP.join()
            self.logger.info("All process finished")
            self.logger.info("With pipeline")
            self.logger.info("Detect count: " + str(self.CnnCounter) + ", track count/2: " + str(self.TrackCounter))

    def newImage(self, frame, counter):
        if (self.serialProcessing):
            if (self.counter % self.detectTrackRatio == 0):
                t1 = time.time()
                self.detections = self.detector.serialDetector(frame)
                t2 = time.time()
                self.cvTracker.refreshTrack(frame, self.detections['detection_out'])
                t3 = time.time()
                self.detectTime = self.detectTime + t2 - t1
                self.refreshTime = self.refreshTime + t3 - t2
                self.detectCount = self.detectCount + 1
            else:
                t3 = time.time()
                (success, boxes) = self.cvTracker.track(frame)
                t4 = time.time()
                self.draw(frame, boxes, self.detections['detection_out'])
                t5 = time.time()
                self.trackTime = self.trackTime + t4 - t3
                self.boxTime = self.boxTime + t5 - t4
                self.trackCount = self.trackCount + 1

            self.counter = self.counter + 1
        else:
            if (not self.processingCNN.value):  # and self.counter%60==0):
                self.trackerCounterQ = 0
                remainTrackQ = self.trackerQ.qsize()
                while (not self.trackerQ.empty()):  # empty Q
                    # print("wait for getting tracker: ",str(self.trackerQ.qsize()))
                    time.sleep(0.1)
                self.logger.info("will add " + str(self.imageQ.qsize()) + ", remained from last: " + str(remainTrackQ))
                while (not self.imageQ.empty()):  # put all image in tracker (image between current cnn and last cnn)
                    self.trackerQ.put([self.imageQ.get(), False])
                self.detectorInQ.put([frame, counter])
                self.detectorImage.put(frame)
            self.imageQ.put(frame)
            self.trackerQ.put([frame, True, counter, time.time()])  # frame, isFirstTime, frameCounter, frameInputTime
            self.trackerCounterQ = self.trackerCounterQ + 1

    def detectorThread(self, detector, detectorInQ, detectorOutQ, processingCNN, initCNN, CnnCounter, getOutFromCNN):
        self.logger.info("detectorThread id = " + str(os.getpid()))
        detector.setRunMode(self.useGpu)
        detector.initNet()
        initCNN.value = True
        while (True):
            if (not detectorInQ.empty()):
                if (detectorInQ.qsize() != 1):
                    self.logger.error("Fatal error, detectorInQ has more than 1 frame!!")
                    qsize = detectorInQ.qsize()
                    imagesInQ = ""
                    for i in range(1, qsize):
                        tmp = detectorInQ.get()
                        imagesInQ = imagesInQ + str(frame[1]) + " ,"
                        detectorInQ.put(tmp)
                    self.logger.error("We had " + str(qsize) + " image that were from numbers: " + imagesInQ)
                    # raise Exception("Fatal error, detectorInQ has more than 1 frame!!")
                processingCNN.value = True
                frame = detectorInQ.get()
                if (frame == self.stopSignal):
                    break
                detections = detector.serialDetector(frame[0])
                detectorOutQ.put([detections['detection_out'], frame[1]])  # detection out, frame num
                # frame[0]=self.draw(frame[0], detections['detection_out'])
                # self.logger.imwrite("detect_images/" + str(CnnCounter.value) + ".jpg", frame[0])
                processingCNN.value = False
                CnnCounter.value = CnnCounter.value + 1
        detectorOutQ.put(self.stopSignal)
        self.logger.info("Done Detector: " + str(detectorInQ.qsize()), True)

    def trackerThread(self, detectorOutQ, detectorImage, trackerQ, resultQ, TrackCounter, RefreshCounter):
        self.logger.info("trackerThread id = " + str(os.getpid()))
        # track_id_list = deque(['1', '2', '3', '4', '5', '6', '7', '7', '8', '9', '10'])
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
                    self.logger.error("Tracker can not keep up to detector: " + str(detectorOutQ.qsize()))
                detectionOut = detectorOutQ.get()
                if (detectionOut == self.stopSignal):
                    self.logger.info("see detectorOut done", True)
                    detectorDone = True
                    continue
                detectionTime = time.time()
                detectionFrameNum = detectionOut[1]

                # if (boxes is not None and len(detection) != 0):
                #     matched, unmatched_dets, unmatched_trks=self.matchTrackerDetection(boxes, priorClass, detectionOut[0])
                detection = detectionOut[0]
                detectionCount = len(detection[0, 0, :, 1]) if detection[0, 0, :, 1][0] != -1 else 0
                priorClass = detection[0, 0, :, 1]
                frame = detectorImage.get()
                not self.cvTracker.refreshTrack(frame, detection, detectionFrameNum)
                RefreshCounter.value = RefreshCounter.value + 1
            if (not trackerQ.empty()):
                frame = trackerQ.get()
                if (frame == self.stopSignal):
                    self.logger.info("See trackerQ done", True)
                    trackerDone = True
                    continue
                (success, boxes) = self.cvTracker.track(frame[0])
                if (frame[1]):
                    # frame frameNum frameInputTime, trackOutTime, detectNum, detectOutTime
                    result = [frame[0], frame[2], frame[3], time.time(), detectionFrameNum, detectionTime,
                              detectionCount, detection, boxes, success, priorClass]
                    # if (self.eval):
                    #     result.append(detection)
                    #     result.append(boxes)
                    #     result.append(success)
                    #     result.append(priorClass)
                    resultQ.put(result)
                    TrackCounter.value = TrackCounter.value + 1
                    fps.update()
        resultQ.put(self.stopSignal)
        fps.stop()
        self.logger.info("Done tracker, " + str(trackerQ.qsize()) + " " + str(resultQ.qsize()), True)
        self.logger.info("Tracker elasped time: {:.2f}".format(fps.elapsed()), True)
        self.logger.info("Tracker approx. FPS: {:.2f}".format(fps.fps()), True)

    def getResultThread(self, resultQ, resultCounter):
        self.logger.info("getResultThread id = " + str(os.getpid()))
        lastInputTime = -1
        lastTrackOutTime = -1
        lastDetectOutTime = -1
        lastFrameDetectNum = -1
        fps = FPS().start()
        while (True):
            if (not resultQ.empty()):
                im = resultQ.get()
                if (im == self.stopSignal):
                    break
                resultCounter.value = resultCounter.value + 1
                fps.update()

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
                boxes = im[8]
                success = im[9]
                priorClass = im[10]

                if (self.eval and len(detection) != 0):
                    boxesGT, labelsGT, trackIdsGT = self.readAnnotation(frameNum)
                    if(lastFrameDetectNum != frameDetectNum): # new detection
                        boxesDet = detection[0, 0, :, 3:7] * np.array([300, 300, 300, 300])
                        clasDet = detection[0, 0, :, 1]
                        matchedAtDetect, unmatchedGTAtDetect, unmatchedDetect = self.matchDetections(boxesGT, labelsGT, boxesDet, clasDet)
                    matchedAtTrack, unmatchedGTAtTrack, unmatchedTrack = self.matchDetections(boxesGT, labelsGT, boxes, priorClass, True)
                    self.logger.csvEval(
                        str(frameNum) + ", " + str(len(matchedAtTrack)) + ", " + str(len(unmatchedTrack)) + ", " + str(
                            len(unmatchedGTAtTrack)) + ", " + str(len(matchedAtDetect)) + ", " + str(
                            len(unmatchedDetect)) + ", " + str(len(unmatchedGTAtDetect)))
                self.logger.imwrite(format(resultCounter.value, '05d') + ".jpg",
                                    self.draw(im[0], detection, success, boxes))
                self.logger.csv(str(frameNum) + ", " + str(frameDetectNum) + ", " + str(detectCount) + ", " + str(
                    inputTime) + ", " + str(inputTime - lastInputTime) + ", " + str(trackOutTime) + ", " + str(
                    trackOutTime - lastTrackOutTime) + ", " + str(detectOutTime) + ", " + str(
                    detectOutTime - lastDetectOutTime) + ", " + str(trackOutTime - inputTime) + ", " + str(
                    trackOutTime - detectOutTime))
                lastInputTime = inputTime
                lastTrackOutTime = trackOutTime
                lastDetectOutTime = detectOutTime
                # cv2.imshow("tracker",im)
                # cv2.waitKey(1)
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

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.imageNetDictionary[label])
            trackIds.append(trackId)

        return boxes, labels, trackIds

    def matchDetections(self, boxesGT, classesGT, boxesDet, classesDet, isInTrackFormat=False, iou_thrd=0.5, h=300, w=300):
        IOU_mat = np.zeros((len(boxesGT), len(boxesDet)), dtype=np.float32)
        if (isInTrackFormat): # just for precision calculation.
            boxesDet = np.asarray([boxesDet[:,0], boxesDet[:,1], boxesDet[:,0] + boxesDet[:,2], boxesDet[:,1] + boxesDet[:,3]]).transpose()
        for g, gt in enumerate(boxesGT):
            # trk = convert_to_cv2bbox(trk)
            for d, det in enumerate(boxesDet):
                # tmp_tracker = self.tracker_list[t]
                IOU_mat[g, d] = self.box_iou2(gt, det) if (classesGT[g] == self.CLASSES[int(classesDet[d])]) else 0  # MNR

        matched_idx = np.asarray(linear_sum_assignment(-IOU_mat)).transpose()

        unmatchedGT, unmatchedDet = [], []
        for g, gt in enumerate(boxesGT):
            if (g not in matched_idx[:,0]):
                unmatchedGT.append(g)

        for d, det in enumerate(boxesDet):
            if (d not in matched_idx[:,1]):
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

    def matchTrackerDetection(self, trackers, priorClasses, detection, iou_thrd=0.3, h=300, w=300):
        cls = detection[0, 0, :, 1]
        conf = detection[0, 0, :, 2]
        # box, conf, cls = (box.astype(np.int32), conf, cls)
        boxes = (detection[0, 0, :, 3:7] * np.array([w, h, w, h])).astype(np.int32)
        IOU_mat = np.zeros((len(trackers), len(boxes)), dtype=np.float32)
        for t, trk in enumerate(trackers):
            # trk = convert_to_cv2bbox(trk)
            for d, det in enumerate(boxes):
                # tmp_tracker = self.tracker_list[t]
                IOU_mat[t, d] = self.box_iou2(trk, det) if (cls[d] == priorClasses[d]) else 0  # MNR

        matched_idx = np.asarray(linear_sum_assignment(-IOU_mat))

        unmatched_trackers, unmatched_detections = [], []
        for t, trk in enumerate(trackers):
            if (t not in matched_idx[:, 0]):
                unmatched_trackers.append(t)

        for d, det in enumerate(boxes):
            if (d not in matched_idx[:, 1]):
                unmatched_detections.append(d)

        matches = []

        # For creating trackers we consider any detection with an
        # overlap less than iou_thrd to signifiy the existence of
        # an untracked object

        for m in matched_idx:
            if (IOU_mat[m[0], m[1]] < iou_thrd):
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

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

    def draw(self, frame, detection, validTrackBoxes=False, trackBoxes=[]):
        for box in trackBoxes:
            (x1, y1, x2, y2) = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            sub_img = frame[y1:y2, x1:x2]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255 if validTrackBoxes else 128

            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

            # Putting the image back to its position
            frame[y1:y2, x1:x2] = res
        if (len(detection) != 0):
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
