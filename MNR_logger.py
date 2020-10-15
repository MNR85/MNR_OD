import os
from datetime import datetime
import time
import cv2
import platform
import logging
from multiprocessing import Queue, current_process
import sys

class MNR_logger():
    def __init__(self, rootPath="results", enable=True):
        self.rootPath=rootPath
        self.childPath=self.rootPath+"/"+datetime.now().strftime("%m%d%Y_%H%M%S")+platform.node()
        self.trackPath=self.childPath+"/trackOut"
        self.detectPath = self.childPath + "/DetectOut"
        self.enable=enable
        self.msgQ = Queue(maxsize=0)  # main
        if not os.path.exists(self.rootPath):
            os.makedirs(self.rootPath)
        os.makedirs(self.childPath)
        os.makedirs(self.trackPath)
        os.makedirs(self.detectPath)
        self.fArbiterResults = open(self.childPath + "/arbiter_result.csv", "w")
        self.fInfo = open(self.childPath + "/info.log", "w")
        self.fError = open(self.childPath + "/error.log", "w")
        self.fAll = open(self.childPath + "/all.log", "w")

    def start(self):
        # if(not self.enable):
        #     return

        self.fArbiterResults.write(
            "frameNumber, framDetectNum, detectCount, inputTime, inputTimeDiff, trackOutTime, trackOutTimeDiff , detectOutTime, detectOutTimeDiff, latency, TDLatency\n")
        os.system('tegrastats --interval 1000 --logfile tegrastats.out &')

    def tegrastat(self):
        fin.read()  # read first time
        fin.seek(0)  # offset of 0
        fin.read()  # read again

    def imwrite(self, fileName, frame, isTrack=True):
        if(isTrack):
            cv2.imwrite(self.trackPath+"/"+fileName, frame)
        else:
            cv2.imwrite(self.detectPath + "/" + fileName, frame)

    def info(self, msg, toSTDout=False, toFile=True):
        # if (not self.enable):
        #     return
        data="[Info] from ["+current_process().name+"] "+msg+" at "+str(time.time())
        if(toSTDout):
            print(data)
        if(toFile):
            self.msgQ.put(data)
            # logging.info(data)
            # self.fInfo.write(data+"\n")
            # self.fAll.write(data + "\n")

    def warning(self, msg, toSTDout=False, toFile=True):
        # if (not self.enable):
        #     return
        data="[Warn] from ["+current_process().name+"] "+msg+" at "+str(time.time())
        if(toSTDout):
            print(data)
        if(toFile):
            self.msgQ.put(data)
            # logging.warning(data)
            # self.fError.write(data+"\n")
            # self.fAll.write(data + "\n")

    def error(self, msg, toSTDout=False, toFile=True):
        # if (not self.enable):
        #     return
        data="[Err] from ["+current_process().name+"] "+msg+" at "+str(time.time())
        if(toSTDout):
            print(data)
        if(toFile):
            self.msgQ.put(data)
            # logging.error(data)
            # self.fError.write(data+"\n")
            # self.fAll.write(data + "\n")

    def csv(self, newLine):
        # if (not self.enable):
        #     return
        self.fArbiterResults.write(newLine)

    def stop(self):
        os.system('tegrastats --stop')
        msgQSize=sys.getsizeof(self.msgQ)
        while(not self.msgQ.empty()):
            data=self.msgQ.get()
            msgQSize = msgQSize+sys.getsizeof(data)
            self.fAll.write(data+"\n")
        print("Size of msgQ was: ",str(msgQSize))
        self.fArbiterResults.close()
        # self.fInfo.close()
        # self.fError.close()
        self.fAll.close()


