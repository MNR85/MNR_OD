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
        self.childPath=self.rootPath+"/"+datetime.now().strftime("%m%d%Y_%H%M%S")+"_"+platform.node()
        self.trackPath=self.childPath+"/trackOut"
        self.detectPath = self.childPath + "/DetectOut"
        self.enable=enable
        self.msgQ = Queue(maxsize=0)  # main

    def start(self):
        # if(not self.enable):
        #     return
        if not os.path.exists(self.rootPath):
            os.makedirs(self.rootPath)
        os.makedirs(self.childPath)
        os.makedirs(self.trackPath)
        os.makedirs(self.detectPath)
        self.fArbiterResults = open(self.childPath + "/arbiter_result.csv", "w")
        # self.fInfo = open(self.childPath + "/info.log", "w")
        # self.fError = open(self.childPath + "/error.log", "w")
        self.fAll = open(self.childPath + "/all.log", "w")
        self.fArbiterResults.write(
            "frameNumber, framDetectNum, detectCount, inputTime, inputTimeDiff, trackOutTime, trackOutTimeDiff , detectOutTime, detectOutTimeDiff, latency, TDLatency\n")
        # os.system('tegrastats --interval 1000 --logfile tegrastats.out &')
        if(platform.node()=="tx2-desktop"):
            # Resource usage
            gpuUsageFile = "/sys/devices/gpu.0/load"
            cpuUsageFile = "/proc/stat"
            memUsageFile = "/proc/meminfo"

            # Resource tempreture
            gpuTempFile = "/sys/devices/virtual/thermal/thermal_zone2/temp"
            bCpuTempFile = "/sys/devices/virtual/thermal/thermal_zone0/temp"
            mCpuTempFile = "/sys/devices/virtual/thermal/thermal_zone1/temp"

            # Resource power
            VDD_INpower = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input"
            VDD_GPUpower = "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input"
            VDD_CPUpower = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power1_input"
            VDD_SOCpower = "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power1_input"
            VDD_DDRpower = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power2_input"
            VDD_Wifipower = "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power2_input"

            self.fCPU=open(cpuUsageFile, "r")
            self.fMem = open(memUsageFile, "r")
            self.cpuTotal= {}
            self.cpuIdle= {}

        else: # General computers
            cpuUsageFile = "/proc/stat"
            memUsageFile = "/proc/meminfo"
            self.fCPU = open(cpuUsageFile, "r")
            self.fMem = open(memUsageFile, "r")
            self.cpuTotal = {}
            self.cpuIdle = {}


    def cpuStat(self):
        self.fCPU.seek(0)
        cpuTotal = {}
        cpuIdle = {}
        cpuPercentage = {}
        for line in self.fCPU:
            if (not line.startswith("cpu")):
                break
            stat = line.split()  # split by space
            cpuTotal[stat[0]] = sum([int(i) for i in stat[1:]])
            cpuIdle[stat[0]] = sum([int(i) for i in stat[4:6]])
            if stat[0] in self.cpuTotal:
                totalD = cpuTotal[stat[0]] - self.cpuTotal[stat[0]]
                idleD = cpuIdle[stat[0]] - self.cpuIdle[stat[0]]
                cpuPercentage[stat[0]] = (((cpuTotal[stat[0]] - self.cpuTotal[stat[0]]) - (
                            cpuIdle[stat[0]] - self.cpuIdle[stat[0]])) / (
                                                      cpuTotal[stat[0]] - self.cpuTotal[stat[0]])) * 100.0
            self.cpuTotal[stat[0]] = cpuTotal[stat[0]]
            self.cpuIdle[stat[0]] = cpuIdle[stat[0]]
        return cpuPercentage
    def memStat(self):
        self.fMem.seek(0)
        memTotal=int(self.fMem.readline().split()[1])
        self.fMem.readline() # MemFree is not required
        memAvail=int(self.fMem.readline().split()[1])
        return ((memTotal-memAvail)/memTotal)*100

    def tegrastat(self):
        return self.cpuStat()


        # fin.read()  # read first time
        # fin.seek(0)  # offset of 0
        # fin.read()  # read again

    def imwrite(self, fileName, frame, isTrack=True):
        if(isTrack):
            cv2.imwrite(self.trackPath+"/"+fileName, frame)
        else:
            cv2.imwrite(self.detectPath + "/" + fileName, frame)

    def info(self, msg, toSTDout=False, toFile=True):
        # if (not self.enable):
        #     return
        data="[Info] from ["+current_process().name+"] \t\t"+msg+" at "+str(time.time())
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
        data="[Warn] from ["+current_process().name+"]  \t\t"+msg+" at "+str(time.time())
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
        data="[Err] from ["+current_process().name+"]  \t\t"+msg+" at "+str(time.time())
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
        self.fAll.write("Size of msgQ was: "+str(msgQSize))
        self.fArbiterResults.close()
        # self.fInfo.close()
        # self.fError.close()
        self.fAll.close()

logger=MNR_logger()
logger.start()
counter=0
while(counter<100):
    print(logger.tegrastat())
    print(logger.memStat())
    time.sleep(1)
    counter=counter+1