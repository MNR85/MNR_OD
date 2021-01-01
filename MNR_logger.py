import os
from datetime import datetime
import time
import cv2
import platform
import logging
from multiprocessing import Queue, current_process, Value, Array
import sys

class MNR_logger():
    def __init__(self, rootPath="results", postfix="", enable=True):
        self.rootPath=rootPath
        self.postfix=postfix
        self.ID=datetime.now().strftime("%m%d%Y_%H%M%S")+"_"+self.postfix+"_"+platform.node()
        self.childPath=self.rootPath+"/"+self.ID
        self.trackPath=self.childPath+"/trackOut"
        self.detectPath = self.childPath + "/DetectOut"
        self.resultPath = self.rootPath+"/results.csv"
        self.enable=enable
        self.msgQ = Queue(maxsize=0)
        self.startTime=0

    def start(self, startTime=0):
        self.startTime=startTime
        if not os.path.exists(self.rootPath):
            os.makedirs(self.rootPath)
        if not os.path.isfile(self.resultPath):
            self.resultFile = open(self.resultPath, "w")
            self.resultFile.write("ID, frameCount, Data, Prototxt, HW, Method, Tracker, Detect/Track, FPS, Accuracy, Precision, Recall, TP, FN, FP, Power"+"\n")
            self.resultFile.close()
        self.resultFile = open(self.resultPath, "a")
        os.makedirs(self.childPath)
        os.makedirs(self.trackPath)
        os.makedirs(self.detectPath)
        self.fArbiterResults = open(self.childPath + "/arbiter_result_"+self.postfix+".csv", "w")
        self.fEval = open(self.childPath + "/eval_"+self.postfix+".csv", "w")
        self.fAll = open(self.childPath + "/all_"+self.postfix+".log", "w")
        strCSV = "frameNumber, framDetectNum, detectCount, inputTime, inputTimeDiff, trackOutTime, trackOutTimeDiff , detectOutTime, detectOutTimeDiff, latency, TDLatency"

        self.platformNod=platform.node()
        if (self.platformNod == "tx2-desktop"):
            platformStatCSV = ", temp-GPU, temp-bCPU, temp-mCPU, power-Total, power-GPU, power-CPU, power-SOC, power-DDR, power-Wifi, usage-GPU, usage-Mem, usage-CPU"
            # Resource usage
            gpuUsageFile = "/sys/devices/gpu.0/load"
            cpuUsageFile = "/proc/stat"
            memUsageFile = "/proc/meminfo"

            # Resource tempreture
            gpuTempFile = "/sys/devices/virtual/thermal/thermal_zone2/temp"
            bCpuTempFile = "/sys/devices/virtual/thermal/thermal_zone0/temp"
            mCpuTempFile = "/sys/devices/virtual/thermal/thermal_zone1/temp"

            # Resource power
            vdd_INpower = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power0_input"
            vdd_GPUpower = "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input"
            vdd_CPUpower = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power1_input"
            vdd_SOCpower = "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power1_input"
            vdd_DDRpower = "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device1/in_power2_input"
            vdd_Wifipower = "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power2_input"

            self.fuCPU = open(cpuUsageFile, "r")  # file usage cpu
            self.fuMem = open(memUsageFile, "r")
            self.fuGPU = open(gpuUsageFile, "r")

            self.ftCPUb = open(bCpuTempFile, "r")  # file temperature cpu b
            self.ftCPUm = open(mCpuTempFile, "r")
            self.ftGPU = open(gpuTempFile, "r")

            self.fpTotal = open(vdd_INpower)  # file power total
            self.fpGPU = open(vdd_GPUpower)
            self.fpCPU = open(vdd_CPUpower)
            self.fpSOC = open(vdd_SOCpower)
            self.fpDDR = open(vdd_DDRpower)
            self.fpWifi = open(vdd_Wifipower)
        else:  # General computers
            platformStatCSV = ", usage-Mem, usage-CPU"
            cpuUsageFile = "/proc/stat"
            memUsageFile = "/proc/meminfo"
            self.fuCPU = open(cpuUsageFile, "r")
            self.fuMem = open(memUsageFile, "r")
        self.fArbiterResults.write(strCSV+ platformStatCSV + "\n")
        self.fEval.write("frameNumber, totalMatched, totalMissed, totalWrong, matchedTrack, unmatchedTrack, unmatchedGTAtTrack, matchedDetect, unmatchedDetect, unmatchedGTAtDetect\n")
        # need for store previous state
        self.cpuTotal = {}
        self.cpuIdle = {}
        self.lastPlatformStat = ""
        self.lastPlatformStatUpdate=time.time()-1
        # # check value before
        # if (self.platformNod == "tx2-desktop"):
        #     firstPlatformStat = self.tegrastat()[:-1]
        # else:
        #     firstPlatformStat = self.platformstat()[:-1]
        # time.sleep(1)
        # if (self.platformNod == "tx2-desktop"):
        #     self.firstPlatformStat = [(e2 + e1) / 2 for (e2, e1) in
        #                               zip(self.tegrastat()[:-1], firstPlatformStat)]
        # else:
        #     self.firstPlatformStat = [(e2 + e1)/2 for (e2, e1) in zip(self.platformstat()[:-1], firstPlatformStat)]
        # self.averagePlatformStat = self.evalResult = Array('i', len(self.firstPlatformStat))#[0] * len(self.firstPlatformStat)
        # self.averagePlatformStat = [0] * len(self.firstPlatformStat)
        self.averagePowerStat = Value('f')
        self.counterPowerStat = Value('i')

    def cpuStat(self):
        self.fuCPU.seek(0)
        cpuTotal = {}
        cpuIdle = {}
        cpuPercentage = {}
        for line in self.fuCPU:
            if (not line.startswith("cpu")):
                break
            stat = line.split()  # split by space
            cpuTotal[stat[0]] = sum([int(i) for i in stat[1:]])
            cpuIdle[stat[0]] = sum([int(i) for i in stat[4:6]])
            if stat[0] in self.cpuTotal:
                cpuPercentage[stat[0]] = (((cpuTotal[stat[0]] - self.cpuTotal[stat[0]]) - (
                            cpuIdle[stat[0]] - self.cpuIdle[stat[0]])) / (
                                                      cpuTotal[stat[0]] - self.cpuTotal[stat[0]])) * 100.0
            self.cpuTotal[stat[0]] = cpuTotal[stat[0]]
            self.cpuIdle[stat[0]] = cpuIdle[stat[0]]
        return cpuPercentage

    def memStat(self):
        self.fuMem.seek(0)
        memTotal=int(self.fuMem.readline().split()[1])
        self.fuMem.readline() # MemFree is not required
        memAvail=int(self.fuMem.readline().split()[1])
        return ((memTotal-memAvail)/memTotal)*100

    def platformstat(self):
        # ---- usage
        cpuS = (self.cpuStat())
        memS = (self.memStat())
        return [memS, cpuS]
        # if not hasattr(self, 'firstPlatformStat'):
        #     return [memS, cpuS]
        # else:
        #     res = [e2-e1 for (e2,e1) in zip(self.firstPlatformStat, [memS])]
        #     res.append(cpuS)
        #     return res

    def tegrastat(self):
        # ---- usage
        cpuS= (self.cpuStat())
        memS= (self.memStat())

        self.fuGPU.seek(0)
        gpuS=(int(self.fuGPU.readline())/10)

        # ----- temp
        self.ftCPUb.seek(0)
        bcpuT=(int(self.ftCPUb.readline())/1000)

        self.ftCPUm.seek(0)
        mcpuT=(int(self.ftCPUm.readline())/1000)

        self.ftGPU.seek(0)
        gpuT=(int(self.ftGPU.readline())/1000)

        # ----- power
        self.fpTotal.seek(0)
        totalP=(int(self.fpTotal.readline()))

        self.fpGPU.seek(0)
        gpuP=(int(self.fpGPU.readline()))

        self.fpCPU.seek(0)
        cpuP=(int(self.fpCPU.readline()))

        self.fpSOC.seek(0)
        socP=(int(self.fpSOC.readline()))

        self.fpDDR.seek(0)
        ddrP=(int(self.fpDDR.readline()))

        self.fpWifi.seek(0)
        wifiP=(int(self.fpWifi.readline()))

        self.averagePowerStat.value = totalP
        self.counterPowerStat.value = self.counterPowerStat.value + 1

        ##
        # order is
        # , temp-GPU, temp-bCPU, temp-mCPU, power-Total, power-GPU, power-CPU, power-DDR, power-Wifi, usage-GPU, usage-Mem, usage-CPU
        return [gpuT, bcpuT, mcpuT, totalP, gpuP, cpuP, socP, ddrP, wifiP, gpuS, memS, cpuS]
        # if not hasattr(self, 'firstPlatformStat'):
        #     return [gpuT, bcpuT, mcpuT, totalP, gpuP, cpuP, socP, ddrP, wifiP, gpuS, memS, cpuS]
        # else:
        #     res = [e2-e1 for (e2,e1) in zip(self.firstPlatformStat, [gpuT, bcpuT, mcpuT, totalP, gpuP, cpuP, socP, ddrP, wifiP, gpuS, memS])]
        #     res.append(cpuS)
        #     return res

    def imwrite(self, fileName, frame, isTrack=True):
        if(isTrack):
            cv2.imwrite(self.trackPath+"/"+fileName, frame)
        else:
            cv2.imwrite(self.detectPath + "/" + fileName, frame)

    def info(self, msg, toSTDout=False, toFile=True):
        data= '{:30s}\t\t'.format("[Info] from ["+current_process().name+"]")+msg+" at "+str(time.time()-self.startTime)
        if(toSTDout):
            print(data)
        if(toFile):
            self.msgQ.put(data)

    def warning(self, msg, toSTDout=False, toFile=True):
        data='{:30s}\t\t'.format("[Warn] from ["+current_process().name+"]")+msg+" at "+str(time.time()-self.startTime)
        if(toSTDout):
            print(data)
        if(toFile):
            self.msgQ.put(data)

    def error(self, msg, toSTDout=False, toFile=True):
        data='{:30s}\t\t'.format("[Err] from ["+current_process().name+"]")+msg+" at "+str(time.time()-self.startTime)
        if(toSTDout):
            print(data)
        if(toFile):
            self.msgQ.put(data)

    def result(self, msg, toSTDout=True, toFile=True):
        msg=self.ID+", "+msg
        data='{:30s}\t\t'.format("[Result] from ["+current_process().name+"]")+msg+" at "+str(time.time()-self.startTime)
        if(toSTDout):
            print(data)
        if(toFile):
            # self.averagePlatformStat = [a / self.counterPlatformStat.value for a in self.averagePlatformStat]
            averagePowerStat=0
            if(self.counterPowerStat.value!=0):
                averagePowerStat = self.averagePowerStat.value/self.counterPowerStat.value
            self.resultFile.write(msg+ "," + str(averagePowerStat)+"\n")
            self.msgQ.put(data)

    def csv(self, newLine):
        nowT=time.time()
        if(nowT-self.lastPlatformStatUpdate>1): # can do cpu measure only speedy
            self.lastPlatformStatUpdate=nowT
            if (self.platformNod == "tx2-desktop"):
                self.lastPlatformStat=self.tegrastat()
            else:
                self.lastPlatformStat=self.platformstat()
            # self.averagePlatformStat = [a+b for (a,b) in zip(self.averagePlatformStat, self.lastPlatformStat[:-1])]
            # self.counterPlatformStat.value = self.counterPlatformStat.value+1
        self.fArbiterResults.write(newLine+"," +str(self.lastPlatformStat)[1:-1]+"\n")
        # self.fArbiterResults.flush()

    def csvEval(self, newLine):
        self.fEval.write(newLine+"\n")

    def flush(self):
        msgQSize = sys.getsizeof(self.msgQ)
        while (not self.msgQ.empty()):
            data = self.msgQ.get()
            msgQSize = msgQSize + sys.getsizeof(data)
            self.fAll.write(data + "\n")
        print("Size of msgQ was: ", str(msgQSize))
        self.fAll.write("Size of msgQ was: " + str(msgQSize)+"\n")
        self.fAll.flush()
        self.fArbiterResults.flush()
        self.fEval.flush()

    def stop(self):
        msgQSize=sys.getsizeof(self.msgQ)
        while(not self.msgQ.empty()):
            data=self.msgQ.get()
            msgQSize = msgQSize+sys.getsizeof(data)
            self.fAll.write(data+"\n")
        print("Size of msgQ was: ",str(msgQSize))
        self.fAll.write("Size of msgQ was: "+str(msgQSize)+"\n")
        self.fArbiterResults.close()
        self.fEval.close()
        self.fAll.close()
        self.resultFile.close()
