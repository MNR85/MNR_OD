import os
from arbiter import Arbiter
import cv2
from imutils.video import FPS
from MNR_logger import MNR_logger

# ------------- Args
videoName="test_images/los_angeles.mp4"
useGPU=True
serial=False
trackType='csrt'
protoFile='ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.prototxt'
caffeModel='ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.caffemodel'

# -------------- Objs
logger=MNR_logger("results")
arbiter = Arbiter(protoFile, caffeModel, useGPU, serial, trackType, logger) #'mosse')
# gst_str = ('v4l2src device=/dev/video{} ! '
#                'video/x-raw, width=(int){}, height=(int){} ! '
#                'videoconvert ! appsink').format(1, 1920, 1080)
# cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture(1)


# -------------- Starts

cap = cv2.VideoCapture(videoName)
# arbiter.logger.start()
arbiter.logger.info("Video is: "+videoName)
arbiter.logger.info("Options: userGPU: "+str(useGPU))

fps = FPS().start()
counter=0
# try:
while(cap.isOpened() and counter < 1000):
    ret, frame = cap.read()
    if ret == True:
        fps.update()
        resized=cv2.resize(frame, (300, 300))
        cv2.putText(resized, str(counter), (20,20), cv2.FONT_ITALIC, 0.6, (0, 0, 255), 1)
        arbiter.newImage(resized, counter)
        counter = counter + 1
    else:
        break
# except Exception as e:
#     print("Exception: ",e)
print("Finished stream")
fps.stop()
arbiter.stop()
arbiter.logger.stop()
cap.release()
print("Input frame:")
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))