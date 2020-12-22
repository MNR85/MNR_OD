import os
from arbiter import Arbiter
import cv2
from imutils.video import FPS
from MNR_logger import MNR_logger
from argparse import ArgumentParser
import time

# ------------- Args
parser = ArgumentParser()
parser.add_argument("-g", '--gpu', required=False,
                    action='store_true', help="Enable GPU boost")
parser.add_argument("-s", "--serial", required=False,
                    action='store_true', help="Serial or parallel detection")
parser.add_argument("-d", '--debug', required=False,
                    action='store_true', help="Debug mode")
parser.add_argument("-e", '--eval', required=False,
                    action='store_true', help="Enable evaluation")
parser.add_argument("-r", "--ratio", required=False, type=int,
                    help="Detect track ratio for fixed ratio")
parser.add_argument("-t", "--tracker", required=False, type=str,
                    default='mosse', help="path to video input file", metavar="FILE")
parser.add_argument("-v", "--video", required=False, type=str,
                    default='test_images/ILSVRC/ILSVRC2017_train_00006000.mp4', help="path to video input file", metavar="FILE")
parser.add_argument("-a", "--annotation", required=False, type=str,
                    default='test_images/ILSVRC/ILSVRC2017_train_00006000/', help="path to annotation folder", metavar="FILE")
parser.add_argument("-p", "--prototxt", required=False, type=str,
                    default='ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.prototxt',
                    help="path to Caffe 'deploy' prototxt file", metavar="FILE")
parser.add_argument("-m", "--model", required=False, type=str,
                    default='ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.caffemodel',
                    help="path to Caffe pre-trained model", metavar="FILE")
args = vars(parser.parse_args())
# parser.add_argument("-f", "--frame", required=False,
#                     type=int, default=20, help="Frame count")
# videoName="test_images/los_angeles.mp4"
fixedRatio=not (args['ratio'] is None)
fixedRatio=True if args['serial'] else fixedRatio
fixedTracker=True
# ratio=10
# eval=True
# serial=False
# gpu=True
# trackType='csrt'

# -------------- Objs
logger=MNR_logger("results")

arbiter = Arbiter(args['prototxt'], args['model'], args['gpu'], args['serial'], args['tracker'],fixedTracker, logger, args['eval'], args['annotation'], args['ratio'], fixedRatio, args['debug'], )
# gst_str = ('v4l2src device=/dev/video{} ! '
#                'video/x-raw, width=(int){}, height=(int){} ! '
#                'videoconvert ! appsink').format(1, 1920, 1080)
# cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture(1)


# -------------- Starts

cap = cv2.VideoCapture(args['video'])
# arbiter.logger.start()
arbiter.logger.info("Video is: "+args['video'])
arbiter.logger.info("Options: userGPU: "+str(args['gpu'])+", tracker: "+args['tracker']+", serial: "+str(args['serial']))
arbiter.logger.info("Prototxt is: "+args['prototxt'])

fps = FPS().start()
counter=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        fps.update()
        resized=cv2.resize(frame, (300, 300))
        if (args['debug']):
            cv2.putText(resized, str(counter), (20,20), cv2.FONT_ITALIC, 0.6, (0, 0, 255), 1)
        arbiter.newImage(resized, counter)
        counter = counter + 1
    else:
        break
arbiter.logger.info("Finished stream", True)
fps.stop()
arbiter.logger.info("Input frame: elasped time: {:.2f}".format(fps.elapsed()), True)
arbiter.logger.info("Input frame: approx. FPS: {:.2f}".format(fps.fps()), True)
arbiter.stop()
arbiter.logger.stop()
cap.release()