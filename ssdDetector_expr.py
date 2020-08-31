import numpy as np
from multiprocessing import Process, Value, Queue
from argparse import ArgumentParser
import cv2
import time
import MNR_Net
from imutils.video import VideoStream, FPS
import os
# os.environ["GLOG_minloglevel"] = "1"
os.environ["GLOG_minloglevel"] = "0"
CLASSES = ('background',
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def getBoxedImage(origimg, net_out):
    h = origimg.shape[0]
    w = origimg.shape[1]
    box = net_out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = net_out['detection_out'][0,0,:,1]
    conf = net_out['detection_out'][0,0,:,2]
    
    box, conf, cls = (box.astype(np.int32), conf, cls)
    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(origimg, p1, p2, COLORS[int(cls[i])], 2)#(0,255,0))
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
        
    return origimg
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-g", '--gpu', required=False,
                        action='store_true', help="Enable GPU boost")
    parser.add_argument("-s", "--serial", required=False,
                        action='store_true', help="Serial or parallel detection")
    parser.add_argument("-p", "--prototxt", required=False, type=str, default='ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.prototxt',
                        help="path to Caffe 'deploy' prototxt file", metavar="FILE")
    parser.add_argument("-m", "--model", required=False, type=str,
                        default='ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.caffemodel', help="path to Caffe pre-trained model", metavar="FILE")
    parser.add_argument("-v", "--video", required=False, type=str,
                        default='../part02.mp4', help="path to video input file", metavar="FILE")
    parser.add_argument("-f", "--frame", required=False,
                        type=int, default=20, help="Frame count")
    parser.add_argument("-n", "--name", required=False, type=str,
                        default='GL552vw', help="Name for log file", metavar="FILE")

    args = vars(parser.parse_args())

    detector = MNR_Net.Detector(args['prototxt'], args['model'])
    detector.setRunMode(args['gpu'])

    cap = cv2.VideoCapture(0)#args['video'])

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    print('mode: serial ',args['serial'], ', gpu ',args['gpu'])

    frameCount = 0
    p = Process(name='popThread',target=detector.getImageFromQThread)#, args=[detector.runThread])
    p.daemon = True
    if not args['serial']:       
        p.start()
    if(not detector.netIsInit.value):
        print('waiting for init net...')
        if args['serial']:
            detector.initNet()
        while not detector.netIsInit:
            a=0
    time.sleep(2)
    #print('net is inited!')
    while(cap.isOpened()):
        # Capture frame-by-frame
        t1 = time.time()
        ret, frame = cap.read()
        if ret == True:# and frameCount <args['frame']:
            if args['serial']:                
                getBoxedImage(frame, detector.serialDetector(frame))
                cv2.imshow("SSD", frame)
                key = cv2.waitKey(1) & 0xFF
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break
            else:
                # detector.pipelineDetectorButWorkSerial(frame)
                detector.addImageToQ(frame)

            frameCount = frameCount+1
            detector.newPreprocess(time.time()-t1)
        # Break the loop
        else:
            break
    if not args['serial']:
        detector.runThread.value = False
        p.join()
    cap.release()
    moreInfo = 'mode: serial '+str(args['serial'])+', gpu '+str(args['gpu'])
    if args['serial']==True:
        method = 'Serial'
    else:
        method = 'Pipeline'
    if args['gpu']==True:
        hw = 'GPU'
    else:
        hw = 'CPU'

    gpuName=args['name']
    print('start saving to file...')
    detector.saveDataToFiles("executionTime_python_" + gpuName+"_"+method+"_"+hw, moreInfo, frameCount, args['serial'])
    print('finish all')
