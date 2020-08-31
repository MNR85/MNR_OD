from imutils.video import VideoStream, FPS
import MNR_Net
import time
import cv2
from argparse import ArgumentParser
from os import system

if __name__ == '__main__':
    system('clear')
    parser = ArgumentParser()
    parser.add_argument("-p", "--prototxt", required=False,type=str, default='ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.prototxt', help="path to Caffe 'deploy' prototxt file", metavar="FILE")
    parser.add_argument("-m", "--model", required=False,type=str, default = 'ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.caffemodel', help="path to Caffe pre-trained model", metavar="FILE")
    parser.add_argument("-co", "--confidence", required=False,type=float, default=0.7, help="minimum probability to filter weak detections")
    parser.add_argument("-g", "--gpu", required=False,type=bool, default=True, help="Enable GPU boost")
    parser.add_argument("-ca", "--caffe", required=False,type=bool, default=False, help="Enable Caffe transformInput")
    parser.add_argument("-he", "--height", required=False,type=int, default=300, help="Height of transformInput")
    parser.add_argument("-w", "--width", required=False,type=int, default=300, help="Width of transformInput")

    args = vars(parser.parse_args())
    mNet = MNR_Net.Detector(args['prototxt'], args['model'])
    mNet.setRunMode(args['gpu'])
    mNet.initNet()
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(0.1)
    fps = FPS().start()
    frameCount=0  
    times = [0,0,0,0,0]
    
    while True:
        time1 = time.time()
        
        frame = vs.read()
        time2 = time.time()
        preFrame = mNet.transformInput(frame)
        time3 = time.time()
        netOut = mNet.MNetForward(preFrame)
        time4 = time.time()
        
        postFrame = mNet.getBoxedImage(frame, netOut)
        time5 = time.time()
        
        cv2.imshow("SSD", postFrame)
        time6 = time.time()
        
        times[0] = times[0]+time2-time1
        times[1] = times[1]+time3-time2
        times[2] = times[2]+time4-time3
        times[3] = times[3]+time5-time4
        times[4] = times[4]+time6-time5

        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # update the FPS counter
        fps.update()
        frameCount = frameCount+1
	
    fps.stop()
    vs.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed())+" frame counter = "+str(frameCount))
    times[0] = (times[0]/frameCount)*1000000
    times[1] = (times[1]/frameCount)*1000000
    times[2] = (times[2]/frameCount)*1000000
    times[3] = (times[3]/frameCount)*1000000
    times[4] = (times[4]/frameCount)*1000000
    print("[INFO] elapsed time part: "+str(times))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
