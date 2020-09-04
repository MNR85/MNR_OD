from arbiter import Arbiter
import cv2
from imutils.video import FPS

arbiter = Arbiter('ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.prototxt','ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.caffemodel', True, 'mosse') #'mosse')
gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(1, 1920, 1080)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
# cap = cv2.VideoCapture("test_images/soccer_01.mp4")#"test_images/soccer_01.mp4")#1)
fps = FPS().start()
counter=0
try:
    while(cap.isOpened() and counter < 20):
        ret, frame = cap.read()
        if ret == True:
            fps.update()
            # cv2.imshow("Raw", frame)
            counter=counter+1
            # key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            # if key == ord("q"):
            #     break
            arbiter.newImage(frame)
except:
    print("Exception")
fps.stop()
arbiter.stop()
cap.release()
# cv2.destroyAllWindows()
print("Input frame:")
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("Frame Input: "+str(counter))
print("Result counter: ", str(arbiter.resultCounter.value),"CnnCounter counter: ", str(arbiter.CnnCounter.value),"TrackCounter counter: ", str(arbiter.TrackCounter.value))