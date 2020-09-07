import cv2
from imutils.video import FPS
fps = FPS().start()
cap = cv2.VideoCapture("test_images/soccer_01.mp4")
counter=0
print("Is opened: ",str(cap.isOpened()))
while(cap.isOpened() and counter < 100):
    ret, frame = cap.read()
    if ret == True:
        fps.update()
    counter = counter + 1

fps.stop()
cap.release()

print("Input frame:")
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("Frame Input: "+str(counter))