import cv2
import numpy as np
import glob
images = glob.glob('results/10202020_204331_tx2-desktop/trackOut/*.jpg')
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter('out.avi', 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
video.release()