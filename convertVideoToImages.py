import cv2
cap = cv2.VideoCapture('./test_images/part02.mp4')
i=0
for i in range(5):
    ret, image = cap.read()
    cv2.imwrite('./test_images/i/'+str(i)+".jpg",image)