import cv2
import numpy as np
kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2,2))
while(True):
    img = cv2.imread("detect_from_scratch/image/1.jpg")
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    _,_,gray=cv2.split(hsv)
    bw=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 12)
    cv2.imshow("sth",bw)
    if cv2.waitKey(0)==27:
        break