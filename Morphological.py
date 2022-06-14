import cv2
import numpy as np
kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(2,2))
while(True):
    img = cv2.imread("detect_from_scratch/image/3.jpg")
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    _,_,imgGrayscale=cv2.split(hsv)
    height, width = imgGrayscale.shape
    
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #tạo bộ lọc kernel
    
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 10)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10)
    cv2.imshow("gray",imgGrayscale )
    if cv2.waitKey(0)==27:
        break