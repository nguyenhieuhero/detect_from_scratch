import imp
import cv2
import numpy as np

lower = np.array([0, 0, 70])
upper = np.array([179, 255, 255])

def maximizeContrast(imgGrayscale):
    height, width = imgGrayscale.shape
    
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #tạo bộ lọc kernel
    
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations = 10)
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations = 10)
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    return imgGrayscalePlusTopHatMinusBlackHat
def white_ratio(img,type='gray'):
    if type=='gray':
        return cv2.countNonZero(img)/img.size
    if type=='gray_rv':
        return 1-(cv2.countNonZero(img)/img.size)
    if type=='hsv':
        mask = cv2.inRange(img, lower, upper)
        return cv2.countNonZero(mask)/mask.size
    else:
        mask = cv2.inRange(cv2.cvtColor(img,cv2.COLOR_BGR2HSV), lower, upper)
        return cv2.countNonZero(mask)/mask.size
def draw(img,tl,tr,bl,br):
    pts = np.array([tl,tr,br,bl], np.int32)
    pts = pts.reshape((-1,1,2))
    return cv2.polylines(img,[pts],True,(255,0,255),3)


