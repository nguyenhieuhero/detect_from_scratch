import numpy as np
import cv2
from utils import white_ratio
frame = cv2.imread("detect_from_scratch/image/2.jpg")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
hsv = cv2.GaussianBlur(hsv,(7,7),0)
lower = np.array([60, 0, 80])
upper = np.array([179, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
cv2.imshow("original",cv2.resize(frame,(700,400)))
cv2.imshow("result",cv2.resize(mask,(700,400)))
print(white_ratio(mask))
cv2.waitKey(0)