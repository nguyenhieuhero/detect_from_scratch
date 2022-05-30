import cv2
import numpy as np

image=cv2.imread("detect_from_scratch\plate_detected\plate1.jpg")
bit=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
bit=np.bitwise_not(bit)
cv2.imshow(f"white {bit.shape}",bit)
numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(bit,connectivity=4)
for i in range(0, numLabels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[i]
    if(area>1000 and area<7000):
        cv2.imshow(f"{int(w),int(h)}",bit[y:y+h,x:x+w])
        # image = cv2.circle(image,(cX, cY), radius=0, color=(255, 0, 255), thickness=20)

cv2.waitKey(0)

cv2.destroyAllWindows()