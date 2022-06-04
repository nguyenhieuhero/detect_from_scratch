import glob
import cv2
import numpy as np
from utils import toNorm
for x in glob.glob("detect_from_scratch/JPEGImages/*"):
    image=cv2.imread(f"{x}")
    bit=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(bit,127,255,cv2.THRESH_BINARY_INV)
    img=cv2.resize(thresh,(400,300))
    infor=[]
    # bit=np.bitwise_not(bit)
    cv2.imshow(f"white {img.shape}",img)
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img,connectivity=4)
    for i in range(0, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        if(w in range(20,110) and h in range(75,170)):
            crop=img[y:y+h,x:x+w]
            infor.append([x,y,w,h])
            # cv2.imshow(f"{int(w),int(h)}",toNorm(crop))
            # cv2.imwrite(f"detect_from_scratch/num_data/{x*y*w*h}.jpg",toNorm(crop))
