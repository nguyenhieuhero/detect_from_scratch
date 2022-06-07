import cv2
from sqlalchemy import true
from utils import toNorm,pred_num,maximizeContrast
import numpy as np
import glob
while(True):
    i=np.random.randint(0,1769)
    image=cv2.imread(f"detect_from_scratch\JPEGImages/{i}.jpg")
    cv2.imshow("original",image)
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    contrast=maximizeContrast(v)
    blur=cv2.GaussianBlur(contrast,(5,5),0)
    thresh=cv2.adaptiveThreshold(blur, 255.0, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    if(thresh.shape[0]>250):
        out=cv2.resize(thresh,(400,300))
    else:
        out=cv2.resize(thresh,(700,160))
    infor=[]
    # bit=np.bitwise_not(bit)
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(out,connectivity=4)
    for i in range(0, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        if(w in range(10,110) and h in range(75,170)):
            infor.append([x,y,w,h])
    if(len(infor)>0):
        name_plate=''
        upper=[]
        lower=[]
        for loc in infor:
            if(loc[1]<70):
                upper.append(loc)
            else:
                lower.append(loc)
        upper.sort(key=lambda row: row[0])
        lower.sort(key=lambda row: row[0])
        sort=upper+lower
        for x,y,w,h in sort:
            norm_output=toNorm(out[y:y+h,x:x+w])
            name_plate+=pred_num(norm_output)
            # cv2.imshow(f"{x,y,w,h}",out[y:y+h,x:x+w])
    cv2.imshow(f"{name_plate}",thresh)
    if cv2.waitKey(0)&0xFF == 27:
        break
    else:
        cv2.destroyAllWindows()