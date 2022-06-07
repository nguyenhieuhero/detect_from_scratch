import cv2
from utils import toNorm,pred_num
import numpy as np
while True:
    index=np.random.randint(0,1000)
    img_src=f"detect_from_scratch/output/syn_{index}.jpg"
    print(img_src)
    image=cv2.imread(img_src,cv2.IMREAD_GRAYSCALE)
    _,thresh = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
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
    cv2.imshow(f"{name_plate}",image)
    if cv2.waitKey(0)&0xFF == 27:
        break
    else:
        cv2.destroyAllWindows()