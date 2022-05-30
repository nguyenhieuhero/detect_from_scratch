import cv2
import numpy as np
from utils import maximizeContrast,white_ratio,draw,toNorm,pred_num
cap=cv2.VideoCapture("detect_from_scratch/video/test_3.mp4")
while(True):
    frame = cv2.imread("detect_from_scratch/image/5.jpg")
    # _,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    # cv2.imshow("gray",cv2.resize(v,(700,400)))
    contrast=maximizeContrast(v)
    # cv2.imshow("tang tuong phan",cv2.resize(contrast,(700,400)))
    blur=cv2.GaussianBlur(contrast,(5,5),0)
    thresh=cv2.adaptiveThreshold(blur, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
    # cv2.imshow("thresh",cv2.resize(thresh,(700,400)))
    cn=cv2.Canny(thresh,100,255)
    # cv2.imshow("blur",cv2.resize(blur,(700,400)))
    # cv2.imshow("show",cv2.resize(cn,(700,400)))
    kernel = np.ones((3,3), np.uint8)
    dilate_img=cv2.dilate(cn,kernel,iterations=1)
    contours,_=cv2.findContours(dilate_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("canny",cv2.resize(cn,(700,400)))
    # cv2.drawContours(frame,contours,-1,(0,0,255),4)
    sorted_cnt= sorted(contours, key = cv2.contourArea, reverse = True)[:5]
    plate_cnt=[]
    coordinate=[]
    for cnt in sorted_cnt:
        peri=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.06*peri,True)
        if len(approx)== 4:
            coordinate.append(approx)
            plate_cnt.append(cnt)
    # for cnt in plate_cnt:
    #     cv2.drawContours(frame,[cnt],0,(0,0,255),4)
    label_index=0
    for coord in coordinate:
        label_index+=1
        points=coord.reshape(4,2)
        Area=[]
        for i in range(4):
            Area.append(points[i,0]*points[i,1])
        tli,bri=np.argmin(Area),np.argmax(Area)
        tl=points[tli]
        br=points[bri]
        new=np.delete(points,[tli,bri],0)
        tr=new[np.argmin(new[:,1])]
        bl=new[np.argmax(new[:,1])]
        # print(tl,tr,bl,br)
        # frame = cv2.circle(frame,tuple(tl), radius=0, color=(255, 0, 255), thickness=20)
        # print(tl,tr,bl)
        # print(np.sum(np.square(tl-tr)))
        # print(np.sum(np.square(tl-bl)))
        check1=np.sum(np.square(tl-tr))/np.sum(np.square(tl-bl))
        check2=np.sum(np.square(bl-br))/np.sum(np.square(tr-br))
        edge_diff=np.abs(check1-check2)
        # cv2.imshow(f"{edge_diff}",frame[tl[1]:br[1],tl[0]:br[0]])
        if(edge_diff<1):
            if(check1 and check2 > 3):
                w,h=700,160
            else:
                w,h=400,300
            pst1=np.float32([tl,tr,bl,br])
            pst2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
            M=cv2.getPerspectiveTransform(pst1,pst2)
            out=cv2.warpPerspective(thresh,M,(w,h))
            out = cv2.medianBlur(out,5)
            ratio=white_ratio(out,type='gray_rv')
            if(ratio>0.60):
                infor=[]
                numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(out,connectivity=4)
                for i in range(0, numLabels):
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    area = stats[i, cv2.CC_STAT_AREA]
                    (cX, cY) = centroids[i]
                    if(w in range(20,110) and h in range(75,170)):
                        # crop=out[y:y+h,x:x+w]
                        infor.append([x,y,w,h])
                        # cv2.imwrite(f"detect_from_scratch/num/{label_index}.{i}.jpg",crop)
                        # cv2.imshow(f"Plate: {label_index} {x,y,w,h}",crop)
                # print(len(infor))
                if(len(infor)>0):
                    frame=draw(frame,tl,tr,bl,br)
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
                        cv2.imshow(f"Plate: {pred_num(norm_output)} {x,y,w,h}",norm_output)
                    frame=cv2.putText(frame,name_plate,tl,cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),2)
    cv2.imshow("original",cv2.resize(frame,(700,400)))
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()