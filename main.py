import cv2
import numpy as np
from utils import maximizeContrast,white_ratio,draw,toNorm,pred_num
frame = cv2.imread("detect_from_scratch/image/10.jpg")
#Chuyển dài màu sang hsv để nhận sáng tốt hơn
hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
# Chia color channel thành 3 channel, channel v là độ sáng
h,s,v=cv2.split(hsv)
# Tăng tương phản sử dụng TopHat và BlackHat
contrast=maximizeContrast(v)
#Làm mờ để giảm nhiễu
blur=cv2.GaussianBlur(contrast,(5,5),0)
#Sử dụng nhị phân hóa ảnh ngưỡng động, nếu chỉ nhị phân bằng cách đặt threshold thì sẽ mất thông tin ảnh
thresh=cv2.adaptiveThreshold(blur, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
#Lấy cạnh của ảnh nhị phân
cn=cv2.Canny(thresh,100,255)
kernel = np.ones((3,3), np.uint8)
#Giãn các pixel trắng của ảnh nhằm đóng vùng những biên của biển bị nhiễu
dilate_img=cv2.dilate(cn,kernel,iterations=1)
# Lấy contours của ảnh đã xử lí
contours,_=cv2.findContours(dilate_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# Vì có rất nhiều contours nhiễu nên ta sắp xếp lại và chỉ lấy những contour có diện tích hơn nhất, trong trường hợp này là 5
sorted_cnt= sorted(contours, key = cv2.contourArea, reverse = True)[:5]
# Contour của biển số
plate_cnt=[]
#Tọa độ của các đỉnh mỗi contour
coordinate=[]
# Biển số đã xác định
plate=[]
for cnt in sorted_cnt:
    # Tính chu vi contour, True ở đây nghĩa là contour đóng kín
    peri=cv2.arcLength(cnt,True)
    # Làm trơn và xấp xỉ hình dạng, approx là tập hợp đỉnh của hình xấp xỉ
    approx=cv2.approxPolyDP(cnt,0.06*peri,True)
    # Những contour xấp xỉ có 4 đỉnh sẽ lưu thông tin contour và tọa độ 4 đỉnh
    if len(approx)== 4:
        coordinate.append(approx)
        plate_cnt.append(cnt)
# for cnt in plate_cnt:
#     cv2.drawContours(frame,[cnt],0,(0,0,255),4)
for coord in coordinate:
    points=coord.reshape(4,2)
    Area=[]
    # Sắp xếp tọa độ 4 đỉnh cho đúng vị trí để xoay ảnh và kiểm tra bằng cách sử dụng diện tích
    for i in range(4):
        # Diện tích 4 đỉnh tính từ tọa độ trên cùng bên trái (0,0)
        Area.append(points[i,0]*points[i,1])
    # đỉnh trên trái có diện tích nhỏ nhất, đỉnh dưới phải có diện tích lớn nhất
    tli,bri=np.argmin(Area),np.argmax(Area)
    tl=points[tli]
    br=points[bri]
    # Xóa tọa độ 2 đỉnh đã xác định
    new=np.delete(points,[tli,bri],0)
    # Đỉnh nào có tọa độ y nhỏ hơn tức là đỉnh đấy cao hơn => Đỉnh trên phải
    tr=new[np.argmin(new[:,1])]
    # Đỉnh dưới phải 
    bl=new[np.argmax(new[:,1])]
    # frame = cv2.circle(frame,tuple(tl), radius=0, color=(255, 0, 255), thickness=20)
    # Tính tỉ lệ cạnh ngang trên chia cạnh dọc trái
    check1=np.sum(np.square(tl-tr))/np.sum(np.square(tl-bl))
    # Tính tỉ lệ cạnh ngang dưới chia cạnh dọc phải
    check2=np.sum(np.square(bl-br))/np.sum(np.square(tr-br))
    # Tính độ chênh lệnh của 2 tỉ lệ
    edge_diff=np.abs(check1-check2)
    # cv2.imshow(f"{edge_diff}",frame[tl[1]:br[1],tl[0]:br[0]])
    # Nếu tđọ chênh lệch < 1 thì đó khả năng cao đó là 1 biển số xe vuông còn biển số xe ngang có sai số lớn hơn 
    if(edge_diff<1 or edge_diff >3):
        # Biển ngang
        if(check1 and check2 > 4.5):
            w,h=700,160
        # Biển vuông
        else:
            w,h=400,300
        # Chuyển phối cảnh về thẳng trước mặt
        pst1=np.float32([tl,tr,bl,br])
        pst2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
        M=cv2.getPerspectiveTransform(pst1,pst2)
        out=cv2.warpPerspective(thresh,M,(w,h))
        # out = cv2.medianBlur(out,5)
        # Tính tỉ lệ nền trắng trong hình đã phối cảnh
        ratio=white_ratio(out,type='gray_rv')
        if(ratio>0.60):
            # cv2.imshow(f"{ratio}",out)
            infor=[]
            # Lấy tọa độ những con số bằng thuật toán Connected components analysis
            numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(out,connectivity=4)
            for i in range(0, numLabels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                (cX, cY) = centroids[i]
                # Lấy tọa độ hình có chiều ngang nằm trong khoảng 10-110 và chiều cao từ 75-170 => Dữ liệu chứa số trong biển
                if(w in range(10,110) and h in range(75,170)):
                    infor.append([x,y,w,h])
            # Nếu trong biển có số thì tiến hành sắp xếp và nhận dạng
            if(len(infor)>0):
                # Tên biển
                name_plate=''
                # Dãy số bên trên
                upper=[]
                # Dãy số biên dưới
                lower=[]
                # Duyệt từng biển trong dữ liệu
                for loc in infor:
                    # Loc chứa x,y,w,h; ta sẽ duyệt y để phân chia 2 hàng
                    if(loc[1]<70):
                        upper.append(loc)
                    else:
                        lower.append(loc)
                # Sắp xếp
                upper.sort(key=lambda row: row[0])
                lower.sort(key=lambda row: row[0])
                # Gộp lại những tọa độ đã sắp xếp
                sort=upper+lower
                for x,y,w,h in sort:
                    # Chuẩn hóa và lấp đầy hình ảnh về size 75x100
                    norm_output=toNorm(out[y:y+h,x:x+w])
                    # Nhận dạng hình ảnh số đã chuẩn hóa
                    name_plate+=pred_num(norm_output)
                # Lắp lại tránh trùng khi nhận 1 biển nhiều lần
                if name_plate not in plate:
                    plate.append(name_plate)
                    # Vẽ bọc xung quanh biển
                    frame=draw(frame,tl,tr,bl,br)
                    # Đặt chữ lên bọc biển
                    frame=cv2.putText(frame,name_plate,tl,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
                print(plate)
cv2.imshow("original",cv2.resize(frame,(700,400)))
cv2.waitKey(0)