import cv2


img=cv2.imread("detect_from_scratch/num/1.2.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rs=100/img.shape[0]
resize=cv2.resize(img,(int(img.shape[1]*rs),int(img.shape[0]*rs)))
x_to_put=int((75-resize.shape[1])/2)
constant= cv2.copyMakeBorder(resize,0,0,x_to_put,x_to_put,cv2.BORDER_CONSTANT)
constant=cv2.resize(constant,(75,100))
cv2.imshow(f"img {img.shape[1],img.shape[0]}",img)
cv2.imshow(f"constant {constant.shape[1],constant.shape[0]}",constant)
cv2.imwrite('detect_from_scratch/num/testT.jpg',constant)

cv2.waitKey(0)