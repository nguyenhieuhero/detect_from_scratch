import imp
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

label=np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C','D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
encoder=OneHotEncoder()
encoder.fit(label.reshape(-1,1))
model=load_model('detect_from_scratch\model (1).h5')
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
def toNorm(img,w=75,h=100):
    rs=100/img.shape[0]
    resize=cv2.resize(img,(int(img.shape[1]*rs),int(img.shape[0]*rs)))
    x_to_put=int((75-resize.shape[1])/2)
    constant= cv2.copyMakeBorder(resize,0,0,x_to_put,x_to_put,cv2.BORDER_CONSTANT)
    return cv2.resize(constant,(75,100))
def pred_num(gray_num_img):
    pred=model.predict(np.array([gray_num_img/255]))
    np.argmax(pred)
    out_norm=np.zeros(35)
    out_norm[np.argmax(pred)]=1
    return encoder.inverse_transform([out_norm])[0][0]