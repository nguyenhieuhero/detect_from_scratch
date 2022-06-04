import cv2
import glob
i=0
Char='Y'
for img_source in glob.glob(f"detect_from_scratch\CNN letter Dataset/{Char}/*"):
    i+=1
    if i%10==0:
        image=cv2.imread(img_source)
        bit=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(bit,100,255,cv2.THRESH_BINARY_INV)
        cv2.imwrite(f"detect_from_scratch/num_data/{Char}/{i}.jpg",thresh)