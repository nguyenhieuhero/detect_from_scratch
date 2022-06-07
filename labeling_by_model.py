import os
import glob
import cv2
from utils import pred_num
for img_src in glob.glob("detect_from_scratch/num_data/*.jpg"):
    print(img_src)
    img=cv2.imread(img_src,cv2.IMREAD_GRAYSCALE)
    try:
        label=pred_num(img)
    except:
        pass
    else:
        cv2.imwrite(f"{os.path.join(os.path.dirname(img_src),label,os.path.basename(img_src))}",img)
        os.remove(img_src)