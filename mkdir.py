import glob
import os
import numpy as np
from utils import toNorm
i=0
for x in glob.glob("detect_from_scratch\JPEGImages/*.jpg"):
    i+=1
    os.rename(x,os.path.join(os.path.dirname(x),f"{i}.jpg"))
