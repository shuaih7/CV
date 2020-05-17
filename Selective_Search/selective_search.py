#-*-coding:utf-8-*-

import os, sys, cv2
import numpy as np
from skimage.segmentation import felzenszwalb

def felzenszwalb_processor(img_file):
    image = cv2.imread(img_file, -1)
    scale = 1.0
    sigma = 5.0
    