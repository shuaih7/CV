# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import glob as gb
import numpy as np
from skimage import data
import skimage.morphology as sm
from matplotlib import pyplot as plt

from skimage import morphology
from skimage.measure import label, regionprops
from skimage.filters import try_all_threshold, threshold_otsu, threshold_local


def is_full_window(img_bn):
    h, w = img_bn.shape[:2]
    sum_x = img_bn.sum(axis=0)
    sum_x = sum_x[sum_x>0]
    sum_y = img_bn.sum(axis=1)
    sum_y = sum_y[sum_y>0]
    
    num_x = sum_x.shape[0]
    num_y = sum_y.shape[0]
    
    if num_x == w and num_y == h: 
        return True
    else:
        return False


def extract_cross_center(img_cross, kernel=(25,25), stride=(3,3)):
    img_cross[img_cross>0] = 1
    img_sk = morphology.skeletonize(img_cross)
    # plt.imshow(img_sk, cmap="gray"), plt.show()
    
    windows = []
    h, w = img_cross.shape[:2]
    
    kh, kw = kernel
    sh, sw = stride
    curh, curw = 0, 0
    
    for i in range(h):
        curw = 0
        if curh > h-kh: break
    
        for j in range(w):
            if curw > w-kw: break
            win = img_sk[curh:curh+kh, curw:curw+kw]
            if is_full_window(win):
                center = (curw+kw//2, curh+kh//2)
                windows.append(center)
            curw += sw
        curh += sh
        
    if len(windows) == 0: return None
    
    x, y = 0, 0
    for window in windows:
        x += window[0]
        y += window[1]
    x /= len(windows)
    y /= len(windows)
    
    return (int(x), int(y))


def extract_cross(image, offset=85, min_len=10):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_h = image_hsv[:,:,0]
    image_cross = np.zeros(image_h.shape, dtype=np.uint8)
    
    image_cross[image_h > 255-offset] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
    image_cross = cv2.dilate(image_cross, kernel)
    
    label_mask = label(image_cross, connectivity = 2)
    properties = regionprops(label_mask)

    index = -1
    cross_roi = None
    major_len, minor_len = 0, 0 
    for prop in properties:
        cur_major_len = prop.major_axis_length
        cur_minor_len = prop.minor_axis_length
        if cur_major_len > major_len: 
            index = prop.label
            major_len = cur_major_len
            minor_len = cur_minor_len
            cross_roi = prop.bbox  # y1, x1, y2, x2
            
    if major_len < min_len: return None, None # Case while the cross center has not been extracted
    
    image_cross[label_mask!=index] = 0
    y1, x1, y2, x2 = cross_roi
    img_cr = image_cross[y1:y2, x1:x2]
    
    plt.subplot(1,3,1), plt.imshow(image[:,:,-1], cmap="gray"), plt.title("Orginal image")
    plt.subplot(1,3,2), plt.imshow(image_h, cmap="gray"), plt.title("HSV image")
    plt.subplot(1,3,3), plt.imshow(image_cross, cmap="gray"), plt.title("Binarized image")
    plt.show()
    
    local_center = extract_cross_center(img_cr)
    if local_center is None: return None, None
    
    center = (x1+local_center[0], y1+local_center[1])
    print(center)
    
    return center, image_cross
   
    
if __name__ == "__main__":
    img_dir = r"E:\Projects\Integrated_Camera\bsi_proj\0\diff"
    img_list = gb.glob(img_dir + r"/*.png")
    
    for img_file in img_list:
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        center, image_cross = extract_cross(image)