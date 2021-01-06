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

from utils import *


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
    
    
def extract_cross_angles(img_cross):
    hor_angle, ver_angle = None, None
    cr_h, cr_w = img_cross.shape[:2]
    
    left_center = get_bd_center(img_cross[:,0])
    right_center = get_bd_center(img_cross[:,-1])
    top_center = get_bd_center(img_cross[0,:])
    bottom_center = get_bd_center(img_cross[-1,:])
    
    if left_center>0 and right_center>0 and top_center>0 and bottom_center>0:
        hor_angle = get_image_angle([0,left_center], [cr_w, right_center])
        ver_angle = get_image_angle([top_center,0], [bottom_center,cr_h])
        
    #print(round(hor_angle, 3), round(ver_angle, 3))
    #plt.imshow(img_cross, cmap="gray"), plt.show()
    
    return hor_angle, ver_angle
    
    
def get_bd_center(array):
    bd_centers = []
    for i in range(array.shape[0]):
        if array[i] > 0:
            bd_centers.append(i)
            
    if len(bd_centers) == 0: return -1
    else: return int(sum(bd_centers)/len(bd_centers))
       

def extract_cross(image, thresh=30, min_len=45):
    if image.shape[-1]==3: 
        image_gray = image[:,:,-1] # The image is by-default the BGR format
    else:
        image_gray = image
    image_cross = np.zeros(image_gray.shape, dtype=np.uint8)
    
    image_cross[image_gray > thresh] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    image_cross = cv2.dilate(image_cross, kernel)
    
    label_mask = label(image_cross, connectivity = 2)
    properties = regionprops(label_mask)

    index = -1
    cross_roi = None
    area_ratio = 1 # feature area to the roi area
    major_len, minor_len = 0, 0
    
    # Find the binarized cross feature
    #    1. Both the major axis length and minor axis length should be larger than min_len
    #    2. Feature has the smallest feature area to the roi area ratio
    for prop in properties:
        cur_major_len = prop.major_axis_length
        cur_minor_len = prop.minor_axis_length
        
        y1, x1, y2, x2 = prop.bbox
        cur_roi = image_cross[y1:y2, x1:x2]
        cur_area_ratio = cur_roi.sum() / (y2-y1) / (x2-x1) / 255
        
        if cur_major_len>min_len and cur_area_ratio<area_ratio: 
            index = prop.label
            cross_roi = prop.bbox  # y1, x1, y2, x2
            major_len = cur_major_len
            minor_len = cur_minor_len
            area_ratio = cur_area_ratio
    
    image_cross[label_mask!=index] = 0
    if cross_roi is None: return None, None, None, image_cross
    
    y1, x1, y2, x2 = cross_roi
    img_cr = image_cross[y1:y2, x1:x2]
    hor_angle, ver_angle = extract_cross_angles(img_cr)
    
    local_center = extract_cross_center(img_cr)
    if local_center is None: return None, None, None, image_cross
    center = (x1+local_center[0], y1+local_center[1])
    
    print("The cross center position is:", center)
    plt.subplot(1,3,1), plt.imshow(image), plt.title("Orginal image")
    plt.subplot(1,3,2), plt.imshow(image_gray, cmap="gray"), plt.title("Red channel image")
    plt.subplot(1,3,3), plt.imshow(image_cross, cmap="gray"), plt.title("Binarized image")
    plt.show()
    
    return center, hor_angle, ver_angle, image_cross
    
    
def extract_feature(image, 
                    pt_thresh=30, 
                    pt_min_d=20,
                    pt_max_d=65,
                    pt_aspect_ratio=1.5,
                    cr_thresh=30):
                    
    image_gray = image.copy()
    center, hor_angle, ver_angle, image_cross = extract_cross(image, cr_thresh, pt_max_d*0.75)
    if image.shape[-1] == 3: image_gray = cv2.cvtColor(image_gray, cv2.COLOR_BGR2GRAY)
        
    image_gray = cv2.blur(image_gray,(5,5))
    image_points = np.zeros(image_gray.shape, dtype=np.uint8)
    image_points[image_gray>pt_thresh] = 255
    image_points[image_cross>0] = 0
    
    label_mask = label(image_points, connectivity = 2)
    properties = regionprops(label_mask)
    
    points = []
    for prop in properties:
        cur_d = prop.equivalent_diameter 
        cur_major_len = prop.major_axis_length
        cur_minor_len = max(0.001, prop.minor_axis_length)
        
        cur_aspect_ratio = cur_major_len / cur_minor_len
        
        if cur_d>pt_min_d and cur_d<pt_max_d and cur_aspect_ratio<pt_aspect_ratio:
            points.append((prop.centroid[1], prop.centroid[0]))
        else:
            image_points[label_mask==prop.label] = 0
    
    """
    plt.subplot(1,2,1), plt.imshow(image), plt.title("Original image")
    plt.subplot(1,2,2), plt.imshow(image_points, cmap="gray"), plt.title("Binarized image")
    plt.show()
    """
    
    return points, center, hor_angle, ver_angle
   
    
if __name__ == "__main__":
    img_dir = r"E:\Projects\Integrated_Camera\bsi_proj\diff"
    img_list = gb.glob(img_dir + r"/*.png")
    
    for img_file in img_list:
        # if img_file != r"E:\Projects\Integrated_Camera\bsi_proj\diff\diff_1_0.png": continue
        print("Processing image file", img_file, "...")
        image = cv2.imread(img_file, cv2.IMREAD_COLOR)
        #center, image_cross = extract_cross(image)
        points, center, hor_angle, ver_angle = extract_feature(image)