# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops


def sort_points(center, points):
    if len(points) == 0: return points
    sorted_points, dists = [], []
    
    for point in points:
        dists.append((point[0]-center[0])**2+(point[1]-center[1])**2)
    sorted_index = np.argsort(dists)
    
    points = np.array(points)[sorted_index] # Sort the point array
    
    return points.tolist()
    
    
def get_angle(center, point) -> float:
    x0, y0 = center[0], center[1]
    x1, y1 = point[0], point[1]
    if y0 == y1 and x1 >= x0: return 0.0
    
    length = math.sqrt((x1-x0)**2+(y1-y0)**2)
    
    cos_value = (x1-x0) / length
    value = math.acos(cos_value)/math.pi * 180
    
    if y1 > y0: return value
    else: return 360.0 - value
    
    
def dye_image(img_cross_origin):
    img_cross = img_cross_origin.copy()
    img_h, img_w = img_cross.shape[:2]
    hor_array = img_cross.sum(axis=0)
    ver_array = img_cross.sum(axis=1)

    h_start, v_start = False, False
    for i in range(hor_array.shape[0]):
        if hor_array[i] > 0 and not h_start:
            h_start = True # Start the horizontal line tracking 
            for j in range(img_h): 
                if img_cross[j,i] > 0:
                    for k in range(i,-1,-1): img_cross[j,k] = 255
                    break
        elif hor_array[i] == 0 and h_start:
            for j in range(img_h): 
                if img_cross[j,i-1] > 0:
                    for k in range(i,img_w,1): img_cross[j,k] = 255
                    break
            
    for j in range(ver_array.shape[0]):
        if ver_array[j] > 0 and not v_start:
            v_start = True
            for i in range(img_w):
                if img_cross[j,i] > 0:
                    for k in range(j,-1,-1): img_cross[k,i] = 255
                    break
        elif ver_array[j] == 0 and v_start: 
            for i in range(img_w): 
                if img_cross[j-1,i] > 0:
                    for k in range(j,img_h,1): img_cross[k,i] = 255
                    break
    
    inv_img_cross = 255 - img_cross
    img_cross_mask = label(inv_img_cross, connectivity = 2)
    img_cross_mask[img_cross_mask>4] = 0
    #if img_cross_mask.max() > 4: 
    #    raise Exception("Error occurs in the dye cross mask process.")
    #else: img_cross_mask += 10
    
    # Rearrange the coordinate
    # properties = regionprops(img_cross_mask)
    # mask_centers = []
    #for prop in properties: 
    #    if prop.area < 100: img_cross_mask[prop.label] = 0
    return img_cross_mask
    
    
if __name__ == "__main__":
    img_cross = cv2.imread("sample2_cr.png", cv2.IMREAD_GRAYSCALE)
    img_cross[img_cross>200] = 255
    img_cross[img_cross<=200] = 0
    img_cross_mask = dye_image(img_cross)
    