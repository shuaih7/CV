#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12.15.2020

@author: haoshuai@handaotech.com
"""

import numpy as np
import cv2, os, sys
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
#from skimage import morphology
from utils import *


def draw_boxes(img, boxes, color=255, thickness=2):
    for box in boxes:
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=color, thickness=thickness)
    return img


def write_cross(img_file="sample1.png", thresh=235):
    """Filter out the cross and save into file
    
    Args:
        img_file: Input image file name
        thresh: Thesholding value for the image binarization
    
    """
    im = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img_orig  = im.copy()

    im = cv2.GaussianBlur(im,(5,5),0)

    thres,im = cv2.threshold(im,thresh,255,cv2.THRESH_BINARY)

    # Create the region proposal mask and filter out the curve
    label_mask = label(im, connectivity = 2)
    properties = regionprops(label_mask)

    max_rad = 80
    for prop in properties:
        diameter = prop.equivalent_diameter
        if diameter < max_rad:
            im[label_mask==prop.label] = 0

    plt.imshow(im, cmap="gray"), plt.title("Cross")
    plt.show()
    
    filename, suffix = os.path.splitext(img_file)
    save_name = filename + "_cr" + suffix
    cv2.imwrite(save_name, im)
    
    
def is_valid_window(img_win) -> bool:
    def check_array(array):
        if array[0] > 0 or array[-1] > 0: return False
        is_start = False
        num_block = 0
        
        for i in range(1, array.shape[0]):
            if array[i-1]==0 and array[i]>0: is_start = True
            elif array[i-1]>0 and array[i]==0 and is_start:
                num_block += 1
                is_start = False
        
        if num_block == 1: return True
        else: return False
        
    top_flag = check_array(img_win[0,:])
    bottom_flag = check_array(img_win[-1,:])
    left_flag = check_array(img_win[:,0])
    right_flag = check_array(img_win[:,-1])
    
    return top_flag and bottom_flag and left_flag and right_flag
    
    
def get_intersection(pt1, pt2, pt3, pt4):
    def get_k_and_b(point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        
        k = (y1-y2)/(x1-x2)
        b = (y1*x2-y2*x1)/(x2-x1)
        
        return k, b
        
    k1, b1 = get_k_and_b(pt1, pt2)
    k2, b2 = get_k_and_b(pt3, pt4)
    
    x0 = (b1-b2)/(k2-k1)
    y0 = (b1*k2-b2*k1)/(k2-k1)
    
    return (x0, y0)
    
    
def get_cross(img, hsize=0, vsize=0, hstep=30, vstep=20):
    """Get the center of the cross
    
    Args:
        img:   Binary image with the cross
        hsize: horizontal kernel size to find the area contains the cross center
        vsize: vertical kernel size to find the area contains the cross center
        hstep: moving step in the horizontal region
        vstep: moving step in the verical region
        
    Returns:
        center: center point of the cross
    
    """
    img_h, img_w = img.shape[:2]
    if hsize == 0: hsize = img_w // 4
    if vsize == 0: vsize = img_h // 4
    
    h_pos, v_pos, max_num, window = 0, 0, 0, None
    
    while h_pos+hsize < img_w:
        while v_pos+vsize < img_h:
            cur_win = img[v_pos:v_pos+vsize, h_pos:h_pos+hsize]
            cur_num = cur_win.sum()//255
            if cur_num > max_num: 
                #label_mask = label(255-cur_win, connectivity = 2)
                #if label_mask.max() == 4:
                if is_valid_window(cur_win):
                    window = [h_pos, v_pos, h_pos+hsize, v_pos+vsize]
                    max_num = cur_num
            v_pos += vstep # Add a vertical step
        v_pos = 0
        h_pos += hstep # Add a horizontal step
    
    for i in range(window[0]+1, window[2], 1):
        if img[window[1],i-1] == 0 and img[window[1],i] > 0: 
            top_left = (i, window[1])
        elif img[window[1],i-1] > 0 and img[window[1],i] == 0:
            top_right = (i, window[1])
        if img[window[3],i-1] == 0 and img[window[3],i] > 0:
            bottom_left = (i, window[3])
        elif img[window[3],i-1] > 0 and img[window[3],i] == 0:
            bottom_right = (i, window[3])
            
    for i in range(window[1]+1, window[3], 1):
        if img[i-1,window[0]] == 0 and img[i,window[0]] > 0:
            left_up = (window[0], i)
        elif img[i-1,window[0]] > 0 and img[i,window[0]] == 0:
            left_dn = (window[0], i)
        if img[i-1,window[2]] == 0 and img[i,window[2]] > 0:
            right_up = (window[2], i)
        elif img[i-1,window[2]] > 0 and img[i,window[2]] == 0:
            right_dn = (window[2], i)
            
    center1 = get_intersection(left_up, right_dn, left_dn, right_up)
    center2 = get_intersection(top_left, bottom_right, top_right, bottom_left)
    center = (int((center1[0]+center2[0])/2), int((center1[1]+center2[1])/2))
    
    # Get the horizontal and vertical slope 
    hor_angle1 = get_image_angle(left_up, right_up)
    hor_angle2 = get_image_angle(left_dn, right_dn)
    ver_angle1 = get_image_angle(bottom_left, top_left)
    ver_angle2 = get_image_angle(bottom_right, top_right)
    
    hor_angle = (hor_angle1+hor_angle2)/2
    ver_angle = (ver_angle1+ver_angle2)/2
    
    print(center, hor_angle, ver_angle)
    if window is not None:     
        img_box = img.copy()
        img_box = draw_boxes(img_box, [window])
        plt.subplot(1,2,1), plt.imshow(img, cmap="gray"), plt.title("Original Cross")
        plt.subplot(1,2,2), plt.imshow(img_box, cmap="gray"), plt.title("Selected Window")
        plt.show()
    
    return center, hor_angle, ver_angle
    
    
if __name__ == "__main__":
    img = cv2.imread("sample2_cr.png", cv2.IMREAD_GRAYSCALE)
    center, hor_angle, ver_angle = get_cross(img)

    



