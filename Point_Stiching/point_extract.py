# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import glob as gb
import numpy as np
from skimage import data
import skimage.morphology as sm
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from skimage.filters import try_all_threshold, threshold_otsu, threshold_local


def try_all_thresholding(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
    plt.show()
    

def try_local_thresholding(img_file):   
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

    global_thresh = threshold_otsu(image)
    binary_global = image > global_thresh

    block_size = 25
    local_thresh = threshold_local(image, block_size, offset=5)
    binary_local = image > local_thresh

    fig, axes = plt.subplots(ncols=3, figsize=(7, 8))
    ax = axes.ravel()
    plt.gray()

    ax[0].imshow(image)
    ax[0].set_title('Original')

    ax[1].imshow(binary_global)
    ax[1].set_title('Global thresholding')

    ax[2].imshow(binary_local)
    ax[2].set_title('Local thresholding')

    for a in ax:
        a.axis('off')

    plt.show()
    

def get_point_diameter(image, min_d=2, max_d=30, aspect_ratio=1.6):
    global_thresh = threshold_otsu(image)
    binary_global = image > global_thresh
    
    label_mask = label(binary_global, connectivity = 2)
    properties = regionprops(label_mask)

    num_pts, pt_len = 0, 0
    for prop in properties:
        diameter = prop.equivalent_diameter
        major_len = prop.major_axis_length
        minor_len = max(1, prop.minor_axis_length)
        if diameter < min_d or diameter > max_d or major_len/minor_len > aspect_ratio:
            binary_global[label_mask==prop.label] = 0
        else:
            pt_len += diameter
            num_pts += 1      
    # plt.imshow(binary_global, cmap="gray"), plt.show()
    
    if num_pts > 0: 
        return int(pt_len / num_pts)
    else:
        return 0
        
        
def point_thresholding(image, min_d=3, max_d=30, aspect_ratio=1.5):
    point_size = get_point_diameter(image, min_d, max_d, aspect_ratio)
    if point_size == 0: return np.zeros(image.shape, dtype=np.uint8)
    
    block_size = point_size*4+1 # The block size must be odd
    offset = point_size//2+1
    local_thresh = threshold_local(image, block_size, offset=offset)
    binary_local = image > local_thresh
    # binary_local = sm.erosion(binary_local,sm.square(max(1, point_size//3)))
    # binary_local = sm.opening(binary_local, sm.disk(point_size//3+1))
    
    img_pt = binary_local
    num_pts, pt_len, loop_num = 0, 0, 1
    
    for i in range(loop_num):
        img_pt = sm.opening(img_pt, sm.disk(point_size//3+1))
        label_mask = label(img_pt, connectivity = 2)
        properties = regionprops(label_mask)
        
        points = []
        for prop in properties:
            diameter = prop.equivalent_diameter
            major_len = prop.major_axis_length
            minor_len = max(1, prop.minor_axis_length)
            if diameter < min_d or diameter > max_d or major_len/minor_len > aspect_ratio:
                img_pt[label_mask==prop.label] = 0
            else:
                points.append((prop.centroid[1], prop.centroid[0]))
    
    fig, axes = plt.subplots(ncols=2, figsize=(7, 8))
    ax = axes.ravel()
    plt.gray()

    ax[0].imshow(binary_local)
    ax[0].set_title('Local thresholding')
    
    ax[1].imshow(img_pt)
    ax[1].set_title('Point image')

    for a in ax:
        a.axis('off')
    plt.show()
    
    return img_pt, points
    
    
if __name__ == "__main__":
    """
    img_file = r"E:\Projects\Integrated_Camera\point_images\2020-12-28_164902_10.png"
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    img_pt, points = point_thresholding(image)
    
    """
    img_dir = r"E:\Projects\Integrated_Camera\point_images_2"
    
    img_list = gb.glob(img_dir + r"/*.png")
    for img_file in img_list:
        image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        img_pt = point_thresholding(image)
