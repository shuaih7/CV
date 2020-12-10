# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops


def extract_points(img, thresh=200, min_rad=20, max_rad=80):
    origin = img.copy()
    img = cv2.GaussianBlur(img,(5,5),0)
    _, img_bn = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    
    # Create the region proposal mask and filter out the curve
    label_mask = label(img_bn, connectivity = 2)
    # mask = label_mask.copy()
    properties = regionprops(label_mask)
    
    points = []
    for prop in properties:
        diameter = prop.equivalent_diameter
        if diameter < min_rad or diameter > max_rad: 
            img_bn[label_mask==prop.label] = 0
            points.append(prop.centroid)
    """   
    plt.subplot(1,3,1), plt.imshow(origin, cmap="gray"), plt.title("Gray Scale Image")
    plt.subplot(1,3,2), plt.imshow(img, cmap="gray"), plt.title("Blurred Gray Scale Image")
    plt.subplot(1,3,3), plt.imshow(img_bn, cmap="gray"), plt.title("Binarized Image")
    plt.show()
    """
   
    return points
    
    
def sort_points(points, pt):
    if len(points) == 0: return points
    sorted_points, dists = [], []
    
    for point in points:
        dists.append((point[0]-center[0])**2+(point[1]-center[1])**2)
    sorted_index = np.argsort(dists)
    
    return sorted_index
    
    
def get_angle(center, point):
    x0, y0 = center[0], center[1]
    x1, y1 = point[0], point[1]
    if y0 == y1 and x1 >= x0: return 0.0
    
    length = math.sqrt((x1-x0)**2+(y1-y0)**2)
    
    cos_value = (x1-x0) / length
    value = math.acos(cos_value)/math.pi * 180
    
    if y1 > y0: return value
    else: return 360.0 - value


def map_points(img1, img2,
               num_stich=20,  # Number of points to match
               max_angle=30,
               tolerance=6):
    # Part 1: extract the anchor points
    center1, center2 = (554,133), (54,159)
    
    # Part 2: stiching the feature points
    points1 = extract_points(img1)
    points2 = extract_points(img2)
    
    if num_stich > len(points1): num_stich = len(points1)
    if tolerance > num_stich: tolerance = len(points1)
    points1 = points1[:num_stich]
    points2 = points2[:num_stich]
        
    indices1 = sort_points(points1, center1)
    indices2 = sort_points(points2, center2)
    
    # ---- 2.1: Select the initial matching point
    cur_pt1 = points1[indices1[0]]
    cur_angle = get_angle(center1, cur_pt1)
    for t in range(tolerance): 
        cur_pt2 = points2[t]
        
        
        
    
if __name__ == "__main__":
    img_file_1 = r"E:\Projects\Integrated_Camera\image_stiching\img1.png"
    img_file_2 = r"E:\Projects\Integrated_Camera\image_stiching\img2.png"
    
    img1 = cv2.imread(img_file_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_file_2, cv2.IMREAD_GRAYSCALE)
    
    print(get_angle((0,0),(-1,0)))