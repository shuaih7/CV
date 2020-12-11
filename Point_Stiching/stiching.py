# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops


def draw_points(img, points, radius=5, color=(0,0,255), thickness=4):
    for point in points:
        point = (int(point[0]), int(point[1]))
        img = cv2.circle(img, point, radius, color, thickness)
    plt.imshow(img, cmap="gray")
    plt.show()
    
    
def display_mapping(img1, img2, map_pos, radius=5, color=(0,0,255), thickness=4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, cur_map in enumerate(map_pos):
        pt1, pt2 = cur_map
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        img1 = cv2.circle(img1, pt1, radius, color, thickness)
        img1 = cv2.putText(img1, str(i), pt1, font, 1.2, color, thickness)
        img2 = cv2.circle(img2, pt2, radius, color, thickness)
        img2 = cv2.putText(img2, str(i), pt2, font, 1.2, color, thickness)
    plt.subplot(1,2,1), plt.imshow(img1, cmap="gray"), plt.title("Image 1")
    plt.subplot(1,2,2), plt.imshow(img2, cmap="gray"), plt.title("Image 2")
    plt.show()


def extract_points(img, thresh=200, min_rad=20, max_rad=80):
    origin = img.copy()
    img = cv2.GaussianBlur(img,(5,5),0)
    _, img_bn = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    
    # Create the region proposal mask and filter out the curve
    label_mask = label(img_bn, connectivity = 2)
    properties = regionprops(label_mask)
    
    points = []
    for prop in properties:
        diameter = prop.equivalent_diameter
        if diameter < min_rad or diameter > max_rad: 
            img_bn[label_mask==prop.label] = 0
        else: points.append([prop.centroid[1], prop.centroid[0]])
    """
    plt.subplot(1,3,1), plt.imshow(origin, cmap="gray"), plt.title("Gray Scale Image")
    plt.subplot(1,3,2), plt.imshow(img, cmap="gray"), plt.title("Blurred Gray Scale Image")
    plt.subplot(1,3,3), plt.imshow(img_bn, cmap="gray"), plt.title("Binarized Image")
    plt.show()
    """
    return points
    
    
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
    
    
def map_single_point(points1,
                     points2,
                     max_angle_sft=30,
                     tolerance=6): 
    cur_pt1 = points1[0]
    cur_angle1 = get_angle(center1, cur_pt1)
    cur_pts2 = points2[:tolerance]
    for index, cur_pt2 in enumerate(cur_pts2):
        cur_angle2 = get_angle(center2, cur_pt2)
        # print(cur_pt1, cur_angle1, cur_pt2, cur_angle2)
        if abs(cur_angle1-cur_angle2) < max_angle_sft: 
            del points1[0]
            del points2[index]
            return points1, points2, [cur_pt1, cur_pt2]
    return points1, points1, None     
    

def map_points(img1, 
               img2,
               center1, 
               center2,
               num_stich=5,     # Number of points to match
               max_angle_sft=30, # Maximum angle shifting between two continuous images
               max_angle_dir=30, # Maximum angle to distinghuish the direction
               tolerance=6):     # ...
    
    # --- 1: stiching the feature points
    map_pos = [] # List to store the mapped point positions
    points1 = extract_points(img1)
    points2 = extract_points(img2)
    
    if num_stich > len(points1): num_stich = len(points1)
    if tolerance > num_stich: tolerance = len(points1)

    points1 = sort_points(center1, points1)
    points2 = sort_points(center2, points2)
    
    # --- 1.1: Select the initial matching point
    for i in range(num_stich):
        points1, points2, pt_map = map_single_point(points1, points2, max_angle_sft=max_angle_sft, tolerance=tolerance)
        if pt_map is not None: map_pos.append(pt_map)
    
    return map_pos    
        
    
if __name__ == "__main__":
    img_file_1 = r"E:\Projects\Integrated_Camera\image_stiching\img1.png"
    img_file_2 = r"E:\Projects\Integrated_Camera\image_stiching\img2.png"
    
    center1, center2 = (554,133), (54,159)
    img1 = cv2.imread(img_file_1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_file_2, cv2.IMREAD_GRAYSCALE)
    
    map_pos = map_points(img1, img2, center1, center2)
    print(map_pos)
    display_mapping(img1, img2, map_pos)
    
    