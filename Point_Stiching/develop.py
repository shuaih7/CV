# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

from utils import *
from cross import get_cross


POINTS = {}


def extract_image(img, thresh=220, min_rad=20, max_rad=80):
    """Binarize the input image and extract multiple info
    
    Args:
        img: Input grayscale image
        thresh: Thesholding value for the image binarization
        min_rad: Minimum point radius
        max_rad: Maximum point radius
        
    Returns:
        img_points: Binarized image contains the points
        img_cross: Binarized image contains the cross
        
    """
    img = cv2.GaussianBlur(img,(5,5),0)
    thres,img_points = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)
    img_cross = img_points.copy()

    # Create the region proposal mask and filter out the curve
    label_mask = label(img_points, connectivity = 2)
    properties = regionprops(label_mask)

    max_rad = 80
    #points = []
    for prop in properties:
        diameter = prop.equivalent_diameter
        if diameter < min_rad:
            img_points[label_mask==prop.label] = 0
            img_cross[label_mask==prop.label] = 0
        elif diameter > max_rad:
            img_points[label_mask==prop.label] = 0
        #else: 
        #    points.append([prop.centroid[1], prop.centroid[0]])
            
    img_cross[img_points>0] = 0
    '''
    plt.subplot(1,2,1), plt.imshow(img_points, cmap="gray"), plt.title("Point Image")
    plt.subplot(1,2,2), plt.imshow(img_cross, cmap="gray"), plt.title("Cross Image")
    plt.show()
    '''
    return img_points, img_cross
    

def get_hor_len(point1, point2):
    # x1, y1 = point1
    # x2, y2 = point2
    # return math.sqrt((x2-x1)**2+(y2-y1)**2)
    return abs(point2[0]-point1[0])
    
    
def get_ver_len(point1, point2):
    # x1, y1 = point1
    # x2, y2 = point2
    # return math.sqrt((x2-x1)**2+(y2-y1)**2)
    return abs(point2[1]-point1[1])
    
    
def register(point, 
             row, 
             col, 
             hor_angle=0,
             ver_angle=0, 
             hor_len=0, 
             ver_len=0):
             
    point_elem = {
        "row": row,
        "col": col,
        "hor_angle": hor_angle,
        "ver_angle": ver_angle,
        "hor_len": hor_len,
        "ver_len":ver_len,
    }
    
    index = (point[0], point[1])
    POINTS[index] = point_elem
    
    
def update_information(center, 
                       point, 
                       row, 
                       col, 
                       hor_angle, 
                       ver_angle, 
                       hor_len, 
                       ver_len, 
                       coordinate,
                       dir="horizontal"):
    """ Update the information for the next search_surround
    
    Args:
        center: center point
        point: reference point
        row: row index of the center point
        col: column index of the center point
        hor_len: horizontal length from the previous
        ver_len: vertical length from the previous
        
    Returns:
        row: row index
        col: column index
        hor_angle: updated horizontal angle
        ver_angle: updated vertical angle
        hor_len: updated horizontal length
        ver_len: updated vertical length
        
    Raises:
        ValueError
        
    """
    x0, y0 = center
    x1, y1 = point
    
    if dir == "horizontal":
        hor_len = get_hor_len(center, point)
        
        if x1 > x0: 
            hor_angle = get_image_angle(center, point)
            if coordinate==1 or coordinate==4: col += 1
            else: col -= 1
        elif x1 < x0:
            hor_angle = get_image_angle(point, center)
            if coordinate==1 or coordinate==4: col -= 1
            else: col += 1
            
    elif dir == "vertical":
        ver_len = get_ver_len(center, point)
        
        if y1 > y0:
            ver_angle = get_image_angle(point, center)
            if coordinate==3 or coordinate==4: row += 1
            else: row -= 1
        elif y1 < y0:
            ver_angle = get_image_angle(center, point)
            if coordinate==3 or coordinate==4: row -= 1
            else: row += 1
    else:
        raise ValueError("Invalid direction value.")
        
    return row, col, hor_angle, ver_angle, hor_len, ver_len
    
    
def search_surround(input_points, 
                    center,
                    row, 
                    col,
                    hor_angle, 
                    ver_angle, 
                    hor_len, 
                    ver_len,
                    coordinate,
                    max_angle_shift=10,
                    max_hor_ratio=1.5,
                    max_ver_ratio=1.5):
    points = input_points.copy()
    points = sort_points(center, points)
    
    x0, y0 = center
    for i, pt in enumerate(points[:8]):
        x1, y1 = pt
        # Two points are identical
        if abs(x1-x0)<0.0001 and abs(y1-y0)<0.0001:
            continue
        # Check the horizontal right
        if abs(get_angle(center, pt)-hor_angle)<max_angle_shift \
                and max(get_hor_len(center,pt),hor_len)/min(get_hor_len(center,pt),hor_len)<max_hor_ratio:
            if (pt[0], pt[1]) not in POINTS:
                nrow, ncol, nhor_angle, nver_angle, nhor_len, nver_len = update_information(center, pt, row, col, 
                                        hor_angle, ver_angle, hor_len, ver_len, coordinate, dir="horizontal")
                register(pt, nrow, ncol)
                search_surround(points, pt, nrow, ncol, nhor_angle, nver_angle, nhor_len, nver_len, coordinate)
        elif abs(get_angle(center, pt)-ver_angle)<max_angle_shift \
                and max(get_ver_len(center,pt),ver_len)/min(get_ver_len(center,pt),ver_len)<max_ver_ratio:
            if (pt[0], pt[1]) not in POINTS:
                nrow, ncol, nhor_angle, nver_angle, nhor_len, nver_len = update_information(center, pt, row, col, 
                                        hor_angle, ver_angle, hor_len, ver_len, coordinate, dir="vertical")
                register(pt, nrow, ncol)
                search_surround(points, pt, nrow, ncol, nhor_angle, nver_angle, nhor_len, nver_len, coordinate) 
    return             
            

def index_single(img, coordinate=4, max_angle_shift=10, max_hor_ratio=1.5, max_ver_ratio=1.5):
    img_points, img_cross = extract_image(img)
    center, hor_angle, ver_angle = get_cross(img_cross)
    
    img_cross_mask = dye_cross_image(img_cross)
    cur_points = img_points.copy()
    cur_points[img_cross_mask!=3] = 0
    
    points = []
    label_mask = label(cur_points, connectivity = 2)
    properties = regionprops(label_mask)
    for prop in properties:
        points.append((prop.centroid[1], prop.centroid[0]))
    
    # 1. Get the closest point as the anchor
    points = sort_points(center, points)
    anchor = points[0]
    
    # 2. Initialize the point grid size
    hor_index, ver_index, hor_anchor, ver_anchor = 0, 0, None, None
    points = sort_points(anchor, points[1:])
    for i, pt in enumerate(points):
        pt_hor_angle = get_image_angle(anchor, pt)  
        pt_ver_angle = 180 + pt_hor_angle # get_image_angle(anchor, pt) 
        if abs(pt_hor_angle-hor_angle) < max_angle_shift and hor_anchor is None:
            hor_anchor = pt
            hor_index = i
        elif abs(pt_ver_angle-ver_angle) < max_angle_shift and ver_anchor is None:
            ver_anchor = pt
            ver_index = i
        if hor_anchor is not None and ver_anchor is not None: break
    
    hor_len = get_hor_len(hor_anchor, anchor)
    ver_len = get_ver_len(ver_anchor, anchor)
    hor_angle = get_image_angle(anchor, hor_anchor) # Update the horizontal angle
    ver_angle = get_image_angle(ver_anchor, anchor) # Update the vertical angle
    
    # 3. Register the anchor points
    register(anchor, row=0, col=0)
    register(hor_anchor, row=0, col=1)
    register(ver_anchor, row=1, col=0)
    print(hor_angle, ver_angle)
    
    # 4. Seaching for the surroundings
    pts_for_hor = points
    del pts_for_hor[hor_index]
    pts_for_ver = points.copy()
    del pts_for_ver[ver_index]
    search_surround(points, hor_anchor, 0, 1, hor_angle, ver_angle, hor_len, ver_len, 
                    coordinate, max_angle_shift, max_hor_ratio, max_ver_ratio)
    search_surround(points, ver_anchor, 1, 0, hor_angle, ver_angle, hor_len, ver_len, 
                    coordinate, max_angle_shift, max_hor_ratio, max_ver_ratio)
        
    # TODO: Determine the case when there is no hor-anchor or ver_anchor ...
    # print(anchor, hor_anchor, ver_anchor)
    # plt.imshow(cur_points, cmap="gray"), plt.show()
    
   
if __name__ == "__main__":
    img_file = "sample1.png"
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    index_single(img)
    for elem in POINTS:
        print(elem)
        print(POINTS[elem])
        print()
    
    


