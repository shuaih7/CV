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


def extract_image(img, thresh=220, min_rad=15, max_rad=80):
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
             coordinate,
             previous=None,
             hor_angle=0,
             ver_angle=0, 
             hor_len=0, 
             ver_len=0):
             
    point_elem = {
        "row": row,
        "col": col,
        "coordinate": coordinate,
        "from": previous
    }
        # "hor_angle": hor_angle,
        # "ver_angle": ver_angle,
        # "hor_len": hor_len,
        # "ver_len":ver_len
    # }
    
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
        hor_angle = get_image_angle(center, point)
        
        if x1 > x0: 
            if coordinate==1 or coordinate==4: col += 1
            else: col -= 1
        elif x1 < x0:
            if coordinate==1 or coordinate==4: col -= 1
            else: col += 1
            
    elif dir == "vertical":
        ver_len = get_ver_len(center, point)
        ver_angle = get_image_angle(center, point)
        
        if y1 > y0:
            if coordinate==3 or coordinate==4: row += 1
            else: row -= 1
        elif y1 < y0:
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
                    
    if len(input_points) == 0 or center is None: return
    points = input_points.copy()
    points = sort_points(center, points)
    
    x0, y0 = center
    for i, pt in enumerate(points[:8]):
        x1, y1 = pt
        
        # Filter out the identical points
        if abs(x1-x0)<0.0001 and abs(y1-y0)<0.0001:
            continue
            
        # Calculate the angle shift
        cur_angle = get_image_angle(center, pt)
        hor_angle_shift = abs(cur_angle - hor_angle)
        ver_angle_shift = abs(cur_angle - ver_angle)
            
        if hor_angle_shift < max_angle_shift \
                and max(get_hor_len(center,pt),hor_len)/min(get_hor_len(center,pt),hor_len)<max_hor_ratio:
            if (pt[0], pt[1]) not in POINTS:
                nrow, ncol, nhor_angle, nver_angle, nhor_len, nver_len = update_information(center, pt, row, col, 
                                        hor_angle, ver_angle, hor_len, ver_len, coordinate, dir="horizontal")
                register(pt, nrow, ncol, coordinate, (row, col))
                search_surround(points, pt, nrow, ncol, nhor_angle, nver_angle, nhor_len, nver_len, coordinate)
                
        elif ver_angle_shift < max_angle_shift \
                and max(get_ver_len(center,pt),ver_len)/min(get_ver_len(center,pt),ver_len)<max_ver_ratio:
            if (pt[0], pt[1]) not in POINTS:
                nrow, ncol, nhor_angle, nver_angle, nhor_len, nver_len = update_information(center, pt, row, col, 
                                        hor_angle, ver_angle, hor_len, ver_len, coordinate, dir="vertical")
                register(pt, nrow, ncol, coordinate, (row, col))
                search_surround(points, pt, nrow, ncol, nhor_angle, nver_angle, nhor_len, nver_len, coordinate) 
    return             
            

def index_coordinate(img_points, 
                 center, 
                 hor_angle,
                 ver_angle,
                 img_cross_mask, 
                 coordinate, 
                 max_angle_shift=10, 
                 max_hor_ratio=1.5, 
                 max_ver_ratio=1.5,
                 start_factor=1.2):
    
    cur_points = img_points.copy()
    cur_points[img_cross_mask!=coordinate] = 0
    
    points = []
    label_mask = label(cur_points, connectivity = 2)
    properties = regionprops(label_mask)
    for prop in properties:
        points.append((prop.centroid[1], prop.centroid[0]))
    
    # 1. Get the closest point as the anchor
    points = sort_points(center, points)
    if len(points) == 0: return # Case when there is no point inside this coordinate
    
    anchor = points[0]
    register(anchor, row=0, col=0, coordinate=coordinate)
    
    # 2. Initialize the point grid size
    hor_anchor, ver_anchor = None, None
    
    if len(points) == 1: return # Case when there is only one point inside this coordinate
    points = sort_points(anchor, points[1:])
    
    for i, pt in enumerate(points):
        pt_angle = get_image_angle(anchor, pt)
         
        if abs(pt_angle-hor_angle) < max_angle_shift*start_factor and hor_anchor is None:
            hor_anchor = pt
            hor_angle = pt_angle # Update the horizontal angle
            hor_len = get_hor_len(hor_anchor, anchor)
            register(hor_anchor, row=0, col=1, coordinate=coordinate, previous=(0,0))
            
        elif abs(pt_angle-ver_angle) < max_angle_shift*start_factor and ver_anchor is None:
            ver_anchor = pt
            ver_angle = pt_angle # Update the vertical angle
            ver_len = get_ver_len(ver_anchor, anchor)
            register(ver_anchor, row=1, col=0, coordinate=coordinate, previous=(0,0))
            
        if hor_anchor is not None and ver_anchor is not None: break
    
    if hor_anchor is None and ver_anchor is None: return # Case when the horizontal and vertical anchors cannot be found

    # Case when there only exits one anchor point
    if hor_anchor is None: hor_len = ver_len
    if ver_anchor is None: ver_len = hor_len
        
    pts_for_hor = points
    pts_for_ver = points.copy()
    
    search_surround(points, hor_anchor, 0, 1, hor_angle, ver_angle, hor_len, ver_len, 
                coordinate, max_angle_shift, max_hor_ratio, max_ver_ratio)
    search_surround(points, ver_anchor, 1, 0, hor_angle, ver_angle, hor_len, ver_len, 
                    coordinate, max_angle_shift, max_hor_ratio, max_ver_ratio)
        
    # print(anchor, hor_anchor, ver_anchor)
    # plt.imshow(cur_points, cmap="gray"), plt.show()
    

def index_image(img, 
                max_angle_shift=20, 
                max_hor_ratio=1.5, 
                max_ver_ratio=1.5,
                start_factor=1.2):
                
    img_points, img_cross = extract_image(img)
    center, hor_angle, ver_angle, img_cross_mask = get_cross(img_cross)
    
    if center is None: 
        raise Exception("Could not find the cross inside the input image.")
    elif abs(ver_angle-hor_angle) < max_angle_shift*start_factor:
        raise Exception("The horizontal and vertical angles are too close")
        
    for coordinate in range(1,5):
        index_coordinate(img_points, center, hor_angle, ver_angle, img_cross_mask, 
                coordinate, max_angle_shift, max_hor_ratio, max_ver_ratio, start_factor)
    
    # plt.subplot(1,2,1), plt.imshow(img_points, cmap="gray"), plt.title("Point Image")
    # plt.subplot(1,2,2), plt.imshow(img_cross, cmap="gray"), plt.title("Cross Image")
    for elem in POINTS:
        print(elem)
        print(POINTS[elem])
        print()
    plt.imshow(img_points, cmap="gray"), plt.title("Points Image"), plt.show()


if __name__ == "__main__":
    img_file = "./test/test.png"
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    index_image(img)
    
    
    


