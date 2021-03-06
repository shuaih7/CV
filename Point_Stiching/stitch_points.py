# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import math
import time
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

from utils import *
from feature_extract import *


POINTS = {}


def display_index(img, color=(0,0,255), size=0.6, thickness=2):
    if img.shape[-1] > 4: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for pos in POINTS:
        elem = POINTS[pos]
        point = (int(pos[0]), int(pos[1]))
        index = elem["index"]
        # img = cv2.circle(img, point, radius, color, thickness)
        img = cv2.putText(img, str(index), point, font, size, color, thickness)
    plt.imshow(img), plt.title("Image Indexing Result")
    plt.show()
    
    return img


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
    
    
def get_len(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2-x1)**2+(y2-y1)**2)
    
    
def register(point, 
             row, 
             col, 
             previous=None,
             hor_angle=0,
             ver_angle=0, 
             hor_len=0, 
             ver_len=0):
             
    point_elem = {
        "from": previous,
        "index": (row, col),
        "hor_angle": hor_angle,
        "ver_angle": ver_angle,
        "hor_len": hor_len,
        "ver_len":ver_len
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
        
        if x1 > x0: col += 1
        elif x1 < x0: col -= 1
            
    elif dir == "vertical":
        ver_len = get_ver_len(center, point)
        ver_angle = get_image_angle(center, point)
        
        if y1 > y0: row -= 1
        elif y1 < y0: row += 1
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
        
        ## Checking box
        """
        if int(x1) == 328 and int(y1) == 268: 
            print("from:", POINTS[(center[0],center[1])]["index"])
            print("angle =", cur_angle, "hor_angle =", hor_angle, "ver_angle =", ver_angle)
            print()
        """
        
        if is_same_dir(hor_angle, cur_angle, max_angle_shift) \
                and max(get_hor_len(center,pt),hor_len)/min(get_hor_len(center,pt),hor_len)<max_hor_ratio:
            
            if (pt[0], pt[1]) not in POINTS:
                nrow, ncol, nhor_angle, nver_angle, nhor_len, nver_len = update_information(center, pt, row, col, 
                                        hor_angle, ver_angle, hor_len, ver_len, dir="horizontal")
                register(pt, nrow, ncol, (row, col), hor_angle, ver_angle)
                search_surround(points, pt, nrow, ncol, nhor_angle, nver_angle, nhor_len, nver_len)
                
        elif is_same_dir(ver_angle, cur_angle, max_angle_shift) \
                and max(get_ver_len(center,pt),ver_len)/min(get_ver_len(center,pt),ver_len)<max_ver_ratio:
                
            if (pt[0], pt[1]) not in POINTS:
                nrow, ncol, nhor_angle, nver_angle, nhor_len, nver_len = update_information(center, pt, row, col, 
                                        hor_angle, ver_angle, hor_len, ver_len, dir="vertical")
                register(pt, nrow, ncol, (row, col), hor_angle, ver_angle)
                search_surround(points, pt, nrow, ncol, nhor_angle, nver_angle, nhor_len, nver_len) 
    return             
            

def index_coordinate(points, 
                     center, 
                     hor_angle,
                     ver_angle,
                     max_angle_shift=10, 
                     max_hor_ratio=1.5, 
                     max_ver_ratio=1.5,
                     start_factor=1.2):
    
    # 1. Get the closest point as the anchor
    points = sort_points(center, points)
    if len(points) < 4: 
        raise PointsNotEnoughError("There are not enough points for the alogorithm to define the locality.")
    
    anchor = points[0]
    if anchor[0] < center[0]: col = 0
    else: col = 1
    if anchor[1] < center[1]: row = 1
    else: row = 0
    register(anchor, row=row, col=col)
    
    # 2. Register the horizontal and vertical anchors
    points = points[1:]
    hor_len, ver_len = None, None
    is_hor_registered = False
    is_ver_registered = False
    
    for pt in points[:3]:
        cur_angle = get_image_angle(anchor, pt)
        
        if is_same_dir(hor_angle, cur_angle, max_angle_shift):
            if not is_hor_registered:
                hor_len = get_hor_len(anchor, pt)
                is_hor_registered = True
            
        elif is_same_dir(ver_angle, cur_angle, max_angle_shift):
            if not is_ver_registered:
                ver_len = get_ver_len(anchor, pt)
                is_ver_registered = True
    
    # 3. Start indexing the image
    if hor_len is None or ver_len is None:
        raise AnchorNotFoundError("Could not find the anchors for locality.")
    search_surround(points, anchor, row, col, hor_angle, ver_angle, hor_len, ver_len, 
                    max_angle_shift, max_hor_ratio, max_ver_ratio)
        
    # print(anchor, hor_anchor, ver_anchor)
    # plt.imshow(cur_points, cmap="gray"), plt.show()
    

def index_image(image, 
                pt_thresh=30,
                pt_min_d=20,
                pt_max_d=65,
                pt_aspect_ratio=1.5,
                cr_thresh=30,
                max_angle_shift=15, 
                max_hor_ratio=1.25, 
                max_ver_ratio=1.25,
                start_factor=1.2):
                
    #img_points, points = extract_points(img, thresh, min_rad, max_rad, aspect_ratio)
    points, center, hor_angle, ver_angle = extract_feature(image, 
        pt_thresh, pt_min_d, pt_max_d, pt_aspect_ratio, cr_thresh)
    
    index_coordinate(points, center, hor_angle, ver_angle, max_angle_shift, 
                        max_hor_ratio, max_ver_ratio, start_factor)
    
    # plt.subplot(1,2,1), plt.imshow(img_points, cmap="gray"), plt.title("Point Image")
    # plt.subplot(1,2,2), plt.imshow(img_cross, cmap="gray"), plt.title("Cross Image")
    # for elem in POINTS:
        # print(elem)
        # print(POINTS[elem])
        # print()
    # plt.imshow(img_points, cmap="gray"), plt.title("Points Image"), plt.show()


if __name__ == "__main__":
    img_dir = r"E:\Projects\Integrated_Camera\bsi_proj\diff"
    img_list = gb.glob(img_dir + r"/*.png")
    
    for img_file in img_list:
        print("Processing", img_file)
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        
        start = time.time()
        try:
            index_image(img)
        except: pass
        #period = time.time() - start
        #print("The running time is %s.", period)
        
        img_draw = display_index(img, size=0.25, color=(0,0,255), thickness=1)
        POINTS = {}
        """
        for index in POINTS:
            if POINTS[index]["index"] == (1,0):
                print(index, POINTS[index])
        """
    
    


