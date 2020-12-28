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


def extract_points(img, thresh=220, min_rad=15, max_rad=80, aspect_ratio=1.8):
    """Binarize the input image and extract the points
    
    Args:
        img: Input grayscale image
        thresh: Thesholding value for the image binarization
        min_rad: Minimum point radius
        max_rad: Maximum point radius
        
    Returns:
        img_points: Binarized image contains the points
        points: List of centroid points
        
    """
    img = cv2.GaussianBlur(img,(5,5),0)
    thres,img_points = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)

    # Create the region proposal mask and filter out the curve
    label_mask = label(img_points, connectivity = 2)
    properties = regionprops(label_mask)

    points = []
    for prop in properties:
        diameter = prop.equivalent_diameter
        major_len = prop.major_axis_length
        minor_len = max(1, prop.minor_axis_length)
        if diameter < min_rad or diameter > max_rad or major_len/minor_len > aspect_ratio:
            img_points[label_mask==prop.label] = 0
        else: points.append((prop.centroid[1], prop.centroid[0]))
    
    '''
    plt.subplot(1,2,1), plt.imshow(img_points, cmap="gray"), plt.title("Point Image")
    plt.subplot(1,2,2), plt.imshow(img_cross, cmap="gray"), plt.title("Cross Image")
    plt.show()
    '''
    return img_points, points
    
    
def check_binary(img_bn, points):
    for point in points:
        if point is not None:
            val = img_bn[point[1], point[0]]
            if val == 0: raise ValueError("Could not find the specified anchor point.")


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
                     hor_anchor=None,
                     ver_anchor=None,
                     max_angle_shift=10, 
                     max_hor_ratio=1.5, 
                     max_ver_ratio=1.5,
                     start_factor=1.2):
    
    # 1. Get the closest point as the anchor
    points = sort_points(center, points)
    if len(points) == 0: return # Case when there is no point inside this coordinate
    
    anchor = points[0]
    register(anchor, row=0, col=0)
    
    # 2. Register the horizontal and vertical anchors
    if len(points) == 1: return # Case when there is only one point inside this coordinate
    points = points[1:]
    
    if hor_anchor is None:
        print("Warning: The horizontal anchor point has not been set, this may cause fatal stiching error to the final result!")
        
        hor_index = -1
        for i, pt in enumerate(points[:8]):
            pt_angle = get_image_angle(anchor, pt)
            
            if abs(pt_angle) < max_angle_shift*start_factor:
                hor_angle = pt_angle
                hor_anchor = pt
                hor_len = get_hor_len(anchor, hor_anchor)
                hor_index = i
                break
                
        if hor_index >= 0: 
            del points[hor_index]
        else:
            raise Exception("Could not find the horizontal anchor automatically.")
        
    else:
        hor_index = 0
        local_dist = 100000
        for i, pt in enumerate(points[:8]):
            # Find the centroid closest to the horizontal anchor point, and update the horizontal anchor
            if get_hor_len(hor_anchor, pt) < local_dist:
                local_dist = get_hor_len(hor_anchor, pt)
                hor_index = i
    
        hor_anchor = points[hor_index]
        hor_angle = get_image_angle(anchor, hor_anchor)
        hor_len = get_hor_len(anchor, hor_anchor)
        del points[hor_index]
    
    # Register the horizontal anchor
    if hor_anchor[0] > anchor[0]:
        hor_anchor_col = 1
    else:
        hor_anchor_col = -1
    register(hor_anchor, row=0, col=hor_anchor_col, previous=(0,0))
        
    if ver_anchor is None:
        print("Warning: The vertical anchor point has not been set, this may cause fatal stiching error to the final result!")
    
        ver_index = -1
        for i, pt in enumerate(points[:8]):
            pt_angle = get_image_angle(anchor, pt)
            
            min_ver_angle = 90.0 - max_angle_shift*start_factor
            max_ver_angle = 90.0 + max_angle_shift*start_factor
            if abs(pt_angle) > 90.0 - max_angle_shift*start_factor:
                ver_angle = pt_angle
                ver_anchor = pt
                ver_len = get_ver_len(anchor, ver_anchor)
                ver_index = i
                break
                
        if ver_index >= 0: 
            del points[ver_index]
        else: 
            raise Exception("Could not find the vertical anchor automatically.")
            
    else:
        ver_index = 0
        local_dist = 100000
        for i, pt in enumerate(points[:8]):
            # Find the centroid closest to the vertical anchor point, and update the vertical anchor
            if get_ver_len(ver_anchor, pt) < local_dist:
                local_dist = get_ver_len(ver_anchor, pt)
                ver_index = i
    
        ver_anchor = points[ver_index]
        ver_angle = get_image_angle(anchor, ver_anchor)
        ver_len = get_ver_len(anchor, ver_anchor)
        del points[ver_index]
        
    # Register the vertical anchor
    if ver_anchor[1] > anchor[1]:
        ver_anchor_row = -1
    else:
        ver_anchor_row = 1
    register(ver_anchor, row=ver_anchor_row, col=0, previous=(0,0))
    
    if hor_anchor is None and ver_anchor is None: return # Case when the horizontal and vertical anchors cannot be found
        
    pts_for_hor = points
    pts_for_ver = points.copy()
    
    search_surround(points, hor_anchor, 0, hor_anchor_col, hor_angle, ver_angle, hor_len, ver_len, 
                    max_angle_shift, max_hor_ratio, max_ver_ratio)
    search_surround(points, ver_anchor, ver_anchor_row, 0, hor_angle, ver_angle, hor_len, ver_len, 
                    max_angle_shift, max_hor_ratio, max_ver_ratio)
        
    # print(anchor, hor_anchor, ver_anchor)
    # plt.imshow(cur_points, cmap="gray"), plt.show()
    

def index_image(img, 
                center, 
                hor_anchor=None,
                ver_anchor=None,
                thresh=210,
                min_rad=15,
                max_rad=60,
                aspect_ratio=1.8,
                max_angle_shift=20, 
                max_hor_ratio=1.5, 
                max_ver_ratio=1.5,
                start_factor=1.2):
                
    img_points, points = extract_points(img, thresh, min_rad, max_rad, aspect_ratio)
    check_binary(img_points, [center, hor_anchor, ver_anchor])
    #plt.imshow(img_points, cmap="gray"), plt.show()
    #sys.exit()
    
    index_coordinate(points, center, hor_anchor, ver_anchor, max_angle_shift, 
                        max_hor_ratio, max_ver_ratio, start_factor)
    
    # plt.subplot(1,2,1), plt.imshow(img_points, cmap="gray"), plt.title("Point Image")
    # plt.subplot(1,2,2), plt.imshow(img_cross, cmap="gray"), plt.title("Cross Image")
    # for elem in POINTS:
        # print(elem)
        # print(POINTS[elem])
        # print()
    # plt.imshow(img_points, cmap="gray"), plt.title("Points Image"), plt.show()


if __name__ == "__main__":
    img_file = r"E:\Projects\Integrated_Camera\2020-12-25\2020-12-28_143741_0.png"
    center = (166,176)
    hor_anchor = (118,183)
    ver_anchor = (172,143)
    
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    
    start = time.time()
    
    index_image(img, center, hor_anchor, ver_anchor, thresh=210, min_rad=3, max_rad=30, aspect_ratio=3)
    period = time.time() - start
    print("The running time is %s.", period)
    
    display_index(img, size=0.25, color=(0,0,255), thickness=1)
    
    #for index in POINTS:
    #    if POINTS[index]["index"] == (-9,4):
    #        print(POINTS[index])
    
    
    


