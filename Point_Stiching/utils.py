# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops


def sort_points(center, points) -> list:
    if len(points) == 0: return points
    sorted_points, dists = [], []
    
    for point in points:
        dists.append((point[0]-center[0])**2+(point[1]-center[1])**2)
    sorted_index = np.argsort(dists)
    
    points = np.array(points)[sorted_index] # Sort the point array
    
    return points.tolist()
    
    
def get_angle(center, point) -> float:
    """
    0 -> 360
    """
    x0, y0 = center[0], center[1]
    x1, y1 = point[0], point[1]
    if y0 == y1 and x1 >= x0: return 0.0
    
    length = math.sqrt((x1-x0)**2+(y1-y0)**2)
    
    cos_value = (x1-x0) / length
    value = math.acos(cos_value)/math.pi * 180
    
    if y1 > y0: return value
    else: return 360.0 - value
    
"""    
def get_image_angle(center, point) -> float:
    '''
    0 -> 180
    '''
    x0, y0 = center[0], center[1]
    x1, y1 = point[0], point[1]
    if y0 == y1: return 0.0
    
    y0, y1 = -y0, -y1
    length = math.sqrt((x1-x0)**2+(y1-y0)**2)
    
    cos_value = abs(x1-x0) / length
    value = math.acos(cos_value)/math.pi * 180
    
    return value
"""

def get_image_angle(center, point) -> float:
    return get_angle(center, point)
    

def is_same_dir(angle1, angle2, max_offset) -> bool:
    angle_offset = abs(angle1 - angle2)
    if angle_offset < max_offset:
        return True
    elif angle_offset > 180-max_offset and angle_offset < 180+max_offset:
        return True
    elif min(angle1, angle2) < max_offset and abs(360-max(angle1, angle2)-min(angle1, angle2)) < max_offset:
        return True
    else: 
        return False
        
    
def create_background(shape, save_name="background.png"):
    img = np.zeros(shape, dtype=np.uint8)
    cv2.imwrite(save_name, img)
    
    
if __name__ == "__main__":
    hor_angle = 130.048
    ver_angle = 242.009
    cur_angle = 238.47
    
    max_angle_shift = 10
    
    if is_same_dir(hor_angle, cur_angle, max_angle_shift): 
        print("Yes")
    else: print("No")
    