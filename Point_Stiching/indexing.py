# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

from utils import *


NUM_DIR = 8

POINT_LIST = []

POINT_ELEM = {
    "angle": -1,
    "points": []
}


def draw_image(img, point, label="", radius=5, color=(0,0,255), thickness=4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    point = (int(point[0]), int(point[1]))
    img = cv2.circle(img, point, radius, color, thickness)
    img = cv2.putText(img, str(label), point, font, 1.2, color, thickness)
    
    plt.imshow(img, cmap="gray"), plt.title("Image Indexing Result")
    plt.show()
    
    return img
    
    
def display_indexing(img, point_list=None, radius=5, color=(0,0,255), thickness=4):
    if point_list is None: point_list = POINT_LIST
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, elem in enumerate(point_list):
        points = elem["points"]
        for pt in points:
            pt = (int(pt[0]), int(pt[1]))
            img = cv2.circle(img, pt, radius, color, thickness)
            img = cv2.putText(img, str(i), pt, font, 1.2, color, thickness)
    plt.imshow(img, cmap="gray"), plt.title("Image Indexing Result")
    plt.show()
    
    return img
    
    
def get_main_angles(center, points, top=12):
    points = points[:top]
    hor_angle, ver_angle = 0.0, 90.0
    hor_cosin, ver_cosin = 0.0, 1.0 # Select larger hor_cosin and smaller ver_cosin
    
    for point in points:
        cur_angle = get_angle(center, point)
        cur_cosin = math.cos(cur_angle)
        
        if cur_cosin > hor_cosin: 
            hor_cosin = cur_cosin
            hor_angle = cur_angle
            hor_pt = point
            
        if cur_cosin < ver_cosin:
            ver_cosin = cur_cosin
            ver_angle = cur_angle
            ver_pt = point
    
    return hor_angle, ver_angle, hor_pt, ver_pt


def extract_points(img, thresh=216, min_rad=20, max_rad=80):
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
    
    
def drop_point(center, point, cutoff_angle=8):
    angle = get_angle(center, point)
    
    for elem in POINT_LIST:
        if abs(elem["angle"]-angle) <= cutoff_angle:
            elem["angle"] = angle # Update the angle value
            elem["points"].append(point)
            return len(POINT_LIST)
            
    if len(POINT_LIST) < NUM_DIR:
        point_elem = POINT_ELEM.copy()
        point_elem["angle"] = angle
        point_elem["points"] = []
        point_elem["points"].append(point)
        POINT_LIST.append(point_elem)
        return len(POINT_LIST)
    else: 
        return 0
    

def indexing(img_file, 
             center, 
             mark_top=16):
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    points = extract_points(img)
    points = sort_points(center, points) # Sort the points based on the distances to the center
    points = points[1:] # Filter the closest point to the center
    
    # hor_angle, ver_angle, hor_pt, ver_pt = get_main_angles(center, points) 
    # print(hor_angle, ver_angle)
    # img = draw_image(img, hor_pt, label="hor")
    # img = draw_image(img, ver_pt, label="ver")
    
    
class ImageIndex(object):
    def __init__(self):
        self.img_origin_bn = None
        self.img_points_bn = None
        self.img_cross_bn = None
        self.points = None
        self.center = None
        self.hor_angle = None
        self.ver_angle = None
        
    def get_index(self, img_file):
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        self.extract_coord(img)
        self._test()
        
    def extract_coord(self, img):
        self._extract_points(img)
        self._extract_cross()
        
        self.center = (0,0)
        self.hor_angle = 0.0
        self.ver_angle = 90.0
        
    def _extract_points(self, img, thresh=221, min_rad=20, max_rad=80):
        img = cv2.GaussianBlur(img,(5,5),0)
        _, img_bn = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        self.img_origin_bn = img_bn.copy()
        
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
        self.img_points_bn = img_bn
        self.points = points
        
    def _extract_cross(self):
        if self.img_origin_bn is None or self.img_points_bn is None: return 
        img_cross_bn = self.img_origin_bn.copy()
        img_cross_bn[self.img_points_bn>0] = 0
        self.img_cross_bn = img_cross_bn
        
    def _test(self):
        plt.subplot(1,3,1), plt.imshow(self.img_origin_bn, cmap="gray"), plt.title("Binary Image")
        plt.subplot(1,3,2), plt.imshow(self.img_points_bn, cmap="gray"), plt.title("Binary Points Image")
        plt.subplot(1,3,3), plt.imshow(self.img_cross_bn, cmap="gray"), plt.title("Binary Cross Image")
        plt.show()
        
    
if __name__ == "__main__":
    img_file = "sample.png"
    
    imageIndex = ImageIndex()
    _, _, _ = imageIndex.get_index(img_file)
   
