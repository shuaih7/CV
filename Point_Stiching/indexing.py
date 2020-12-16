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
    """Create indeices for each highlighted points on the image

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """
    def __init__(self):
        self.img_origin_bn = None
        self.img_points_bn = None
        self.img_cross_bn = None
        self.points = None
        self.center = None
        self.hor_angle = None
        self.ver_angle = None
        
    def get_indices(self, img_file):
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        self.extract_coord(img)
        #self._test()
        
    def extract_coord(self, img):
        self._extract_features(img)
        self._parse_cross(img)
        
        self.center = (0,0)
        self.hor_angle = 0.0
        self.ver_angle = 90.0
        
    def _extract_features(self, img, thresh=221, min_rad=20, max_rad=80):
        """Extract crucial features from the input image

        Binarize the input image and extract the points and cross features.
        It is assumed that the points and cross are the only over-exposure region on the image.

        Args:
            img: Input numpy array of the grayscale image
            thresh: Thesholding value for the image binarization
            min_rad: Minimum radius of the points
            max_rad: Maximum radius of the points
            
        """
        img = cv2.GaussianBlur(img,(5,5),0)
        _, img_bn = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        self.img_origin_bn = img_bn.copy()
        self.img_cross_bn = np.zeros(img_bn.shape, dtype=np.uint8)
        
        # Create the region proposal mask and filter out the curve
        label_mask = label(img_bn, connectivity = 2)
        properties = regionprops(label_mask)
        
        points = []
        for prop in properties:
            diameter = prop.equivalent_diameter
            if diameter < min_rad:
                img_bn[label_mask==prop.label] = 0
            elif diameter > max_rad: 
                img_bn[label_mask==prop.label] = 0
                self.img_cross_bn[label_mask==prop.label] = 255
            else: 
                points.append([prop.centroid[1], prop.centroid[0]])

        self.img_points_bn = img_bn
        self.points = points
        
    def _parse_cross(self, img, thresh1=35, thresh2=135):
        if self.img_cross_bn is None: return
        # Method 1 - Using Hough Transform
        contours, hierarchy = cv2.findContours(self.img_cross_bn,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        edges = cv2.drawContours(np.zeros(img.shape, dtype=np.uint8),contours,-1,255,3)  
        # edges = cv2.Canny(self.img_cross_bn, 35, 135)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 220)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            print((x0, y0), rho)
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            img = cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
        plt.subplot(1,2,1), plt.imshow(edges, cmap="gray"), plt.title("Cross Edges")
        plt.subplot(1,2,2), plt.imshow(img, cmap="gray"), plt.title("Lines")
        plt.show()

        
    def _get_hor_cross_center(self, step=5, stride=1):
        if self.img_cross_bn is None: return
        else: 
            start_count = False
            pos = 0
        
        while pos < self.img_origin_bn.shape[1]:
            
            pos += step
            
    def _count_blocks(self, array, stride=1) -> int:
        pos, num_blocks, val = stride, 0, array[stride]
        if val > 0: num_blocks += 1
        
        while pos < array.shape[0]:
            if array[pos] == 0 and array[pos-stride] > 0:
                num_blocks += 1
            pos += stride
            
        return num_blocks
  
    def _test(self):
        array = np.array([0,0,1,1,1,1,1,0,0,0,0,1,1,1,0,0,1,1,1,1,1])
        print(self._count_blocks(array, stride=2))
        """
        # Show the binarized images
        plt.subplot(1,3,1), plt.imshow(self.img_origin_bn, cmap="gray"), plt.title("Binary Image")
        plt.subplot(1,3,2), plt.imshow(self.img_points_bn, cmap="gray"), plt.title("Binary Points Image")
        plt.subplot(1,3,3), plt.imshow(self.img_cross_bn, cmap="gray"), plt.title("Binary Cross Image")
        plt.show()
        """
        
    
if __name__ == "__main__":
    img_file = "sample2.png"
    
    imageIndex = ImageIndex()
    imageIndex.get_indices(img_file)
   
