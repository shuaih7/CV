# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

from utils import *
from cross import get_cross


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
    

def index_single(img):
    img_points, img_cross = extract_image(img)
    center, hor_angle, ver_angle = get_cross(img_cross)
    
    img_cross_mask = dye_cross_image(img_cross)
    cur_points = img_points.copy()
    cur_points[img_cross_mask!=3] = 0
    # plt.imshow(cur_points, cmap="gray"), plt.show()
    
    
    
   
if __name__ == "__main__":
    img_file = "sample1.png"
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    index_single(img)


