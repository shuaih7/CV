# -*- coding: UTF-8 -*-

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops


def extract_curve(img_file, thresh=200):
    # Load the grayscale image and binarize
    grayimage = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    image = cv2.GaussianBlur(grayimage,(5,5),0)
    _, img_bn = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    
    # Erosion to cut off the line from the background
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img_bn, kernel, iterations=1)
    
    # Create the region proposal mask and filter out the curve
    label_mask = label(erosion, connectivity = 2)
    properties = regionprops(label_mask)
    
    for prop in properties:
        if prop.MajorAxisLength
    
    plt.subplot(1,3,1), plt.imshow(grayimage, cmap="gray"), plt.title("Gray Scale Image")
    plt.subplot(1,3,2), plt.imshow(image, cmap="gray"), plt.title("Blurred Gray Scale Image")
    plt.subplot(1,3,3), plt.imshow(erosion, cmap="gray"), plt.title("Binarized Image")
    plt.show()
    
    
if __name__ == "__main__":
    img_file_1 = r"E:\Projects\Integrated_Camera\laser_pattern\2020-12-10_142752_0.png"
    img_file_2 = r"E:\Projects\Integrated_Camera\laser_pattern\2020-12-10_143000_0.png"
    
    extract_curve(img_file_1)
