#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:43:10 2017

@author: zhao
"""

import numpy as np
import cv2, os, sys
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage import morphology

im = cv2.imread('sample2.png', cv2.IMREAD_GRAYSCALE)
img_orig  = im.copy()

im = cv2.GaussianBlur(im,(5,5),0)

thres,im = cv2.threshold(im,221,255,cv2.THRESH_BINARY)

# Create the region proposal mask and filter out the curve
label_mask = label(im, connectivity = 2)
properties = regionprops(label_mask)

max_rad = 80
for prop in properties:
    diameter = prop.equivalent_diameter
    if diameter < max_rad:
        im[label_mask==prop.label] = 0

contours, hierarchy = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#print(np.array(img).shape)
#print(np.array(contours).shape)

# contours, hierarchy = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
# im = cv2.drawContours(np.zeros(im.shape, dtype=np.uint8),contours,-1,255,3)  

back = np.zeros(im.shape, dtype=np.uint8)
edges = morphology.skeletonize(im/255)
back[edges] = 255
edges = back.copy()
lines = cv2.HoughLines(edges, 1, np.pi/180, 57)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    img = cv2.line(back,(x1,y1),(x2,y2),255,1)
    print((x0,y0), "rho=", rho, "theta=", theta)
plt.subplot(1,2,1), plt.imshow(edges, cmap="gray"), plt.title("Edges")
plt.subplot(1,2,2), plt.imshow(img, cmap="gray"), plt.title("Lines")
plt.show()



