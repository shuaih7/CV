# -*- coding: utf-8 -*-

import os, sys, cv2
import numpy as np
from skimage.morphology import reconstruction

"""
# Part 1: Reginal maximal - dilation
# Intensity of seed image must be less than that of the mask image for reconstruction by dilation.

mask = np.array([[10,10,10,10,10,10,10,10,10,10],
                 [10,14,14,14,10,10,11,10,11,10],
                 [10,14,14,14,10,10,10,11,10,10],
                 [10,14,14,14,10,10,11,10,11,10],
                 [10,10,10,10,10,10,10,10,10,10],
                 [10,11,10,10,10,18,18,18,10,10],
                 [10,10,10,11,10,18,18,18,10,10],
                 [10,10,11,10,10,18,18,18,10,10],
                 [10,11,10,11,10,10,10,10,10,10],
                 [10,10,10,10,10,10,11,10,10,10]], dtype=np.uint8)
                 
marker = np.array([[0,0,0,0,0,1,0,0,0,0],
                   [0,0,0,0,0,0,0,0,11,0],
                   [0,0,13,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0,16,0,0],
                   [0,0,0,0,0,0,0,1,0,0],
                   [0,0,0,0,0,0,0,0,0,0]], dtype=np.uint8)
                   
                   print("The original marker image:")
print(marker)
print()

output = reconstruction(marker, mask, method='dilation')
print("The morphology reconstructed output:")
print(output)
print()
"""

# Part 2: Reginal minimal - erosion
# Intensity of seed image must be greater than that of the mask image for reconstruction by erosion.

mask = np.array([[10,10,10,10,10,10,10,10,10,10],
                 [10,7,7,7,10,10,11,10,11,10],
                 [10,7,7,7,10,10,10,11,10,10],
                 [10,7,7,7,10,10,11,10,11,10],
                 [10,10,10,10,10,10,10,10,10,10],
                 [10,11,10,10,10,2,2,2,10,10],
                 [10,10,10,11,10,2,2,2,10,10],
                 [10,10,11,10,10,2,2,2,10,10],
                 [10,11,10,11,10,10,10,10,10,10],
                 [10,10,10,10,10,10,11,10,10,10]], dtype=np.uint8)
                 
marker = np.array([[20,20,20,20,20,21,20,20,20,20],
                   [20,20,20,20,20,20,20,20,11,20],
                   [20,20,13,20,20,20,20,20,20,20],
                   [20,20,20,20,20,20,20,20,20,20],
                   [20,20,20,20,20,20,20,20,20,20],
                   [20,20,20,20,20,20,20,22,20,20],
                   [20,20,20,20,20,20,9,9,20,20],
                   [20,20,20,20,15,12,9,9,20,20],
                   [20,20,20,20,20,20,20,21,20,20],
                   [20,20,20,20,20,20,20,20,20,20]], dtype=np.uint8)

print("The original marker image:")
print(marker)
print()

output = reconstruction(marker, mask, method='erosion')
print("The morphology reconstructed output:")
print(output)
print()