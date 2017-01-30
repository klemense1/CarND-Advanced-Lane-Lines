#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 21:31:32 2017

@author: Klemens
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
cv2.line(img,(0,0),(511,511),(255,0,0),5)
plt.figure()
plt.imshow(img)