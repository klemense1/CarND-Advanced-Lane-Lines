#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 10:58:21 2017

@author: Klemens
"""

import numpy as np
import cv2
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

plt.close("all")

def undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

    
def perspective_transform(image):
    """
    
    """

    img_size = (image.shape[1], image.shape[0])

    source_pts = np.float32(
                     [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
                     [((img_size[0] / 6) - 10), img_size[1]],
                     [(img_size[0] * 5 / 6) + 60, img_size[1]],
                     [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]]
                     )
    dest_pts = np.float32(
                     [[(img_size[0] / 4), 0],
                     [(img_size[0] / 4), img_size[1]],
                     [(img_size[0] * 3 / 4), img_size[1]],
                     [(img_size[0] * 3 / 4), 0]]
                     )
 
    M = cv2.getPerspectiveTransform(source_pts, dest_pts)

    topdown_image = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)
    
    return topdown_image, source_pts, dest_pts


if __name__ == "__main__":

    fname = 'test_images/straight_lines2.jpg'
    img = mpimg.imread(fname)
    ### Camera calibration
    
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
    ### Distortion correction
    undist = undistort(img, mtx, dist)

    ### Color/gradient threshold

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(S)
    ax2.set_title('S Channel', fontsize=30)
    ax3.imshow(binary, 'gray')
    ax3.set_title('Binary Channel', fontsize=30)

    ### Perspective transform

    warped, src, dst = perspective_transform(undist)
    ### Detect lane lines
    
    ### Determine the lane curvature
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    for pt in src:
        ax1.plot(pt[0], pt[1], '.')
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(warped)
    ax2.set_title('Perspective Transformed Image', fontsize=30)
    for pt in dst:
        ax2.plot(pt[0], pt[1], '.')