#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 11:26:20 2017

@author: Klemens
"""

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

nx = 9#TODO: enter the number of inside corners in x
ny = 6#TODO: enter the number of inside corners in y

# map coordinates of corners of 2d image (imgpoints) to 3d coordinates of real undistorted chessboard corners (objpoints)
objpoints = [] # 3d points in real word space
imgpoints = [] # 2d points in image plane

# (1,0,0), (2,0,0) ... (7,5,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

images = glob.glob('camera_cal/calibration*.jpg')
# Make a list of calibration images
for fname in images:
#fname = 'camera_cal/calibration1.jpg'
    img = cv2.imread(fname)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret==True:
        imgpoints.append(corners)
        objpoints.append(objp)
    else:
        print('No corners found in', fname)

# Test undistortion on an image
test_file_name = 'camera_cal/calibration1.jpg'
img = cv2.imread(test_file_name)
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


dst = cv2.undistort(img, mtx, dist, None, mtx)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image: ' + test_file_name.split('/')[-1], fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)    

plt.savefig('output_images/undistort_output.jpg')