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

# map coordinates of corners of 2d image (imgpoints) to 3d coordinates of real undistorted chessboard corners (objpoints)
objpoints = [] # 3d points in real word space
imgpoints = [] # 2d points in image plane

for nx in [6, 7, 8, 9]:
    for ny in [5, 6]:

        # (1,0,0), (2,0,0) ... (7,5,0)
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
        
        images = glob.glob('camera_cal/calibration*.jpg')
        image_list_used = []
        # Make a list of calibration images
        for fname in images:
            img = cv2.imread(fname)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
            if ret==True:

                imgpoints.append(corners)
                objpoints.append(objp)
                image_list_used.append(fname)
            
                print('Found corners in', fname, 'with nx=', nx, ', ny=', ny)

                cv2.drawChessboardCorners(img, (nx,ny), corners, ret)

                image_name=fname.split('/')[-1]
                write_name = 'camera_cal/corners/corners_found_ny' + str(ny) + '_nx' + str(nx) + '_' + image_name

                cv2.imwrite(write_name, img)

if images == image_list_used:
    print('All Images used')
else:
    print('Not all images used')

for test_file_name in images:
# Test undistortion on an image
#test_file_name = 'camera_cal/calibration1.jpg'
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

    # Visualize undistortion
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image: ' + test_file_name.split('/')[-1], fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)    
    
    plt.savefig('camera_cal/undistored/undistort_output_' + test_file_name.split('/')[-1])