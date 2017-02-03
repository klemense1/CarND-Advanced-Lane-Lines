#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:05:46 2017

@author: Klemens
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

plt.close("all")

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny_edge_threshold(img, thresh = (50, 150), smoothing_kernel = 5):
    
    image_gray = grayscale(img)

    # gaussian smoothing
    image_blured = gaussian_blur(image_gray, smoothing_kernel)

    image_edges = canny(image_blured, thresh[0], thresh[1])

    canny_binary = np.zeros_like(image_edges)

    canny_binary[image_edges>0] = 1

    return canny_binary

def saturation_threshold(img, thresh = (90, 255)):
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]

    sat_binary = np.zeros_like(S)
    sat_binary[(S > thresh[0]) & (S <= thresh[1])] = 1
           
    return sat_binary

def lightness_threshold(img, thresh = (40, 255)):
    
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    L = hls[:,:,1]

    lht_binary = np.zeros_like(L)
    lht_binary[(L > thresh[0]) & (L <= thresh[1])] = 1
           
    return lht_binary

def red_threshold(img, thresh = (200, 255)):
    
#    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    R = img[:,:,0]

    red_binary = np.zeros_like(R)
    red_binary[(R > thresh[0]) & (R <= thresh[1])] = 1
           
    return red_binary

def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient=='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient=='y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output

def magnitude_threshold(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude
    abs_sobelxy = np.sqrt(np.square(sobelx) + np.square(sobely))
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelxy/np.max(abs_sobelxy))
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return mag_binary

def direction_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold

    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    angle_sobelxy = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(angle_sobelxy)
    dir_binary[(angle_sobelxy >= thresh[0]) & (angle_sobelxy <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return dir_binary

def combined_binary(img, plotting=False):
    
    sx_binarx = abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(20, 100))

    sy_binary = abs_sobel_threshold(img, orient='y', sobel_kernel=3, thresh=(20, 100))

    mag_binary = magnitude_threshold(img, sobel_kernel=9, mag_thresh=(50, 100))

    dir_binary = direction_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))

    sat_binary = saturation_threshold(img, thresh = (120, 255))

    can_binary = canny_edge_threshold(img, thresh = (50, 100), smoothing_kernel = 5)

    red_binary = red_threshold(img, thresh = (230, 255))

    lightness_binary = lightness_threshold(img, thresh = (30, 255))
    combined = np.zeros_like(dir_binary)

#    combined[((sx_binarx == 1) | ((mag_binary == 1) & (dir_binary == 1))) | ((sat_binary == 1)) | (red_binary == 1)] = 1
    combined[((sx_binarx == 1) | ((mag_binary == 1) & (dir_binary == 1))) | ((sat_binary == 1)  & (lightness_binary==1)) | (red_binary == 1)] = 1

    if plotting:
        # Plot the result
        f, axes = plt.subplots(4, 3, figsize=(12, 12))
        f.tight_layout()
        
        axes[0,0].imshow(sx_binarx, cmap='gray')
        axes[0,0].set_title('X Gradient')
    
        axes[0,1].imshow(sy_binary, cmap='gray')
        axes[0,1].set_title('Y Gradient')
    
        color_binary = np.dstack(( np.zeros_like(sx_binarx), sx_binarx, sy_binary))
    
        axes[0,2].imshow(color_binary)
        axes[0,2].set_title('Combined X and Y gradient')
    
        axes[1,0].imshow(mag_binary, cmap='gray')
        axes[1,0].set_title('Magnitude Gradient')   
    
        axes[1,1].imshow(dir_binary, cmap='gray')
        axes[1,1].set_title('Directional Gradient')
        
        color_binary = np.dstack(( np.zeros_like(mag_binary), mag_binary, dir_binary))
    
        axes[1,2].imshow(color_binary)
        axes[1,2].set_title('Combined Mag and Dir gradient')
    
        axes[2,0].imshow(sat_binary, cmap='gray')
        axes[2,0].set_title('Saturation')

        axes[2,1].imshow(red_binary, cmap='gray')
        axes[2,1].set_title('Red Channel')

        axes[2,2].imshow(can_binary, cmap='gray')
        axes[2,2].set_title('Canny Edge')

        axes[3,0].imshow(lightness_binary, cmap='gray')
        axes[3,0].set_title('Lightness Image')

        axes[3,1].imshow(img)
        axes[3,1].set_title('Original Image')
    
        axes[3,2].imshow(combined, cmap='gray')
        axes[3,2].set_title('Combined Binary Gradient')
        
#        plt.subplots_adjust(bottom=0.1, right=1.8, top=0.9)
        plt.tight_layout()
        plt.savefig('output_images/binary_comparison', dpi=150)#    plt.savefig('output_images/binary_combo_example.jpg')
    return combined

if __name__ == '__main__':
    image = mpimg.imread('test_images/test1.jpg')
    
    omb = combined_binary(image, True)
    
#    plt.figure()
#    plt.imshow(omb, 'gray')
#    plt.savefig('output_images/binary_combo_example.jpg')