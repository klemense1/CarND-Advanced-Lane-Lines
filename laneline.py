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

import thresholds

plt.close("all")

def get_peaks(hist):
    
    maxlength = len(hist)

    hist_left_plane = np.array(hist)
    hist_left_plane[:int(maxlength/2)] = 0
                    
    hist_right_plane = np.array(hist)
    hist_right_plane[int(maxlength/2):] = 0

    peak_left = hist_left_plane.argmax()
    peak_right = hist_right_plane.argmax()

    return peak_left, peak_right
    
def undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

    
def perspective_transform(image, inverse=False):
    """
    
    """

    img_size = (image.shape[1], image.shape[0])

    source_pts = np.float32(
                     [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
                     [((img_size[0] / 6)), img_size[1]],
                     [(img_size[0] * 5 / 6) + 50, img_size[1]],
                     [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]]
                     )
    dest_pts = np.float32(
                     [[(img_size[0] / 4), 0],
                     [(img_size[0] / 4), img_size[1]],
                     [(img_size[0] * 3 / 4), img_size[1]],
                     [(img_size[0] * 3 / 4), 0]]
                     )
    if inverse==False:
        M = cv2.getPerspectiveTransform(source_pts, dest_pts)
    
        transformed_image = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
        
        return transformed_image, source_pts, dest_pts
    else:
        Minv = cv2.getPerspectiveTransform(dest_pts, source_pts)
    
        transformed_image = cv2.warpPerspective(image, Minv, img_size, flags=cv2.INTER_LINEAR)
        
        return transformed_image, dest_pts, source_pts


def calculate_curvature(ploty, leftx, rightx):
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

def draw_lane_area_to_road(img, warped, left_fitx, right_fitx, yvals):
    ### Drawing the lines back down onto the road
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    yvals = ploty
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp, _, __ = perspective_transform(color_warp, inverse=True)
#    # Combine the result with the original image
    combined = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    return combined

def lines_from_hist(warped):
    img_size = warped.shape
    leftx = np.empty((img_size[0],1))
    rightx = np.empty((img_size[0],1))
    ploty = np.linspace(0, 719, num=720)

    printing_size = 2*72
    windowsize = int(img_size[0]/4)
    for slide in range(0, img_size[0], printing_size):
        hist_idx_start = max(img_size[0]-slide-windowsize, 0)
        hist_idx_end = img_size[0]-slide
        save_idx_start = img_size[0]-printing_size-slide
        save_idx_end = img_size[0]-slide
        if hist_idx_start >=0:

            histogram = np.sum(warped[hist_idx_start:hist_idx_end,:], axis=0)
            left_peak, right_peak = get_peaks(histogram)
            leftx[save_idx_start:save_idx_end] = left_peak
            rightx[save_idx_start:save_idx_end] = right_peak

    return leftx, rightx, ploty

def plot_image(image, title, debug_mode):
    if debug_mode:
        fig = plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(title)

def plot_transformed_perspective_binary(orig, transformed, saveasname, src=None, dst=None):
    

    red_color_int = (255,0,0)
    red_color_float = (1,0,0)
    linewidth = 2

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    if src is not None:
        cv2.line(orig['Image'], tuple(src[0]), tuple(src[1]), red_color_int, linewidth) # left
        cv2.line(orig['Image'], tuple(src[1]), tuple(src[2]), red_color_int, linewidth) # bottom
        cv2.line(orig['Image'], tuple(src[2]), tuple(src[3]), red_color_int, linewidth) # right
        cv2.line(orig['Image'], tuple(src[3]), tuple(src[0]), red_color_int, linewidth) # top

    ax1.imshow(orig['Image'])
    ax1.set_title(orig['Title'])

    if dst is not None:
        trimage = transformed['Image'].copy()
        trimage_lined = transformed['Image'].copy()
    
        cv2.line(trimage_lined, tuple(dst[0]), tuple(dst[1]), red_color_float, linewidth) # left
        cv2.line(trimage_lined, tuple(dst[1]), tuple(dst[2]), red_color_float, linewidth) # bottom
        cv2.line(trimage_lined, tuple(dst[2]), tuple(dst[3]), red_color_float, linewidth) # right
        cv2.line(trimage_lined, tuple(dst[3]), tuple(dst[0]), red_color_float, linewidth) # top
        final = np.dstack((trimage_lined, trimage, trimage))

        ax2.imshow(final)
    else:
        ax2.imshow(transformed['Image'], 'gray')

    ax2.set_title(transformed['Title'])
    plt.tight_layout()

    plt.savefig(saveasname)

if __name__ == "__main__":


    fname = 'test_images/test5.jpg'
    img = mpimg.imread(fname)

    ### Camera calibration

    dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    
    ### Distortion correction

    undist = undistort(img, mtx, dist)

    ### Color/gradient threshold

    binary = thresholds.combined_binary(undist)
    plot_image(binary, 'binary ... combined thresholds', True)

    ### Perspective transform

    warped, src, dst = perspective_transform(binary)
    mask = np.zeros(warped.shape, dtype=bool)
    mask.astype(bool)
    mask[:, 150:-150] = True
    warped[~mask] = 0

    original_dict = {'Image': img.copy(),
                     'Title': 'Original Image\n' + fname.split('/')[-1]}

    transformed_dict = {'Image': warped.copy(),
                        'Title': 'Perspective Transformed Image'}
 
    figpath = 'output_images/' + fname.split('/')[-1].split('.')[0] + '_transformed_perspective.jpg'
    
    plot_transformed_perspective_binary(original_dict, transformed_dict, figpath, src, dst)

    
    
    ### Detect lane lines
    leftx, rightx, ploty = lines_from_hist(warped)

    plt.figure()
    plt.imshow(warped, 'gray')
    plt.plot(leftx, ploty, '.', color='red', linewidth=2)
    plt.plot(rightx, ploty, '.', color='blue', linewidth=2)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.gca().invert_yaxis()
    
    figpath = 'output_images/' + fname.split('/')[-1].split('.')[0] + '_detected_lanes.jpg'
    plt.savefig(figpath)
    
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    plt.figure()
    plt.plot(leftx, ploty, '.', color='red', linewidth=2)
    plt.plot(rightx, ploty, '.', color='blue', linewidth=2)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, ploty, color='green', linewidth=2)
    plt.plot(right_fitx, ploty, color='green', linewidth=2)
    plt.gca().invert_yaxis() # to visualize as we do the images

    figpath = 'output_images/' + fname.split('/')[-1].split('.')[0] + '_plotted_lines.jpg'
    plt.savefig(figpath)

    ### Determine the lane curvature
    left_curverad, right_curverad = calculate_curvature(ploty, leftx, rightx)

    print('Radius of curvature:', left_curverad, 'm', right_curverad, 'm')
    
    result = draw_lane_area_to_road(img, warped, left_fitx, right_fitx, ploty)

    plt.figure()
    plt.imshow(result)

