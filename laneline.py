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

DEBUG_MODE = False

class Line():
    def __init__(self, queuelength):
        self.queuelength = queuelength
        # was the line detected in the last iteration?
        self.detected = False
        
        # x values of the last n fits of the line
        self.recent_fit_xval = []
        #average x values of the fitted line over the last n iterations
        self.avg_fit_xval = None # self.bestx = None
        # x values of most recent fit
        self.current_fit_xval = [np.array([False])]
        
        # x values of the last n fits of the line
        self.recent_fit_coeff = [] 
        #polynomial coefficients averaged over the last n iterations
        self.avg_fit_coeff = None  # self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit_coeff = [np.array([False])]   ###
        #difference in fit coefficients between last and new fits
        self.diff_fit_coeff = np.array([0,0,0], dtype='float') 

        # y values that lane shall be fitted for
        self.fit_yval = np.linspace(0, 719, num=720)
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = 0 
        #x values for detected line pixels
        self.allx = None  ###
        #y values for detected line pixels
        self.ally = None #  ###

        self.width_lookup = 2*np.linspace(150, 75, self.queuelength +1).astype(int)
        # window apllied to find the lane in pixels
        self.lane_width_window = self.width_lookup[0]

    def accept_lane(self):
        flag = True
        maxdist = 2.8  # distance in meters from the lane
        if abs(self.line_base_pos) > maxdist:
            print('lane too far away')
            flag  = False        
        if len(self.recent_fit_xval) > 0:
            relative_delta = self.diff_fit_coeff / self.avg_fit_coeff
            # allow maximally this percentage of variation in the fit coefficients from frame to frame
            if not (abs(relative_delta)<np.array([0.7,0.5,0.15])).all():
                print('fit coeffs too far off [%]',relative_delta)
                flag=False
                
        return flag
    
    def set_current_fit_xval(self):
        
        quadratic = self.current_fit_coeff[0]*self.fit_yval**2
        linear = self.current_fit_coeff[1]*self.fit_yval
        offset = self.current_fit_coeff[2]
        self.current_fit_xval = quadratic + linear + offset


    def set_radius_of_curvature(self):
        
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image

        y_eval = np.max(self.fit_yval)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        line_fit_cr = np.polyfit(self.fit_yval*ym_per_pix, self.avg_fit_xval*xm_per_pix, 2)
        # Calculate the new radii of curvature
        line_curverad = ((1 + (2*line_fit_cr[0]*y_eval*ym_per_pix + line_fit_cr[1])**2)**1.5) / np.absolute(2*line_fit_cr[0])

        self.radius_of_curvature = line_curverad


    def set_line_base_pos(self):

        y_eval = np.max(self.fit_yval)

        quadratic = self.current_fit_coeff[0]*y_eval**2
        linear = self.current_fit_coeff[1]*y_eval
        offset = self.current_fit_coeff[2]
        line_pos = quadratic + linear + offset
    
        basepos = 640 # half of image size
        meter_per_pixel = 3.7/1000

        self.line_base_pos = (basepos - line_pos)* meter_per_pixel


    def set_diff_fit_coeff(self):

        if len(self.recent_fit_coeff)>0:
            self.diff_fit_coeff = self.current_fit_coeff - self.avg_fit_coeff
        else:
            self.diff_fit_coeff = np.array([0,0,0], dtype='float')            


    def set_averages(self):
        if len(self.recent_fit_coeff)>0:
            self.avg_fit_coeff = np.array(self.recent_fit_coeff).mean(axis=0)
        if len(self.recent_fit_xval)>0:
            self.avg_fit_xval = np.array(self.recent_fit_xval).mean(axis=0)


    def add_data_to_buffer(self):
        if len(self.recent_fit_xval)==self.queuelength:
            self.recent_fit_xval.pop(0)
        self.recent_fit_xval.append(self.current_fit_xval)
        
        if len(self.recent_fit_coeff)==self.queuelength:
            self.recent_fit_coeff.pop(0)
        self.recent_fit_coeff.append(self.current_fit_coeff)


    def remove_data_from_buffer(self):
        self.recent_fit_xval.pop()

    def set_lane_width_window(self):
        idx = len(self.recent_fit_coeff) 
        self.lane_width_window = self.width_lookup[idx]

    def update(self,line_xval, line_yval):
        
        self.allx = line_xval
        self.ally = line_yval
        
        self.current_fit_coeff = np.polyfit(self.ally, self.allx, 2)

        self.set_current_fit_xval()


        self.set_diff_fit_coeff()
        
        if self.accept_lane():
            self.detected=True
            self.add_data_to_buffer()

        else:
            self.detected=False            
            self.remove_data_from_buffer()

        self.set_averages()

        self.set_radius_of_curvature()

        self.set_line_base_pos()
        
        self.set_lane_width_window()

        return self.detected

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

    
def draw_lane_area_to_road(img, warped, line_left, line_right):
    
    left_fitx = line_left.avg_fit_xval
    lefty = line_left.ally

    right_fitx = line_right.avg_fit_xval
    righty = line_right.ally

        
    ### Drawing the lines back down onto the road
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, lefty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, righty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp, _, __ = perspective_transform(color_warp, inverse=True)
#    # Combine the result with the original image
    combined = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    return combined
    
def draw_line_window_to_road(img, line_mask):

    # Create an image to draw the lines on
    empty = np.zeros_like(line_mask).astype(np.uint8)

    mask_colour = 255 * np.dstack((empty,
                                   empty,
                                   line_mask.astype(np.uint8)))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    mask_warped, _, __ = perspective_transform(mask_colour, inverse=True)

    # Combine the result with the original image
    combined = cv2.addWeighted(img, 1, mask_warped, 0.2, 0)

    return combined

def lines_from_hist(warped):
    img_size = warped.shape
    leftx = np.empty((img_size[0],1))
    rightx = np.empty((img_size[0],1))

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

    return leftx, rightx


def mask_image(image):#, mask_leftx, mask_rightx):
    
    
    
    image_masked = image.copy()
    mask = np.zeros(image_masked.shape, dtype=bool)
    pixels_per_side = int(image.shape[1]/8)
    mask[:, pixels_per_side:-pixels_per_side] = True

    image_masked[~mask] = 0
    
    return image_masked

def right_line_mask(shape, line_inst):
    
    if len(line_inst.recent_fit_xval) == 0:
        xvals = np.ones(720)*950
        yvals = np.linspace(0, 719, num=720)

    else:
        yvals = line_inst.ally
        xvals = line_inst.avg_fit_xval

    half_mask_width = int(line_inst.lane_width_window/2)

    right_mask = create_line_mask(shape, xvals, yvals, half_mask_width)
    
    return right_mask

def left_line_mask(shape, line_inst):
    
    if len(line_inst.recent_fit_xval) == 0:
        xvals = np.ones(720)*300
        yvals = np.linspace(0, 719, num=720)
    
    else:
        yvals = line_inst.ally
        xvals = line_inst.avg_fit_xval

    half_mask_width = int(line_inst.lane_width_window/2)

    left_mask = create_line_mask(shape, xvals, yvals, half_mask_width)
    
    return left_mask

def create_line_mask(shape, xvals, yvals, shift_pixels):

    mask = np.zeros(shape, dtype=bool)

    for idx, pixel_y in enumerate(yvals):
        mask[pixel_y, xvals[idx]-shift_pixels:xvals[idx]+shift_pixels] = True

    return mask

def plot_image(image, title, debug_mode=False):
    
    if debug_mode:
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(title)

def plot_detected_lines(image, line_left, line_right, debug_mode=False):
    
    leftx = line_left.allx
    lefty = line_left.ally

    rightx = line_right.allx
    righty = line_right.ally

    if debug_mode:
        plt.figure()
        plt.imshow(image, 'gray')
        plt.plot(leftx, lefty, '.', color='red', linewidth=2)
        plt.plot(rightx, righty, '.', color='blue', linewidth=2)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.gca().invert_yaxis()
        
        figpath = 'output_images/detected_lanes.jpg'
        plt.savefig(figpath)

def plot_fitted_curve(line_left, line_right, debug_mode=False):
    
    if debug_mode:

        leftx = line_left.allx
        lefty = line_left.ally
        left_fitx = line_left.current_fit_xval

        rightx = line_right.allx
        righty = line_right.ally
        right_fitx = line_right.current_fit_xval
        
        plt.figure()
        plt.plot(leftx, lefty, '.', color='red', linewidth=2)
        plt.plot(rightx, righty, '.', color='blue', linewidth=2)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.plot(left_fitx, lefty, color='green', linewidth=2)
        plt.plot(right_fitx, righty, color='green', linewidth=2)
        plt.gca().invert_yaxis() # to visualize as we do the images

        figpath = 'output_images/plotted_lines.jpg'
        plt.savefig(figpath)

def plot_transformed_perspective_binary(orig, transformed, src=None, dst=None, debug_mode=False):
    
    if debug_mode:
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
        
        figpath = 'output_images/transformed_perspective.jpg'

        plt.savefig(figpath)

def detect(img):
    
    ### 1. Camera calibration

    dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    ### 2. Distortion correction

    img_undist = undistort(img, mtx, dist)

    ### 3. Color/gradient threshold

    img_binary = thresholds.combined_binary(img_undist)
    plot_image(img_binary, 'binary ... combined thresholds', DEBUG_MODE)

    ### 4. Perspective transform
    
    img_warped, src, dst = perspective_transform(img_binary)

    left_line = Line(8)
    right_line = Line(8)

    left_lane_mask = left_line_mask(img_warped.shape, left_line)#, mask_leftx, mask_lefty)

    right_lane_mask = right_line_mask(img_warped.shape, right_line)

    plot_image(left_lane_mask, 'Mask left', True)
    plot_image(right_lane_mask, 'Mask right', True)

    img_warped[(left_lane_mask == 0) & (right_lane_mask == 0)] = 0

    original_dict = {'Image': img.copy(),
                     'Title': 'Original Image'}

    transformed_dict = {'Image': img_warped.copy(),
                        'Title': 'Perspective Transformed Image (masked)'}

    plot_transformed_perspective_binary(original_dict, transformed_dict, src, dst, DEBUG_MODE)
    
    ### 5. Detect lane lines
    
    leftx_detected, rightx_detected = lines_from_hist(img_warped)
    lefty_detected = np.linspace(0, 719, num=720)
    righty_detected = np.linspace(0, 719, num=720)

    left_line.update(leftx_detected, lefty_detected)
    right_line.update(rightx_detected, righty_detected)

    plot_detected_lines(img_warped, left_line, right_line, DEBUG_MODE)
    
    plot_fitted_curve(left_line, right_line, DEBUG_MODE)

    ### 6. Determine the lane curvature
    
    left_curverad = left_line.radius_of_curvature
    right_curverad = right_line.radius_of_curvature
    
    offset = np.mean([left_line.line_base_pos, right_line.line_base_pos])

    img_lanearea = draw_lane_area_to_road(img, img_warped, left_line, right_line)

    img_lanearea = draw_line_window_to_road(img_lanearea, left_lane_mask)
    img_lanearea = draw_line_window_to_road(img_lanearea, right_lane_mask)

    return img_lanearea, left_curverad, right_curverad, offset

def process_image(img):
    
    img_processed, __, __, __ = detect(img)
    
    return img_processed

if __name__ == "__main__":

    fname = 'test_images/straight_lines1.jpg'
    img = mpimg.imread(fname)
    plot_image(img, 'Original Image', True)

    img_lanedetected, left_curverad, right_curverad, off_cent = detect(img)
    
    plot_image(img_lanedetected, 'Detected Lane', True)

    print('Radius of curvature, left:', left_curverad, 'm, right:', right_curverad, 'm')
    print('Offcenter', off_cent, 'm')