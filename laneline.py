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
from moviepy.editor import VideoFileClip

import thresholds

plt.close("all")


class Line():
    def __init__(self, queuelength):
        self.queuelength = queuelength
        # was the line detected in the last iteration?
        self.detected = False
        
        self.buffer = 0
        
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

        # table with line width decay
        self.width_decay = 2*np.linspace(220, 100, self.queuelength +1).astype(int)
        # window apllied to find the lane in pixels
        self.lane_width_window = self.width_decay[0]


    def set_current_fit_xval(self):
        """
        calculates x values from fitted polynomial
        """
        quadratic = self.current_fit_coeff[0]*self.fit_yval**2
        linear = self.current_fit_coeff[1]*self.fit_yval
        offset = self.current_fit_coeff[2]
        self.current_fit_xval = quadratic + linear + offset


    def set_radius_of_curvature(self):
        """
        calculates radius of curvature from avg_fit_xval
        """
        if self.buffer>0:
            y_eval = np.max(self.fit_yval)
    
            y_meter_per_pix = 30/720
            x_meter_per_pix = 3.7/700
            
            line_fit_cr = np.polyfit(self.fit_yval*y_meter_per_pix, self.avg_fit_xval*x_meter_per_pix, 2)
            line_curverad = ((1 + (2*line_fit_cr[0]*y_eval*y_meter_per_pix + line_fit_cr[1])**2)**1.5) / np.absolute(2*line_fit_cr[0])
    
            self.radius_of_curvature = line_curverad


    def set_line_base_pos(self):
        """
        sets position, where line hits bottom of window
        """
        y_eval = np.max(self.fit_yval)

        quadratic = self.current_fit_coeff[0]*y_eval**2
        linear = self.current_fit_coeff[1]*y_eval
        offset = self.current_fit_coeff[2]
        line_pos = quadratic + linear + offset
    
        basepos = 640
        meter_per_pixel = 3.7/1000

        self.line_base_pos = (basepos - line_pos) * meter_per_pixel


    def set_diff_fit_coeff(self):
        """
        sets difference between current fitting coefficients of line 
        and averaged fitting coefficients
        """
        if self.buffer>0:
            self.diff_fit_coeff = self.current_fit_coeff - self.avg_fit_coeff
        else:
            self.diff_fit_coeff = np.array([0,0,0], dtype='float')            


    def set_averages(self):
        """
        set average values for fitting coefficient and fitted x values
        """
        if self.buffer>0:
            self.avg_fit_coeff = np.array(self.recent_fit_coeff).mean(axis=0)
        if self.buffer>0:
            self.avg_fit_xval = np.array(self.recent_fit_xval).mean(axis=0)


    def add_data_to_buffer(self):
        """
        adds new data to buffer. If buffer has maximum length, the oldes value
        value will be dropped
        """
        if self.buffer==self.queuelength:
            self.recent_fit_xval.pop(0)

        self.recent_fit_xval.append(self.current_fit_xval)
        
        if self.buffer==self.queuelength:
            self.recent_fit_coeff.pop(0)

        self.recent_fit_coeff.append(self.current_fit_coeff)
        
        self.buffer = len(self.recent_fit_xval)


    def remove_data_from_buffer(self):
        """
        removes data from buffer
        """
        self.recent_fit_xval.pop()
        self.recent_fit_coeff.pop()

        self.buffer = len(self.recent_fit_xval)


    def set_lane_width_window(self):
        idx = self.buffer
        self.lane_width_window = self.width_decay[idx]


    def check_position_with_other_line(self, current_fit_xval):
        """
        Checks, whether fitted xvalues are plausibel with other line's values.
        If not, the lastly added values are droped and resulting information
        are re-calculated
        """
        if self.buffer > 0:
            lines_too_far = abs(self.current_fit_xval - current_fit_xval).max() > 850
            lines_too_close = abs(self.current_fit_xval - current_fit_xval).min() < 550

            if lines_too_far or lines_too_close:
                print('Not plausible with other line')
                self.remove_data_from_buffer()
        
                self.set_averages()
        
                self.set_radius_of_curvature()
        
                self.set_line_base_pos()
                
                self.set_lane_width_window()


    def update(self, line_xval, line_yval):
        """
        updates line class with new detected x and y values of line
        """
        self.allx = line_xval
        self.ally = line_yval
        
        self.current_fit_coeff = np.polyfit(self.ally, self.allx, 2)

        self.set_current_fit_xval()

        self.set_diff_fit_coeff()
        
        self.add_data_to_buffer()

        self.set_averages()

        self.set_radius_of_curvature()

        self.set_line_base_pos()
        
        self.set_lane_width_window()

        return self.detected


def get_peaks(hist):
    """
    calculates peak for left and right half plane of given histogram
    
    parameters:
        hist ... histogram
        
    returns:
        peak_left ... location of left peak in hist
        peak_right ... location of right peak in hist
    """
    maxlength = len(hist)

    hist_left_plane = np.array(hist)
    hist_left_plane[:int(maxlength/2)] = 0
                    
    hist_right_plane = np.array(hist)
    hist_right_plane[int(maxlength/2):] = 0

    peak_left = hist_left_plane.argmax()
    peak_right = hist_right_plane.argmax()

    return peak_left, peak_right


def undistort(img, mtx, dist):
    """
    undistores the image
    
    parameters:
        img ... distored image
        mtx ... input camera matrix
        dist ... input vector of distortion coefficients
    
    returns:
        undist ... undistored image
    """
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

    
def perspective_transform(image, inverse=False):
    """
    performs a perspective transform. The inverse operation is also possible
    
    parameters:
        image ... input image
        inverse ... boolean allowing for transformation or inverse transf.
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

    
def draw_lane_area_to_road(img, warped_shape, line_left, line_right):
    """
    Draws the area beween left and right line to an image. The fitted values
    in the line objects are recasted into usable format for cv2.fillPoly().
    The drawn polygon is then transformed back on the road.
    
    parameters:
        img ... input image
        warped_shape ... warped image
        line_left ... line object of left line
        line_right ... line object of right line
    
    returns:
        combined ... output image with lane area drawn at
    """

    left_fitx = line_left.avg_fit_xval
    lefty = line_left.ally

    right_fitx = line_right.avg_fit_xval
    righty = line_right.ally
        
    warp_zero = np.zeros(warped_shape).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, lefty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, righty])))])
    pts = np.hstack((pts_left, pts_right))
    
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    newwarp, _, __ = perspective_transform(color_warp, inverse=True)

    combined = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    return combined
    
def draw_transformed_mask_to_road(img, line_mask):
    """
    Draws the line window, where line is seeked in, to the road. The line_mask
    is stacked to get a blue colour, then transformed to the road perspective
    and then added to the image.
    
    parameters:
        img ... input image
        line_mask ... perspective transformed binary mask
    """
    empty = np.zeros_like(line_mask).astype(np.uint8)

    mask_colour = 255 * np.dstack((empty,
                                   empty,
                                   line_mask.astype(np.uint8)))

    mask_warped, _, __ = perspective_transform(mask_colour, inverse=True)

    combined = cv2.addWeighted(img, 1, mask_warped, 0.2, 0)

    return combined

def detect_lines_in_warped(warped):
    """
    detects lines from warped image using a sliding histogram
    
    parameters:
        warped ... warped binary
        
    returns:
        leftx ... x values of left line
        rightx ... x values of right line
    """

    img_size = warped.shape
    leftx = np.empty((img_size[0], 1))
    rightx = np.empty((img_size[0], 1))

    printing_size = 2*72
    windowsize = int(img_size[0]/4)

    for slide in range(0, img_size[0], printing_size):

        hist_idx_start = max(img_size[0]-slide-windowsize, 0)
        hist_idx_end = img_size[0]-slide

        save_idx_start = img_size[0]-printing_size-slide
        save_idx_end = img_size[0]-slide

        if hist_idx_start >=0:

            histogram = np.sum(warped[hist_idx_start:hist_idx_end, :], axis=0)

            left_peak, right_peak = get_peaks(histogram)

            leftx[save_idx_start:save_idx_end] = left_peak
            rightx[save_idx_start:save_idx_end] = right_peak

    return leftx, rightx


def right_line_mask(shape, line_inst):
    """
    creates right line mask with a window width of line_inst.lane_width_window
    
    parameters:
        shape ... shape of mask
        line_inst ... line instance
    """
    if line_inst.buffer == 0:

        xvals = np.ones(720)*950
        yvals = np.linspace(0, 719, num=720)

    else:

        yvals = line_inst.ally
        xvals = line_inst.avg_fit_xval

    half_mask_width = int(line_inst.lane_width_window/2)

    right_mask = create_line_mask(shape, xvals, yvals, half_mask_width)
    
    return right_mask


def left_line_mask(shape, line_inst):
    """
    creates left line mask with a window width of line_inst.lane_width_window
    
    parameters:
        shape ... shape of mask
        line_inst ... line instance
    """
    if line_inst.buffer == 0:

        xvals = np.ones(720)*300
        yvals = np.linspace(0, 719, num=720)
    
    else:

        yvals = line_inst.ally
        xvals = line_inst.avg_fit_xval

    half_mask_width = int(line_inst.lane_width_window/2)

    left_mask = create_line_mask(shape, xvals, yvals, half_mask_width)
    
    return left_mask


def create_line_mask(shape, xvals, yvals, mask_width_halfed):
    """
    create a line mask
    
    parameters:
        shape ... shape of mask
        xvals ... fitted x values of last detected line
        yvals ... fitted y values of last detected line
        mask_width_halfed ... half width of unmasked area around detected line
    returns:
        mask ... mask
    """
    mask = np.zeros(shape, dtype=bool)

    for idx, pixel_y in enumerate(yvals):
        mask[pixel_y, xvals[idx]-mask_width_halfed:xvals[idx]+mask_width_halfed] = True

    return mask


def plot_gray_image(image, title, debug_mode=False):
    """
    plots image in gray
    
    parameters:
        image ... input image
        title ... title of plot
        debug_mode ... [False] only plot if true
    """

    if debug_mode:
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.title(title)
        
        plt.savefig('output_images/' + title.replace(' ','').replace('.', ''), dpi=150)


def plot_detected_lines(warped, line_left, line_right, debug_mode=False):
    """
    plots detected lines on warped image 
    and saves image in 'output_images/detected_lanes'
    
    parameters:
        warped ... warped input image
        line_left ... left line object
        line_right ... right line object
        debug_mode ... [False] only plot if true
    
    """
    leftx = line_left.allx
    lefty = line_left.ally

    rightx = line_right.allx
    righty = line_right.ally

    if debug_mode:
        plt.figure()
        plt.imshow(warped, 'gray')
        plt.plot(leftx, lefty, '.', color='red', linewidth=2)
        plt.plot(rightx, righty, '.', color='blue', linewidth=2)
        plt.xlim(0, 1280)
        plt.ylim(0, 720)
        plt.gca().invert_yaxis()
        
        figpath = 'output_images/detected_lanes'
        plt.savefig(figpath, dpi=150)


def plot_fitted_curve(line_left, line_right, debug_mode=False):
    """
    plots fitted line curves
    and saves image in 'output_images/plotted_lines'
    
    parameters:
        line_left ... left line object
        line_right ... right line object
        debug_mode ... [False] only plot if true

    """
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
        plt.gca().invert_yaxis()

        figpath = 'output_images/plotted_lines'
        plt.savefig(figpath, dpi=150)


def plot_transformed_perspective_binary(orig, transformed, src=None, dst=None, debug_mode=False):
    """
    plots original transformed perspective binary and draws source and 
    destination points as a polygon.
    The image is then saved as 'output_images/transformed_perspective'
    
    parameters:
        orig ... original image
        transformed ... transformed perspective binary image
        src ... [None] source points used for perspective transformation
        dst ... [None] destination points used for perspective transformation
        debug_mode ... [False] only plot if true

    """
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
        
        figpath = 'output_images/transformed_perspective'

        plt.savefig(figpath, dpi=150)


def detect(img):
    """
    Detects right line and left line in a given image.
    The information are then processed in the line objects.
    The detected lines and some other information are finally drawn to the 
    picture.
    
    parameters:
        img ... input image
        
    returns:
        img_lanearea ... image with detected lane area
    """

    global left_line
    global right_line

    ### 1. Camera calibration

    dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    ### 2. Distortion correction

    img_undist = undistort(img, mtx, dist)
    plot_gray_image(img_undist, 'undistored', DEBUG_MODE)

    ### 3. Color/gradient threshold

    img_binary = thresholds.combined_binary(img_undist)
    plot_gray_image(img_binary, 'binary ... combined thresholds', DEBUG_MODE)

    ### 4. Perspective transform
    
    img_warped, src, dst = perspective_transform(img_binary)

    left_lane_mask = left_line_mask(img_warped.shape, left_line)
    right_lane_mask = right_line_mask(img_warped.shape, right_line)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(left_lane_mask, 'gray')
    ax1.set_title('Mask left')
    ax2.imshow(right_lane_mask, 'gray')
    ax2.set_title('Mask right')
    plt.savefig('output_images/masks', dpi=150)

    img_warped[(left_lane_mask == 0) & (right_lane_mask == 0)] = 0

    original_dict = {'Image': img.copy(),
                     'Title': 'Original Image'}

    transformed_dict = {'Image': img_warped.copy(),
                        'Title': 'Perspective Transformed Image (masked)'}

    plot_transformed_perspective_binary(original_dict, transformed_dict, src, dst, DEBUG_MODE)
    
    ### 5. Detect lane lines
    
    leftx_detected, rightx_detected = detect_lines_in_warped(img_warped)

    lefty_detected = np.linspace(0, 719, num=720)
    righty_detected = np.linspace(0, 719, num=720)

    
    left_line.update(leftx_detected, lefty_detected)
    right_line.update(rightx_detected, righty_detected)

    left_current_fit_xval = left_line.current_fit_xval
    right_current_fit_xval = right_line.current_fit_xval

    left_line.check_position_with_other_line(right_current_fit_xval)
    right_line.check_position_with_other_line(left_current_fit_xval)

    
    plot_detected_lines(img_warped, left_line, right_line, DEBUG_MODE)
    
    plot_fitted_curve(left_line, right_line, DEBUG_MODE)

    ### 6. Determine the lane curvature
    
    offset = round(np.mean([left_line.line_base_pos, right_line.line_base_pos]), 2)

    img_lanearea = draw_lane_area_to_road(img, img_warped.shape, left_line, right_line)

    img_lanearea = draw_transformed_mask_to_road(img_lanearea, left_lane_mask)
    img_lanearea = draw_transformed_mask_to_road(img_lanearea, right_lane_mask)

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    str_offset = str('Offset: '+str(offset)+'m')
    cv2.putText(img_lanearea, str_offset, (430,70), font, 1, (255,0,0), 2, cv2.LINE_AA)
    
    if left_line.radius_of_curvature and right_line.radius_of_curvature:

        curvature = round(np.mean([left_line.radius_of_curvature, right_line.radius_of_curvature]), 1)
        
        str_curvature = str('radius of curvature: '+str(curvature)+'m')
        
        cv2.putText(img_lanearea, str_curvature, (430,110), font, 1, (255,0,0), 2, cv2.LINE_AA)    
    
#    cv2.putText(img_lanearea, 'left position: ' + str(float(left_line.line_base_pos.round(1))), (70,170), font, 1, (255,0,0), 2, cv2.LINE_AA)
#    cv2.putText(img_lanearea, 'right position: ' + str(float(right_line.line_base_pos.round(1))), (730,170), font, 1, (255,0,0), 2, cv2.LINE_AA)
# 
#    cv2.putText(img_lanearea, 'left window: ' + str(float(left_line.lane_width_window.round(1))), (70,210), font, 1, (255,0,0), 2, cv2.LINE_AA)
#    cv2.putText(img_lanearea, 'right window: ' + str(float(right_line.lane_width_window.round(1))), (730,210), font, 1, (255,0,0), 2, cv2.LINE_AA)
    
    coeff = left_line.current_fit_coeff
    str_coeff1 = '{}, {}, {}'.format(coeff[0].round(2), coeff[1].round(1), coeff[2].round(0))
    cv2.putText(img_lanearea, 'fit coeff: ' + str_coeff1, (130,250), font, 1, (255,0,0), 2, cv2.LINE_AA)

    coeff = right_line.current_fit_coeff
    str_coeff2 = '{}, {}, {}'.format(coeff[0].round(2), coeff[1].round(1), coeff[2].round(0))
    cv2.putText(img_lanearea, 'fit coeff: ' + str_coeff2, (730,250), font, 1, (255,0,0), 2, cv2.LINE_AA)

    string = 'Buffer ' + ''.join(['I']*left_line.buffer)
    cv2.putText(img_lanearea, string, (30,60), font, 1, (0,0,0), 2, cv2.LINE_AA)

    return img_lanearea

def process_image(img):
    
    global left_line
    global right_line

    img_processed = detect(img)
    
    return img_processed

if __name__ == "__main__":

    PIPELINE_VIDEO = False
    DEBUG_MODE = not(PIPELINE_VIDEO)

    if PIPELINE_VIDEO:
    
        white_output = 'project_video_processed.mp4'
        clip1 = VideoFileClip("project_video.mp4")
        white_clip = clip1.fl_image(process_image)
        white_clip.write_videofile(white_output, audio=False)

    else:
        
        left_line = Line(8)
        right_line = Line(8)
    
        fname = 'test_images/test1.jpg'
        img = mpimg.imread(fname)
        plot_gray_image(img, 'Original Image', True)
    
        img_lanedetected = detect(img)
        
        plt.figure()
        plt.imshow(img_lanedetected)
        plt.savefig('output_images/detected_lane', dpi=150)
