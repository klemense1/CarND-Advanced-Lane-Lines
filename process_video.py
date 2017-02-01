#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 19:30:54 2017

@author: Klemens
"""

from moviepy.editor import VideoFileClip
import laneline
import pickle

if __name__ == "__main__":

    white_output = 'project_video_processed.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(laneline.process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
