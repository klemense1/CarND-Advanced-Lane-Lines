## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image_calibration1]: ./camera_cal/corners/corners_found_ny6_nx8_calibration2.jpg "Corners with ny=6 and nx=8"
[image_calibration2]: ./camera_cal/corners/corners_found_ny6_nx9_calibration2.jpg "Corners with ny=6 and nx=9"
[image_calibration3]: ./camera_cal/undistored/undistort_output_calibration1.jpg "Undistorted image 1"

[image_original]: test_images/test1.jpg "Undistorted lane image"
[image_undistored]: output_images/undistored.png "Undistorted lane image"
[image_binary]: output_images/binarycombinedthresholds.png "Binary lane image"
[image_warped]: output_images/transformed_perspective.png "Warped image"
[image_detected_lines]: output_images/detected_lanes.png "Detected Lines"
[image_masks]: output_images/masks.png "Binary masks for left and right lane"
[image_plotted_lines]: output_images/plotted_lines.png "Fitted curve through detected Lines"


[image_binary_comparison]: output_images/binary_comparison.png "Comparison of different masks"

[image_final]: output_images/detected_lane.png "Output image"

[video1]: ./project_video_processed.mp4 "Video Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Camera Calibration

The code for this step is contained in  `calibratecamera.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the real world space. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I read in the image, convert it to gray-scale, and use the opencv function findChessboardCorners to find the corners matching the specified number of corners in each direction (nx and ny, respectivly).
  ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
If the corners could be identified, I keep the found corners and their matching object points.

    if ret==True:
      imgpoints.append(corners)
      objpoints.append(objp)

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

I first started out with nx = 9 and ny = 6, which lead to missing out the following files:

    No corners found in camera_cal/calibration1.jpg
    No corners found in camera_cal/calibration4.jpg
    No corners found in camera_cal/calibration5.jpg

After having a closer look on them, I realized that those images only had a fraction of the chessboard, which meant they did not have the fill number of corners in each direction. Step by step, I added less possible number of corners in each direction, finally coming up with:

    for nx in [6, 7, 8, 9]:
      for ny in [5, 6]:
        ...

Identifying not all corners does not hurt, as can be seen in the following pictures:

Corners with ny=6 and nx=8

![alt text][image_calibration1]

Corners with ny=6 and nx=9

![alt text][image_calibration2]

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image_calibration3]

Finally, I saved the calibration data as a pickle, so that the calibration does not need to be repeated again.

### Pipeline for single images

The image is getting processed in the detect-function in `laneline.py`. I am going through the processing steps by using the following image as an example.

![alt text][image_original]
#### 1. Distortion correction
First, I am loading loading the camera matrix and the vector of distortion coefficients from the pickle. I then use `cv2.undistort()`, which is implemented in the function `undistort()`.

The output of this step can be seen in the following:

![alt text][image_undistored]

#### 2. Binary thresholding

I used a combination of color and gradient thresholds to generate a binary image (thresholding is performed in the function `combined_binary()` in `thresholds.py`).  Here's an example of my output for this step.

![alt text][image_binary]

How did I get there? I applied the following thresholds:
- gradient gradient in x and y direction
- magnitude gradient from x and y
- directional gradient
- saturation channel
- red channel
- lightness
- canny edge detection

First, I changed the threshold boundaries to get a good output from each operation. Then I tried out various combinations and came up with the following:

    combined[((sx_binarx == 1) | ((mag_binary == 1) & (dir_binary == 1))) |
             ((sat_binary == 1)  & (lightness_binary==1)) |
             (red_binary == 1)] = 1
The outputs of the various thresholds are displayed here:
![alt text][image_binary_comparison]

#### 3. Perspective transformation

To get from a camera view to a birds view, one needs to perform a perspective transformation on the image. The function `perspective_transformation()` function takes as inputs an image (`img`) and a boolean value allowing for both the normal and the inverse transformation. The source (`src`) and destination (`dst`) points are hardcoded in the function itself in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 60), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 580, 460      | 320, 0        |
| 213, 720      | 320, 720      |
| 1116, 720     | 960, 720      |
| 700, 460      | 960, 0        |

As stated above, I am allowing for normal and inverse transformation. The transformation matrix is calculated from:

    if inverse==False:
      M = cv2.getPerspectiveTransform(source_pts, dest_pts)
      ...
    else:
      Minv = cv2.getPerspectiveTransform(dest_pts, source_pts)
      ...

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image_warped]

#### 4. Identification of lane-line pixels
4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I am applying masks for the right and left line to the warped binary image, so the algorithm will not need to search the line in an area where I don't expect it to be. How do I know where to look?

I am applying a window size to where I detected the line before and leave out everything else. As I don't know anything about the line when having only one picture, I starting with a big window size and reduce it when having a video stream and a pretty good idea, where the line should be.

So for a single picture (or for a video stream), the initial masks look like the following:
![alt text][image_masks]

The function `detect_lines_in_warped()` takes a warped binary as an input and uses a sliding histogram to identify the peaks. I am using a window-size in y-direction of 180 pixels (one fourth of the x-length). The histogram is applied to that part of the image. The window is moved with a step-size of 144 (one fifth of the x-length). With those settings, I got a robust line detection. It is displayed in the following:

![alt text][image_detected_lines]

I then fitted a 2nd order polynomial through the detected pixels. The result (green curves) can be seen in the following picture:

![alt text][image_plotted_lines]

#### 5. Radius of lane curvature and position of the vehicle

In the method `set_radius_of_curvature()` of the line class, I am calculating the radius of the line curvature. I am then taking the mean from the curvature of left and right line.

In the method `set_line_base_pos()` of the line class, I am calculating the position, where the line hits the bottom of the window. I am then taking the mean from the base postions of left and right line to calculate the offset.

#### 6. Lane identification

I finally draw the lines interpolated from the fitted polynomial back to the stree. That is done in `draw_lane_area_to_road()`, which takes the original image, the shape of the warped image, and the objects of left and right line as an input.

First, to draw a coloured polygon, I define an array with image shape and three channels.

    warp_zero = np.zeros(warped_shape).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

I then transform the x and y pairs of fitted lines to fit to the `cv2.fillPoly()` function.

    pts_left = np.array([np.transpose(np.vstack([left_fitx, lefty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, righty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

I now got a drawn polygon with the identified lane in `color_warp`, but it is still in birds view. I then transform the colored warp back to the steet and combine it with the input image.

    newwarp, _, __ = perspective_transform(color_warp, inverse=True)

    combined = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

In addition to that, I am also drawing the line window of the mask in blue, which I described before. This is done in `draw_transformed_mask_to_road()`, which works similar to the function stated above.

I am also displaying the fitting coefficients of each line, the offset and the radius of curvature.

![alt text][image_final]

---

### Pipeline for a video stream

The video result can be seen here:

[link to my video result][video1]

I am displaying the buffer of my line classes to see previous image's informations are still in the pipeline. This allows for quick debugging, as a dropping buffer in the mid of the video would indiciate that the line identification was repeatingly unsuccessful.

---

### Discussion

Another great project. Getting from a working image pipeline to a video pipeline by using the line class allowed for much better results.
I think my binary masks and the histogram detection could both be improved. Without using information from previous images, the identifaction used to fail when experiencing lightness or darkness.
The peak detection using histograms might be computationally inefficient. Udacity published code with a sligthly different approach than mine, but as mine was working, I did not really take a look on that. To get faster and more accurate, I will need to improvie this part of the code.
