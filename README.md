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
[image_plotted_lines]: output_images/plotted_lines.png "Fitted curve through detected Lines"


[image_binary_comparison]: output_images/binary_comparison.png "Comparison of different masks"

[image_final]: output_images/detected_lane.png "Output image"

[video1]: ./project_video_processed.mp4 "Video Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

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

### Pipeline (single images)

The image is getting processed in the detect-function in `laneline.py`. I am going through the processing steps by using the following image as an example.

![alt text][image_original]
#### 1. Distortion correction
First, I am loading loading the camera matrix and the vector of distortion coefficients from the pickle. I then use `cv2.undistort()`, which is implemented in the function `undistort()`.

The output of this step can be seen in the following:

![alt text][image_undistored]

#### Binary thresholding

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

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image_warped]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

![alt text][image_detected_lines]

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image_plotted_lines]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image_final]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
