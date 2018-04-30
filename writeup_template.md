##Advanced Lane Finding
###This pipeline detects well on the first project_vedio.mp4 even on some frames with suddenly brightness change. So it meets the basic requirement of this project. 
###For another challenge_vedio.mp4 and harder_challenge_vedio.mp4, this pipeline's performance is not satisfactory. It works from start to end in challenge_vedio.mp4 but with distorted lane boundary. For the harder_challenge_vedio.mp4, it even can not detect any lane for a long time. 

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

[image1]: ./output_images/chess_undistorted.PNG "Undistorted"
[image2]: ./output_images/undistorted.PNG "Undistorted"
[image3]: ./output_images/test1_undistorted_thresholded.PNG "Binary Example"
[image4]: ./output_images/test1_warped.PNG "Warp Example"
[image5]: ./output_images/test1_detect.jpg "Fit Visual"
[image6]: ./output_images/test1_wraped_back_txt.jpg "Output"
[image7]: ./examples/color_fit_lines.jpg "Fit Show"
[video1]: ./out_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

Another important document for this pipeline is adv_lane.ipynb.
In jupyte notebook, this pipeline starts from scratch step by step according to project guideline listed above for one image. 

Then all these codes are intergrated together to deal with vedio frame by frame. 
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./adv_lane.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Something should be mentioned that there are 3 images in camera_cal folder which can not found chesscorner successfully. It is luck that it does not influence further work. 

These 3 images have been reviewed that they are distorted heavily that can not match the pre-assigned parameter 9 and 6. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


![alt text][image1]


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
I make usef of these `objpoints` and `imgpoints` and apply `cv2.calibrateCamera()` and `cv2.undistort()` function to a test image and get these following results like this one:

It is not easy to tell the difference for this test image with orginan and undistorted but really works. 
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I tried several methods from class to check the influence.
Finally I used a combination of HLS color and gradient thresholds to generate a binary image (jupyter notebook 3rd cell "Use color transforms, gradients, etc., to create a thresholded binary image" or fuciton `def binary_transf(self, img, s_thresh=(90, 255), sx_thresh=(20, 100)))`.  Here's an example of my output for this step.

These parameters are carrying over from class but also be confirmed that is pretty optimized results. 

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a opencv function called `cv2.warpPerspective(threshold_image, M, img_size)`, which appears in the 4th code cell of the IPython notebook and `def process_image(self, image))`.  The `cv2.warpPerspective()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  

This shows the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 260, 0        | 
| 700, 460      | 1040, 0       |
| 1040, 680     | 1040, 720     |
| 260, 680      | 260, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

There are two functions here to detect lane-line. 

The first one is detecting by sliding windows while the other is detecting by the previous detected lane. 

In order to make sure detect lane effectively and quickly, `Class Line().detected` is used as a sign to tell pipeline to call `self.detect_lane_slidingwindow(warped)` or to call `self.detect_lane(warped, self.left_fits[-1], self.right_fits[-1])` directly. 

		# self.detected is used here to decicde to detect by slidingwindow or not
 
        if self.detected == False:
            left_fit, right_fit, out_img = self.detect_lane_slidingwindow(warped)
            left_fit, right_fit, out_img = self.detect_lane(warped, left_fit, right_fit)
        else:
            # use last detect left_fit and right_fit to detect 
            left_fit, right_fit, out_img = self.detect_lane(warped, self.left_fits[-1], self.right_fits[-1])`

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:
![alt text][image7]
![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `def radius_position(self, binary_warped, left_fit, right_fit)`

There are some error for these codes copied from class. The corrected code are shown below.

		# Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        center = (((left_fit[0]*720**2+left_fit[1]*720+left_fit[2]) +(right_fit[0]*720**2+right_fit[1]*720+right_fit[2]) ) /2 - 640)*xm_per_pix


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines

		# Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = draw_left_fit[0]*ploty**2 + draw_left_fit[1]*ploty + draw_left_fit[2]
        right_fitx = draw_right_fit[0]*ploty**2 + draw_right_fit[1]*ploty + draw_right_fit[2]`
    
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, img_size) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
        #Write curvature and center in image
        TextL = "Left curv: " + str(int(left_curverad)) + " m"
        TextR = "Right curv: " + str(int(right_curverad))+ " m"
        TextC = "Center offset: " + str(round( center,2)) + "m"

        fontScale=1
        thickness=2
    
        fontFace = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(result, TextL, (130,40), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
        cv2.putText(result, TextR, (130,70), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
        cv2.putText(result, TextC, (130,100), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

In order to make the pipeline as robust as possible, after function `def sanity
()` checking every detected lane to decide whether append or ignore it. I use the last 5 average detected lane to show. 

This seem not contribute a lot for the 2nd and 3rd vedio. 

Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

This pipeline detects well on the first project_vedio.mp4 even on some frames with suddenly brightness change. So it meets the basic requirement of this project. 

For another challenge_vedio.mp4 and harder_challenge_vedio.mp4, this pipeline's performance is not satisfactory. It works from start to end in challenge_vedio.mp4 but with distorted lane boundary. For the harder_challenge_vedio.mp4, it even can not detect any lane for a long time. 

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First a simple sanity code are used here to only check:

1 whether left_fit and right_fit is empty or not.

2 whetehr A are possible are negative or not.  

A more detail sanity codes should be used here to gurantee detects are reasonalbe or not.

For these failure detects in challenge and harder challenger vedio, more robust color transform and flexible mask region should be used here to improve pipeline performance. 
