[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

<!-- 





output_images/S_channel_1.jpg
output_images/S_channel_2.jpg
output_images/w_y_hls_1.jpg
output_images/w_y_hls_2.jpg

output_images/sobel_origin.jpg
output_images/sobel_abs_x.jpg
output_images/sobel_abs_y.jpg
output_images/sobel_mag.jpg
output_images/sobel_dir.jpg
output_images/sobel_combin.jpg

output_images/color_grad_combin.jpg
output_images/sobel_combin.jpg
output_images/aoi.jpg
output_images/pers_trans_org.jpg
output_images/pers_trans_curv.jpg
output_images/pers_trans_str_line.jpg

output_images/all_thresh_org.jpg
output_images/all_thresh_binary.jpg

output_images/all_thresh_org.jpg
output_images/all_thresh_binary_pers_trans.jpg

output_images/histogram_1.jpg
output_images/histogram_2.jpg

output_images/detect_lane_pix_and_fit_curv
output_images/detect_lane_pix_and_fit_straight

output_images/det_curv_lane_org.jpg
output_images/det_curv_lane_draw_laneline.jpg

./output_images/project_video_frame_2.0.jpg
./output_images/project_video_frame_14.0.jpg
./output_images/project_video_frame_21.0.jpg
./output_images/project_video_frame_24.0.jpg
./output_images/project_video_frame_29.0.jpg
./output_images/project_video_frame_39.0.jpg
./output_images/project_video_frame_41.0.jpg
./output_images/project_video_frame_45.0.jpg

./output_images/project_video_frame_out_2.0.jpg
./output_images/project_video_frame_out_14.0.jpg
./output_images/project_video_frame_out_21.0.jpg
./output_images/project_video_frame_out_24.0.jpg
./output_images/project_video_frame_out_29.0.jpg
./output_images/project_video_frame_out_39.0.jpg
./output_images/project_video_frame_out_41.0.jpg
./output_images/project_video_frame_out_45.0.jpg
 -->

<!-- <table>
    <tr>
        <td>
            <img src='./examples/undistort_output.png' alt="Drawing" style="width: 10px;">
        </td>
        <td>
            <img src='./test_images/test1.jpg' alt="Drawing" style="width:10px;">
        </td>
    </tr>
</table> -->

<!-- <img src="./examples/undistort_output.png" width="125"/> <img src="./test_images/test1.jpg" width="125"/>  -->

 
# **Advanced Lane Finding Project**

<img src="./output_images/project_video_frame_out_45.0.jpg" width="600"/>

---

The goals / steps of this project is to build a Lane Finding Pipeline as the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


---

### Camera Calibration

#### 1. The below description is how I computed the camera matrix and distortion coefficients with an example of a distortion corrected calibration image.

```python
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2) #that will shape coordinate back to two colomns x and y
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
# Step through the list and search for chessboard corners
for idx, fname in enumerate(images): #idx start from 0 to len(images)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
```

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


|<img src="./output_images/calibr_1.jpg" width="400"/> <img src="./output_images/calibr_2.jpg" width="400"/> 
|:--:| 
|*Original Image  _________________________    Undistoted Image*|

<!-- ![alt-text-1](./output_images/calibr_1.jpg "title-1") ![alt-text-2](./output_images/calibr_2.jpg "title-2") -->
<!-- *hello* -->



<!-- ![alt text][image1] -->

### Pipeline (single images)

#### 1. Distortion-corrected image.
----

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

|<img src="./output_images/calibr_1.jpg" width="400"/> <img src="./output_images/undist_2.jpg" width="400"/> 
|:--:| 
|*Original test image  _________________________    Undistoted test image*|

<!-- ![alt text][image2] -->

So, when we apply the distortion correction on out test images the result will be as below:

|<img src="./output_images/undist_test_2.jpg" width="400"/> <img src="./output_images/undist_test_1.jpg" width="400"/> 
|:--:| 
|*Original image  _________________________    Undistoted image*|


#### 2. Color transforms, gradients or other methods to create a thresholded binary image. 
----

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `P2.py`).  
A color space is a specific organization of colors; color spaces provide a way to categorize colors and represent them in digital images.

RGB is red-green-blue color space. You can think of this as a 3D space, in this case a cube, where any color can be represented by a 3D coordinate of R, G, and B values. For example, white has the coordinate (255, 255, 255), which has the maximum value for red, green, and blue.

I splitted up the image to three channels as below:

```python
R = undistorted_test_image[:,:,0]
G = undistorted_test_image[:,:,1]
B = undistorted_test_image[:,:,2]
```

Here's an example of my output for this step.

|<img src="./output_images/rbg_test_1.jpg" width="250"/> <img src="./output_images/rbg_test_2.jpg" width="250"/> <img src="./output_images/rbg_test_3.jpg" width="250"/>
|:--:| 
|*R-Channel____________________G-Channel______________________B-Channel*|

Then I repeated the smae for HLS channels

```python
hls = cv2.cvtColor(undistorted_test_image, cv2.COLOR_RGB2HLS)
H_channel = hls[:,:,0]
L_channel = hls[:,:,1]
S_channel = hls[:,:,2]
```
Here's an example of my output for this step.

|<img src="./output_images/hls_test_1.jpg" width="250"/> <img src="./output_images/hls_test_2.jpg" width="250"/> <img src="./output_images/rbg_test_3.jpg" width="250"/>
|:--:| 
|*H-Channel____________________L-Channel______________________S-Channel*|

From these examples, you can see that the R channel does a reasonable job of highlighting the lines and the **S channel** is probably the best bet to picks up the lines well.

To identify White and Yellow Color in the Images we need to apply the below code

```python
def get_y_w_hls_images(img):
    
    hls_image =  cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # White isolated from HLS components
    w_hls = np.zeros_like(hls_image[:,:,0])
    w_hls[((hls_image[:,:,0] >= 0)   & (hls_image[:,:,0] <= 255))
        & ((hls_image[:,:,1] >= 210) & (hls_image[:,:,1] <= 255))
        & ((hls_image[:,:,2] >= 0)   & (hls_image[:,:,2] <= 255))] = 1
    # Yellow isolated from HLS components
    y_hls = np.zeros_like(hls_image[:,:,0])
     y_hls[((hls_image[:,:,0] >= 20)  & (hls_image[:,:,0] <= 30))
        & ((hls_image[:,:,1] >= 20)  & (hls_image[:,:,1] <= 200))
        & ((hls_image[:,:,2] >= 90) & (hls_image[:,:,2] <= 255))] = 1
    # OR yellow with white
    hls_w_y = np.zeros_like(hls_image[:,:,0])
    hls_w_y[(w_hls == 1) | (y_hls == 1) ] = 1
```

|<img src="./output_images/w_y_hls_1.jpg" width="400"/> <img src="./output_images/w_y_hls_2.jpg" width="400"/> 
|:--:| 
|*Undistorted Image  _________________________ W_Y Color Thresholded Image*|


#### 3. Applying Sobel Thresholding
----
**Calculate the absolute threshold value for x and y**

Calculate the derivative in the xx and yy direction (the 1, 0 at the end denotes xx direction) and (the 0, 1 at the end denotes yy direction).

Pass in img and set the parameter orient as 'x' or 'y' to take either the xx or yy gradient. Set thresh_min, and thresh_max to specify the range to select for binary output. We can use exclusive (<, >) or inclusive (<=, >=) thresholding.
```python
# Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
# Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
```
***NOTE:*** YOur output should be an array of the same size as the input image. The output array elements should be 1 where gradients were in the threshold range, and 0 everywhere else.
Here's an example of my output for this step:

|<img src="./output_images/sobel_origin.jpg" width="280"/> <img src="./output_images/sobel_abs_x.jpg" width="280"/> <img src="./output_images/sobel_abs_y.jpg" width="280"/> 
|:--:|
|*Original image____________________Abs_Sobel_X______________________Abs_Sobel_Y*|


**Magnitude and Direction of the Gradient**
The magnitude, direction, or absolute value, of the gradient is just the square root of the squares of the individual x and y gradients. For a gradient in both the xx and yy directions, the magnitude is the square root of the sum of the squares.

It's important to note here that the kernel size should be an odd number.
For the direction sobel:
```python
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude 
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1
```
For the magnitude sobel:

```python
Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
```

Here's an example of my output for this step.

|<img src="./output_images/sobel_origin.jpg" width="280"/> <img src="./output_images/sobel_mag.jpg" width="280"/> <img src="./output_images/sobel_dir.jpg" width="280"/> 
|:--:|
|*Original image____________________Magnitude Sobel______________________Direction Sobel*|

#### Direction sobel outout is extremely noisy. So, we need to combin sobels as below

```pyhton
def combin_sobel(abs_sobel_x, abs_sobel_y, mag_sobel, dir_sobel):
    combin = np.zeros_like(dir_sobel)
    # Sobel X returned the best output so we keep all of its results. We perform a binary and on all the other sobels    
    combin[(abs_sobel_x == 1) | ((abs_sobel_y == 1) & (mag_sobel == 1) & (dir_sobel == 1))] = 1
    return combin
```
|<img src="./output_images/sobel_origin.jpg" width="400"/> <img src="./output_images/sobel_combin.jpg" width="400"/> 
|:--:| 
|*Original image_________________________ Combined Sobel*|

#### Color and Gradient Thresholding

Now it's time to combine what you know about color and gradient thresholding to get the best of both worlds.

```python
# Stack each channel to view their individual contributions in green and blue respectively
# This returns a stack of the two binary images, whose components you can see as different colors
color_binary = np.dstack((np.zeros_like(comb_sobel), comb_sobel, undistorted_y_w_hls)) * 255
color_binary = color_binary.astype(np.uint8)
```
|<img src="./output_images/sobel_origin.jpg" width="280"/> <img src="./output_images/color_grad_combin.jpg" width="280"/> <img src="./output_images/sobel_combin.jpg" width="280"/> 
|:--:|
|*Original image____________________Stacked thresholds___________________Combined S-channel and gradient thresh.*|


#### 4. Apply a perspective transform to rectify binary image ("birds-eye view").
----

Draw polyline and Fit a polynomial
```pyhton
(width, length) = (test_image.shape[0] - 1, test_image.shape[1] - 1) 
aoi = np.array([[210,width],[595,450],[690,450], [1110, width]], np.int32)
cv2.polylines(test_image,[aoi],True,(255,0,0), 10)
```
|<img src="./output_images/aoi.jpg" width="500"/>  
|:--:|
|*AOI*|

After applying calibration, and thresholding, we should then appy a perspective transform to a road image as below:

```pyhton
# Warp an image using the perspective transform, M
def get_perspective_transform(img, src, dst):   
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped
```

And here's the example:

|<img src="./output_images/pers_trans_org.jpg" width="280"/> <img src="./output_images/pers_trans_curv.jpg" width="280"/> <img src="./output_images/pers_trans_str_line.jpg" width="280"/> 
|:--:|
|*Original image____________________Pres. trans. for curve___________________Pres. trans. for straight line*|

Then we will apply the conbined thresholds with prespective transfor on the images as below:

```python
def combine_allthresh_and_transform(img):
    pers_img = get_perspective_transform(img, src, dst)
    binary_img = get_all_thresh_binary_image(pers_img)
    return binary_img
```
|<img src="./output_images/all_thresh_org.jpg" width="280"/> <img src="./output_images/all_thresh_binary.jpg" width="280"/> <img src="./output_images/all_thresh_binary_pers_trans.jpg" width="280"/> 
|:--:|
|*Original image____________________All Thresh___________________All Thresh and Pres. Trans*|

#### 5. Detect Lane pixels and Determine the curvature
----
Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

The details about hpw I did this as below.

After applying calibration, thresholding, and a perspective transform to a road image, we should have a binary image where the lane lines stand out clearly. However, we still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

Plotting a histogram of where the binary activations occur across the image is one potential solution for this.

```python
def hist(binary_warped):
    # Lane lines are likely to be mostly vertical nearest to the car
    # Sum across image pixels vertically
    # The highest areas of vertical lines should be larger values
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    histogram = np.dstack((binary_warped, binary_warped, binary_warped))*255
    return histogram
# Create histogram of image binary activations
histogram = hist(binary_warped)
```

|<img src="./output_images/histogram_1.jpg" width="400"/> <img src="./output_images/histogram_2.jpg" width="400"/> 
|:--:| 
|*Perspective Transform______________________________ Histogram*|

#### Implement Sliding Windows and Fit a Polynomial
As shown in the previous animation, we can use the two highest peaks from our histogram as a starting point for determining where the lane lines are, and then use sliding windows moving upward in the image (further along the road) to determine where the lane lines go.
***Set up windows and window hyperparameters***
Our next step is to set a few hyperparameters related to our sliding windows, and set them up to iterate across the binary activations in the image. We have some base hyperparameters below

```python
# HYPERPARAMETERS
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Set height of windows - based on nwindows above and image shape
window_height = np.int(binary_warped.shape[0]//nwindows)
# Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated later for each window in nwindows
leftx_current = leftx_base
rightx_current = rightx_base
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []
```
***Iterate through nwindows to track curvature***
Now that we've set up what the windows look like and have a starting point, we'll want to loop for nwindows, with the given window sliding left or right if it finds the mean position of activated pixels within the window to have shifted.

Then we have to fit a polynomial. We have found all our pixels belonging to each line through the sliding window method, it's time to fit a polynomial to the line. First, we have a couple small steps to ready our pixels.

```python
# Concatenate the arrays of indices (previously was a list of lists of pixels)
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)
# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
# Assum we have `left_fit` and `right_fit` from `np.polyfit` before
# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```
|<img src="./output_images/detect_lane_pix_and_fit_curv.jpg" width="400"/> <img src="./output_images/detect_lane_pix_and_fit_straight.jpg" width="400"/> 
|:--:| 
|*Detect lane lane pixels for curve______________________________Detect lane lane pixels for straight line*|

<!-- This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
 -->

#### 6: Warp the detected lane boundaries back onto the original image.
----

The function below I used to plot on the image such that the lane area is identified clearly

```python
def draw_poly(img, left_fit, right_fit):
    
    # Calculate points.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_pnts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_pnts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((left_pnts, right_pnts))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    return cv2.addWeighted(img, 1, newwarp, 0.3, 0)
```

|<img src="./output_images/det_curv_lane_org.jpg" width="400"/> <img src="./output_images/det_curv_lane_draw_laneline.jpg" width="400"/> 
|:--:| 
|*Undistorted image_____________________________Image with lane line*|


#### 7. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position
---
We've calculated the radius of curvature based on pixel values, so the radius we are reporting is in pixel space, which is not the same as real world space. So we actually need to repeat this calculation after converting our x and y values to real world space.

```python
def visual_lane_cur_pos(img, fontScale=2):
    # get the left and right fit to calculate the curvature 
    left_fit, right_fit, left_fit_m, right_fit_m = get_fit_and_lanes(img)
    output = draw_poly(img, left_fit, right_fit)
    # Calculate curvature
    l_curvature = get_curvature(y_val, left_fit_m) 
    r_curvature = get_curvature(y_val, right_fit_m)
    
    # Calculate vehicle center
    max_x = img.shape[1]*xm_per_pix
    max_y = img.shape[0]*ym_per_pix
    center = max_x / 2
    l_line = left_fit_m[0]*max_y**2 + left_fit_m[1]*max_y + left_fit_m[2]
    r_line = right_fit_m[0]*max_y**2 + right_fit_m[1]*max_y + right_fit_m[2]
    m_line = l_line + (r_line - l_line)/2
    diff = m_line - center
    if diff > 0:
        message = '{:.2f} m right'.format(diff)
    else:
        message = '{:.2f} m left'.format(-diff)
```
Original image                                     Visual lane current position

<img src="./output_images/project_video_frame_2.0.jpg" width="400"/> <img src="./output_images/project_video_frame_out_2.0.jpg" width="400"/>

<img src="./output_images/project_video_frame_14.0.jpg" width="400"/> <img src="./output_images/project_video_frame_out_14.0.jpg" width="400"/>

<img src="./output_images/project_video_frame_21.0.jpg" width="400"/> <img src="./output_images/project_video_frame_out_21.0.jpg" width="400"/>

<img src="./output_images/project_video_frame_24.0.jpg" width="400"/> <img src="./output_images/project_video_frame_out_24.0.jpg" width="400"/>

<img src="./output_images/project_video_frame_29.0.jpg" width="400"/> <img src="./output_images/project_video_frame_out_29.0.jpg" width="400"/>

<img src="./output_images/project_video_frame_39.0.jpg" width="400"/> <img src="./output_images/project_video_frame_out_39.0.jpg" width="400"/>

<img src="./output_images/project_video_frame_41.0.jpg" width="400"/> <img src="./output_images/project_video_frame_out_41.0.jpg" width="400"/>

<img src="./output_images/project_video_frame_45.0.jpg" width="400"/> <img src="./output_images/project_video_frame_out_45.0.jpg" width="400"/>




### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
