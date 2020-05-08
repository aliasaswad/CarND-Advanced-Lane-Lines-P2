## Advanced Lane Line Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
![Lanes Image](./examples/example_output.jpg)

In this project, my goal is to write a software pipeline to identify the lane boundaries in a video. A detailed writeup of the project is provided to givie more insight about the steps we went through to achive or goal in the project.

## Overview
---
Detect lane lines on any road is a somthing common that performed by all drivers to keep the vehicles within the lane lines. That will insure safe traffic and minimim collisions. In this project we will discuss how to apply differents techniques to identify and plot the inside road lanes lines.


The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames. The folder called `output_images`, contain the output from each stage of your pipeline. The video called `project_video.mp4` is the video tat my pipeline worked on.  

The `challenge_video.mp4` video is an extra (and optional) challenge to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

## Conclusion
---

To conclude, we should metion to the the importance of the computer vision process enables since it is the eyes of self-driving cars to make sense of their surroundings.


## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

