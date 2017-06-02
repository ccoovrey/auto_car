# **Finding Lane Lines on the Road** 
When we drive a car, we use the lane lines of the road to guide us on our journey. These
lines are a constant reference for the driver to stay in the lane of the highway. These
lane lines can also be used as a reference for autonomous vehicles. In this project
I am going to use different computer vision algorithms to automatically detect
the lane lines of a highway. This project utilizes the Python library **opencv** (Open
Source Computer Vision). In this project, video images of a car traveling
down a highway are joined with an algorithm that draws red lines on these lane lines. The
algorithm does the following:
* the video color image is transformed into grayscale
* the video is smoothed using Gaussian Blurring
* canny edge detection is used on the video to show the edges inside the image
* hough transform converts the image into a hough space so as to search for 
  lines in the image that potentially could be lane lines
* lane lines in the image are made thicker and extrapolated to the bottom of 
  the image

[start_image]: ./test_images/solidWhiteRight.jpg 
[canny_image]: ./test_images/canny_solidWhiteRight.jpg 
[extrap_lines]: ./test_images/extrap_lines_solidWhiteRight.jpg 
[final_image]: ./test_images/final_solidWhiteRight.jpg 

## Lane Line Detection Algorithm
The lane line detection algorithm can be seen here as an image is transformed. We start
with the color image of drivers view:

![starting image][start_image]

### Canny edge detection
John Canny developed this algorithm in 1986. The goal of the algorithm is to identify
the edges of an object in an image. The algorithm:
* calculates the gradient of each pixel in the image
* detects edges by their shape from these gradients

Here we can see the image transformed using canny edge detection with Gaussian Blurring:

![Canny Image][canny_image]

### Hough transform
In 1962 Paul Hough devised a method for representing lines in a new space known as 
**hough space**. For example, if we had a line in two dimensional space. We could transform
this line into a space represented by its slope and intercept (hough space). A line in
two dimensional space is transformed into a point in hough space. We use a hough space
represented in polar coordinates (rho and theta). 

### Extrapolating Raw Hough Lines
It would be good to extend these raw hough lines into lines that represent the drive
lane for the vehicle. From the slope of these lines, one can deduce if it is a right
or left lane line. We can fit a line to these set of raw hough lines for each of 
the left and right lanes. We can also use a running average of these fitted lines
and use this to extend the right and left lane lines:

![Extrapolating Raw Hough][extrap_lines]

Here we have the final transformed image joined with the initial image using linear
blending:

![Final Image][final_image]

[prob_video]: ./test_videos/final_solidYellowLeft.mp4 

## Video Image
An example of video transformation can be seen in the following videos in the directory
"test_videos":
* initial video: solidWhiteRight.mp4
* transformed video: final_solidWhiteRight.mp4

## Potential shortcoming of pipeline
One can see in this transformed video image that the lane line
wobbles in some frames:
* initial video: solidYellowLeft.mp4
* transformed video: final_solidYellowLeft.mp4

One possible improvement would be to monitor the change in the slope
of each of the lane lines and when this change of slope breaks a certain threshold
with either of the lane lines take a more conservative change in slope.
