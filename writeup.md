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

[image1]: ./report_images/calibration.jpg "Undistorted"
[image2]: ./report_images/undistortion.jpg "Road Transformed"
[image3]: ./report_images/thresh_bin_img.jpg "Binary Example"
[image4]: ./report_images/thresh_bin_img2.jpg "Binary Example2"
[image5]: ./report_images/region1.jpg "Perspective transform straight"
[image6]: ./report_images/region2.jpg "Perspective transform curved"
[image8]: ./report_images/pipeline.jpg "Pipeline"
[image9]: ./report_images/pipeline_area1.jpg "Pipeline area"
[image11]: ./report_images/pipeline1_3.jpg "Pipeline 1-3"
[image12]: ./report_images/pipeline4_6.jpg "Pipeline 4-6"
[video1]: ./video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first section called "Camera calibration" of the IPython notebook `lane_finding.ipynb`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function from OpenCV library.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is an example of image distortion correction for one of the test images:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The main routines are in the "Image processing pipeline" sectin of `lane_finding.ipynb` file.

Image thresholding is the logical disjunction (pythonic `|`) of the following:

* LAB color space turned out to be very good for yellow color thresholding, so the steps are the following (`channel_thresholding(rgb2lab(img), ch=2, thresh=(150, 255))` function in the code):
    * Convert image to LAB color space.
    * Threshold image based on B channel values: **H: 150 - 255**
* White color thresholding (`white_select(...)` function in the code):
    * Convert image to HLS color space.
    * Determine the range of H, L, and S components for white color: **H: 0 - 360, L: 210 - 255, S: 0 - 255.** 
    * Threshold image based on these values
* Sobel gradient thresholding for x coordinate (`sobel_abs(...)` function in the code).

Here's an example of the sequential application of methods above:

![alt text][image3]

The original undistorted picture is on the left, next we see the result of application of `yellow_select(...)` only, then we add white color thresholding and picture on the right is the result of all thresholding methods combined together (via logical "or"). 

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the function called `perspective(img, mode='f', scr=src, dst=dst):`. This function takes `img` as an input and, based on `mode`, returns inverse or forward perspective transform of this image. The source(`src`) and destination(`dst`) points were chosen in the following manner:

```python
src = np.float32([[0, img_size[1]], 
                  [575, 450],
                  [705, 450],
                  [img_size[0], img_size[1]]])
dst = np.float32([[100, img_size[1]],
                  [100, 0],
                  [img_size[0]-100, 0], 
                  [img_size[0]-100, img_size[1]]])
```
This resulted in the following source and destination points:

| Source (x, y)       | Destination (x, y)  | 
|:-------------:|:-------------:| 
| (0, 720)      | (100, 720)    | 
| (575, 450)    | (100, 0)      |
| (705, 450)    | (1180, 0)     |
| (1280, 720)   | (1180, 720)   |

Then the function calculates transform matrix and applies it to the image using OpenCV routines `getPerspectiveTransform` and `cv2.warpPerspective`:

```python
def perspective(img, mode='f', scr=src, dst=dst):
    
    img_size = (img.shape[1], img.shape[0])
   
    if mode == 'f':
        M = cv2.getPerspectiveTransform(src, dst)
    elif mode == 'inv':
        M = cv2.getPerspectiveTransform(dst, src)
        
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here is an example for straight lines:

![alt text][image5]

This example shows the perspective transform for curved lines:

![alt text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The line fitting algortihm combines convolution method and method of histogramms described in the lessons, and based on 3 key steps:

* At the start the algorithm calculates convolution of the **50px window** with the bottom half of the image. It allows to calculate positions for regions of interest (RoI) for left and right lines. Here is how I do it for the left line:
```python
   # Sum quarter bottom of image to get slice
   l_sum = np.sum(image[int(img_y/2):,:center], axis=0)
   conv_l = np.convolve(w, l_sum, 'same')
   leftx_base = np.argmax(conv_l)
```

* Then it goes through the image, slice by slice ( 1 slice correspond to 1/9 of the image height ) and gather all the points that lie inside RoI (function `detect_lines()`). The position of the next region of iterest is calculated from the maximum of the convolution of the current region of interest. I am also filtering this parameter in order to increase robustness with respect to outliers - it helps to avoid situations when the position of the next RoI is shifted too much, that is not realistic for the monotone lane line. Here is the code, showing this idea (`alpha` - parameter of the filter):

```python
   img_layer = image[win_y_low:win_y_high, win_xleft_low:win_xleft_high]
   img_layer_hist = np.sum(img_layer, axis=0)
   conv_left = np.convolve(w, img_layer_hist, 'same')
   mean = np.argmax(conv_left) + win_xleft_low
   leftx_current = np.int(alpha*mean + (1-alpha)*leftx_current)
```

After lines are detected I use the fitting data to recalculate the RoI position at the next video frame. It is implemented in the function called `line_search()`

* Finally the algorithm fits second order polynomes to the data. However instead of `np.polyfit()` I decided to write my own function that does least squares fit with [regularisation](https://see.stanford.edu/materials/lsoeldsee263/07-ls-reg.pdf) - `fit2_rls(x, y, lam)`, where parameter `lam` defines the strength of regularisation. Using the regularisation allows introducing additional constraints to our problem, i.e. if A is our data matrix and y is the vector of measurements and we want to find x, such that ||Ax - y||^2 is minimised (the original formulation of data fitting problem), we can require some components of x to satisfy certain conditions. In the fitting problem, our x consists of 3 elements, and we know that the element corresponding to x^2 cannot be very large, otherwise the fitted curve would have very high curvature, which is impractical for roads (at least for roads on the provided videos). Here is the code that does it:

```python
def fit2_rls(x, y, lam):
    # the regularised problem looks like min||Ax-y||+||Bx - g||
    # here we rewrite it in the matrix form
    x = np.array(x)
    y = np.array(y)
    # add component to y (since in our case g = 0)
    y = np.concatenate((y, np.array([0])))
    # construct matrix A
    A = np.array([np.ones(x.shape[0]), x, x**2])
    # append matrix B = np.array([0, 0, lam]) in our case
    A = np.vstack((A, np.array([0, 0, lam])))
    # get the estimate and return it in the reversed order
    est = np.linalg.lstsq(a=A, b=y)[0]
    return est[::-1
```
Another reason to use regularisation is bad conditioning of fitting problem - i.e. it is quite sensitive to noise in the data. Also I have added [exponential smoothing](https://www.wikiwand.com/en/Exponential_smoothing) in order to increase robustness against detection errors:

```python
   left_fit = betha*np.polyfit(lefty, leftx, 2) + (1-betha)*left_fit
   right_fit = betha*np.polyfit(righty, rightx, 2) + (1-betha)*right_fit
```

The procedure above gives the following resulting fit:

![alt text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The procedure of curvature calculation is in lines 121-136 of `detect_lines()` function (see `lane_finding.ipynb`).

The goal is to calculate curvature of the line based on coefficients identified after least squares polinomial fit. However in order to give the answer in sensible measurement units we need to convert pixel values to meters. To find y-axis coefficient, I approximately measured the distance between two vertical lines (in perspective view) in pixels and given that the lane width is about 3.7 meters, the `ym_per_pix` value is about 3.7/750. The value for `xm_per_pix` can be identified similarly, given that the length of the dashed line is about 3 meters: we project the lane with very distinguishable dashed lines and count how many of them can fit into 720 pixels. I got about 8, so the coefficient is 24/720.

```python
# Define conversions in x and y from pixels space to meters
ym_per_pix = 24/720 # meters per pixel in y dimension
xm_per_pix = 3.7/750 # meters per pixel in x dimension
```
For the position identification I do a simple calculation that gives me offset of the vehicle in meters:

```python
offset = ((right_line_bottom + left_line_bottom)/2 - center)*xm_per_pix
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

The images for each stage of the pipeline are provided below:

![alt text][image11]
![alt text][image12]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Drawbacks of the current implementation:

* Gradient thresholding. Since the algortihm uses sobel gradient thresholding, it is sensitive to the fake lines on the road.
* Color dependence. In general any road lines have yellow or white color, but the algorithm is sensitive to bright and dark parts of the image anyway. In the current realisation I automatically decrease the filtering constant, but there should be better solution.
* Another drawback of the current solution is the need for properly chosen coordinates for perspective transformation. Also it assumes the flat road which is not always the case.
* The algortihm will have problems when there are many cars on the road in front of our car (i.e. traffic jam situation).

There are different ways to improve this project:

* Hidden Markov Model (HMM) approach for region of interest positioning. I started to think about it but due to time limitation decided to implement it a bit later. HMM approach uses the notion of states and transition probabilities between states. The idea is to treat the convolutions of image layers with window as the probability densities and different image layers as discrete states. Then the most probable path, corresponding to the line can be found using Viterbi algorithm. Also, based on probability densities we pick the points for fitting, this gives us better outlier rejection. The disadvantage here is the computational complexity that can be regulated by the number of (states) image layers.

* Adaptive thresholding. For example gradient is not useful if the image is very bright or dark, as well as S channel of HLS color space allows distinguishing lines on bright images, but requires threshold readjustment. It might be a good idea to make it dependant on image brightness.
