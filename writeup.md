## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar_example.jpg
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/Compare_Color.jpg
[image4]: ./output_images/Compare_Channel.jpg
[image5]: ./output_images/Compare_HOG.jpg
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. HOG Features

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

There are five parameters in `HOG_features()` for extracting of features: `color_space`, `orient`, `pix_per_cell`,  `cell_per_block`, and `hog_channel`. I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Here is an example using the `RGB` color space and HOG parameters of `orient=9`, `pix_per_cell=8` and `cell_per_block=2`, hog_channel():

![alt text][image2]

#### 2. Feature Extraction

I compared the results with different color spaces and same other parameters. The second HOG image with HSV and the forth image with HLS are seem worse than with other color spaces. I just chose the first one RGB color space. The results are as follows:

![alt text][image3]

I also tried different HOG channels, and same other parameters. The result with the third channel seems worse than other with other channels. The choose of channel will be decided next with the classification accuracy. The comparison results are in the following picture:

![alt text][image4]

Then I tried to vary the orientation, pixels_per_cell, cells_per_block. The last two HOG image of car image and notcar image are obtained with pix_per_block is equal to 4 and 16. It is obvious that the performance is not good as results with other parameters. The results with original parameters, with changed parameter cell_per_block=4 and cell_per_block=1 are quite small, whereas the results with the results with changed parameter orient=8 and cell_per_block=10 are a little different. But I can not say which one is better. And I just use the original parameter. The results are as follows:
![alt text][image5]


#### 3. Classifier

At first, I obtained all the features of car images and notcar images. Then I created a labels vector to save the expected results, which means, the result is 1 if this is car image, and the result is 0 if this is notcar image. After that, I shuffled all the data, slit it to train data and test data. Then `StandardScaler` implements the Transformer to computer the mean and standard deviation on the training fetures. And this transformation is also applied to test features. 
After I obtained the transformed train features and test features, I used a linear SVC to train a classifier. 

| CHALLEL        		|     Color Space    | orientation   |  pix_per_cell   | cell_per_block   |Training Accuracy	| Test Accuracy    |
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| '0'     		|     RGB        					|9   | 8   |2  |0.988	| 0.919    |
| 'ALL'     		|     RGB        				|9   | 8   |2  |0.988 	| 0.921   |
| 'ALL'     		|     HSV      					|9   | 8   |2  | 0.970	| 0.894   |
| 'ALL'     		|     LUV      					|9   | 8   |2  | 0.990 	| 0.914 |
| 'ALL'     		|     HLS      					|9   | 8   |2  | 	0.968| 0.898 |
| 'ALL'     		|     YUV      					|9   | 8   |2  |0.990 	| 0.908   |
| 'ALL'     		|    YCrCb     					|9   | 8   |2  | 0.992	| 0.921  |
| 'ALL'     		|     RGB        					|10  | 8   |2  |	0.996| 0.914  |
| 'ALL'     		|     RGB        					|10   | 8   |1  | 	|  |
| 'ALL'     		|    YUV        					|10   | 8   |2  |0.996 	| 0.911   |



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decide to use different level windows series

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

