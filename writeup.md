
## Vehicle Detection Project

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
[image6]: ./output_images/window_example.jpg
[image7]: ./output_images/example1.jpg
[image8]: ./output_images/example2.jpg
[image9]: ./output_images/example3.jpg
[image10]: ./output_images/example4.jpg
[image11]: ./output_images/example5.jpg
[image12]: ./output_images/1.jpg
[video1]: ./output_images/example3.jpg


### Histogram of Oriented Gradients (HOG)

#### 1. HOG Features

I started by reading in all the `vehicle` and `non-vehicle` images, which is implemented in line 127 through 141 in `Classifier.py`. Here is an example of one of each of the `vehicle` and `non-vehicle` classes: 

![alt text][image1]

There are five parameters in `HOG_features()` for extracting of features: `color_space`, `orient`, `pix_per_cell`,  `cell_per_block`, and `hog_channel`. I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Here is an example using the `RGB` color space and HOG parameters of `orient=9`, `pix_per_cell=8` and `cell_per_block=2`, `hog_channel=0`:

![alt text][image2]

#### 2. Feature Extraction
For HOG extraction I tried different combination of parameters to check the HOG features of car and notcar image, which is implemented in function `get_hog_features()`: line 71 through 125 and function `get_hog_features()`: line 50 through 69 in `Classifier.py`. But it is difficult to estimate which one is better from HOG features. So I have also tried different parameter combination of HOG in the classifier part and chosen the combination with best test accuracy.
I compared the results with different color spaces and same other parameters. The results are as follows:

![alt text][image3]

I also tried different HOG channels, and same other parameters. The result with the third channel seems worse than other with other channels. The choose of channel will be decided next with the classification accuracy. The comparison results are in the following picture:

![alt text][image4]

Then I tried to vary the orientation, pixels_per_cell, cells_per_block. The last two HOG image of car image and notcar image are obtained with pix_per_block is equal to 4 and 16. It is obvious that the performance is not good as results with other parameters. The results with original parameters, with changed parameter cell_per_block=4 and cell_per_block=1 are quite small, whereas the results with the results with changed parameter orient=8 and cell_per_block=10 are a little different. But I can not decide which one is better. The results are as follows:
![alt text][image5] 

Except HOG features I also used color histogram features (in function `color_hist()`: line 39 through 47 in `Classifier.py`), spatial binning features (in function `bin_spatial()`: line 33 through 38 in `Classifier.py`).

#### 3. Classifier
At first, I obtained all the features of car images and notcar images. Then I created a labels vector to save the expected results, which means, the result is 1 if this is car image, and the result is 0 if this is notcar image. After that, I shuffled all the data, slit it to train data and test data. Then `StandardScaler` implements the Transformer to computer the mean and standard deviation on the training fetures. And this transformation is also applied to test features. The code of this part is implemented in line 148 through 180 in `Classifier.py`.
After I obtained the transformed train features and test features, I used a linear SVC to train a classifier. The results of different combination of parameters are as follows:

| CHALLEL        		|     Color Space    | orientation   |  pix_per_cell   | cell_per_block   |Training Accuracy	| Test Accuracy    |
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| '0'     		|     RGB        					|9   | 8   |2  |0.988	| 0.919  |
| 'ALL'     		|     RGB        				|9   | 8   |2  |0.988 | 0.972  |
| 'ALL'     		|     HSV      					|9   | 8   |2  |0.970	| 0.932  |
| 'ALL'     		|    YCrCb     					|9   | 8   |2  |1.0	| 0.982  |
| 'ALL'     		|    YCrCb     					|10   | 8   |2  |1.0	| 0.990  |

After the test accuracy reaches 0.99, I stoped to try other combinations. Therefore the parameters for classifier are: YCrCb color space, orientation=10, pix_per_cell=8, cell_per_block=2, All channels.

### Sliding Window Search

For the sliding windows I have chosen 4 different sizes. The vehicle that appears smaller, will be near the horizon. Therefore for the larger window size, the overlap will be smaller. The code of this part is implemented in line 236 through 256 and function `slide_window()`in 122 through 158 in `car_detection.py`.

| Number  |   x_start_stop        | y_start_stop   |  window size   | overlap       |
|:-------:|:-------------:|:-------------:|:-------------:|:-------------:|
| 1  		|        [600, 1280]  					|[400, 464]    |  64 |0.85  |
| 2     |     [500, 1280]      				|[400, 480]  | 80 | 0.8  |
| 3     |     [500, 1280]     					|[400, 612]  |96	| 0.7  |
| 3     |     [None, None]     					|[400, 680]  |128	| 0.5  |

![alt text][image6]


### Heatmap
At first, I obtained the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. Then I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap and constructed bounding boxes to cover the area of each blob detected.  The code of this part is implemented in line 191 through 223 with function `add_heat()`, `apply_threshold` and `draw_labeled_bboxes` in 122 through 158 in `car_detection.py`.

Here are five test images with positive detections and their corresponding heatmaps:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

Here is the example with the resulting bounding boxes are drawn onto a test image:
![alt text][image12]

### Video Implementation

Here's a [https://github.com/Hailey0528/CarND-Vehicle-Detection/blob/master/project_video.mp4](./project_video.mp4)
---

### Discussion

In this project, At first I learned to extrat features: spatial binning of color, gradient features, HOG features. Based on the features of car images and notcar images I used support vector machines method to train a classifier. The test accuracy of the classifier reaches 99%. But even the test accuracy of classifier is very good, the prediction accuracy of new images with sliding windows is not high. There are many false positives. And the sliding window has huge effect on the results. I have to say this method is time consuming but not robust. With the predicted pisitive detections, I created a heatmap and then threshold that map to identify the vehicle positions. But the results are still not that good. The classifier predicts some shadow image as a car. And I also tried to average the positive windows of last 15 frames, the result is quite the same.
