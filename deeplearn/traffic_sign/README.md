# **Traffic Sign Recognition Project** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/construction.png "Construction Sign"
[image2]: ./examples/training_labels.png "Training Labels"
[image3]: ./examples/train_test.png "Train/Test Distribution"
[image4]: ./examples/original.png "Orginal Image"
[image5]: ./examples/contrast.png "Contrast Image"
[timage1]: ./german_images/60.jpg "Traffic Sign: 60 km/h"
[timage2]: ./german_images/wild_animals "Traffic Sign: wild animal crossing"
[timage3]: ./german_images/give_way.jpg "Traffic Sign: yield"
[timage4]: ./german_images/kinder.jpg "Traffic Sign: childrens crossing"
[timage5]: ./german_images/stop.jpg "Traffic Sign: stop sign"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The data set has a set of traffic sign images from Germany. Here is 
an example of a traffic sign image, which is for construction:
![Construction Sign][image1]

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set are 43

#### 2. Include an exploratory visualization of the dataset.

Here is a scatterplot of the training dataset label observations. The
labels have anywhere from 200 to 2000 observations for each label.

![Training Labels][image2]

We can see that the training and test dataset have the same distribution shape.

![Train/Test Distribution][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

I looked at different techniques to convert the images: 
* no conversion
* min max normalization
* absolute normalization
* greyscale
* contrast normalization

To get a more generalized model, I also looked at applying different transformations of the image such as:
* scaling
* translating
* rotation

I experimented using the final model architecture and these different preprocess techniques on the images.
The best combinations I found was to apply a random transformation of each image and using a contrast
normalization conversion. For example, here is an original image:

![orginal][image4]

Converted by contrasted normalization:

![contrast][image5]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was very close to a LeNet Model and consisted of the following layers:
| Layer                         |     Description
|:-----------------------------:|:-----------------------------------------------:
| Input                         | 32x32x3 RGB Image
| Convolution                   | 1x1 stride, same padding, outputs 32x32x32
| RELU                          | 
| Max Pooling with dropout      | 2x2 stride, outputs 16x16x32
| Convolution                   | 1x1 stride, same padding, outputs 16x16x128
| RELU                          |
| Max Pooling with dropout      | 4x4 stride, same padding, outputs 4x4x128
| Flatten                       | outputs 2048
| Fully Connected               | outputs 128
| RELU with dropout             |
| Fully Connected               | outputs 128
| RELU with dropout             | 
| Fully Connected               | outputs class labels
										

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 64 and used 30 epochs, where the accuracy stablized to a 
consistent value. Experimenting with different hyperparameters I found the best model to have:
* mu = 0
* sigma = 0.1
* learning rate = 0.0007

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

I used the LeNet architecture. I wanted to start the simplest model possible, before moving to architectures that were
more complex. I obtained very good accuracy using the test set as evidence.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![60 km/h][timage1] ![wild animal crossing][timage2] ![yield][timage3] 
![childrens crossing][timage4] ![stop][timage5]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			| Prediction	        					
|:---------------------:|:---------------------------------------------:
| 60 km/h      		| Stop sign   						
| Wild Animal Crossing 	| U-turn 									
| Yield                 | Yield											
| Childrens Crossing	| Childrens Crossing					 				
| Stop			| Stop      							


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	| Prediction	        					
|:---------------------:|:---------------------------------------------:| 
| .60         		| Stop sign   						
| .20     		| U-turn 							
| .05			| Yield									
| .04	      		| Bumpy Road					 				
| .01		        | Slippery Road      							



