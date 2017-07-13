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
[image4]: ./examples/image.png "Pre-Process Images"
[timage1]: ./german_images/60.jpg "Traffic Sign: 60 km/h"
[timage2]: ./german_images/wild_animals.jpg "Traffic Sign: wild animal crossing"
[timage3]: ./german_images/give_way.jpg "Traffic Sign: yield"
[timage4]: ./german_images/kinder.jpg "Traffic Sign: childrens crossing"
[timage5]: ./german_images/stop.jpg "Traffic Sign: stop sign"
[eimage1]: ./other_images/beware_ice_snow.jpg "Beware Ice/Snow"
[eimage2]: ./other_images/no_entry.jpg "No Entry"

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
The best combinations I found was to apply a random transformation of each image and use a contrast
normalization conversion. For example, here is an original image with various pre-processing 
techniques that I used:

![orginal][image4]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model was very close to a LeNet Model and consisted of the following layers:

```
| Layer                         |     Description                                |
|:-----------------------------:|:-----------------------------------------------:
| Input                         | 32x32x3 RGB Image                              |
| Convolution                   | 1x1 stride, same padding, outputs 32x32x32     |
| RELU                          |                                                |
| Max Pooling with dropout      | 2x2 stride, outputs 16x16x32                   |
| Convolution                   | 1x1 stride, same padding, outputs 16x16x128    |
| RELU                          |                                                |
| Max Pooling                   | 4x4 stride, same padding, outputs 4x4x128      |
| Flatten                       | outputs 2048                                   |
| Fully Connected               | outputs 128                                    |
| RELU with dropout             |                                                |
| Fully Connected               | outputs 128                                    |
| RELU with dropout             |                                                |
| Fully Connected               | outputs class labels                           |
```
				
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 64 and used 30 epochs, where the accuracy stablized to a 
consistent value. Experimenting with different hyperparameters I found the best model to have:
* learning rate = 0.0007

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of stabilizing between 96.5 and 97.5% 
* test set accuracy of 96.3%

I first started out with the LeNet-5 architecture shown in the convolutional neural network module. I wanted to start with the simplest model possible, before
moving to achitecutures that were more complex. Initially this architecture did not give good accuracy, so I next focused on adding dropouts in the layer 1 and
2 pooling and RELU activation layers 3 and 4. I found the best combination for adding dropouts was in the first pooling layer and layers 3 and 4 RELUs. I started with
a learning rate of 0.001, to improve accuracy I kept on decreasing the learning rate to finally using 0.0007. Since this is my first classifier in deep learning I
used the LeNet-5 architecture as a design choice, because I wanted to see if I could get the desired accuracy above 93% with this basic model. The next model
architecture I was going to try was the ZF Net 7 layer model if LeNet-5 couldn't reach the threshold. By adding dropouts and tuning the learning rate I got above 
the 93% threshold so the LeNet-5 model proved good for the project. 
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![60 km/h][timage1] ![wild animal crossing][timage2] ![yield][timage3] 
![childrens crossing][timage4] ![stop][timage5]

Some qualities that might cause my model to misclassify for these German traffic signs I found on the web, could be the writing below
the childrens crossing and the brightness and contrast of the signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			| Prediction	        					
|:---------------------:|:---------------------------------------------:
| 60 km/h      		| 60 km/h   						
| Wild Animal Crossing 	| Wild Animal Crossing 									
| Yield                 | Yield											
| Childrens Crossing	| Beware of Ice and Snow					 				
| Stop			| No Entry      							


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. Given that the accuracy of the captured images is
60% while it was 96.3% on the testing set, seems to point to the conclusion that the model is overfitting. Both signs that were missclassified could
be because of word classification embedded in the signs. The childrens crossing sign in the example did have a name under the sign (Ice and Snow signs have different words underneath the sign such as ICE or SNOW). The stop sign had the word STOP (No Entry is a long white stripe in the sign). One can see examples of these type of signs below: 

![Beware Ice/Snow][eimage1] ![No Entry][eimage2]

This gives a compelling need to have tagged understandings of words
embedded in images for better classification of traffic signs. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 25th cell of the Ipython notebook. The top 5 softmax probabilities for 
each image were:
```
Image 0, Top 5 Probabilities: [ 1.  0.  0.  0.  0.]
Image 1, Top 5 Probabilities: [ 1.  0.  0.  0.  0.]
Image 2, Top 5 Probabilities: [ 1.  0.  0.  0.  0.]
Image 3, Top 5 Probabilities: [ 1.  0.  0.  0.  0.]
Image 4, Top 5 Probabilities: [ 1.  0.  0.  0.  0.]
```
