# **Behavioral Cloning with Deep Learning** 

---

**Behavioral Cloning Project**

The goals of this project are the following: In this project, deep learning is
used to drive a simulated car autonomously. The Python libary Keras is used to build
a Convolutional Neural Network (CNN) to predict the steering angle response to 
navigate a car in a simulated track.

The project consists of the following:
* model.py: script to train the model.
* drive.py: script to drive the car.
* /experiments: folder showing experiments iterating thru model and data design.
* model.h5: model neural network weights.
* behavior_clone.ipynb: ipython notebook showing final experiment.
* final_model.mp4: video showing autonomous driving using final model.

[//]: # (Image References)

[image2]: ./examples/center.jpg "Center Lane Driving"
[image3]: ./examples/recover1.jpg "Recovery Image"
[image4]: ./examples/recover2.jpg "Recovery Image"
[image5]: ./examples/recover3.jpg "Recovery Image"
[image6]: ./examples/center.jpg "Normal Image"
[image7]: ./examples/flipped_image.jpg "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

This project contains the following files:
* model.py: script to train the model.
* drive.py: script to drive the car.
* /experiments: folder showing experiments iterating thru model and data design.
* model.h5: model neural network weights.
* behavior_clone.ipynb: ipython notebook showing final experiment.
* final_model.mp4: video showing autonomous driving using final model.

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

The model is based on nvidia model. This model has the following 5 convolutional layers:
+ convolutional layer 1: 24 5x5 filters with 2x2 stride
+ convolutional layer 2: 36 5x5 filters with 2x2 stride
+ convolutional layer 3: 48 5x5 filters with 2x2 stride
+ convolutional layer 4: 64 3x3 filters with single stride
+ convolutional layer 5: 64 3x3 filters with single stride with dropout
+ fully connected layer

Training data consisted of the following strategy:
+ 2 laps of center lane driving
+ 1 lap recovery from side driving
+ 1 lap of smooth curve driving
+ reverse direction: 2 laps of center lane driving
+ reverse direction: 1 lap recovery from side driving
+ reverse direction: 1 lap of smooth curve driving

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 61-65) 

The model includes RELU layers to introduce nonlinearity (code lines 61-65, 68), and the data is normalized in the model using a Keras lambda layer (code line 59). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 66). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 76).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I used an iterative approach starting from a simple model to complex in a series of experiments. The series of experiments involved either changes in the models architecture, changes in training data or data processing. These experiments consisted of:
+ basic model: model flattens into an output, one lap of driving data with no data augmentation or processing. The experiment is at 'experments/behavior_clone_basic.ipynb'.
+ basic_plus model:  model has 2 layers, max pooling and a dropout. The model used one lap of driving data with no data augmentation and processing. The experiment is at 'experiments/behavior_clone_basic_plus.ipynb'.
+ basic_data model: model has 2 layers, max pooling and dropout. The model used one lap of driving data which was normalized and mean centered. The data was also augmented by flipping the image and inversing the steering measurements. The experiment is at 'experiments/behavior_clone_datamodel.ipynb.
+ basic_data2 model: model is similar to basic_data, but the images were cropped and training data consisted of multiple camera angles. The experiment is at 'experiments/behavior_clone_datamodel2.ipynb'.
+ basic_nvidia model: this experiment used the final model architecture and the data with processing and augmentation similar to basic_data2 model. The experiment is at 'experiments/behavior_clone_nvidia.ipynb'.

Each experiment led me to a more complex model architecture or data choice. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat overfitting I included dropout in the model architecture. The final step was to run the simulator to see how well the car was driving around track one. With each progressive experiments car performance improved. Each experiment shows a video of autonomous driving with each of the models. This performance started with a car autonomously driving in a circle out of the road to driving autonomously around the track without leaving the road. 

#### 2. Final Model Architecture

The final model architecture (model.py lines 58-69) consisted of a convolution neural network with the following layers and layer sizes:
+ convolutional layer 1: 24 5x5 filters with 2x2 stride
+ convolutional layer 2: 36 5x5 filters with 2x2 stride
+ convolutional layer 3: 48 5x5 filters with 2x2 stride
+ convolutional layer 4: 64 3x3 filters with single stride
+ convolutional layer 5: 64 3x3 filters with single stride with dropout
+ fully connected layer

#### 3. Creation of the Training Set & Training Process

In the final model to capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover if going close to the side of the road. These images show what a recovery looks:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Further data points were made having smooth driving around curves. I then reversed the car on track one and did the following in counter direction:
+ 2 laps of center lane driving
+ 1 lap of recovery from sides
+ 1 lap of smooth curve driving

To augment the data sat, I also flipped images and angles thinking that this would generalize the model further. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 10,240 data points. I then preprocessed this data by:
+ normalizing the data and mean centering
+ using images from multiple camera images
+ cropping images

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
