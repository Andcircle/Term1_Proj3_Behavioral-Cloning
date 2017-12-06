# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pic/Original.png "Original Image"
[image2]: ./pic/Flip.png "Flipped Image"
[image3]: ./pic/Brightness.png "Brightness Argumented Image"
[image4]: ./pic/RGB.png "RGB Argumented Image"
[image5]: ./pic/center.png
[image6]: ./pic/left.png
[image7]: ./pic/right.png



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* nv_model_github.py containing the script to create and train the model
* nv_generator_github.py containing the script to create the training data
* drive.py for driving the car in autonomous mode
* nv_model_github.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_lz_nv1_v1_ok.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Layer (type)                     Output Shape          Param #     Connected to                     
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
Conv1 (Convolution2D)            (None, 43, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
Conv2 (Convolution2D)            (None, 20, 77, 36)    21636       Conv1[0][0]                      
____________________________________________________________________________________________________
Conv3 (Convolution2D)            (None, 8, 37, 48)     43248       Conv2[0][0]                      
____________________________________________________________________________________________________
Conv4 (Convolution2D)            (None, 6, 35, 64)     27712       Conv3[0][0]                      
____________________________________________________________________________________________________
Conv5 (Convolution2D)            (None, 4, 33, 64)     36928       Conv4[0][0]                      
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 4, 33, 64)     0           Conv5[0][0]                      
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 8448)          0           dropout_1[0][0]                  
____________________________________________________________________________________________________
FC1 (Dense)                      (None, 100)           844900      flatten_1[0][0]                  
____________________________________________________________________________________________________
FC2 (Dense)                      (None, 50)            5050        FC1[0][0]                        
____________________________________________________________________________________________________
FC3 (Dense)                      (None, 10)            510         FC2[0][0]                        
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1)             11          FC3[0][0]                        
____________________________________________________________________________________________________

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.
At the same time, for the first 3 Conv2D layer, 'subsample' has been set to (2,2), in order to avoid generate too many parameters.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate is set to 0.0001

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 
But, I believe in my dataset, I use too many recovering data, that's why in the final test (in the video), the car will keep driving off the center line and recovering back. In order to get better performance, the dataset shoule be fine tuned.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The purpose of this proj is similar to this paper https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
So I start from the Nvidia CNN, but in order to prevent overfitting one dropout layer has been added.

#### 2. Creation of the Training Set & Training Process

First, I used original data sample, the training result is fine, but I can't make the whole lap, so I added my own training data set. I did one dataset for center driving, one dataset for off centerline recovery. The final result turns out to be OK, but the recovery dataset maybe too big, so the vehicle will keep driving off center line and then recovering. The performance can be improved by fine tune the dataset.

In order to improve training result with minimum dataset, I did following things:
1. During training, I used all the left, center, right cameras

![alt text][image5]
![alt text][image6]
![alt text][image7]
2. I randomly fliped the image in training dataset, and changed brightness (HSV), RGB color. During this process generator is a very powerful tool, because it can keep changing the traing data in each training epoch.

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]

### Next Step
1. Fine tune the training dataset to reduce the recovery data fraction
2. More training data argumentation, e.g. image rotation, image shift, to make the model more robust
