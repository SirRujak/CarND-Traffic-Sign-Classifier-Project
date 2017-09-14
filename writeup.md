#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup-data/image_type_distribution.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./new_signs/1_30kph_sign.png "Traffic Sign 1"
[image5]: ./new_signs/12_priority_road_sign.png "Traffic Sign 2"
[image6]: ./new_signs/17_no_entry_sign.png "Traffic Sign 3"
[image7]: ./new_signs/17_no_entry_sign_90_deg.png "Traffic Sign 4"
[image8]: ./new_signs/21_double_curve.jpg "Traffic Sign 5"
[image9]: ./new_signs/28_children_crossing_sign.png "Traffic Sign 6"
[image10]: ./new_signs/29_bicycle_crossing_sign.png "Traffic Sign 7"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup

You're reading it! and here is a link to my [project code](https://github.com/SirRujak/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration



I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the training set. As can be seen, the number of training examples for each category spans a range of less than two hundred to over two thousand.

![alt text][image1]

### Design and Test a Model Architecture

The data was formed into four groups for testing:

1. The base dataset.
2. Base data with gaussian blurs.
3. Base data normalized.
4. Base plus gaussian normalized.

The normalization step was achieved through the Tensorflow per_image_standardization function with a map_fn call to apply the function to every image within a batch. Gaussian blurs were applied to each image at 3, 5, and 7 pixel kernels.

The network was trained on each of these datasets to compare results. In the final training run the normalized base dataset achieved the highest validation accuracy and was therefore used in the remainder of the project. It is important to note though that there was on average a 1-2% difference between all of the methods during training which suggests that all forms of training data would achieve the desired results. Along with this, it is quite possible that the extended gaussian blur datasets would lead to a more generalized network due to the enforced spacial uncertainty of the training set.

### My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x54 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x54 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x54 				|
| Flatten	    | 1350x1      									|
| Fully connected		| 100        									|
| Fully connected		| 110        									|
| Fully connected		| 43        									|
| RELU					|												|
| Softmax				|         									|
|						|												|
|						|												|
 


To train the model, I used a cross entropy calculation and followed with a function to reduce mean accuracy. The Tensorflow Adam optimizer was chosen with a learning rate of 0.0005 due to extensive testing. A batch size of 16 was used to increace step wise error and allow the optimizer to escape potential wells that would have been problematic otherwise given the low training rate. Finally, due to the slow training rate the training process lasted 75 epochs. Most models would fit within 30-40 epochs but the system would save the model when a new high record for validation accuracy was set in order to reduce overfitting.

## Results

My final model results were:
* validation set accuracy of 96.7%
* test set accuracy of 93.8%

The initial architecture of the model is based upon LeNet and the final is very similar. The main changes that were made were in the size parameters of each layer and the addition of thre ReLu layers. After ReLu was added the model performed at over 93% accuracy on the validation set and therefore all other changes were in an attempt to make training more reliable. By lowering the sizes of each layer as much as possible the network would find a solution set far faster that would be at a more reliable accuracy. The trade off of this is that it no longer was able to achieve a maximal validation accuracy.

LeNet was chosen as a basis due to the proven high accuracy on image based neural systems. This choice was validated by the minimal changes that were required to achieve very high accuracy on this dataset.

### Test a Model on New Images

Here are seven German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10]

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double Curve			| Beware of Ice/Snow      							|
| No Entry					| No Entry											|
| 30 km/h      		| 30 km/h   									| 
| Bicycle Crossing			| 30 km/h      							|
| Children Crossing			| Bumpy Road      							|
| Priority Road     			| Priority Road 										|
| No Entry	      		| 60 km/h					 				|


The model was able to correctly guess 3 of the 7 traffic signs, which gives an accuracy of 43%. This does not compare favorably to the test set but can mostly be explained. The new images used were taken from the perspective of pedestrians and bicyclists rather than from automobiles. As such the angles that signs will appear may be rather different than the training set. This combined with the choice of intentionally hard examples for the new test leads to the conclusion that while the system would be able to generalize over most images from within a car it would have trouble understanding them from new perspectives. It is possible that the gaussian extended training set would be more capable of generalizing over this type of change.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 29th cell of the Ipython notebook.

For the first image, the model is rather sure that this is a beware of ice/snow sign (probability of 0.061), instead the image contains a double curve sign. It is possible the mistake was made because the sign was covered in ice. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.061         			| Beware of Ice/Snow   									| 
| 0.022    				| Children Crossing 										|
| 0.022				| Road Narrows on the Right							|
| 0.022      			| Slippery Road					 				|
| 0.022			    | Right-of-Way at the Next Intersection      							|


For the second image, the model is rather sure that this is a no entry (probability of 0.061), this is the case. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.061         			| No Entry   									| 
| 0.022    				| Go Straight or Right 										|
| 0.022				| End of all Speed and Passing Limits							|
| 0.022      			| 60 km/h					 				|
| 0.022			    | 120 km/h      							|


For the third image, the model is rather sure that this is a speed limit 30 km/h sign (probability of 0.061), this is the case. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.061         			| 30 km/h  									| 
| 0.022    				| 50 km/h										|
| 0.022				| 120 km/h							|
| 0.022      			| 80 km/h					 				|
| 0.022			    | Bicycles Crossing      							|

For the fourth image, the model is rather sure that this is a 30 km/h sign (probability of 0.061), instead the image contains a bicycle crossing sign. This could have been mistaken due to the unusual angle of this sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.061         			| 30 km/h   									| 
| 0.022    				| Right-of-Way at the Next Intersection 										|
| 0.022				| General Caution							|
| 0.022      			| 20 km/h					 				|
| 0.022			    | End of Speed Limit 80 km/h      							|

For the fifth image, the model is rather sure that this is a road narrows to the right sign (probability of 0.061), instead the image contains a children crossing sign. This could have been mistaken due to the unusual angle of this sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.061         			| Road Narrows to the Right   									| 
| 0.022    				| Road Work 										|
| 0.022				| Double Curve							|
| 0.022      			| General Caution					 				|
| 0.022			    | 80 km/h      							|

For the sixth image, the model is very sure that this is a priority road sign (probability of 0.061), this is correct. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.061         			| Priority Road   									| 
| 0.022    				| Road Work 										|
| 0.022				| End of No Passsing							|
| 0.022      			| End of All Speed and Passing Limits					 				|
| 0.022			    | Right-of-Way at the Next Intersection      							|

For the seventh image, the model is very sure that this is a 60 km/h sign (probability of 0.054), instead the image contains a no entry sign. This could have been mistaken due to the sign being at a 90 degree angle from the average sign of its type. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.054         			| 60 km/h   									| 
| 0.024    				| 80 km/h 										|
| 0.023				| 20 km/h							|
| 0.022      			| Yield					 				|
| 0.022			    | Children Crossing      							|

For the purpose of these values, "rather sure" indicates that it it over twice as confident in this answer as it is of any other.

