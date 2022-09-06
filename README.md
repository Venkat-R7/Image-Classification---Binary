# Image-Classification---Binary

**Introduction**

Many real-world computer vision problems start with image classification. Image classification is important in a variety of fields, from medicine to security systems. The basic aim of image classification is for a model to identify a picture by the term of its class, based on how it was trained using an image dataset.
This project's objective is to develop an algorithm that can distinguish between cats and dogs. A dataset was utilized to train different machine learning models, and an algorithm was then used to differentiate images.


**Introduction to the dataset**

The dataset used for this project was sourced from Kaggle under the project Playground Prediction Competition, the dataset consists of a total of 1002 to images as training data and 100 images as test data.
The train dataset is equally distributed between dogs and cats with 501 images in each category, the initial observation is that all the images are of different shapes and size.

**Preprocessing Steps**

The dataset was pre-processed before the same cwas used In model training, for this purpose the following steps were carried out
1. Loading image from the train data path
2. Resizing all image to constant size 350*350
3. Normalising the array to store the values between 0 to 1
4. Storing the normalised data to a NumPy Array
5. Principle component analysis

**Principal Component Analysis**

The steps involved and performed as part of PCA are as follows:
The images used in this project are colour images with three channels: red, green, and blue. Each image's dimensions are 350 * 350 * 3, with 350 representing the image's height and breadth and 3 representing the three colours red, green, and blue.
Each picture channel was split from the original image using the CV2 module, and the values of each channel were stored in three independent NumPy arrays.
The original image, as well as the three different colour channels that were split, are displayed in the sample image below.

**Classification Models Used**

With the dataset pre-processed, split and prepared, machine learning models have been trained on top of it for image classification.
For model were trained using the training dataset:
1.	K-Nearest Neighbours Algorithm - KNN 
2.	Support Vector Machine Algorithm – SVM
3.	Random Forest Algorithm
4.	Convolutional Neural Network Algorithm – CNN

All of the models were trained on the train dataset, and the accuracy was calculated by predicting the validation data using the labels.


**Data Augmentation**

In data processing, data augmentation refers to approaches for increasing the amount of data by adding slightly changed copies of current data or creating new synthetic information from existing data. When training a machine learning model, it functions as a regularize and helps to reduce overfitting.
The following data augmentations were applied to the dataset before passing to CNN model:
•	Random flip of images – horizontally and vertically
•	Random zoom of images, based on both height and width
•	Random rotation of images 

**Results**





**Conclusion**

The dataset was created using numerous pre-processing techniques such as normalizing, principal component analysis, and others, then machine learning techniques were utilized to build an effective model for image categorization of dog vs cat.
Four ML algorithms namely, KNN, Support vector machine, Random Forest and CNN were trained using the train dataset and accuracy predicted using validation data.
Hyperparameters were tuned for CNN and random forest as both showed equal performance metrics in training stage.
With multiple parameters tuned, it was observed that the CNN model performed best with an accuracy of 63.18 %.

The best model, CNN, was then used to predict the unseen test data.
