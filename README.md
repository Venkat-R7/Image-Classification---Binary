# Image-Classification---Binary

**Introduction**

Many real-world computer vision problems start with image classification. Image classification is important in a variety of fields, from medicine to security systems. The basic aim of image classification is for a model to identify a picture by the term of its class, based on how it was trained using an image dataset.
This project's objective is to develop an algorithm that can distinguish between cats and dogs. A dataset was utilized to train different machine learning models, and an algorithm was then used to differentiate images.


**Introduction to the dataset**

The dataset used for this project was sourced from Kaggle under the project Playground Prediction Competition, the dataset consists of a total of 1002 to images as training data and 100 images as test data.
The train dataset is equally distributed between dogs and cats with 501 images in each category, the initial observation is that all the images are of different shapes and size.

**Steps**

The dataset was pre-processed before the same cwas used In model training, for this purpose the following steps were carried out
1. Loading image from the train data path
2. Resizing all image to constant size 350*350
3. Normalising the array to store the values between 0 to 1
4. Storing the normalised data to a NumPy Array
5. Principle component analysis
