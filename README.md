# CNN_DogsNdCats

# Image Classification using Convolutional Neural Networks (CNN)
This is a simple implementation of image classification using CNN. The dataset used for training and validation is a subset of the Kaggle's 'Dogs vs. Cats' dataset. The code was implemented using Python and the Keras library with the TensorFlow backend.
##Prerequisites
*   Python
*   Keras
*   TensorFlow
*   Plotly
*   PIL
*   Matplotlib

## Data Preparation
The first step is to prepare the data. The dataset is divided into training and validation sets, each containing subdirectories for each class. In this implementation, we have two classes, dogs and cats. The training set contains 8000 images (4000 images for each class), while the validation set contains 2000 images (1000 images for each class).

We then visualize the distribution of the classes in the training data using a pie chart. Additionally, we display the first 10 images from each class using the PIL library.
##Data Augmentation
To avoid overfitting, we apply data augmentation. Data augmentation artificially creates new training samples by randomly transforming the existing ones. We apply rescaling and random horizontal and vertical flips to the images.
##Models Architectures
1.   Model 1:
*  Architecture: A sequential model with multiple convolutional layers, max pooling layers, dropout layers, and dense layers.
*  Input shape: (224, 224, 3)
*  Output activation function: Sigmoid
*  Loss function: Binary cross-entropy
*  Optimizer: RMSprop
*  Callbacks: Early stopping
*  Trained for 35 epochs
2.   Model 2:


*  Architecture: A sequential model with multiple convolutional layers, max pooling layers, dropout layers, and dense layers.
*  Input shape: (224, 224, 3)
*  Output activation function: Sigmoid
*  Loss function: Binary cross-entropy
*  Optimizer: RMSprop
*  Callbacks: Early stopping and learning rate reduction
*  Trained for 35 epochs
3.   Model 3:

*  Architecture: Transfer learning with VGG16 pre-trained model without the top *  layers, followed by a custom top layer.
*  Input shape: (224, 224, 3)
*  Output activation function: Sigmoid
*  Loss function: Binary cross-entropy
*  Optimizer: Adam with learning rate of 0.001
*  Callbacks: Early stopping and learning rate reduction
*  Trained for 10 epochs
