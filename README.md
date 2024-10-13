# Digit-Recognizer using ANN
![peoject](C:/Users/DELL/Downloads/digit-photo.png)

## Overview:
- This project focuses on building a digit recognition system using Artificial Neural Networks (ANN), specifically implemented with Keras and TensorFlow.
-  The dataset used for this project is from [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/data), which consists of grayscale images of handwritten digits from 0 to 9. The dataset is composed of 42,000 images for training and 28,000 images for testing, with each image having a dimension of 28x28 pixels, flattened into 784 features.

## Project Steps:
### Data Representation in Keras:
The dataset is loaded and the images are reshaped into a 1D array of 784 features for compatibility with fully connected layers in the neural network.

### Data Normalization:
Each pixel value in the images is normalized by dividing by 255 to ensure the values range between 0 and 1. This step improves the model's convergence during training and helps with performance

### Densely Connected Networks in Keras:
A densely connected neural network was built using Keras’ Sequential API.
The network includes fully connected layers where every neuron in one layer is connected to every neuron in the subsequent layer.

### Sequential Class in Keras:
The Sequential class was used to stack layers of the neural network in a linear fashion, which simplifies model creation and management.

### Model Architecture:
1- Input Layer: The input layer has 128 neurons with the ReLU (Rectified Linear Unit) activation function. The input shape is set to 784, representing the flattened images.

2- Hidden Layer: A second dense layer with 256 neurons, also using the ReLU activation function, was added to introduce non-linearity and increase model capacity.

3- Output Layer: The output layer contains 10 neurons, corresponding to the 10 classes of digits (0-9). The softmax activation function is applied here to output class probabilities for the predictions.

### Model Summary:
A summary of the model architecture was generated using model.summary(), which provides details of each layer, including the number of parameters and the output shape at each stage.

### Model Compilation:
The model was compiled using the SGD (Stochastic Gradient Descent) optimizer, which updates the model's weights based on the gradient of the loss function with respect to each weight, aiming for faster convergence and performance.
The loss function used is categorical_crossentropy, which is appropriate for multi-class classification problems like digit recognition, where the model predicts one of the ten digit classes (0-9).
The performance metric chosen for evaluation was accuracy, which measures how well the model correctly classifies the digits during training and testing.

### Training Evaluation of the Model:
The model was trained for 50 epochsand then  evaluated using the test dataset. The test accuracy achieved was around 98%, which indicates that the model performs well in recognizing handwritten digits.

### Confusion Matrix:
A confusion matrix was generated to visualize the performance of the classifier on each digit. It provides insights into the correct and incorrect predictions made by the model, helping identify any misclassifications.

## Conclusion:
This project demonstrates the effectiveness of artificial neural networks in performing digit recognition using the MNIST dataset. With an accuracy of 98%, the model is highly capable of distinguishing between different handwritten digits. By implementing a densely connected neural network with ReLU and softmax activations, this project successfully leverages deep learning techniques to achieve high performance in image classification tasks.
