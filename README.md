# Introduction to AI: Neural Networks 

This repository contains neural network models built for the "Introduction to AI" course in University of Puerto Rico. The networks in this repository are implemented using PyTorch and are designed for a variety of machine learning tasks such as regression, classification, and convolutional tasks.

## Neural Networks

### Regression Models 

1. **Network1**: This is a simple linear regression model with one layer followed by a ReLU activation function.

2. **Network2**: This model has a multi-layer perceptron structure with 10, 20, 10 nodes in each layer, all followed by ReLU activation functions.

3. **Network3**: This model is a slightly more complex multi-layer perceptron with a 10, 20, 30, 20 structure, all followed by ReLU activation functions.

### Classification Models

1. **Network1**: A four-layer fully connected network for classification tasks. 

2. **Network2**: A six-layer fully connected network for more complex classification tasks.

3. **Network3**: A variant of the six-layer fully connected network with a different number of nodes in each layer.

### Convolutional Models 

1. **Network1**: This network uses a simple convolutional block with a ReLU activation function.

2. **Network2**: This network has a convolutional block with two convolutional layers and ReLU activations.

3. **Network3**: This network includes two convolutional blocks each with batch normalization and max pooling.

4. **Bono**: This network has two convolutional blocks, each with batch normalization and ReLU activations. It also includes max pooling and dropout layers, and a sequence of fully connected layers with ReLU activations.
