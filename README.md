## MINST-NN-and-CNN-models

Creating models for MINST database

The MINST database is formed by digits images with labels.
# Problem
Build and traing a classifing model for correct classify the digits from 1 to 10.
The images are made by 28x28 pixels in one channel color gray color.
<img src='https://miro.medium.com/max/800/1*LyRlX__08q40UJohhJG9Ow.png' width="600" height="auto">

# Solutions
I provided two different model to solve this classification task.
The solutions are:
1. Full connected NeuralNetwork
2. Convolutional Neural Network

# Full connected NeuralNetwork
The Neural Network id made by:
- 784 nodes in input
- 50 nodes in the Hidden Layer
- 10 output nodes

784 nodes are caused by the dimension of the image 28x28 = 784, we reshape a matrix(28,28) to an array(784).
<img src='https://simone-rizzo.github.io/MINST-NN-and-CNN-models/nn.bmp' width="600" height="auto">

- Training: Got 55786 / 60000 with accuracy 92.98%
- Test: Got 9295 / 10000 with accuracy 92.95%

# Convolutional NeuralNetwork

- Training: Got 59089 / 60000 with accuracy 98.48%
- Test: Got 9842 / 10000 with accuracy 98.42%
