# Machine Learning from scratch

## About
This ML repository is all about coding Machine Learning algorithms from scratch by Numpy with the math under the hood without Auto-Differentiation frameworks like Tensorflow, Pytorch, etc. Some advanced models in Computer Vision, NLP require Tensorflow to quickly get the idea written in paper.

## Repository structure
As a software engineer, I follow the principle of OOP to construct the repository. You can see that `NeuralNetwork` class will use `FCLayer`, `BatchNormLayer`, `ActivationLayer` class and `CNN` class will use `ConvLayer`, `PoolingLayer`, `FCLayer`, `ActivationLayer`,... This helps me easily reuse every piece of code I wrote as well as for readable code. 

## Table of contents
- Machine Learning models:
  * [Linear Regression](https://github.com/giangtranml/ml-from-scratch/blob/master/linear_regression/linear_regression.py)
  * [Logistic Regression](https://github.com/giangtranml/ml-from-scratch/blob/master/logistic_regression/logistic_regression.py)
  * [Softmax Regression](https://github.com/giangtranml/ml-from-scratch/blob/master/softmax_regression/softmax_regression.py)
  * [Neural Network](https://github.com/giangtranml/ml-from-scratch/blob/master/neural_network/neural_network.py)
  * [Convolutional Neural Network](https://github.com/giangtranml/ml-from-scratch/blob/master/convolutional_neural_network/convolutional_neural_network.py)
  * [Support Vector Machine](https://github.com/giangtranml/ml-from-scratch/blob/master/svm/svm.py)
  
- Deep Learning layers:
  * [Fully-Connected Layer](https://github.com/giangtranml/ml-from-scratch/blob/master/nn_components/layers.py#L43)
  * [Convolutional Layer](https://github.com/giangtranml/ml-from-scratch/blob/master/nn_components/layers.py#L107)
  * [Pooling Layer](https://github.com/giangtranml/ml-from-scratch/blob/master/nn_components/layers.py#L245)
  * [Activation Layer](https://github.com/giangtranml/ml-from-scratch/blob/master/nn_components/layers.py#L372)
  * [BatchNorm Layer](https://github.com/giangtranml/ml-from-scratch/blob/master/nn_components/layers.py#L436)
  * [Dropout Layer](https://github.com/giangtranml/ml-from-scratch/blob/master/nn_components/layers.py#L407)
  
- Optimization algorithms:
  * [SGD](https://github.com/giangtranml/ml-from-scratch/blob/master/optimizations_algorithms/optimizers.py#L16)
  * [SGD with Momentum](https://github.com/giangtranml/ml-from-scratch/blob/master/optimizations_algorithms/optimizers.py#L24)
  * [RMSProp](https://github.com/giangtranml/ml-from-scratch/blob/master/optimizations_algorithms/optimizers.py#L37)
  * [Adam](https://github.com/giangtranml/ml-from-scratch/blob/master/optimizations_algorithms/optimizers.py#L51)
- Weights initialization:
  * [He initialization](https://github.com/giangtranml/ml-from-scratch/blob/master/nn_components/initializers.py#L3)
  * [Xavier/Glorot initialization](https://github.com/giangtranml/ml-from-scratch/blob/master/nn_components/initializers.py#L24)
- Advanced models:
  * [Attention Mechanism (Bahdanau and Luong Attention)](https://github.com/giangtranml/ml-from-scratch/blob/master/attention_mechanism/Attention_Mechanism.ipynb)
  * [Transformer](https://github.com/giangtranml/ml-from-scratch/blob/master/transformer/Transformer_Pytorch.ipynb)
