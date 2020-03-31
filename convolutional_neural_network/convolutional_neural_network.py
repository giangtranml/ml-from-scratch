"""
Author: Giang Tran
Email: giangtran204896@gmail.com
Docs: https://giangtranml.github.io/ml/conv-net.html
"""
import numpy as np
from neural_network.neural_network import NeuralNetwork
from nn_components.layers import ConvLayer, ActivationLayer, PoolingLayer, FlattenLayer, FCLayer, BatchNormLayer
from nn_components.losses import CrossEntropy

class CNN(NeuralNetwork):

    def __init__(self, optimizer:object, layers:list, loss_func:object=CrossEntropy()):
        """
        A Convolutional Neural Network.

        Parameters
        ----------
        optimizer: (object) optimizer class to use (sgd, sgd_momentum, rms_prop, adam)
        layers: (list) a list of sequential layers of cnn architecture.
        loss_func: (object) the type of loss function we want to optimize. 
        """
        super().__init__(optimizer, layers, loss_func)

    def _structure(self, layers):
        """
        Structure function that initializes convolutional neural network architecture.

        Parameters
        ----------
        layers: (list) a list of sequential layers. For convolutional neural network, it should have [ConvLayer, PoolingLayer, FCLayer,
                                                                                        ActivationLayer, BatchnormLayer, DropoutLayer]
        """
        for layer in layers:
            if isinstance(layer, (ConvLayer, FCLayer, BatchNormLayer)):
                layer.initialize_optimizer(self.optimizer)
        return layers

    def _backward(self, Y, Y_hat, X):
        """
        CNN backward propagation.

        Parameters
        ----------
        Y: one-hot encoding label.
            shape = (m, C).
        Y_hat: output values of forward propagation NN.
            shape = (m, C).
        X: training dataset.
            shape = (m, iW, iH, iC).
        """
        dA_prev = self._backward_last(Y, Y_hat)
        for i in range(len(self.layers)-3, 0, -1):
            dA_prev = self.layers[i].backward(dA_prev, self.layers[i-1])
        _ = self.layers[i-1].backward(dA_prev, X)
