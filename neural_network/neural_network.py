"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtranml.github.io/ml/neural-network.html
"""

import numpy as np
from nn_components.layers import FCLayer, ActivationLayer, BatchNormLayer, DropoutLayer
from nn_components.losses import CrossEntropy
from tqdm import tqdm

class NeuralNetwork:

    def __init__(self, optimizer:object, layers:list, loss_func:object=CrossEntropy()):
        """
        Deep neural network architecture.

        Parameters
        ----------
        optimizer: (object) optimizer object uses to optimize the loss.
        layers: (list) a list of sequential layers. For neural network, it should have [FCLayer, ActivationLayer, BatchnormLayer, DropoutLayer]
        loss_func: (object) the type of loss function we want to optimize. 
        """
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.layers = self._structure(layers)

    def _structure(self, layers):
        """
        Structure function that initializes neural network architecture.

        Parameters
        ----------
        layers: (list) a list of sequential layers. For neural network, it should have [FCLayer, ActivationLayer, BatchnormLayer, DropoutLayer]
        """
        for layer in layers:
            if isinstance(layer, (FCLayer, BatchNormLayer)):
                layer.initialize_optimizer(self.optimizer)
        return layers

    def _forward(self, train_X, prediction=False):
        """
        NN forward propagation level.

        Parameters
        ----------
        train_X: training dataset X.
                shape = (N, D)
        prediction: whether this forward pass is prediction stage or training stage.

        Returns
        -------
        Probability distribution of softmax at the last layer.
            shape = (N, C)
        """
        inputs = train_X
        for layer in self.layers:
            if isinstance(layer, (BatchNormLayer, DropoutLayer)):
                inputs = layer.forward(inputs, prediction=prediction)
                continue
            inputs = layer.forward(inputs)
        output = inputs
        return output

    def _backward_last(self, Y, Y_hat):
        """
        Special formula of backpropagation for the last layer.
        """
        delta = self.loss_func.backward(Y_hat, Y)
        dW = self.layers[-3].output.T.dot(delta)
        self.layers[-2].update_params(dW)
        dA_prev = delta.dot(self.layers[-2].W.T)
        return dA_prev

    def _backward(self, Y, Y_hat, X):
        """
        NN backward propagation level. Update weights of the neural network.

        Parameters
        ----------
        Y: one-hot encoding label.
            shape = (N, C).
        Y_hat: output values of forward propagation NN.
            shape = (N, C).
        X: training dataset.
            shape = (N, D).
        """
        dA_prev = self._backward_last(Y, Y_hat)
        for i in range(len(self.layers)-3, 0, -1):
            dA_prev = self.layers[i].backward(dA_prev, self.layers[i-1])
        _ = self.layers[i-1].backward(dA_prev, X)
    
    def backward(self, Y, Y_hat, X):
        return self._backward(Y, Y_hat, X)

    def __call__(self, X, prediction=False):
        return self._forward(X, prediction)

    def predict(self, test_X):
        """
        Predict function.
        """
        y_hat = self._forward(test_X, prediction=True)
        return np.argmax(y_hat, axis=1)