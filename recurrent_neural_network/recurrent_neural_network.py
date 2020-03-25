"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtranml.github.io/ml/machine-learning/recurrent-neural-network
Note: not correctly implemented yet!
"""
from nn_components.activations import softmax, tanh, tanh_grad
from neural_network.neural_network import NeuralNetwork
import numpy as np
from tqdm import tqdm


class RecurrentNeuralNetwork(NeuralNetwork):

    def __init__(self, hidden_units, epochs, optimizer, batch_size):
        """
        Constructor for Recurrent Neural Network. 
        """
        self.hidden_units = hidden_units
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size

    def _loss(self, Y, Y_hat):
        """
        Loss function for many to many RNN.

        Parameters
        ----------
        Y: one-hot encoding label tensor. shape = (N, T, C)
        Y_hat: output at each time step. shape = (N, T, C)
        """
        return -np.mean(np.sum(Y*np.log(Y_hat), axis=(1, 2)))

    def _forward(self, X):
        """
        RNN forward propagation.

        Parameters
        ----------
        X: time series input, shape = (N, T, D)

        Returns
        -------
        Y_hat: softmax output at every step, shape = (N, T, C)
        """
        m, timesteps, _ = X.shape
        h0 = np.zeros(shape=(m, self.hidden_units))
        self.states = np.zeros(shape=(m, timesteps, self.hidden_units))
        self.states[:, 0, :] = tanh(np.dot(X[:, 0, :], self.Wax) + np.dot(h0, self.Waa) + self.ba)
        for t in range(1, timesteps):
            self.states[:, t, :] = tanh(np.dot(X[:, t, :], self.Wax) + np.dot(self.states[:, t-1, :], self.Waa) + self.ba)
        Y_hat = np.einsum("nth,hc->ntc", self.states, self.Wy)
        Y_hat = softmax(Y_hat + self.by)
        return Y_hat

    def _backward(self, X_train, Y_train, Y_hat):
        """
        X_train: shape=(m, time_steps, vector_length)
        Y_train: shape=(m, time_steps, vocab_length)
        Y_hat: shape=(m, time_steps, vocab_length)
        """
        m, time_steps, vector_len = X_train.shape
        dWaa = np.zeros(shape=self.Waa.shape)
        dWax = np.zeros(shape=self.Wax.shape)
        dba = np.zeros(shape=self.ba.shape)

        delta = (Y_hat - Y_train)/m
        dWy = np.einsum("ntc,nth->hc", delta, self.states)
        dby = np.sum(delta, axis=(0, 1))

        delta = np.einsum("ntc,hc->nth", delta, self.Wy)
        d_states = np.einsum("ntk,hk->nth", tanh_grad(self.states), self.Waa)

        for t in reversed(range(time_steps)):
            for i in range(0, t):
                dWaa += delta[:, t, :] * np.prod(d_states[:, i+1:t, :], axis=1)
        # self.update_params(dWy, dby, dWaa, dWax, dba)

    def train(self, X_train, Y_train):
        """
        X_train: shape=(m, time_steps, vector_length)
        Y_train: shape=(m, time_steps, vocab_length)
        """
        m, time_steps, vector_len = X_train.shape
        _, _, vocab_len = Y_train.shape
        self.Wax = np.random.normal(size=(vector_len, self.hidden_units))
        self.Waa = np.random.normal(size=(self.hidden_units, self.hidden_units))
        self.Wy = np.random.normal(size=(self.hidden_units, vocab_len))
        self.ba = np.zeros(shape=(1, self.hidden_units))
        self.by = np.zeros(shape=(1, vocab_len))
        super().train(X_train, Y_train)

    def update_params(self, dWy, dby, dWaa, dWax, dba):
        """
        Update parameters of RNN by its gradient.
        """
        dWy = self.optimizer.minimize(dWy)
        dby = self.optimizer.minimize(dby)
        dWaa = self.optimizer.minimize(dWaa)
        dWax = self.optimizer.minimize(dWax)
        dba = self.optimizer.minimize(dba)

        self.Wy -= dWy
        self.by -= dby
        self.Waa -= dWaa
        self.Wax -= dWax
        self.ba -= dba
