"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtranml.github.io/ml/machine-learning/optimization-algorithms
"""
import numpy as np

class _Optimizers: 

    def __init__(self):
        pass

    def minimize(self, grad):
        raise NotImplementedError("Child class must implement minimize() function")

class SGD(_Optimizers):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def minimize(self, grad):
        return self.alpha*grad

class SGDMomentum(_Optimizers):

    def __init__(self, alpha=0.01, beta=0.9):
        self.alpha = alpha
        self.beta = beta
        self.v = None

    def minimize(self, grad):
        if self.v is None:
            self.v = np.zeros(shape=grad.shape)
        self.v = self.beta*self.v + (1-self.beta)*grad
        return self.alpha * self.v

class RMSProp(_Optimizers):

    def __init__(self, alpha=0.01, beta=0.9, epsilon=1e-9):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.s = None

    def minimize(self, grad):
        if self.s is None:
            self.s = np.zeros(shape=grad.shape)
        self.s = self.beta*self.s + (1-self.beta)*grad**2
        return self.alpha * (1/(np.sqrt(self.s + self.epsilon))) * grad

class Adam(_Optimizers):
    
    def __init__(self, alpha=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-9):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.v = None
        self.s = None

    def minimize(self, grad):
        if self.v is None and self.s is None:
            self.v = np.zeros(shape=grad.shape)
            self.s = np.zeros(shape=grad.shape)
        self.v = self.beta_1*self.v + (1-self.beta_1)*grad
        self.s = self.beta_2*self.s + (1-self.beta_2)*grad**2
        return self.alpha * (self.v / (np.sqrt(self.s + self.epsilon)))
        