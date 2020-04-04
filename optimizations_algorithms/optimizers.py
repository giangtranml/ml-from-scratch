"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtranml.github.io/ml/machine-learning/optimization-algorithms
"""
import numpy as np

class _Optimizers: 

    def __init__(self):
        pass

    def step(self, grads, layers):
        raise NotImplementedError("Child class must implement step() function")

class SGD(_Optimizers):

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def step(self, grads, layers):
        for grad, layer in zip(grads, layers):
            layer.update_params(grad)

class SGDMomentum(_Optimizers):

    def __init__(self, alpha=0.01, beta=0.9):
        self.alpha = alpha
        self.beta = beta
        self.v = []
    
    def step(self, grads, layers):
        if len(self.v) == 0:
            self.v = [np.zeros_like(grad) for grad in grads]
        for i, (grad, layer) in enumerate(zip(grads, layers)):
            self.v[i] = self.beta*self.v[i] + (1-self.beta)*grad
            grad = self.alpha * self.v[i]
            layer.update_params(grad)

class RMSProp(_Optimizers):

    def __init__(self, alpha=0.01, beta=0.9, epsilon=1e-9):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.s = []

    def step(self, grads, layers):
        if len(self.s) == 0:
            self.s = [np.zeros_like(grad) for grad in grads]
        for i, (grad, layer) in enumerate(zip(grads, layers)):
            self.s[i] = self.beta*self.s[i] + (1-self.beta)*grad**2
            grad = self.alpha * (grad/(np.sqrt(self.s[i]) + self.epsilon))
            layer.update_params(grad)

class Adam(_Optimizers):
    
    def __init__(self, alpha=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-9):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.v = []
        self.s = []
        self.t = 1

    def step(self, grads, layers):
        if len(self.s) == 0 and len(self.v) == 0:
            self.v = [np.zeros_like(grad) for grad in grads]
            self.s = [np.zeros_like(grad) for grad in grads]
        for i, (grad, layer) in enumerate(zip(grads, layers)):
            self.v[i] = (self.beta_1*self.v[i] + (1-self.beta_1)*grad)
            self.s[i] = (self.beta_2*self.s[i] + (1-self.beta_2)*grad**2)
            v_correct = self.v[i] / (1-self.beta_1**self.t)
            s_correct = self.s[i] / (1-self.beta_2**self.t)
            grad = self.alpha * (v_correct / (np.sqrt(s_correct) + self.epsilon))
            layer.update_params(grad)
        self.t += 1
        