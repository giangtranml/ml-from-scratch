import numpy as np

def sigmoid(z):
    """
    Sigmoid activation function.
        g(z) = 1 / (1 + e^-z)
    """
    return 1/(1+np.exp(-z))

def tanh(z):
    """
    Tanh activation function.
        g(z) = tanh(z)
    """
    return np.tanh(z)

def relu(z):
    """
    Relu activation function.
        g(z) = max(0, z)
    """
    return z*(z > 0)

def softmax(z, axis=-1):
    """
    Softmax activation function. Use at the output layer.
        g(z) = e^z / sum(e^z)
    """
    z_prime = z - np.max(z, axis=axis, keepdims=True)
    return np.exp(z_prime) / np.sum(np.exp(z_prime), axis=axis, keepdims=True)

def sigmoid_grad(z):
    """
    Sigmoid derivative.
        g'(z) = g(z)(1-g(z))
    """
    return z*(1-z)

def tanh_grad(z):
    """
    Tanh derivative.
        g'(z) = 1 - g^2(z).
    """
    return 1 - z**2

def relu_grad(z):
    """
    Relu derivative.
        g'(z) = 0 if g(z) <= 0
        g'(z) = 1 if g(z) > 0
    """
    return 1*(z > 0)