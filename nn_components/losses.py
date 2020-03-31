import numpy as np


class CrossEntropy:
    
    def __init__(self, weights=1):
        self.weights = 1

    def __call__(self, Y_hat, Y):
        """
        Compute cross-entropy loss.

        Parameters
        ----------
        Y: one-hot encoding label. shape=(num_dataset, num_classes)
        Y_hat: softmax probability distribution over each data point. 
            shape=(num_dataset, num_classes)

        Returns
        -------
        J: cross-entropy loss.
        """
        assert Y.shape == Y_hat.shape, "Unmatch shape."
        loss = np.sum(self.weights * Y * np.log(Y_hat), axis=-1)
        return -np.mean(loss)

    def backward(self, Y_hat, Y):
        """
        Compute gradient of CE w.r.t linear (LINEAR -> SOFTMAX -> CE)

        Parameters
        ----------
        Y: one-hot encoding label. shape=(num_dataset, num_classes)
        Y_hat: softmax probability distribution over each data point. 
            shape=(num_dataset, num_classes)

        Returns
        -------
        grad CE w.r.t LINEAR
        """
        m = Y.shape[0]
        return (Y_hat - Y)/m

class FocalLoss:

    def __init__(self, alpha=1, gamma=2):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, Y, Y_hat):
        loss = np.sum(Y * self.alpha * (1-Y_hat)**self.gamma * np.log(Y_hat), axis=-1)
        return -np.mean(loss)

    def backward(self):
        pass

class MSE:
    
    def __init__(self):
        pass

    def __call__(self, y_hat, y):
        """
        Mean squared error
        """
        m = len(y)
        loss = np.sum((y_hat - y)**2)/(2*m)
        return loss

    def backward(self, y_hat, y):
        """
        Compute gradient of MSE w.r.t y_hat

        Parameters
        ----------
        y_hat: output from linear transformation. shape = (num_dataset, )
        y: ground truth, real values. shape = (num_dataset, )
        """
        m = len(y)
        grad = (y_hat - y)/m
        return grad

class BinaryCrossEntropy:
    pass
