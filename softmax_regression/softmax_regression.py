"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtranml.github.io/ml/softmax-regression.html
"""
import numpy as np
import sys
sys.path.append("..")
from libs.utils import load_dataset_mnist, preprocess_data, Trainer
from libs.mnist_lib import MNIST
from optimizations_algorithms.optimizers import SGD
from nn_components.activations import softmax
from nn_components.losses import CrossEntropy


class SoftmaxRegression:

    def __init__(self, feature_dim:int, num_classes:int, optimizer:object, loss_func:object=CrossEntropy()):
        """
        Constructor for softmax regression. It can be seen as the simplest neural network with input layer and output layer. 

        Parameter
        ---------
        feature_dim: (int) dimension of input feature.
        num_classes: (int) number of output classes.
        optimizer: (object) optimizer used to optimize loss w.r.t W.
        loss_func: (object) the type of loss function to optimize.
        """
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.W = np.random.normal(size=(feature_dim, num_classes))

    def _forward(self, X:np.ndarray):
        """
        Compute softmax transformation.
            LINEAR: Z = X.dot(W)
            SOFTMAX: A = softmax(Z)

        Returns
        -------
        Softmax: e^z / sum(e^z).
            Shape = (N, num_class)
        """
        Z = X.dot(self.W)
        A = softmax(Z)
        return A

    def backward(self, Y:np.ndarray, Y_hat:np.ndarray, X:np.ndarray):
        """
        Compute gradient matrix.

        Parameters
        ----------
        X: training set.
            Shape = (N, D).
        y: training one-hot label vectors.
            Shape = (N, num_class).
        y_hat: predicted probabilities.
            Shape = (N, num_class).
        Returns
        -------
        Gradient of cross entropy loss respect to W: 1/N*(X.T.dot(Y_hat - Y))
            Shape = (D, num_class).
        """
        dZ = self.loss_func.backward(Y_hat, Y)
        dW = X.T.dot(dZ)
        dW = self.optimizer.minimize(dW)
        self.W = self.W - dW

    def __call__(self, X:np.ndarray):
        return self._forward(X)

    def predict(self, X_test:np.ndarray):
        return np.argmax(X_test.dot(self.W), axis=-1)


if __name__ == '__main__':
    load_dataset_mnist("../libs")
    mndata = MNIST('../libs/data_mnist')

    images, labels = mndata.load_training()
    images, labels = preprocess_data(images, labels)
    optimizer = SGD(0.01)
    batch_size = 64
    epochs = 20
    loss_func = CrossEntropy()

    softmax_regression = SoftmaxRegression(feature_dim=images.shape[1], num_classes=labels.shape[1], optimizer=optimizer, loss_func=loss_func)

    trainer = Trainer(softmax_regression, batch_size, epochs)
    trainer.train(images, labels)

    images_test, labels_test = mndata.load_testing()
    images_test, labels_test = preprocess_data(images_test, labels_test, test=True)

    pred = softmax_regression.predict(images_test)

    print("Accuracy:", len(pred[labels_test == pred]) / len(pred))
    from sklearn.metrics.classification import confusion_matrix

    print("Confusion matrix: ")
    print(confusion_matrix(labels_test, pred))