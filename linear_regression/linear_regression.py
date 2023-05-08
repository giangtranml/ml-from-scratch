"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtran.me/machine-learning/linear-regression
"""
import sys
sys.path.append("..")
import numpy as np

class LinearRegression:

    def __init__(self, alpha, epochs=1000, lambda_=0.1):
        self.alpha = alpha
        self.epochs = epochs
        self.lambda_ = lambda_
        self.w = None
        self.b = None

    def _hypothesis(self, X):
        return np.dot(X, self.w) + self.b

    def _mse_loss(self, X, y_hat, y):
        m = y.shape[0]
        return np.sum((y_hat - y)**2)/(2*m) + self.lambda_*np.linalg.norm(self.w, 2)**2 / (2*m)

    def _gradient(self, X, y_hat, y):
        m = X.shape[0]
        return 1/m * np.dot(X.T, y_hat - y) + (self.lambda_/m*self.w)

    def _gradient_bias(self, y_hat, y):
        m = y.shape[0]
        return 1/m * np.sum(y_hat - y)  

    def _train(self, X_train, y_train):
        for e in range(self.epochs):
            y_hat = self._hypothesis(X_train)
            print("Loss at epoch %s: %f" % (e, self._mse_loss(X_train, y_hat, y_train)))
            w_grad = self._gradient(X_train, y_hat, y_train)
            b_grad = self._gradient_bias(y_hat, y_train)
            self._update_params(w_grad, b_grad)
            if np.linalg.norm(w_grad, 2) < 1e-6:
                break

    def _update_params(self, w_grad, b_grad):
        self.w -= self.alpha*w_grad
        self.b -= self.alpha*b_grad

    def train(self, X_train, y_train):
        self.w = np.random.normal(size=(X_train.shape[1], 1))
        self.b = np.mean(y_train)
        self._train(X_train, y_train)

    def predict(self, X_test):
        assert X_test.shape[1] == self.w.shape[0], "Incorrect shape."
        return self._hypothesis(X_test)

    def r2_score(self, y_hat, y_test):
        total_sum_squares = np.sum((y_test - np.mean(y_test))**2)
        residual_sum_squares = np.sum((y_test - y_hat)**2)
        return 1 - residual_sum_squares/total_sum_squares