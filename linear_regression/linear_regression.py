"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtran.me/machine-learning/linear-regression
"""
import sys
sys.path.append("..")
import numpy as np
import copy
import imageio
import io
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, alpha, epochs=1000, lambda_=0.1, do_visualize=False):
        self.alpha = alpha
        self.epochs = epochs
        self.lambda_ = lambda_
        self.w = None
        self.b = None
        self.do_visualize = do_visualize
        self.vis_elems = {
            "loss": [],
            "iteration": [],
            "weight": [],
            "bias": []
        }

    def _hypothesis(self, X, w, b):
        return np.dot(X, w) + b

    def _mse_loss(self, X, y_hat, y):
        m = y.shape[0]
        return np.sum((y_hat - y)**2)/(2*m) + self.lambda_*np.linalg.norm(self.w, 2)**2 / (2*m)

    def _gradient(self, X, y_hat, y):
        m = X.shape[0]
        return 1/m * np.dot(X.T, y_hat - y) + (self.lambda_/m*self.w)

    def _gradient_bias(self, y_hat, y):
        m = y.shape[0]
        return 1/m * np.sum(y_hat - y)  
    
    def _train_one_epoch(self, X_train, y_train, e):
        y_hat = self._hypothesis(X_train, self.w, self.b)
        loss = self._mse_loss(X_train, y_hat, y_train)
        print("Loss at epoch %s: %f" % (e, loss))
        w_grad = self._gradient(X_train, y_hat, y_train)
        b_grad = self._gradient_bias(y_hat, y_train)
        self._update_params(w_grad, b_grad)
        w_grad_norm = np.linalg.norm(w_grad, 2)
        return loss, w_grad_norm

    def _train(self, X_train, y_train):
        for e in range(self.epochs):
            loss, w_grad_norm = self._train_one_epoch(X_train, y_train, e)
            if self.do_visualize and (e+1) % 5 == 0:
                self.vis_elems["loss"].append(loss)
                self.vis_elems["iteration"].append(e)
                self.vis_elems["weight"].append(copy.deepcopy(self.w))
                self.vis_elems["bias"].append(copy.deepcopy(self.b))

            if w_grad_norm < 1e-6:
                break

    def _update_params(self, w_grad, b_grad):
        self.w -= self.alpha*w_grad
        self.b -= self.alpha*b_grad

    def _plot(self, w, b, loss, iteration, X, X_transform, y):
        y_plot = self._hypothesis(X_transform, w, b)
        plt.figure(0, figsize=(6, 6))
        plt.clf()
        plt.title("Loss: " + str(loss))
        plt.scatter(X[:, 0], y, color='r')
        label = "Iteration: " + str(iteration)
        for ind, t in enumerate(loss):
            label += "\nTheta %s: %.2f" % (ind, t)
        plt.plot(X, y_plot, '-', label=label)
        plt.legend()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        return imageio.imread(img_buf)

    def create_gif(self, X, X_transform, y):
        imgs = []
        for l, i, w, b in zip(self.vis_elems["loss"], self.vis_elems["iteration"], self.vis_elems["weight"], self.vis_elems["bias"]):
            imgs.append(self._plot(w, b, l, i, X, X_transform, y))
        imageio.mimsave("linear_regressionss.gif", imgs, fps=5)

    def train(self, X_train, y_train):
        self.w = np.random.normal(size=(X_train.shape[1], 1))
        self.b = np.mean(y_train)
        self._train(X_train, y_train)

    def predict(self, X_test):
        assert X_test.shape[1] == self.w.shape[0], "Incorrect shape."
        return self._hypothesis(X_test, self.w, self.b)

    def r2_score(self, y_hat, y_test):
        total_sum_squares = np.sum((y_test - np.mean(y_test))**2)
        residual_sum_squares = np.sum((y_test - y_hat)**2)
        return 1 - residual_sum_squares/total_sum_squares