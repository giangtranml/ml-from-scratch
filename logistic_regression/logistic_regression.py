"""
Author: Giang Tran
Email: giangtran240896@gmail.com
Docs: https://giangtranml.github.io/ml/logistic-regression.html
"""
import pandas as pd
import re
import json
import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.append("..")
from optimizations_algorithms.optimizers import SGD


class LogisticRegression:

    def __init__(self, epochs, optimizer, batch_size):
        """
        Constructor for logistic regression.

        Parameter
        ---------
        epochs: number of epoch to train logistic regression.
        optimizer: optimizer algorithm to update weights.
        batch_size: number of batch size using each iteration.
        """
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size

    def _sigmoid(self, X):
        """
        Sigmoid activation function:
            s(z) = 1/(1+e^-x)

        Parameter
        ---------
        X: matrix of dataset. shape = (n, d) with n is number of training, d
            is dimension of each vector.

        Return
        ---------
        s(x): value of activation.
        """
        assert X.shape[1] == self.w.shape[0], "Invalid shape."
        z = X.dot(self.w)
        return 1/(1+np.exp(-z))

    def _cross_entropy_loss(self, y_true, y_pred):
        """
        Compute cross entropy loss.
        """
        m = y_true.shape[0]
        return -np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1 - y_pred))/m

    def _gradient(self, X, y_true, y_pred):
        """
        Compute gradient of J with respect to `w`.
        """
        m = X.shape[0]
        return (X.T.dot(y_pred - y_true))/m

    def _train(self, X_train, y_train):
        """
        Main training function. 
        """
        for e in range(self.epochs):
            batch_loss = 0
            num_batches = 0
            it = 0
            while it < X_train.shape[0]:
                y_hat = self._sigmoid(X_train[it:it+self.batch_size])
                loss = self._cross_entropy_loss(y_train[it:it+self.batch_size], y_hat)
                batch_loss += loss
                grad = self._gradient(X_train[it:it+self.batch_size], y_train[it:it+self.batch_size], y_hat)
                self.w -= self.optimizer.minimize(grad)
                it += self.batch_size
                num_batches += 1
            print("Loss at epoch %s: %f" % (e + 1 , batch_loss / num_batches))

    def train(self, train_X, train_y):
        """
        Wrapper training function, check the prior condition first.
        """
        assert type(train_X) is np.ndarray, "Expected train X is numpy array but got %s" % type(train_X)
        assert type(train_y) is np.ndarray, "Expected train y is numpy array but got %s" % type(train_y)
        train_y = train_y.reshape((-1, 1))
        self.w = np.random.normal(size=(train_X.shape[1], 1))
        self._train(train_X, train_y)

    def predict(self, test_X):
        """
        Output sigmoid value of trained parameter w, b.
        Choose threshold 0.5
        """
        pred = self._sigmoid(test_X)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

def clean_sentences(string):
    label_chars = re.compile("[^A-Za-z \n]+")
    string = string.lower()
    return re.sub(label_chars, "", string)


def main():
    df = pd.read_csv("data/amazon_baby_subset.csv")
    reviews = df.loc[:, 'review'].values
    for ind, review in enumerate(reviews):
        if type(review) is float:
            reviews[ind] = ""

    reviews = clean_sentences("\n".join(reviews))
    with open("data/important_words.json") as f:
        important_words = json.load(f)
    reviews = reviews.split("\n")
    n = len(reviews)
    d = len(important_words)
    X = np.zeros((n, d))
    y = df.loc[:, 'sentiment'].values
    y[y == -1] = 0

    for ind, review in enumerate(reviews):
        for ind_w, word in enumerate(important_words):
            X[ind, ind_w] = review.count(word)
    ones = np.ones((n, 1))
    X = np.concatenate((X, ones), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    epochs = 20
    learning_rate = 0.1
    batch_size = 64
    optimizer = SGD(alpha=learning_rate)
    logistic = LogisticRegression(epochs, optimizer, batch_size)
    logistic.train(X_train, y_train)
    pred = logistic.predict(X_test)
    y_test = y_test.reshape((-1, 1))
    print("Accuracy: " + str(len(pred[pred == y_test])/len(pred)))

if __name__ == '__main__':
    main()
