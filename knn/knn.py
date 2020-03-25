"""
Author: Giang Tran.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import itemfreq
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    """
    Characteristics of KNN algorithm:
        - Non-parametric ML algorithm. It means KNN doesn't have training phase to find optimal parameter.
        - Therefore, training phase and predicting phase in 1 phase make the algorithm very slow as the data larger
            and larger, also the dimension of vector is big.
        - ...

    Idea of KNN algorithm:
        - Given a specific dataset, for each row is a n-D dimension vector and the label.
        - Pre-processing dataset (normalization, standardization) into same scale (optional).
        - For any new point in predicting phase, the algorithm finds the distance between that point and all other
            points in training set (L_1, L_2, L_inf).
        - Base on K hyper-parameter, the algorithm will find K nearest neighbor and classify that point into
            which class.

    """
    _metrics = {'euclidean': '_l2_distance', 'manhattan': '_l1_distance', 'cosine': '_cosine_similarity'}

    def __init__(self, K, X, y, metric='euclidean'):
        self.K = K
        assert type(X) is np.ndarray, "X must be a numpy array"
        assert type(y) is np.ndarray, "y must be a numpy array"
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.metric = metric

    def _l1_distance(self, X_new):
        """
        l1 = abs(x_1 - x_2) + abs(y_1 - y_2)
        :param X_new:
        :return: ndarray manhattan distance of X_new versus all other points X.
        """
        return cdist(X_new, self.X, 'cityblock')

    def _l2_distance(self, X_new):
        """
        l2 = sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2)
        :param X_new:
        :return: ndarray euclidean distance of X_new versus all other points X.
        """
        return cdist(X_new, self.X, 'euclidean')

    def _cosine_similarity(self, X_new):
        """
        similarity = cos(alpha) = dot(A, B) / (len(A)*len(B))
        :param X_new:
        :return: ndarray cosine similarity of X_new versus all other points X.
        """
        return cdist(X_new, self.X, 'cosine')

    def predict(self, X_new):
        assert type(X_new) is np.ndarray, "Use numpy array instead."
        assert X_new.shape[1] == self.X.shape[1], "Mismatch shape."
        if self.metric not in self._metrics.keys():
            self.metric = 'euclidean'
        func = getattr(self, self._metrics[self.metric])
        dist = func(X_new)
        dist = np.argsort(dist, axis=1)
        k_nearest = dist[:, :self.K]
        labels = self.y[k_nearest]
        res = []
        for label in labels:
            label, count = np.unique(label, return_counts=True)
            res.append(label[np.argmax(count)])
        return np.array(res)


def experiment(X, y, X_test, y_test):
    print("--- Experiment ---")
    ks = [1, 3, 5, 7, 9, 11]
    metrics = ['manhattan', 'euclidean', 'cosine']
    for metric in metrics:
        for k in ks:
            knn = KNN(k, X, y, metric=metric)

            y_pred = knn.predict(X_test)

            print("KNN with K = %d and metric = %s | Accuracy: %f" % (k, metric, len(y_test[y_pred == y_test]) / len(y_test)))

        print("-"*50)

def main():
    df = pd.read_csv("./data/train.csv")
    X = df.loc[:, :].values
    y = pd.read_csv("./data/trainDirection.csv").iloc[:, 0].values

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    df_test = pd.read_csv("./data/testing.csv")
    X_test = df_test.drop('Direction', axis=1).iloc[:, 1:].values
    y_test = df_test.loc[:, 'Direction'].values

    print("X test shape:", X_test.shape)
    print("y test shape:", y_test.shape)

    debug = True

    if debug:
        experiment(X, y, X_test, y_test)
        return

    k = 3

    knn = KNN(k, X, y, metric='manhattan')

    pred = knn.predict(X_test)

    print("My KNN accuracy:", len(y_test[pred == y_test]) / len(y_test))

    # Check with Sk learn KNN.
    sk_knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    sk_knn.fit(X, y)

    y_sk = sk_knn.predict(X_test)

    print("Sk-learn KNN accuracy:", len(y_test[y_sk == y_test]) / len(y_test))


if __name__ == '__main__':
    main()
