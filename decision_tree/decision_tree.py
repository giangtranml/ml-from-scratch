"""
Author: Giang Tran.
"""

import numpy as np
from math import log2


class NodeDT:
    """
    Class Node represents in Decision Tree
    """

    def __init__(self, X, y, feature_name):
        self.feature_name = feature_name
        self.X = X
        self.y = y
        self.is_leaf = False
        self.label = None
        self.used = []

    def entropy(self):
        """
        Compute entropy at a given node.
        E(X) = - sum_v(p(X_v) * log_2(p(X_v))) with X_v is a subset of X = (X_1, X_2, ..., X_n)
        :return: entropy coefficient.
        """
        n = len(self.y)
        sum_ = 0
        for i in np.unique(self.y):
            v = len(self.y[self.y == i])
            sum_ += -((v/n) * log2(v/n))
        return sum_

    def classification_error(self):
        pass


class DecisionTree:
    """
    Metrics: either entropy/information gain or classification error.
    """

    _metrics = {'ce': '_classification_error', 'ig': '_information_gain'}

    def __init__(self, max_depth=None, criterion='ig'):
        """
        :param max_depth: define what depth of the tree should be.
        :param criterion: either 'ce' or 'ig'.
        """
        self.max_depth = max_depth
        self.criterion = criterion
        if self.criterion not in self._metrics.keys():
            self.criterion = 'ig'
        self.num_class = 0
        self.tree = None
        self.thresholds = {}

    def _is_numerical(self, feature):
        return len(np.unique(feature)) >= 100

    def _find_threshold(self, feature, y_train, num_class):
        """
        The main point is find a good threshold that is the optimal split label.
        A good threshold is the threshold that minimize mis-classification error.

        The algorithm:
            - If there are `n` data points in feature data => there are `n-1` available thresholds.
            - For each available threshold, split feature data to 2 partitions.
            - For each partition, we check and compute mis-classification error for each label.

        :param feature: numerical value of `feature`.
        :param y_train: label.
        :param num_class: number of class
        :return: categorical value of `feature`.
        """
        assert len(num_class) == 2, "This function only assumes work with binary classification."
        best_threshold = 0.0
        max_exact_classification = 0.0
        is_positive_negative = False
        sorted_feature = sorted(np.unique(feature))
        for i in range(len(sorted_feature)-1):
            # assume the value less than threshold is negative (0), greater than threshold is positive (1)
            threshold = (sorted_feature[i] + sorted_feature[i+1]) / 2
            left_partition = y_train[feature < threshold]
            right_partition = y_train[feature > threshold]
            negative_positive = ((len(left_partition[left_partition == 0]) + len(right_partition[right_partition == 1]))
                                 / len(feature))
            # assume the value less than threshold is positive (1), greater than threshold is negative. (0)
            positive_negative = ((len(left_partition[left_partition == 1]) + len(right_partition[right_partition == 0]))
                                 / len(feature))
            # make decision here
            is_positive_negative = positive_negative > negative_positive
            choose = positive_negative if is_positive_negative else negative_positive
            if max_exact_classification < choose:
                max_exact_classification = choose
                best_threshold = threshold
        return best_threshold, is_positive_negative

    def _entropy(self, feature, node):
        """
        Compute entropy each partition of specific feature in a given node.
        :param feature: specific feature in dataset of `node`.
        :param node: a node we're checking on.
        :return: an entropy scalar that measure the uncertainty of a feature in data.
        """
        entropy = 0
        categories = np.unique(feature)
        num_point = len(feature)
        for category in categories:
            # for each category in that feature
            num_category = len(feature[feature == category])
            for c in self.num_class:
                # count the number of each class
                num_category_class = len(feature[np.logical_and(feature == category, node.y == c)])
                if num_category_class == 0:
                    continue
                # compute entropy/information gain or classification error
                entropy += num_category / num_point * (
                        -num_category_class / num_category * log2(num_category_class / num_category))
        return entropy

    def _information_gain(self, feature, node):
        """
        Compute information gain between a node with that feature.
        :param feature:
        :param node:
        :return: information gain coefficient.
        """
        return node.entropy() - self._entropy(feature, node)

    def _classification_error(self, feature, node):
        pass

    def _stop(self, node):
        """
        Stop condition:
            - Reach max depth or already reach all features.
            - If entropy of that node is 0
        :return: True if the node meets stop condition. False otherwise.
        """
        return len(node.used) == node.X.shape[1] or len(node.used) == self.max_depth or node.entropy() == 0

    def _build_dt(self, root, column_name):
        """
        Algorithm:
            - Start from the root. Find the best feature that has optimum entropy/information gain or classification error.
            - From that best feature, loop through all categories to build subtree.
            ...
            - If entropy/classification erorr is 0, or reach all features then that node is leaf or reach the max depth,
                then stop and move to other subtrees
        :param root: root node at current level
        :return:
        """
        N, D = root.X.shape
        best_coef = 0.0
        best_feature = 0
        for d in range(D):
            if column_name[d] in root.used:
                continue
            feature = root.X[:, d]
            coef = getattr(self, self._metrics[self.criterion])(feature, root)
            if best_coef < coef:
                best_coef = coef
                best_feature = d
        # after choose the best feature to split.
        # loop through all its categories to build subtree
        feature = root.X[:, best_feature]
        categories = np.unique(feature)
        for category in categories:
            node = NodeDT(root.X[feature == category], root.y[feature == category], column_name[best_feature])
            node.used = root.used + [column_name[best_feature]]
            setattr(root, 'feature_' + str(category), node)
            setattr(root, 'feature_split', best_feature)
            if not self._stop(node):
                self._build_dt(node, column_name)
            else:
                node.is_leaf = True
                node.label = 1 if len(node.y[node.y == 1]) >= len(node.y[node.y == 0]) else 0

    def _train(self, X_train, y_train, column_name):
        self.tree = NodeDT(X_train, y_train, 'root')
        self._build_dt(self.tree, column_name)

    def train(self, X_train, y_train, column_name):
        self.num_class = np.unique(y_train)
        _, D = X_train.shape
        for d in range(D):
            feature = X_train[:, d]
            if self._is_numerical(feature):
                threshold, is_positive_negative = self._find_threshold(feature, y_train, self.num_class)
                feature[feature < threshold] = int(is_positive_negative)
                feature[feature > threshold] = int(not is_positive_negative)
                X_train[:, d] = feature
                self.thresholds[d] = (threshold, is_positive_negative)
        self._train(X_train, y_train, column_name)

    def _predict(self, X_new, node):
        if not node.is_leaf:
            node = getattr(node, 'feature_' + str(X_new[node.feature_split]))
            return self._predict(X_new, node)
        return node.label

    def predict(self, X_new):
        # First convert numerical feature to categorical feature.
        for key, (threshold, is_positive_negative) in self.thresholds.items():
            X_new[key] = int(is_positive_negative) if X_new[key] < threshold else int(not is_positive_negative)
        tree = self.tree
        label = self._predict(X_new, tree)
        return label

    def representation(self):
        print(self.tree)
    

if __name__ == '__main__':
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier

    df = pd.read_csv('data/titanic_train.csv')
    X = df.loc[:, :].drop(['Survived', 'PassengerId'], axis=1).values
    y = df.loc[:, 'Survived'].values

    dt = DecisionTree(criterion='ig', max_depth=5)
    dt.train(X, y, df.columns.drop(['Survived', 'PassengerId']))

    df_test = pd.read_csv('data/titanic_test.csv')
    X_test = df_test.loc[:, :].drop(['Survived', 'PassengerId'], axis=1).values
    y_test = df_test.loc[:, 'Survived'].values
    predicts = []
    for x in X_test:
        predicts.append(dt.predict(x))
    predicts = np.asarray(predicts)
    print("Accuracy:", len(predicts[predicts == y_test])/len(predicts))

    dt_sk = DecisionTreeClassifier(max_depth=5)
    X[X[:, 7] == 'male', 7] = 1
    X[X[:, 7] == 'female', 7] = 0

    X_test[X_test[:, 7] == 'male', 7] = 1
    X_test[X_test[:, 7] == 'female', 7] = 0
    dt_sk.fit(X, y)
    y_pred = dt_sk.predict(X_test)
    print("Accuracy of Sk-learn:", len(y_pred[y_pred == y_test]) / len(y_pred))



