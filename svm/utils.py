from scipy import io
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
import numpy as np
import re
from nltk.stem import PorterStemmer


def load_mat_file(mat_file):
    mat = io.loadmat(mat_file)
    if 'Xtest' in mat:
        return mat['Xtest'], mat['ytest']
    return mat['X'], mat['y']


class Plotter:

    def __init__(self, X, y, interactive='off'):
        if interactive == 'on':
            plt.ion()
        self.X = X
        self.y = y

    def plot_data(self):
        y = self.y.reshape((-1, ))
        class_1 = self.X[y == -1]
        class_2 = self.X[y == 1]
        plt.scatter(class_1[:, 0], class_1[:, 1], c='r', marker='+')
        plt.scatter(class_2[:, 0], class_2[:, 1], c='b', marker='o')
        plt.show()

    def visualize_boundary_linear(self, W, b):
        W = W.T.reshape((2, ))
        xp = np.linspace(np.min(self.X[:, 0]), np.max(self.X[:, 0]), 100)
        yp = -(W[0] * xp + b)/W[1]
        plt.plot(xp, yp)
        plt.pause(10)

    def visualize_boundary(self, svm_model):
        x1_plot = np.linspace(np.min(self.X[:, 0]), np.max(self.X[:, 0]), 100)
        x2_plot = np.linspace(np.min(self.X[:, 1]), np.max(self.X[:, 1]), 100)
        X1, X2 = np.meshgrid(x1_plot, x2_plot)
        vals = np.zeros(X1.shape)
        for i in range(X1.shape[1]):
            this_X = np.array([X1[:, i], X2[:, i]]).T
            vals[:, i] = svm_model.predict(this_X)
        plt.contour(X1, X2, vals, [0])
        plt.show()
        plt.pause(20)


def load_vocabulary(file_name):
    vocabs = []
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            _, word = line.strip().split('\t')
            vocabs.append(word)
    return vocabs


def process_email(email_content, vocabs):
    word_indices = []
    email_content = email_content.lower()
    email_content = re.sub('<[^<>]+>', ' ', email_content)
    email_content = re.sub('[0-9]+', 'number', email_content)
    email_content = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_content)
    email_content = re.sub('(http|https)://[^\s]*', 'httpaddr', email_content)
    email_content = re.sub('[$]+', 'dollar', email_content)
    email_content = re.sub('[^a-zA-Z0-9]', ' ', email_content)

    ps = PorterStemmer()
    list_words = email_content.split(' ')
    for word in list_words:
        word = ps.stem(word)
        if len(word) < 1:
            continue
        if word in vocabs:
            word_indices.append(vocabs.index(word))
    return word_indices


def email_feature(word_indices, vocabs):
    n = len(vocabs)
    x = np.zeros((n, ))
    for indices in word_indices:
        x[indices] = 1
    return x