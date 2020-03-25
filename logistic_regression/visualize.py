import io
import imageio
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, alpha=0.01, server=False):
        self.alpha = alpha
        self.server = server
        self.threshold = 0.01
        if server:
            self.imgs = []
        else:
            matplotlib.use('Tkagg')
            plt.ion()

    def _prepare_data(self, num_points=100):
        X1 = np.random.multivariate_normal([5, 6], [[5, 1], [1, 5]], num_points)
        X2 = np.random.multivariate_normal([14, 15], [[4, 0], [0, 4]], num_points)
        y1 = np.ones(shape=(num_points, 1))
        y2 = np.zeros(shape=(num_points, 1))
        return X1, X2, y1, y2

    def _hypothesis(self, X, w):
        return X.dot(w)

    def _sigmoid(self, X, w):
        z = self._hypothesis(X, w)
        return 1/(1+np.exp(-z))

    def _cross_entropy_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1 - y_pred))/m

    def _gradient(self, X, y_true, y_pred):
        m = X.shape[0]
        return (X.T.dot(y_pred - y_true))/m

    def _gradient_descent(self, X, y, alpha):
        previous_w = None
        cost = []
        thetas = []
        iterations = []
        while self.iteration < 2000:
            y_pred = self._sigmoid(X, self.w)
            if previous_w is None or np.mean(np.abs(self.w - previous_w)) >= self.threshold:
                previous_w = self.w
                thetas.append(self.w)
                cost.append(self._cross_entropy_loss(y, y_pred))
                iterations.append(self.iteration)
            self.w = self.w - alpha*self._gradient(X, y, y_pred)
            if np.linalg.norm(self._gradient(X, y, y_pred), 2) < 1e-12:
                break
            self.iteration += 1
        return cost, thetas, iterations

    def _plot(self, theta, cost, iteration, X, num_points):
        X1_plot = np.linspace(0, 20, num=num_points).reshape((-1, 1))
        X2_plot = np.linspace(0, 20, num=num_points).reshape((-1, 1))
        bias = np.ones(shape=(num_points, 1))
        X_plot = np.concatenate((X1_plot, X2_plot, bias), axis=1)
        y_plot = self._sigmoid(X_plot, theta)
        y_plot = np.squeeze(y_plot)
        X1 = X[:num_points, :]
        X2 = X[num_points:, :]
        y = self.predict(X, theta)
        y1 = y[:num_points, :]
        y2 = y[num_points:, :]
        fig = plt.figure(0, figsize=(6, 6))
        plt.clf()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("Loss: " + str(cost))
        ax.scatter(X1[:, 0], X1[:, 1], y1, c="b", label="Class 1")
        ax.scatter(X2[:, 0], X2[:, 1], y2,  c="r", label="Class 0")
        label = "Iteration: " + str(iteration)
        ax.plot(X1_plot, X2_plot, y_plot, '-', label=label)
        ax.set_xlabel("x axis")
        ax.set_ylabel("y axis")
        ax.set_zlabel("z axis")
        plt.legend()
        if not self.server:
            plt.show()
            plt.pause(0.1)
        else:
            self.save_fig(plt)

    def save_fig(self, plot):
        img_buf = io.BytesIO()
        plot.savefig(img_buf, format='png')
        img_buf.seek(0)
        self.imgs.append(imageio.imread(img_buf))

    def _train(self, X, y):
        self.w = np.random.normal(size=(X.shape[1], 1))
        self.iteration = 0
        return self._gradient_descent(X, y, self.alpha)

    def exec(self):
        num_points = 100
        X1, X2, y1, y2 = self._prepare_data(num_points=num_points)
        X = np.concatenate((X1, X2), axis=0)
        X = np.concatenate((X, np.ones(shape=(2*num_points, 1))), axis=1)
        y = np.concatenate((y1, y2), axis=0)
        cost, thetas, iterations = self._train(X, y)
        for c, t, i in zip(cost, thetas, iterations):
            self._plot(t, c, i, X, num_points)
        if self.server:
            imageio.mimsave("logistic_regression.gif", self.imgs, fps=5)
        else:
            plt.pause(10)

    def predict(self, test_X, w):
        """
        Output sigmoid value of trained parameter w, b.
        Choose threshold 0.5
        """
        pred = self._sigmoid(test_X, w)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Logistic Regression Visualization.")
    parser.add_argument("--save", action="store_true", help="Use this option to save in a file.")
    args = parser.parse_args()
    lr = LogisticRegression(server=args.save)
    lr.exec()