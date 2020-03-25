import io
import imageio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Visualize the convergence of gradient descent of linear regression on 2D space.
    """
    def __init__(self, alpha=0.01, noise=10, degree=1, server=False):
        self.alpha = alpha
        self.threshold = 0.5
        self.noise = noise
        self.server = server
        self.degree = degree
        if server:
            self.imgs = []
        else:
            matplotlib.use('Tkagg')
            plt.ion()

    def _prepare_data(self, num_points=100):
        X = np.linspace(-2, 2, num_points)
        X = X.reshape(-1, 1)
        coef = []
        for d in range(self.degree):
            coef.append(np.random.uniform(-20, 40))
        coef.append(np.random.uniform(0, 2))
        coef = np.array(coef)
        X_transform = self._transform_space(X)
        bias = np.ones((X.shape[0], 1))
        X_transform = np.concatenate((X_transform, bias), axis=1)
        y = X_transform.dot(coef).reshape((num_points, 1)) + np.random.uniform(1, self.noise, (num_points, 1))
        return X, X_transform, y

    def _hypothesis(self, X, theta):
        assert X.shape[1] == theta.shape[0], "Incorrect shape."
        return np.dot(X, theta)

    def _cost(self, X, y):
        # either use np.linalg.norm(y_hat - y, 2)**2
        # or use (y_hat - y)**2
        m = y.shape[0]
        return .5/m*np.sum((self._hypothesis(X, self.theta) - y)**2)

    def _gradient(self, X, y):
        m = X.shape[0]
        return 1/m*np.dot(X.T, self._hypothesis(X, self.theta) - y)

    def _transform_space(self, X):
        X_temp = X[:]
        for d in range(2, self.degree + 1):
            X_temp = np.concatenate(((X[:, 0] ** d).reshape(-1, 1), X_temp), axis=1)
        return X_temp

    def _gradient_descent(self, X, y, alpha):
        previous_theta = None
        cost = []
        thetas = []
        iterations = []
        while True:
            if previous_theta is None or np.mean(np.abs(self.theta - previous_theta)) >= self.threshold:
                previous_theta = self.theta
                thetas.append(self.theta)
                cost.append(self._cost(X, y))
                iterations.append(self.iteration)
            self.theta = self.theta - alpha*self._gradient(X, y)
            if abs(np.mean(self._gradient(X, y))) < 1e-6:
                break
            self.iteration += 1
        return cost, thetas, iterations

    def _plot(self, theta, cost, iteration, X, X_transform, y):
        y_plot = self._hypothesis(X_transform, theta)
        plt.figure(0, figsize=(6, 6))
        plt.clf()
        plt.title("Cost: " + str(cost))
        plt.scatter(X[:, 0], y, color='r')
        label = "Iteration: " + str(iteration)
        for ind, t in enumerate(theta):
            label += "\nTheta %s: %.2f" % (ind, t)
        plt.plot(X, y_plot, '-', label=label)
        plt.legend()
        if not self.server:
            plt.show()
            plt.pause(0.5)
        else:
            self.save_fig(plt)

    def save_fig(self, plot):
        img_buf = io.BytesIO()
        plot.savefig(img_buf, format='png')
        img_buf.seek(0)
        self.imgs.append(imageio.imread(img_buf))

    def _train(self, X, y):
        self.theta = np.random.normal(size=(X.shape[1], 1))
        self.iteration = 0
        return self._gradient_descent(X, y, self.alpha)

    def exec(self):
        X, X_transform, y = self._prepare_data()
        cost, thetas, iterations = self._train(X_transform, y)
        for c, t, i in zip(cost, thetas, iterations):
            self._plot(t, c, i, X, X_transform, y)
        if self.server:
            imageio.mimsave("linear_regression_" + str(self.degree) +".gif", self.imgs, fps=5)
        else:
            plt.pause(10)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Linear Regression Visualization.")
    parser.add_argument("--save", action="store_true", help="Use this option to save in a file.")
    args = parser.parse_args()

    lr = LinearRegression(alpha=0.01, noise=20, degree=1, server=args.save)
    lr.exec()
    lr = LinearRegression(alpha=0.01, noise=30, degree=2, server=args.save)
    lr.exec()