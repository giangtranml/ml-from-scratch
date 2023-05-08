import numpy as np
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression

def standardize_regression(X, y):
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    y_mean = np.mean(y)
    y_std = np.std(y)
    return ((X - x_mean)/x_std, x_mean, x_std), ((y - y_mean) / y_std, y_mean, y_std)


def main():
    X = np.loadtxt('prostate.data.txt', skiprows=1)
    columns = ['lcavol', 'lweight',	'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
    y = X[:, -1]
    X = X[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    (X_train, _, _), (y_train, _, _) = standardize_regression(X_train, y_train)
    y_train = y_train.reshape((-1, 1))

    alpha = 0.01
    epochs = 500
    lambda_ = 0
    linear_regression = LinearRegression(alpha, epochs, lambda_)
    linear_regression.train(X_train, y_train)

    (X_test, x_mean, x_std), (y_test, y_mean, y_std) = standardize_regression(X_test, y_test)
    pred = linear_regression.predict(X_test)
    y_test = y_test.reshape((-1, 1))
    print("Test score: %f" % linear_regression.r2_score(pred, y_test))


if __name__ == '__main__':
    main()

