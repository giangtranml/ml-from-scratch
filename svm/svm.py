"""
Author: Giang Tran.
"""
from cvxopt import matrix, solvers
import numpy as np
from scipy.spatial.distance import cdist


class SVM:

    kernels = {"linear": "_linear_kernel", "poly": "_polynomial_kernel", "rbf": "_gaussian_kernel",
               "sigmoid": "_sigmoid_kernel"}

    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='auto', r=0.0,debug=False, is_saved=False):
        self.C = C
        if kernel not in list(self.kernels.keys()):
            self.kernel = 'linear'
        else:
            self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.r = r
        self.debug = debug
        self.is_saved = is_saved
        self.support_vectors, self.dual_coef = None, None

    def _linear_kernel(self, x, z):
        return np.dot(x, z.T)

    def _polynomial_kernel(self, x, z):
        """
        k(x, z) = (r + gamma*(np.dot(x, z))**degree
        """
        return (self.r + self.gamma*np.dot(x, z.T))**self.degree

    def _gaussian_kernel(self, x, z):
        """
        k(x, z) = exp(-gamma*(norm_2(x, z)**2))
        """
        return np.exp(-self.gamma*(cdist(x, z)**2))

    def _sigmoid_kernel(self, x, z):
        """
        k(x, z) = tanh(gamma*np.dot(x, z) + r)
        """
        def tanh(s):
            return (np.exp(s)-np.exp(-s))/(np.exp(s)+np.exp(-s))
        return tanh(self.gamma*np.dot(x, z.T) + self.r)

    def _solve_lagrange_dual_function(self, X_train, y_train):
        """
        note: maximize f(x) <==> minimize -f(x)

        ==> Using cvxopt.qp

        minimize: f(x) = (1/2)*x'*P*x + q'*x
        s.t: G*x <= h
             A*x = b

        V = [x_1*y_1, x_2*y_2, ..., x_n*y_n]

        P = V.T * V
        q = (-1).T

        G = [[-1, 0, 0, ..., 0],
             [0, -1, 0, ..., 0],
             [0, 0, -1, ..., 0],
             ....              ,
             [0, 0, 0, ..., -1],
             --------------
             [1, 0, 0, ..., 0],
             [0, 1, 0, ..., 0],
             [0, 0, 1, ..., 0],
             ....             ,
             [0, 0, 0, ..., 1]]

            => G.shape = (2*N, N)

        h = [[0, 0, 0, ..., 0].T, [C, C, C, ..., C].T]

            => h.shape = (2*N, 1)

        A = y

        b = np.zeros(N)

        """
        N, D = X_train.shape

        X = getattr(self, self.kernels[self.kernel])(X_train, X_train)
        y = y_train.dot(y_train.T)
        P = matrix(X*y)  # shape = (N, N)
        q = matrix(-np.ones((N, 1)))

        G = matrix(np.concatenate((-np.eye(N), np.eye(N)), axis=0))  # shape = (2N, N)
        h = matrix(np.array([0] * N + [self.C] * N).reshape(-1, 1))  # shape = (2N, 1)

        A = matrix(y_train.T)
        b = matrix(np.zeros((1, 1)))

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)

        lambda_ = np.array(sol['x'])
        return lambda_

    def _solve_svm(self, X, y, lambda_):
        """
        lambda_: sparse vector we found from solving lagrange dual function above.
        --------------------------------------------------------------------------------------------------------
        Let S = {n: 0 < lambda_n <= C (epsilon < lambda_n <= C)} support vectors set use for compute w.
        Let M = {m: 0 < lambda_m < C (epsilon < lambda_m < C)} points that lie exactly on margins, use for compute b.
        """
        epsilon = 1e-6
        S = np.where(np.logical_and(lambda_ > epsilon, lambda_ <= self.C))[0]
        M = np.where(np.logical_and(lambda_ > epsilon, lambda_ < self.C))[0]
        X_S = X[S, :]
        y_S = y[S, :]
        lambda_S = lambda_[S]
        X_M = X[M, :]
        y_M = y[M, :]
        return X_S, lambda_S*y_S, X_M, y_M

    def _train(self, X_train, y_train):
        """
        Solve SVM by using Lagrange duality

        g(z) = -1/2*z'*V'*V*z + 1'*z

        z = argmax_z g(z) <==> argmin_z -g(z)

        z = argmin_z (1/2)*z'*V'*V*z - 1'*z
        s.t: -z <= 0
              z <= C
              y'*z = 0

        After get z, find w, b.
        ----------------------------------------
        """
        if self.gamma == 'auto':
            self.gamma = 1/X_train.shape[1]
        # elif self.gamma == 'scale':
        #     self.gamma = (1/X_train.shape[1])*np.var(X_train, axis=0)
        lambda_ = self._solve_lagrange_dual_function(X_train, y_train)
        return self._solve_svm(X_train, y_train, lambda_)

    def train(self, X_train, y_train):
        assert len(np.unique(y_train)) == 2, "This SVM assumes only work for binary classification."
        assert type(X_train) is np.ndarray and type(y_train) is np.ndarray, \
            "Expect numpy array but got %s" % (type(X_train) if type(X_train) is not np.ndarray else type(y_train))
        self.support_vectors, self.dual_coef, self.X_M, self.y_M = self._train(X_train, y_train)
        if self.debug:
            self._check_with_sklearn(X_train, y_train)

    def _check_with_sklearn(self, X, y):
        print("-"*50)
        print("------------ Training phrase --------------")
        print("My SVM support vectors:", self.support_vectors)

        from sklearn.svm import SVC
        sk_svm = SVC(C=self.C, gamma=self.gamma, kernel=self.kernel, degree=self.degree, coef0=self.r)
        sk_svm.fit(X, y)
        print("Sk-learn SVM support vectors:", sk_svm.support_vectors_)
        print("-"*50)

    def decision(self, X_test):
        assert type(X_test) is np.ndarray, "Expect numpy array but got %s" % (type(X_test))
        w = self.dual_coef.T.dot(getattr(self, self.kernels[self.kernel])(self.support_vectors, X_test))
        b = np.mean(self.y_M - self.dual_coef.T.dot(getattr(self, self.kernels[self.kernel])(self.support_vectors, self.X_M)))
        pred = w + b
        return pred

    def predict(self, X_test):
        """
        w = np.dot(dual_coef, kernel(support_vector, X_test))
        b = (1/N_M)*sum_M(y_M - sum_S(dual_coef * kernel(support_vector, X_M)))
        """
        pred = self.decision(X_test)
        pred[pred >= 0] = 1
        pred[pred < 0] = -1
        return pred
