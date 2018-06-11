import sys
sys.path.append('..')

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from whp_decorate import func_cost
from utils import plot_x_y, load_iris

_show_cost=True

class LinearModel:
    def __init__(self):
        self.lr = LinearRegression()
        self.sgdr = SGDRegressor(penalty='l2')
        self.ridge = Ridge(alpha=1, solver="cholesky")   
        self.lasso = Lasso(alpha=0.1)
        self.elast = ElasticNet(alpha=0.1, l1_ratio=0.5)
        self.lgsr = LogisticRegression()
        self.softmax = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)

    def lgs_demo(self, plt):
        iris = load_iris()
        X = iris["data"][:, 3:]
        y = (iris["target"] == 2)
        self.lgsr.fit(X, y)
        X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
        y_proba = self.lgsr.predict_proba(X_new)
        plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
        plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
        plt.show()

    def lr_demo(self, r=100, c=1, theta = (4, 3), need_return=False, need_plot=True):
        if (c+1) != len(theta):
            raise Exception('ERROR: columns or theta counts error')
        theta = np.array(theta).reshape((c + 1, 1))
        noice = np.random.randn(r, 1)
        X = np.random.rand(r, c)
        y = theta[0] + np.dot(X,theta[1:,:]) + noice
        X_b = np.c_[np.ones((r, 1)), X]

        theta_hat = self.analytic_theta(X_b, y)
        if need_return:
            return X, X_b, y, theta

        self.lr.fit(X, y)
        print(theta_hat[0], ' - ', self.lr.intercept_)
        print(theta_hat[1:], ' - ', self.lr.coef_)
        
        gd_theta = self.gradient_decent(y, X_b, theta)
        print(theta.T, '\n', gd_theta.T)

        sgd_theta = self.stochastic_gradient_descent(y, X_b, theta)
        print(theta.T, '\n', sgd_theta.T)
    
        if need_plot:
            y_pred = X_b.dot(theta_hat)
            y_gd = X_b.dot(gd_theta)
            y_sgd = X_b.dot(sgd_theta)
            X = X_b[:,1]
            plot_x_y(X, y, min(X), max(X), min(y), max(y), label='normal')
            plot_x_y(X, y_pred, min(X), max(X), min(y), max(y), kind='r.', label='pred')
            plot_x_y(X, y_gd, min(X), max(X), min(y), max(y), kind='g.', label='gd')
            plot_x_y(X, y_sgd, min(X), max(X), min(y), max(y), kind='y.', label='sgd', _show=True)

    def poly_demo(self, r=100, c=1, need_plot=True):
        
        X = np.random.rand(r, c)
        y = self.polynomial(X, [2, 6.5, 0.5])

        X_poly = self.polynomial_fit(X, 2)
        mod = self.lr.fit(X_poly, y)
        y_pred = self.polynomial(X, [mod.coef_[:,0], mod.coef_[:,1], mod.intercept_]) 
        if need_plot:
            plot_x_y(X, y, min(X), max(X), min(y), max(y), label='random')
            plot_x_y(X_poly[:,0], y_pred, min(X), max(X), min(y), max(y), kind='r.', label='reg', _show=True)

    def polynomial(self, X, a:list):
        index = len(a)
        y = 0
        for i in range(index):
            y += a[i] * (X**i)
        return y

    def polynomial_fit(self, X, idx, include_bias=False):
        poly_features = PolynomialFeatures(degree=idx,include_bias=include_bias)
        return poly_features.fit_transform(X)

    @func_cost(show=_show_cost)
    def analytic_theta(self, X_b, y):
        theta_hat = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        return theta_hat

    @func_cost(show=_show_cost)
    def gradient_decent(self, y_real, x, theta, eta=0.1, epochs=100, threshold=0.01):
        m = len(y_real)
        gradient = 0
        for i in range(epochs):
            next_gradient = 2 / m * x.T.dot(x.dot(theta) - y_real)
            theta = theta -  eta * next_gradient
            gap = sum(abs(next_gradient - gradient))
            if gap < threshold:
                print('less than threshold in index: ', i)
                break
            gradient = next_gradient
        return theta

    @func_cost(show=_show_cost)
    def stochastic_gradient_descent(self, y_real, x, theta, t0=(5, 50), epochs=50, threshold=0.01):
        _learning_rate = lambda t, t0: (t0[0] / (t0[1] + t))
        gradient = 0
        m = len(y_real)
        for e in range(epochs):
            gap = 0
            for i in range(m):
                idx = np.random.randint(m)
                xi = x[idx:idx+1]
                yi = y_real[idx:idx+1]
                next_gradient = 2 * xi.T.dot(xi.dot(theta) - yi)
                eta = _learning_rate(e * m + i, t0)
                theta = theta - eta * next_gradient
                gap += sum(abs(next_gradient - gradient))
                gradient = next_gradient
            if gap < threshold:
                print('less than threshold in epoach: {}'.format(e))
                break
        return theta

