import sys
sys.path.append('..')

import numpy as np
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from whp_decorate import func_cost
from utils import plot_x_y, load_iris

class SvmModel:
    def __init__(self):
        self.standrad = StandardScaler()
        self.ployfea = PolynomialFeatures(degree=3)
        self.lsvc = LinearSVC(C=10, loss='hinge')
        self.poly_svc = SVC(kernel='poly', degree=3, coef0=1, C=5)
        self.rbs_svc = SVC(kernel='rbf', gamma=5, C=0.001)
        self.svr = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
        X, Y = load_iris(return_xy=True)
        self.X = X[:, (2, 3)]
        self.Y = (Y == 2).astype(np.float64)    # Iris-Virginica

    def pipeline(self, steps):
        return Pipeline(steps=steps)

    def linear_svc(self):
        pass

    def msvm_demo(self):
        # model = self.pipeline(
        #     steps=[
        #         ('ploy_features', self.ployfea),
        #         ('scalar', self.standrad),
        #         ('linear_svc', self.lsvc)
        #     ]
        # )
        model = self.pipeline(
            steps=[
                ("scaler", self.standrad),
                ("poly_svc", self.poly_svc)
            ]
        )
        model.fit(self.X, self.Y)
        print(model.predict([[5.5, 1.7]]))

    def lsvc_demo(self):
        model = self.pipeline(
            steps=[
                ('scalar', self.standrad),
                ('linear_svc', self.lsvc)
            ]
        )
        model.fit(self.X, self.Y)
        print(model.predict([[5.5, 1.7]]))
        