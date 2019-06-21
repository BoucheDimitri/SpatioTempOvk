import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from functools import partial


class RandomFourierFeatures:

    def __init__(self, sigma, D, d):
        self.sigma = sigma
        self.D = D
        self.d = d
        self.w = np.random.normal(0, 1, (self.d, self.D))
        self.b = np.random.uniform(0, 2 * np.pi, (1, self.D))

    def eval(self, X):
        return np.sqrt(2 / self.D) * np.cos(self.sigma * X.dot(self.w) + self.b)

    def get_feature(self, i):
        return lambda x: np.sqrt(2 / self.D) * np.cos(self.sigma * self.w[:, i].dot(x) + self.b[:, i])

    def features_basis(self):
        return [self.get_feature(i) for i in range(self.D)]


class RFFRidge:

    def __init__(self, lamb, rffeats):
        self.lamb = lamb
        self.rffeats = rffeats
        self.w = None

    def fit(self, X, y):
        Z = self.rffeats.eval(X)
        ridge = Ridge(alpha=self.lamb, fit_intercept=False)
        ridge.fit(Z, y)
        self.w = ridge.coef_.flatten()

    def predict(self, X):
        Z = self.rffeats.eval(X)
        return Z.dot(self.w)


# n = 10000
# samp = np.linspace(0, 1, n)
# y = np.array([0.5 * np.cos(2 * np.pi * x) + 0.5 * np.cos(4* np.pi * x) + np.random.normal(0, 0.1) for x in samp]).reshape((n, 1))
# X = samp.reshape((n, 1))
#
# feats = RandomFourierFeatures(10, 50, 1)
# test = RFFRidge(1, feats)
# test.fit(X, y)
# ypred = test.predict(X)
#
# plt.plot(samp, ypred)
# plt.scatter(samp, y)