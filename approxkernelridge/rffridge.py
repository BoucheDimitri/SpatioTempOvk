import numpy as np
from sklearn.linear_model import Ridge


class RandomFourierFeatures:

    def __init__(self, sigma, D, d, seed=0):
        self.sigma = sigma
        self.D = D
        self.d = d
        np.random.seed(seed)
        self.w = np.random.normal(0, 1, (self.d, self.D))
        np.random.seed(seed)
        self.b = np.random.uniform(0, 2 * np.pi, (1, self.D))
        self.seed = seed

    def eval(self, X):
        return np.sqrt(2 / self.D) * np.cos(self.sigma * X.dot(self.w) + self.b)

    def add_features(self, nfeatures):
        np.random.seed(self.seed)
        wadd = np.random.normal(0, 1, (self.d, nfeatures))
        np.random.seed(self.seed)
        badd = np.random.uniform(0, 2 * np.pi, (1, nfeatures))
        self.w = np.concatenate((self.w, wadd), axis=1)
        self.b = np.concatenate((self.b, badd), axis=1)
        self.D = self.D + nfeatures

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
