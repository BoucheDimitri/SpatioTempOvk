import numpy as np


class FourierDict:

    @staticmethod
    def cos_atom(n, P, x):
        return np.cos(2 * np.pi * n * x / P)

    @staticmethod
    def sin_atom(n, P, x):
        return np.sin(2 * np.pi * n * x / P)

    @staticmethod
    def sin_atom_prime(n, P, x):
        return (2 * np.pi * n / P) * np.cos(2 * np.pi * n * x / P)

    def __init__(self, interval, nfreq, d=1):
        self.P = interval[1] - interval[0]
        self.D = 2 * nfreq
        self.nfreq = nfreq
        self.d = d
        nvec = np.arange()

    def __init__(self, alpha, nfreq, interval):
        self.interval = interval
        P = interval[1] - interval[0]
        self.P = P
        self.nfreq = nfreq
        basis_cos = [partial(FourierBased.cos_atom, i, P) for i in range(1, nfreq + 1)]
        basis_sin = [partial(FourierBased.sin_atom, i, P) for i in range(1, nfreq + 1)]
        super(FourierBased, self).__init__(alpha, basis_cos + basis_sin)

    def eval(self, X):
        cosvec = np.cos()
        return np.sqrt(2 / self.D) * np.cos(self.sigma * X.dot(self.w) + self.b)

    def get_feature(self, i):
        return lambda x: np.sqrt(2 / self.D) * np.cos(self.sigma * self.w[:, i].dot(x) + self.b[:, i])

    def features_basis(self):
        return [self.get_feature(i) for i in range(self.D)]


