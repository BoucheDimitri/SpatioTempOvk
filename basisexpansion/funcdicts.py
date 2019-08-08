import numpy as np
from functools import partial


class FuncDict:

    def __init__(self, d=1):
        self.d = 1
        pass

    def eval(self, X):
        pass


class MexHatDict(FuncDict):

    @staticmethod
    def atom(t0, sigma, t):
        c = 2 / (np.sqrt(3) * np.power(np.pi, 0.25))
        return c * (1 - ((t - t0)/sigma) ** 2) * np.exp(-0.5 * ((t - t0)/sigma) ** 2)

    def __init__(self, interval, tgrid, sigmagrid, d=1):
        super(MexHatDict, self).__init__(d)
        self.interval = interval
        self.tgrid = tgrid
        self.sigmagrid = sigmagrid
        grid2d = np.meshgrid(self.tgrid, self.sigmagrid)
        self.tvec = grid2d[0].flatten()
        self.sigmavec = grid2d[1].flatten()
        self.D = len(self.tgrid) * len(self.sigmagrid)

    def eval(self, X):
        return MexHatDict.atom(self.tvec, self.sigmavec, X)

    def get_feature(self, i):
        return partial(MexHatDict.eval, self.tvec[i], self.sigmavec[i])

    def features_basis(self):
        return [self.get_feature(i) for i in range(self.D)]


class FourierDict:

    @staticmethod
    def cos_atom(n, P, x):
        return np.cos(2 * np.pi * n * x / P)

    @staticmethod
    def sin_atom(n, P, x):
        return np.sin(2 * np.pi * n * x / P)

    def __init__(self, interval, nfreq, d=1):
        self.P = interval[1] - interval[0]
        self.D = 2 * nfreq
        self.nfreq = nfreq
        self.d = d

    def eval(self, X):
        nvec = np.arange(0, self.nfreq, 1)
        cosZ = np.cos(2 * np.pi * X * nvec / self.P)
        sinZ = np.cos(2 * np.pi * X * nvec / self.P)
        return np.concatenate((cosZ, sinZ), axis=1)
