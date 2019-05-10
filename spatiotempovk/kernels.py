import numpy as np


class Kernel:

    def __init__(self, storeK=True):
        self.storeK = storeK
        self.K = None

    def __call__(self, x0, x1):
        pass

    def compute_K(self, X):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                k = self(X[i], X[j])
                K[i, j] = k
                K[j, i] = k
        if self.storeK:
            self.K = K
        return K

    def compute_Knew(self, X, Xnew):
        n = len(X)
        m = len(Xnew)
        Knew = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                Knew[i, j] = self(Xnew[i], X[j])
        return Knew


class ConvKernel(Kernel):

    def __init__(self, kernelx, kernely, storeK=True):
        super(ConvKernel, self).__init__(storeK)
        self.storeK = storeK
        self.K = None
        self.kernelx = kernelx
        self.kernely = kernely

    def __call__(self, s0, s1):
        Kx = self.kernelx.compute_Knew(s0[0], s1[0])
        Ky = self.kernely.compute_Knew(s0[1], s1[1])
        return np.mean(Kx * Ky)

    def from_mat(self, index0, index1):
        return np.mean(self.kernelx.K[index0, :][:, index1] * self.kernely.K[index0, :][:, index1])

    def compute_K_from_mat(self, Ms):
        T = len(Ms)
        K = np.zeros((T, T))
        for t0 in range(T):
            for t1 in range(t0, T):
                tm0 = sum(Ms[:t0])
                tm1 = sum(Ms[:t1])
                index0 = range(tm0, tm0 + Ms[t0])
                index1 = range(tm1, tm1 + Ms[t1])
                k = self.from_mat(index0, index1)
                K[t0, t1] = k
                K[t1, t0] = k
        if self.storeK:
            self.K = K
        return K


class GaussianKernel(Kernel):

    def __init__(self, sigma, storeK=True):
        super(GaussianKernel, self).__init__(storeK)
        self.sigma = sigma

    def __call__(self, x0, x1):
        return np.exp(- ((np.linalg.norm(x0 - x1)) ** 2) / (2 * self.sigma ** 2))
