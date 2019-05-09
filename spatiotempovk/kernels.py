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
        Knew = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                Knew[i, j] = self(Xnew[i], X[j])
        return Knew


class ConvKernel:

    def __init__(self, kernelx, kernely, storeK=True):
        self.storeK = storeK
        self.K = None
        self.kernelx = kernelx
        self.kernely = kernely

    # def __call__(self, s0=None, s1=None, index0=None, index1=None, Kspace=None, Kmeasures=None):
    #     Kx = None
    #     Ky = None
    #     if index0 is not None and index1 is not None and Kspace is not None:
    #         Kx = Kspace[index0, :][:, index1]
    #     if index0 is not None and index1 is not None and Kmeasures is not None:
    #         Ky = Kmeasures[index0, :][:, index1]
    #     # if Kx is None and s0 is not None and s1 is not None:
    #     #     Kx = self.kernelx.compute_Knew(s0[0], s1[0])
    #     # else:
    #     #     raise Exception("Either s0 and s1 or index0, index1 and Kspace should be passed")
    #     # if Ky is None and s0 is not None and s1 is not None:
    #     #     Ky = self.kernely.compute_Knew(s0[1], s1[1])
    #     # else:
    #     #     raise Exception("Either s0 and s1 or index0, index1 and Kmeasures should be passed")
    #     return np.sum(Kx * Ky)

    def __call__(self, index0, index1, Kspace, Kmeasures):
        Kx = Kspace[index0, :][:, index1]
        Ky = Kmeasures[index0, :][:, index1]
        return np.mean(Kx * Ky)

    def compute_K(self, Spt, Kspace, Kmeasures):
        K = np.zeros((Spt.T, Spt.T))
        for i in range(Spt.T):
            for j in range(i, Spt.T):
                index0 = range(Spt.flat_index(i, 0), Spt.flat_index(i, Spt.Ms[i]))
                index1 = range(Spt.flat_index(j, 0), Spt.flat_index(j, Spt.Ms[j]))
                k = self(index0=index0, index1=index1, Kspace=Kspace, Kmeasures=Kmeasures)
                K[i, j] = k
                K[j, i] = k
        if self.storeK:
            self.K = K
        return K


class GaussianKernel(Kernel):

    def __init__(self, sigma, storeK=True):
        super(GaussianKernel, self).__init__(storeK)
        self.sigma = sigma

    def __call__(self, x0, x1):
        return np.exp(- ((np.linalg.norm(x0 - x1)) ** 2) / (2 * self.sigma ** 2))
