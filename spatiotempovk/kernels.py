import numpy as np
from geopy import distance


class Kernel:

    def __init__(self):
        pass

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
        return K

    def compute_Knew(self, X, Xnew):
        n = len(X)
        m = len(Xnew)
        Knew = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                Knew[i, j] = self(Xnew[i], X[j])
        return Knew.T


class ConvKernel(Kernel):

    """
    One layer convolutional kernel

    Parameters
    ----------
    kernelx: spatiotempovk.kernels.Kernel
        kernel for space
    kernely: spatiotempovk.kernels.Kernel
        kernel for measurements
    Kx: numpy.ndarray
        Kernel matrix associated with kernelx on the training dataset
    Ky: numpy.ndarray
        Kernel matrix associated with kernely on the training dataset

    Attributes
    ----------
    kernelx: spatiotempovk.kernels.Kernel
        kernel for space
    kernely: spatiotempovk.kernels.Kernel
        kernel for measurements
    Kx: numpy.ndarray
        Kernel matrix associated with kernelx on the training dataset
    Ky: numpy.ndarray
        Kernel matrix associated with kernely on the training dataset

    """

    def __init__(self, kernelx, kernely, Kx, Ky=None, sameloc=False):
        super(ConvKernel, self).__init__()
        self.kernelx = kernelx
        self.Kx = Kx
        self.kernely = kernely
        self.Ky = Ky
        self.sameloc = sameloc

    def __call__(self, s0, s1):
        if not self.sameloc:
            Kx = self.kernelx.compute_Knew(s0[0], s1[0])
        else:
            Kx = self.Kx
        Ky = self.kernely.compute_Knew(s0[1], s1[1])
        return np.mean(Kx * Ky)

    def from_mat(self, index0, index1):
        if not self.sameloc:
            return np.mean(self.Kx[index0, :][:, index1] * self.Ky[index0, :][:, index1])
        else:
            return np.mean(self.Kx * self.Ky[index0, :][:, index1])

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
        return K


class GaussianKernel(Kernel):
    """
    Gaussian kernel

    Parameters
    ----------
    sigma: float
        bandwidth parameter of the kernel

    Attributes
    ----------
    sigma: float
        bandwidth parameter of the kernel
    """
    def __init__(self, sigma):
        super(GaussianKernel, self).__init__()
        self.sigma = sigma

    def __call__(self, x0, x1):
        return np.exp(- ((np.linalg.norm(x0 - x1)) ** 2) / (2 * self.sigma ** 2))


class GaussianGeoKernel(Kernel):
    """
    Gaussian kernel when distances are in geographic coordinates

    Parameters
    ----------
    sigma: float
        bandwidth parameter of the kernel

    Attributes
    ----------
    sigma: float
        bandwidth parameter of the kernel

    """
    def __init__(self, sigma):
        super(GaussianGeoKernel, self).__init__()
        self.sigma = sigma

    def __call__(self, x0, x1):
        dist = distance.geodesic(x0, x1).km
        return np.exp(- (dist ** 2) / (2 * self.sigma ** 2))
