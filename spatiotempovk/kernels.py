import numpy as np
from geopy import distance
from smoothing import representer
from smoothing import fourierrandom
from smoothing import parametrized_func


class Kernel:

    def __init__(self, func=None, normalize=False):
        self.normalize = normalize
        self.func=func

    def __call__(self, x0, x1):
        if self.func is not None:
            return self.func(x0, x1)
        else:
            pass

    def compute_K(self, X):
        n = len(X)
        K = np.zeros((n, n))
        # Compute diagonal first for normalization
        for i in range(n):
            K[i, i] = self(X[i], X[i])
        for i in range(n):
            for j in range(i+1, n):
                k = self(X[i], X[j])
                K[i, j] = k
                K[j, i] = k
                if self.normalize:
                    knorm = (1 / (np.sqrt(K[i, i]) * np.sqrt(K[j, j])))
                    K[i, j] *= knorm
                    K[j, i] *= knorm
        if self.normalize:
            for i in range(n):
                K[i, i] = 1.0
        return K

    def compute_Knew(self, X, Xnew):
        n = len(X)
        m = len(Xnew)
        Knew = np.zeros((m, n))
        if self.normalize:
            normsX = [self(X[i], X[i]) for i in range(n)]
            normsXnew = [self(Xnew[i], Xnew[i]) for i in range(m)]
        for i in range(m):
            for j in range(n):
                k = self(Xnew[i], X[j])
                if self.normalize:
                    knorm = 1 / (np.sqrt(normsXnew[i]) * np.sqrt(normsX[j]))
                    Knew[i, j] = k * knorm
                else:
                    Knew[i, j] = k
        return Knew.T


class GaussianSameLoc(Kernel):

    def __init__(self, sigma, normalize=False):
        super(GaussianSameLoc, self).__init__(normalize)
        self.sigma = sigma

    def __call__(self, s0, s1):
        return np.exp(- ((np.linalg.norm(s0[1] - s1[1])) ** 2) / (2 * self.sigma ** 2))


class GaussianFuncKernel(Kernel):

    def __init__(self, sigma, rffeats, mu, normalize=False):
        super(GaussianFuncKernel, self).__init__(normalize)
        self.sigma = sigma
        self.rffeats = rffeats
        self.smoother = fourierrandom.RFFRidgeSmoother(self.rffeats, mu)

    def __call__(self, w0, w1):
        return np.exp(- ((np.linalg.norm(w0 - w1)) ** 2) / (2 * self.sigma ** 2))

    def compute_K(self, X):
        n = len(X)
        x = [X[i][0] for i in range(n)]
        y = [X[i][1] for i in range(n)]
        w, base = self.smoother(x, y)
        K = np.zeros((n, n))
        # Compute diagonal first for normalization
        for i in range(n):
            K[i, i] = self(w[i], w[i])
        for i in range(n):
            for j in range(i + 1, n):
                k = self(w[i], w[j])
                K[i, j] = k
                K[j, i] = k
                if self.normalize:
                    knorm = (1 / (np.sqrt(K[i, i]) * np.sqrt(K[j, j])))
                    K[i, j] *= knorm
                    K[j, i] *= knorm
        if self.normalize:
            for i in range(n):
                K[i, i] = 1.0
        return K

    def compute_Knew(self, X, Xnew):
        n = len(X)
        m = len(Xnew)
        x = [X[i][0] for i in range(n)]
        y = [X[i][1] for i in range(n)]
        xnew = [Xnew[i][0] for i in range(m)]
        ynew = [Xnew[i][1] for i in range(m)]
        w, base = self.smoother(x, y)
        wnew, basenew = self.smoother(xnew, ynew)
        Knew = np.zeros((m, n))
        if self.normalize:
            normsX = [self(w[i], w[i]) for i in range(n)]
            normsXnew = [self(wnew[i], wnew[i]) for i in range(m)]
        for i in range(m):
            for j in range(n):
                k = self(wnew[i], w[j])
                if self.normalize:
                    knorm = 1 / (np.sqrt(normsXnew[i]) * np.sqrt(normsX[j]))
                    Knew[i, j] = k * knorm
                else:
                    Knew[i, j] = k
        return Knew.T


#
# class GaussianFuncKernel(Kernel):
#
#     def __init__(self, kernelx, mu, sigma=1, bounds=(0, 1), precision=100, normalize=False):
#         super(GaussianFuncKernel, self).__init__(normalize)
#         self.bounds = bounds
#         self.precision = precision
#         self.evalgrid = np.linspace(bounds[0], bounds[1], precision)
#         self.sigma = sigma
#         self.kernelx = kernelx
#         self.mu = mu
#
#     def __call__(self, f, g):
#         fvec = np.array([f(x) for x in self.evalgrid])
#         gvec = np.array([g(x) for x in self.evalgrid])
#         return np.exp(- ((np.linalg.norm(fvec - gvec)) ** 2) / (2 * self.sigma ** 2))
#
#     def compute_K(self, X):
#         ridgesmoother = representer.RidgeSmoother(self.kernelx, self.mu)
#         n = len(X)
#         x = [X[i][0] for i in range(n)]
#         y = [X[i][1] for i in range(n)]
#         alpha, base = ridgesmoother(x, y)
#         fs = [parametrized_func.ParametrizedFunc(alpha[i], base) for i in range(n)]
#         K = np.zeros((n, n))
#         # Compute diagonal first for normalization
#         for i in range(n):
#             K[i, i] = self(fs[i], fs[i])
#         for i in range(n):
#             for j in range(i + 1, n):
#                 k = self(fs[i], fs[j])
#                 K[i, j] = k
#                 K[j, i] = k
#                 if self.normalize:
#                     knorm = (1 / (np.sqrt(K[i, i]) * np.sqrt(K[j, j])))
#                     K[i, j] *= knorm
#                     K[j, i] *= knorm
#         if self.normalize:
#             for i in range(n):
#                 K[i, i] = 1.0
#         return K
#
#     def compute_Knew(self, X, Xnew):
#         ridgesmoother = representer.RidgeSmoother(self.kernelx, self.mu)
#         n = len(X)
#         m = len(Xnew)
#         x = [X[i][0] for i in range(n)]
#         y = [X[i][1] for i in range(n)]
#         xnew = [X[i][0] for i in range(m)]
#         ynew = [X[i][1] for i in range(m)]
#         alpha, base = ridgesmoother(x, y)
#         alphanew, basenew =
#         fs = [parametrized_func.ParametrizedFunc(alpha[i], base) for i in range(n)]
#         K = np.zeros((n, n))
#         # Compute diagonal first for normalization
#         for i in range(n):
#             K[i, i] = self(fs[i], fs[i])
#         for i in range(n):
#             for j in range(i + 1, n):
#                 k = self(fs[i], fs[j])
#                 K[i, j] = k
#                 K[j, i] = k
#                 if self.normalize:
#                     knorm = (1 / (np.sqrt(K[i, i]) * np.sqrt(K[j, j])))
#                     K[i, j] *= knorm
#                     K[j, i] *= knorm
#         if self.normalize:
#             for i in range(n):
#                 K[i, i] = 1.0
#         return K


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
        Kernel matrix associated with kernelx on the training dataset, if sameloc is true should be the kernel
        matrix for the locations at which the measures are done at each time
    Ky: numpy.ndarray or NoneType
        Kernel matrix associated with kernely on the training dataset
    sameloc: bool
        Are the measurements always made at the same locations
    """

    def __init__(self, kernelx, kernely, Kx, Ky=None, sameloc=False, normalize=True):
        super(ConvKernel, self).__init__(normalize)
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
        if self.normalize:
            for t0 in range(T):
                for t1 in range(t0+1, T):
                    k = 1 / (np.sqrt(K[t0, t0]) * np.sqrt(K[t1, t1]))
                    K[t0, t1] *= k
                    K[t1, t0] *= k
            for t0 in range(T):
                K[t0, t0] = 1.0
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
