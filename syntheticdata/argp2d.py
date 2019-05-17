import numpy as np
import matplotlib.pyplot as plt


class GaussianKernel:

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x0, x1):
        return np.exp(-(np.linalg.norm(x0 - x1) ** 2) / (2 * self.sigma ** 2))

    def kernel_matrix(self, X):
        n = X.shape[1]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                k = self(X[:, i], X[:, j])
                K[i, j] = k
                K[j, i] = k
        return K


def covariance_matrix(nx, ny, kernel):
    X = np.zeros((2, nx * ny))
    for i in range(nx):
        for j in range(ny):
            X[:, i * ny + j] = [i, j]
    K = kernel.kernel_matrix(X)
    return K


def draw_gp2d(K, nx, ny):
    n = K.shape[0]
    mu = np.zeros(n)
    draws = np.random.multivariate_normal(mu, K)
    return draws.reshape((nx, ny))


def draw_ar1_gp2d(nx=50, ny=50, T=5, sigma=5, rho=0.75):
    ker = GaussianKernel(sigma)
    K = covariance_matrix(nx, ny, ker)
    ar = [draw_gp2d(K, nx, ny)]
    for t in range(1, T):
        innov = draw_gp2d(K, nx, ny)
        ar.append(rho * ar[t-1] + (1 - rho) * innov)
    return ar


def draw_observations(nobs, argp):
    obs = []
    nx, ny = argp[0].shape
    T = len(argp)
    for t in range(T):
        X = np.zeros((nobs, 2))
        Y = np.zeros(nobs)
        for i in range(nobs):
            x0, x1 = np.random.randint(0, nx), np.random.randint(0, ny)
            X[i, :] = [x0, x1]
            Y[i] = argp[t][x0, x1]
        obs.append((X, Y))
    return obs


def draw_observations_sameloc(nobs, argp):
    obs = []
    nx, ny = argp[0].shape
    T = len(argp)
    x0, x1 = np.random.randint(0, nx), np.random.randint(0, ny)
    for t in range(T):
        X = np.zeros((nobs, 2))
        Y = np.zeros(nobs)
        for i in range(nobs):
            X[i, :] = [x0, x1]
            Y[i] = argp[t][x0, x1]
        obs.append((X, Y))
    return obs

# fig, axes = plt.subplots(2, 2)
# axes[0, 0].imshow(argp[0])
# axes[0, 1].imshow(argp[1])
# axes[1, 0].imshow(argp[2])
# axes[1, 1].imshow(argp[3])
