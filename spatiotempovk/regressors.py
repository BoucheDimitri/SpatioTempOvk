import numpy as np
import scipy.optimize as optimize


class DiffSpatioTempRegressor:

    def __init__(self, loss, spacereg, timereg, mu, lamb, kernelx, kernels):
        self.loss = loss
        self.spacereg = spacereg
        self.timereg = timereg
        self.mu = mu
        self.lamb = lamb
        self.kernelx = kernelx
        self.kernels = kernels
        self.alpha = None

    @staticmethod
    def predict(alpha, kx, ks):
        return ks.T.dot(alpha).dot(kx)

    def data_fitting(self, alpha, Ms, y, Kx, Ks):
        xi = 0
        T = Ks.shape[0]
        for t in range(T):
            xit = 0
            for m in range(Ms[t]):
                tm = sum(Ms[:t]) + m
                xit += self.loss(y[tm], DiffSpatioTempRegressor.predict(alpha, Kx[tm], Ks[t]))
            xi += (1 / Ms[t]) * xit
        return (1 / T) * xi

    def data_fitting_prime(self, alpha, Ms, y, Kx, Ks):
        xi_prime = np.zeros(alpha.shape)
        T = Ks.shape[0]
        for t in range(T):
            for m in range(Ms[t]):
                tm = sum(Ms[:t]) + m
                k = Ks[t].reshape((Ks.shape[0], 1)).dot(Kx[tm].reshape((1, Kx.shape[0])))
                xi_prime += (1 / Ms[t]) * self.loss.prime(y[tm],
                                                          DiffSpatioTempRegressor.predict(alpha, Kx[tm], Ks[t])) * k
        return (1 / T) * xi_prime

    def objective(self, alpha, Ms, y, Kx, Ks):
        return self.data_fitting(alpha, Ms, y, Kx, Ks) \
               + self.mu * self.spacereg(alpha, Kx, Ks) \
               + self.lamb * self.timereg(alpha, Kx, Ks)

    def objective_func(self, Ms, y, Kx, Ks):
        MT = Kx.shape[0]
        T = Ks.shape[0]

        def obj(alpha):
            alpha_res = alpha.reshape((T, MT))
            return self.objective(alpha_res, Ms, y, Kx, Ks)

        return obj

    def objective_prime(self, alpha, Ms, y, Kx, Ks):
        return self.data_fitting_prime(alpha, Ms, y, Kx, Ks) \
               + self.mu * self.spacereg.prime(alpha, Kx, Ks) \
               + self.lamb * self.timereg.prime(alpha, Kx, Ks)

    def objective_grad_func(self, Ms, y, Kx, Ks):
        MT = Kx.shape[0]
        T = Ks.shape[0]

        def grad(alpha):
            alpha_res = alpha.reshape((T, MT))
            return self.objective_prime(alpha_res, Ms, y, Kx, Ks)

        return grad

