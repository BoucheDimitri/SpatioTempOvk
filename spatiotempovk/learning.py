import numpy as np


class SpatioTempFuncRidge:

    def __init__(self, mu, lamb):
        self.mu = mu
        self.lamb = lamb


class PredictFunc:

    def __init__(self, alpha, mu, lamb):
        self.alpha = alpha
        self.mu = mu
        self.lamb = lamb

    def __call__(self, kx, ks):
        return ks.T.dot(self.alpha).dot(kx)

    def Xi(self, loss, Ms, y, Kx, Ks):
        xi = 0
        T = Ks.shape[0]
        for t in range(T):
            xit = 0
            for m in range(Ms[t]):
                tm = sum(Ms[:t]) + m
                xit += loss(y[tm], self(Kx[tm], Ks[t]))
            xi += (1 / Ms[t]) * xit
        return (1 / T) * xi

    def compute_A(self, Kx):
        T = self.alpha.shape[0]
        A = np.zeros((T, T))
        for t0 in range(T):
            for t1 in range(t0, T):
                a = self.alpha[t0, :].dot(Kx).dot(self.alpha[t1, :].T)
                A[t0, t1] = a
                A[t1, t0] = a
        return A

    def Gamma(self, Ks, A):
        T = Ks.shape[0]
        reg = 0
        for t in range(T):
            reg += Ks[t, :].T.dot(A).dot(Ks[t, :])
        return (1 / T) * reg

    def Omega(self, Ks, A):
        return np.sum(Ks * A)

    def regularization(self, Kx, Ks):
        A = self.compute_A(Kx)
        return self.mu * self.Gamma(Ks, A) + self.lamb * self.Omega(Ks, A)

    def Gamma_prime(self, Kx, Ks):
        return (2 / Ks.shape[0]) * Ks.T.dot(Ks).dot(self.alpha).dot(Kx)

    def Omega_prime(self, Kx, Ks):
        return 2 * Ks.dot(self.alpha).dot(Kx)

    def Xi_prime(self, loss_prime, Ms, y, Kx, Ks):
        xi_prime = np.zeros(self.alpha.shape)
        T = Ks.shape[0]
        for t in range(T):
            for m in range(Ms[t]):
                tm = sum(Ms[:t]) + m
                k = Ks[t].reshape((Ks.shape[0], 1)).dot(Kx[tm].reshape((1, Kx.shape[0])))
                xi_prime += (1 / Ms[t]) * loss_prime(y[tm], self(Kx[tm], Ks[t])) * k
        return (1 / T) * xi_prime

    def regularization_prime(self, Kx, Ks):
        return self.mu * self.Gamma_prime(Kx, Ks) + self.lamb * self.Omega_prime(Kx, Ks)

    def objective_gradient(self, loss_prime, Ms, y, Kx, Ks):
        return self.Xi_prime(loss_prime, Ms, y, Kx, Ks) + self.regularization_prime(Kx, Ks)


def predict(alpha, kx, ks):
    return ks.T.dot(alpha).dot(kx)


def Xi(alpha, loss, Ms, y, Kx, Ks):
    xi = 0
    T = Ks.shape[0]
    for t in range(T):
        xit = 0
        for m in range(Ms[t]):
            tm = sum(Ms[:t]) + m
            xit += loss(y[tm], predict(alpha, Kx[tm], Ks[t]))
        xi += (1 / Ms[t]) * xit
    return (1 / T) * xi


def compute_A(alpha, Kx):
    T = alpha.shape[0]
    A = np.zeros((T, T))
    for t0 in range(T):
        for t1 in range(t0, T):
            a = alpha[t0, :].dot(Kx).dot(alpha[t1, :].T)
            A[t0, t1] = a
            A[t1, t0] = a
    return A


def Gamma(Ks, A):
    T = Ks.shape[0]
    reg = 0
    for t in range(T):
        reg += Ks[t, :].T.dot(A).dot(Ks[t, :])
    return (1 / T) * reg


def Omega(Ks, A):
    return np.sum(Ks * A)


def regularization(alpha, Kx, Ks, mu, lamb):
    A = compute_A(alpha, Kx)
    return mu * Gamma(Ks, A) + lamb * Omega(Ks, A)


def objective(alpha, loss, Ms, y, Kx, Ks, mu, lamb):
    return Xi(alpha, loss, Ms, y, Kx, Ks) + regularization(alpha, Kx, Ks, mu, lamb)


def Gamma_prime(alpha, Kx, Ks):
    return (2 / Ks.shape[0]) * Ks.T.dot(Ks).dot(alpha).dot(Kx)


def Omega_prime(alpha, Kx, Ks):
    return 2 * Ks.dot(alpha).dot(Kx)


def Xi_prime(alpha, loss_prime, Ms, y, Kx, Ks):
    xi_prime = np.zeros(alpha.shape)
    T = Ks.shape[0]
    for t in range(T):
        for m in range(Ms[t]):
            tm = sum(Ms[:t]) + m
            k = Ks[t].reshape((Ks.shape[0], 1)).dot(Kx[tm].reshape((1, Kx.shape[0])))
            xi_prime += (1 / Ms[t]) * loss_prime(y[tm], predict(alpha, Kx[tm], Ks[t])) * k
    return (1 / T) * xi_prime


def regularization_prime(alpha, Kx, Ks, mu, lamb):
    return mu * Gamma_prime(alpha, Kx, Ks) + lamb * Omega_prime(alpha, Kx, Ks)


def objective_gradient(alpha, loss_prime, Ms, y, Kx, Ks, mu, lamb):
    return Xi_prime(alpha, loss_prime, Ms, y, Kx, Ks) + regularization_prime(alpha, Kx, Ks, mu, lamb)


def get_gradient_func(loss_prime, Ms, y, Kx, Ks, mu, lamb):
    def gradient_func(alpha):
        g = objective_gradient(alpha.reshape((Ks.shape[0], Kx.shape[0])), loss_prime, Ms, y, Kx, Ks, mu, lamb)
        return g.flatten()
    return gradient_func


def get_objective_func(loss, Ms, y, Kx, Ks, mu, lamb):
    def objective_func(alpha):
        return objective(alpha.reshape((Ks.shape[0], Kx.shape[0])), loss, Ms, y, Kx, Ks, mu, lamb)
    return objective_func


# def gradient_descent(alpha0, loss, loss_prime, Ms, y, Kx, Ks, mu, lamb, nu, maxit, eps=1):



