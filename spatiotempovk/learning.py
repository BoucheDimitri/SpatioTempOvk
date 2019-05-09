import numpy as np


class PredictFunc:

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, kx, ks):
        return ks.T.dot(self.alpha).dot(kx)

    def compute_A(self, Kx):
        T = self.alpha.shape[0]
        A = np.zeros((T, T))
        for t0 in range(T):
            for t1 in range(t0, T):
                a = self.alpha[t0, :].dot(Kx).dot(self.alpha[t1, :].T)
                A[t0, t1] = a
                A[t1, t0] = a
        return A

    def compute_Gamma(self, Kx, Ks):
        T = Ks.shape[0]
        A = self.compute_A(Kx)
        reg = 0
        for t in range(T):
            reg += Ks[t, :].T.dot(A).dot(Ks[t, :])ù
        return (1 / T) * reg

    def compute_Omega(self):