import numpy as np


class DoubleRepresenterRegularizer:

    """
    Parameters
    ----------


    Attributes
    ----------
    """

    def __init__(self):
        pass

    def __call__(self, alpha, Kx, Ks):
        pass

    def prime(self, alpha, Kx, Ks):
        pass


class TikhonovSpace(DoubleRepresenterRegularizer):

    def __init__(self):
        super(TikhonovSpace, self).__init__()

    @staticmethod
    def compute_A(alpha, Kx):
        T = alpha.shape[0]
        A = np.zeros((T, T))
        for t0 in range(T):
            for t1 in range(t0, T):
                a = alpha[t0, :].dot(Kx).dot(alpha[t1, :].T)
                A[t0, t1] = a
                A[t1, t0] = a
        return A

    def __call__(self, alpha, Kx, Ks):
        T = Ks.shape[0]
        A = TikhonovTime.compute_A(alpha, Kx)
        reg = 0
        for t in range(T):
            reg += Ks[t, :].T.dot(A).dot(Ks[t, :])
        return (1 / T) * reg

    def prime(self, alpha, Kx, Ks):
        return (2 / Ks.shape[0]) * Ks.T.dot(Ks).dot(alpha).dot(Kx)


class TikhonovTime(DoubleRepresenterRegularizer):

    def __init__(self):
        super(TikhonovTime, self).__init__()

    @staticmethod
    def compute_A(alpha, Kx):
        T = alpha.shape[0]
        A = np.zeros((T, T))
        for t0 in range(T):
            for t1 in range(t0, T):
                a = alpha[t0, :].dot(Kx).dot(alpha[t1, :].T)
                A[t0, t1] = a
                A[t1, t0] = a
        return A

    def __call__(self, alpha, Kx, Ks):
        A = TikhonovTime.compute_A(alpha, Kx)
        return np.sum(Ks * A)

    def prime(self, alpha, Kx, Ks):
        return 2 * Ks.dot(alpha).dot(Kx)

