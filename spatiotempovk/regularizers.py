import numpy as np


class DoubleRepresenterRegularizer:

    """
    Abstract class for regularization terms for functions parametrized using 2 representer theorems
    """

    def __init__(self):
        pass

    def __call__(self, Kx, Ks, alpha):
        pass

    def prime(self, Kx, Ks, alpha):
        pass


class TikhonovSpace(DoubleRepresenterRegularizer):

    def __init__(self):
        super(TikhonovSpace, self).__init__()

    @staticmethod
    def compute_A(Kx, alpha):
        T = alpha.shape[0]
        A = np.zeros((T, T))
        for t0 in range(T):
            for t1 in range(t0, T):
                # Weird order of matrix product for compatibility with algebra.repeated_matrix.RepSymMatrix
                a = alpha[t0, :].dot(Kx.dot(alpha[t1, :].T))
                # # Test, not compatible with algebra.repeated_matrix
                # a = alpha[t0, :].dot(Kx).dot(alpha[t1, :].T)
                A[t0, t1] = a
                A[t1, t0] = a
        return A

    def __call__(self, Kx, Ks, alpha):
        T = Ks.shape[0]
        MT = Kx.shape[0]
        if alpha.ndim == 1:
            # Support for flat alpha for scipy optimizers
            # This does not modify alpha outside of the function as we just assign a new view to it inside the function
            # but do not touch the memory
            alpha = alpha.reshape((T, MT))
        A = TikhonovTime.compute_A(Kx, alpha)
        reg = 0
        for t in range(T):
            reg += Ks[t, :].T.dot(A).dot(Ks[t, :])
        return (1 / T) * reg

    def prime(self, Kx, Ks, alpha):
        T = Ks.shape[0]
        MT = Kx.shape[0]
        flatind = False
        if alpha.ndim == 1:
            # Support for flat alpha for scipy optimizers
            # This does not modify alpha outside of the function as we just assign a new view to it inside the function
            # but do not touch the memory
            alpha = alpha.reshape((T, MT))
            flatind = True
        # Weird order of matrix product for compatibility with algebra.repeated_matrix.RepSymMatrix
        grad = (2 / Ks.shape[0]) * Ks.T.dot(Ks).dot((Kx.dot(alpha.T)).T)
        # # Test incompatible with RepSymMat
        # grad = (2 / T) * Ks.dot(Ks).dot(alpha).dot(Kx)
        if flatind:
            return grad.flatten()
        else:
            return grad


class TikhonovTime(DoubleRepresenterRegularizer):

    def __init__(self):
        super(TikhonovTime, self).__init__()

    @staticmethod
    def compute_A(Kx, alpha):
        T = alpha.shape[0]
        A = np.zeros((T, T))
        for t0 in range(T):
            for t1 in range(t0, T):
                # Weird order of matrix product for compatibility with algebra.repeated_matrix.RepSymMatrix
                a = alpha[t0, :].dot(Kx.dot(alpha[t1, :].T))
                A[t0, t1] = a
                A[t1, t0] = a
        return A

    def __call__(self, Kx, Ks, alpha):
        T = Ks.shape[0]
        MT = Kx.shape[0]
        if alpha.ndim == 1:
            # Support for flat alpha for scipy optimizers
            # This does not modify alpha outside of the function as we just assign a new view to it inside the function
            # but do not touch the memory
            alpha = alpha.reshape((T, MT))
        A = TikhonovTime.compute_A(Kx, alpha)
        return np.sum(Ks * A)

    def prime(self, Kx, Ks, alpha):
        T = Ks.shape[0]
        MT = Kx.shape[0]
        flatind = False
        if alpha.ndim == 1:
            # Support for flat alpha for scipy optimizers
            # This does not modify alpha outside of the function as we just assign a new view to it inside the function
            # but do not touch the memory
            alpha = alpha.reshape((T, MT))
            flatind = True
        # Weird order of matrix product for compatibility with algebra.repeated_matrix.RepSymMatrix
        grad = 2 * Ks.dot((Kx.dot(alpha.T)).T)
        if flatind:
            return grad.flatten()
        else:
            return grad

