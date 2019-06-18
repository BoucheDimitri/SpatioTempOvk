import numpy as np
from functools import partial


class ParametrizedFunc:

    def __init__(self, alpha, func_basis):
        self.alpha = alpha
        self.basis = func_basis

    def eval_matrix(self, X):
        """
        Matrix of evaluation at the different elements of X for the basis functions

        Parameters
        ----------
        X: numpy.ndarray
            either X.shape = (dim_space, ) or X.shape = (n_evaluations, dim_space)

        Returns
        -------
        K: numpy.ndarray
            Evaluation matrix, K[i, j] = self.basis[j](X[i])
        """
        if not isinstance(X, np.ndarray):
            K = np.array([self.basis[j](X) for j in range(len(self.basis))])
        elif X.ndim == 1 or not isinstance(X, np.ndarray):
            K = np.array([self.basis[j](X) for j in range(len(self.basis))])
        else:
            K = np.zeros((X.shape[0], len(self.basis)))
            for i in range(X.shape[0]):
                K[i] = np.array([self.basis[j](X[i]) for j in range(len(self.basis))]).flatten()
        return K

    def __call__(self, X):
        K = self.eval_matrix(X)
        return K.dot(self.alpha)


class PolynomialBased(ParametrizedFunc):

    def __init__(self, alpha):
        self.degree = len(alpha) - 1
        basis = [partial(lambda x2, x1: np.power(x1, x2), n) for n in range(self.degree + 1)]
        super(PolynomialBased, self).__init__(alpha, basis)

    # @classmethod
    # def new_instance(cls, alpha):
    #     return cls(alpha)

    def prime(self):
        alpha_prime = np.array([i * self.alpha[i] for i in range(1, self.degree + 1)])
        return PolynomialBased(alpha_prime)


class FourierBased(ParametrizedFunc):

    @staticmethod
    def cos_atom(n, P, x):
        return np.cos(2 * np.pi * n * x / P)

    @staticmethod
    def cos_atom_prime(n, P, x):
        return - (2 * np.pi * n / P) * np.sin(2 * np.pi * n * x / P)

    @staticmethod
    def sin_atom(n, P, x):
        return np.sin(2 * np.pi * n * x / P)

    @staticmethod
    def sin_atom_prime(n, P, x):
        return (2 * np.pi * n / P) * np.cos(2 * np.pi * n * x / P)

    def __init__(self, alpha, nfreq, interval):
        self.interval = interval
        P = interval[1] - interval[0]
        self.P = P
        self.nfreq = nfreq
        basis_cos = [partial(FourierBased.cos_atom, i, P) for i in range(1, nfreq + 1)]
        basis_sin = [partial(FourierBased.sin_atom, i, P) for i in range(1, nfreq + 1)]
        super(FourierBased, self).__init__(alpha, basis_cos + basis_sin)

    def prime(self):
        alpha1 = [-(2 * np.pi * i / self.P) * self.alpha[i - 1] for i in range(1, self.nfreq + 1)]
        alpha2 = [(2 * np.pi * i / self.P) * self.alpha[self.nfreq + i - 1] for i in range(1, self.nfreq + 1)]
        return FourierBased(alpha2 + alpha1, self.nfreq, self.interval)