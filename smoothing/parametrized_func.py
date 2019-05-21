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
        if X.ndim == 1:
            K = np.array([self.basis[j](X) for j in range(len(self.basis))])
        else:
            K = np.zeros((X.shape[0], len(self.basis)))
            for i in range(X.shape[0]):
                K[i] = np.array([self.basis[j](X[i]) for j in range(len(self.basis))])
        return K

    def __call__(self, X):
        K = self.eval_matrix(X)
        return K.T.dot(self.alpha)


class Polynomial(ParametrizedFunc):

    def __init__(self, alpha):
        self.degree = len(alpha) - 1
        basis = [partial(lambda x2, x1: np.power(x1, x2), n) for n in range(self.degree + 1)]
        super(Polynomial, self).__init__(alpha, basis)

    # @classmethod
    # def new_instance(cls, alpha):
    #     return cls(alpha)

    def prime(self):
        alpha_prime = np.array([i * self.alpha[i] for i in range(1, self.degree + 1)])
        return Polynomial(alpha_prime)
