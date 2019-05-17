import numpy as np
from sklearn import kernel_ridge
from functools import partial

import smoothing.parametrized_func as param_func


class Smoother:

    def __init__(self):
        pass

    def __call__(self, X, Y):
        """
        Parameters
        ----------
        X: list
            len(X) = n_problems, X[t].shape = (nobs_t, dim_input)
        Y: list
            len(Y) = n_problems, Y[t].shape = (nobs_t, ) or (nobs_t, 1)

        Returns
        --------
        alpha: numpy.ndarray
            alpha.shape = (n_problems, smoothing_dim)class sklearn.kernel_ridge.KernelRidge(alpha=1, kernel=’linear’, gamma=None, degree=3, coef0=1, kernel_params=None)[source]

        """
        pass


class RidgeSmoother(Smoother):

    def __init__(self, kernel=None, mu=1):
        super(RidgeSmoother).__init__()
        self.kernel = kernel
        self.mu = mu

    def __call__(self, X, Y):
        nprobs = len(X)
        if len(Y) != nprobs:
            raise ValueError("Number of problems for input ({}) and target ({}) do not match".format(nprobs, len(Y)))
        alpha = np.zeros((nprobs, X[0].shape[0]))
        for t in range(nprobs):
            clf = kernel_ridge.KernelRidge(alpha=self.mu, kernel=self.kernel)
            clf.fit(X[t], Y[t])
            alpha[t, :] = clf.dual_coef_.flatten().copy()
        return alpha, [partial(self.kernel, x) for x in X[0]]






