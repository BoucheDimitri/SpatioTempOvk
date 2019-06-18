

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