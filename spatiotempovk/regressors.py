import numpy as np
import scipy.optimize as optimize


class DiffSpatioTempRegressor:

    """
    Class for spatio temporal regressor which contain only differentiable terms

    Parameters
    ----------
    loss: spatiotempovk.losses.DiffLoss
        Loss chosen for data fitting term
    spacereg: spatiotempovk.regularizers.DoubleRepresenterRegularizer
        Regularization chosen for space
    timereg: spatiotempovk.regularizers.DoubleRepresenterRegularizer
        Regularization chosen for time
    mu: float
        Regularization parameter for space regularization
    lamb: float
        Regularization parameter for time regularization
    kernelx: spatiotempovk.kernels.Kernel
        Kernel to compare locations
    kernels: spatiotempovk.kernels.ConvKernel
        Convolutional kernel between a kernelx and a kernel comparing measurements


    Attributes
    ----------
    loss: spatiotempovk.losses.DiffLoss
        Loss chosen for data fitting term
    spacereg: spatiotempovk.regularizers.DoubleRepresenterRegularizer
        Regularization chosen for space
    timereg: spatiotempovk.regularizers.DoubleRepresenterRegularizer
        Regularization chosen for time
    mu: float
        Regularization parameter for space regularization
    lamb: float
        Regularization parameter for time regularization
    kernelx: spatiotempovk.kernels.Kernel
        Kernel to compare locations
    kernels: spatiotempovk.kernels.ConvKernel
        Convolutional kernel between a kernelx and a kernel comparing measurements
    alpha: numpy.ndarray
        Optimal parameter set when fitted
    S: spatiotempovk.spatiotempdata.SpatioTempData
        Training data
    """

    def __init__(self, loss, spacereg, timereg, mu, lamb, kernelx, kernels):
        self.loss = loss
        self.spacereg = spacereg
        self.timereg = timereg
        self.mu = mu
        self.lamb = lamb
        self.kernelx = kernelx
        self.kernels = kernels
        self.alpha = None
        self.S = None

    @staticmethod
    def eval(alpha, kx, ks):
        """
        Prediction function

        Parameters
        ----------
        alpha: np.ndarray
            Representer theorem coefficients, alpha.shape = (T, barM)
        kx: np.ndarray
            Kernel comparison between new location and all training location kx.shape = (barM, )
        ks: np.ndarray
            Kernel comparison between new sample and all training samples, ks.shape = (T, )

        Returns
        -------
        F: float
            prediction at time T+1 for location x
        """
        return ks.T.dot(alpha).dot(kx)

    def data_fitting(self, alpha, Ms, y, Kx, Ks):
        """
        Compute data fitting term

        Parameters
        ----------
        alpha: np.ndarray
            Representer theorem coefficients, alpha.shape = (T, barM)
        Ms: list
            Number of locations per time step : [M_1,...,M_T]
        y: np.ndarray
            Flatten measurements, y.shape = (barM, )
        Kx: np.ndarray
            Training locations kernel matrix, Kx.shape = (barM, barM)
        Ks: np.ndarray
            Time samples kernel matrix, Ks.shape = (T, T)

        Returns
        -------
        xi: float
            evaluation of the datafitting term in alpha
        """
        xi = 0
        T = Ks.shape[0]
        for t in range(T):
            xit = 0
            for m in range(Ms[t]):
                tm = sum(Ms[:t]) + m
                xit += self.loss(y[tm], DiffSpatioTempRegressor.eval(alpha, Kx[tm], Ks[t]))
            xi += (1 / Ms[t]) * xit
        return (1 / T) * xi

    def data_fitting_prime(self, alpha, Ms, y, Kx, Ks):
        """
        Gradient of data fitting term

        Parameters
        ----------
        alpha: np.ndarray
            Representer theorem coefficients, alpha.shape = (T, barM)
        Ms: list
            Number of locations per time step : [M_1,...,M_T]
        y: np.ndarray
            Flatten measurements, y.shape = (barM, )
        Kx: np.ndarray
            Training locations kernel matrix, Kx.shape = (barM, barM)
        Ks: np.ndarray
            Time samples kernel matrix, Ks.shape = (T, T)

        Returns
        -------
        grad_xi: float
            gradient of data fitting term evaluated in alpha
        """
        T = Ks.shape[0]
        xi_prime = np.zeros(alpha.shape)
        for t in range(T):
            for m in range(Ms[t]):
                tm = sum(Ms[:t]) + m
                k = Ks[t].reshape((Ks.shape[0], 1)).dot(Kx[tm].reshape((1, Kx.shape[0])))
                xi_prime += (1 / Ms[t]) * self.loss.prime(y[tm],
                                                          DiffSpatioTempRegressor.eval(alpha, Kx[tm], Ks[t])) * k
        return (1 / T) * xi_prime

    def objective(self, alpha, Ms, y, Kx, Ks):
        """
        Complet objective function

        Parameters
        ----------
        alpha: np.ndarray
            Representer theorem coefficients, alpha.shape = (T, barM)
        Ms: list
            Number of locations per time step : [M_1,...,M_T]
        y: np.ndarray
            Flatten measurements, y.shape = (barM, )
        Kx: np.ndarray
            Training locations kernel matrix, Kx.shape = (barM, barM)
        Ks: np.ndarray
            Time samples kernel matrix, Ks.shape = (T, T)

        Returns
        -------
        obj_eval: float
            value of the full objective in alpha
        """
        return self.data_fitting(alpha, Ms, y, Kx, Ks) \
               + self.mu * self.spacereg(alpha, Kx, Ks) \
               + self.lamb * self.timereg(alpha, Kx, Ks)

    def objective_func(self, Ms, y, Kx, Ks):
        """
        Fix all parameters but alpha to optimize the objective function

        Parameters
        ----------
        Ms: list
            Number of locations per time step : [M_1,...,M_T]
        y: np.ndarray
            Flatten measurements, y.shape = (barM, )
        Kx: np.ndarray
            Training locations kernel matrix, Kx.shape = (barM, barM)
        Ks: np.ndarray
            Time samples kernel matrix, Ks.shape = (T, T)

        Returns
        -------
        obj: function
            objective function
        """
        MT = Kx.shape[0]
        T = Ks.shape[0]

        def obj(alpha):
            alpha_res = alpha.reshape((T, MT))
            return self.objective(alpha_res, Ms, y, Kx, Ks)

        return obj

    def objective_prime(self, alpha, Ms, y, Kx, Ks):
        """
        Gradient of the objective function

        Parameters
        ----------
        alpha: np.ndarray
            Representer theorem coefficients, alpha.shape = (T, barM)
        Ms: list
            Number of locations per time step : [M_1,...,M_T]
        y: np.ndarray
            Flatten measurements, y.shape = (barM, )
        Kx: np.ndarray
            Training locations kernel matrix, Kx.shape = (barM, barM)
        Ks: np.ndarray
            Time samples kernel matrix, Ks.shape = (T, T)

        Returns
        -------

        obj: function
            objective function as a function of only alpha
        """
        return self.data_fitting_prime(alpha, Ms, y, Kx, Ks) \
               + self.mu * self.spacereg.prime(alpha, Kx, Ks) \
               + self.lamb * self.timereg.prime(alpha, Kx, Ks)

    def objective_grad_func(self, Ms, y, Kx, Ks):
        """
        Fix all parameters but alpha to in the objective gradient for optimization

        Parameters
        ----------
        Ms: list
            Number of locations per time step : [M_1,...,M_T]
        y: np.ndarray
            Flatten measurements, y.shape = (barM, )
        Kx: np.ndarray
            Training locations kernel matrix, Kx.shape = (barM, barM)
        Ks: np.ndarray
            Time samples kernel matrix, Ks.shape = (T, T)

        Returns
        -------
        grad: function
            gradient of objective function as a function of only alpha
        """
        MT = Kx.shape[0]
        T = Ks.shape[0]

        def grad(alpha):
            alpha_res = alpha.reshape((T, MT))
            return self.objective_prime(alpha_res, Ms, y, Kx, Ks).flatten()

        return grad

    def fit(self, S, solver='L-BFGS-B', tol=1e-5, Kx=None, Ks=None):
        self.S = S
        if Kx is None:
            Kx = self.kernelx.compute_K(S["x_flat"])
        if Ks is None:
            Ks = self.kernels.compute_K(S["xy_tuple"])
        alpha0 = np.zeros(S.get_T() * S.get_barM())
        obj = self.objective_func(S.get_Ms(), S["y_flat"], Kx, Ks)
        grad = self.objective_grad_func(S.get_Ms(), S["y_flat"], Kx, Ks)
        sol = optimize.minimize(fun=obj, x0=alpha0, jac=grad, tol=tol, method=solver)
        self.alpha = sol["x"].reshape((S.get_T(), S.get_barM()))

    def predict(self, Slast, Xnew):
        Kxnew = self.kernelx.compute_Knew(self.S["x_flat"], Xnew)
        Ksnew = self.kernels.compute_Knew(self.S["xy_tuple"], Slast["xy_tuple"])
        return Ksnew.T.dot(self.alpha).dot(Kxnew)
