import numpy as np
import scipy.optimize as optimize
import functools

import algebra.repeated_matrix as repmat
import spatiotempovk.spatiotempdata as spatiotempdata


class DiffLocObsOnFuncReg:
    """
    Regressor (Reg) of a localized observations set (LocObs) onto another which outputs a function (Func)
    in the case where the loss is differentiable (Diff)
    """

    def __init__(self, loss, smoothreg, globalreg, mu, lamb, kernelx, kernels):
        self.loss = loss
        self.smoothreg = smoothreg
        self.globalreg = globalreg
        self.mu = mu
        self.lamb = lamb
        self.kernelx = kernelx
        self.kernels = kernels
        self.alpha = None
        self.training_input = None
        self.training_output = None
        self.sameloc = False

    @staticmethod
    def eval(kx, ks, alpha):
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

    def data_fitting(self, Ms, y, Kx, Ks, alpha):
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
        MT = Kx.shape[0]
        T = Ks.shape[0]
        if alpha.ndim == 1:
            # Support for flat alpha for scipy optimizers
            # This does not modify alpha outside of the function as we just assign a new view to it inside the function
            # but do not touch the memory
            alpha = alpha.reshape((T, MT))
        for t in range(T):
            xit = 0
            for m in range(Ms[t]):
                tm = sum(Ms[:t]) + m
                xit += self.loss(y[tm], DiffLocObsOnFuncReg.eval(Kx[tm], Ks[t], alpha))
            xi += (1 / Ms[t]) * xit
        return (1 / T) * xi

    def data_fitting_prime(self, Ms, y, Kx, Ks, alpha):
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
        MT = Kx.shape[0]
        T = Ks.shape[0]
        flatind = False
        if alpha.ndim == 1:
            # Support for flat alpha for scipy optimizers
            # This does not modify alpha outside of the function as we just assign a new view to it inside the function
            # but do not touch the memory
            alpha = alpha.reshape((T, MT))
            flatind = True
        xi_prime = np.zeros(alpha.shape)
        for t in range(T):
            for m in range(Ms[t]):
                tm = sum(Ms[:t]) + m
                k = Ks[t].reshape((Ks.shape[0], 1)).dot(Kx[tm].reshape((1, Kx.shape[0])))
                xi_prime += (1 / Ms[t]) * self.loss.prime(y[tm],
                                                          DiffLocObsOnFuncReg.eval(Kx[tm], Ks[t], alpha)) * k
        if flatind:
            return (1 / T) * xi_prime.flatten()
        else:
            return (1 / T) * xi_prime

    def objective(self, Ms, y, Kx, Ks, alpha):
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
        return self.data_fitting(Ms, y, Kx, Ks, alpha) \
            + self.mu * self.smoothreg(Kx, Ks, alpha) \
            + self.lamb * self.globalreg(Kx, Ks, alpha)

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
        return functools.partial(self.objective, Ms, y, Kx, Ks)

    def objective_prime(self, Ms, y, Kx, Ks, alpha):
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
        return self.data_fitting_prime(Ms, y, Kx, Ks, alpha) \
            + self.mu * self.smoothreg.prime(Kx, Ks, alpha) \
            + self.lamb * self.globalreg.prime(Kx, Ks, alpha)

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
        return functools.partial(self.objective_prime, Ms, y, Kx, Ks)

    def fit(self, S, V, solver='L-BFGS-B', tol=1e-5, Kx=None, Ks=None):
        # Memorize training set
        # if not isinstance(S, spatiotempdata.LocObsSet):
        #     self.training_input = spatiotempdata.LocObsSet(S)
        # else:
        #     self.training_input = S
        # if not isinstance(V, spatiotempdata.LocObsSet):
        #     self.training_output = spatiotempdata.LocObsSet(V)
        # else :
        #     self.training_output = V
        self.training_input = S
        self.training_output = V
        # Check if same location can be exploited in the locations
        if self.training_output.sameloc:
            self.sameloc = True
        if Kx is None:
            # Exploit sameloc=True by using the RepSymMatrix container to avoid storing a huge redundant matrix
            if self.sameloc:
                Kx = repmat.RepSymMatrix(self.kernelx.compute_K(V["x"][0]), rep=(V.get_T(), V.get_T()))
            else:
                Kx = self.kernelx.compute_K(V["x_flat"])
        if Ks is None:
            Ks = self.kernels.compute_K(S["xy_tuple"])
        # alpha0 = np.zeros(S.get_T() * S.get_barM())
        alpha0 = np.random.normal(0, 1, V.get_T() * V.get_barM())
        obj = self.objective_func(V.get_Ms(), V["y_flat"], Kx, Ks)
        grad = self.objective_grad_func(V.get_Ms(), V["y_flat"], Kx, Ks)
        record = []
        sol = optimize.minimize(fun=obj, x0=alpha0, jac=grad, tol=tol,
                                method=solver, callback=lambda X: record.append(obj(X)))
        self.alpha = sol["x"].reshape((V.get_T(), V.get_barM()))
        sol["record"] = record
        return sol

    def predict(self, Snew, Xnew):
        # Exploit sameloc=True by using the RepSymMatrix container to avoid storing a huge redundant matrix
        if self.sameloc:
            Kxnew = repmat.RepSymMatrix(self.kernelx.compute_K(self.training_output["x"][0]),
                                        rep=(self.training_output.get_T(), 1))
        else:
            Kxnew = self.kernelx.compute_Knew(self.training_output["x_flat"], Xnew)
        Ksnew = self.kernels.compute_Knew(self.training_input["xy_tuple"], Snew["xy_tuple"])
        # Weird order of matrix product for compatibility with algebra.repeated_matrix.RepSymMatrix
        return (Kxnew.transpose().dot(self.alpha.T).dot(Ksnew)).T




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
    sameloc: bool
        Will be inherited from S attribute, whether computation acceleration using the fact that locations
        are always the same should be used
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
        self.sameloc = False

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
               + self.mu * self.spacereg(Kx, Ks, alpha) \
               + self.lamb * self.timereg(Kx, Ks, alpha)

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
               + self.mu * self.spacereg.prime(Kx, Ks, alpha) \
               + self.lamb * self.timereg.prime(Kx, Ks, alpha)

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
        """
        Fit regressor

        Parameters
        ----------
        S: spatiotempovk.spatiotempdata.SpatioTempData
            The data to fit the regressor on

        """
        self.S = S
        # Inherits sameloc flag from the data it is fitted on
        # if sameloc is true it is exploited to speed up computations
        if self.S.sameloc:
            self.sameloc = True
        else:
            self.sameloc = False
        if Kx is None:
            # Exploit sameloc=True by using the RepSymMatrix container to avoid storing a huge redundant matrix
            if self.sameloc:
                Kx = repmat.RepSymMatrix(self.kernelx.compute_K(S["x"][0]), rep=(S.get_T(), S.get_T()))
            else:
                Kx = self.kernelx.compute_K(S["x_flat"])
        if Ks is None:
            Ks = self.kernels.compute_K(S["xy_tuple"])
        # alpha0 = np.zeros(S.get_T() * S.get_barM())
        alpha0 = np.random.normal(0, 1, S.get_T() * S.get_barM())
        obj = self.objective_func(S.get_Ms(), S["y_flat"], Kx, Ks)
        grad = self.objective_grad_func(S.get_Ms(), S["y_flat"], Kx, Ks)
        sol = optimize.minimize(fun=obj, x0=alpha0, jac=grad, tol=tol, method=solver)
        self.alpha = sol["x"].reshape((S.get_T(), S.get_barM()))
        print(sol["success"])

    def predict(self, Slast, Xnew):
        # Exploit sameloc=True by using the RepSymMatrix container to avoid storing a huge redundant matrix
        if self.sameloc:
            Kxnew = repmat.RepSymMatrix(self.kernelx.compute_K(self.S["x"][0]), rep=(self.S.get_T(), 1))
        else:
            Kxnew = self.kernelx.compute_Knew(self.S["x_flat"], Xnew)
        Ksnew = self.kernels.compute_Knew(self.S["xy_tuple"], Slast["xy_tuple"])
        # Weird order of matrix product for compatibility with algebra.repeated_matrix.RepSymMatrix
        return (Kxnew.transpose().dot(self.alpha.T).dot(Ksnew)).T
