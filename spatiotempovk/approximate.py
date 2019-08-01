import numpy as np
import scipy.optimize as optimize
import functools


class RFFSampleinFuncout:

    def __init__(self, loss, lamb, kers, rffs):
        self.loss = loss
        self.rffs = rffs
        self.kers = kers
        self.lamb = lamb
        self.training_input = None
        self.training_output = None
        self.C = None

    def data_fitting(self, Ls, v, w, Ks, C):
        xi = 0
        T = Ks.shape[0]
        if C.ndim == 1:
            # Support for flat alpha for scipy optimizers
            # This does not modify alpha outside of the function as we just assign a new view to it inside the function
            # but do not touch the memory
            C = C.reshape((self.rffs.D, T))
        for t in range(T):
            xit = 0
            for m in range(Ls[t]):
                tm = sum(Ls[:t]) + m
                phi = self.rffs.eval(v[tm, :])
                xit += self.loss(w[tm], Ks[t].T.dot(C.T).dot(phi.flatten()))
            xi += (1 / Ls[t]) * xit
        return (1 / T) * xi

    def data_fitting_prime(self, Ls, v, w, Ks, C):
        T = Ks.shape[0]
        flatind = False
        if C.ndim == 1:
            # Support for flat alpha for scipy optimizers
            # This does not modify alpha outside of the function as we just assign a new view to it inside the function
            # but do not touch the memory
            C = C.reshape((self.rffs.D, T))
            flatind = True
        xi_prime = np.zeros(C.shape)
        for t in range(T):
            for m in range(Ls[t]):
                tm = sum(Ls[:t]) + m
                phi = self.rffs.eval(v[tm, :])
                k = phi.T.dot(np.expand_dims(Ks[t], axis=0))
                xi_prime += (1 / Ls[t]) * k * self.loss.prime(w[tm], Ks[t].T.dot(C.T).dot(phi.flatten()))
        if flatind:
            return (1 / T) * xi_prime.flatten()
        else:
            return (1 / T) * xi_prime

    def regularization(self, Ks, C):
        T = Ks.shape[0]
        if C.ndim == 1:
            # Support for flat alpha for scipy optimizers
            # This does not modify alpha outside of the function as we just assign a new view to it inside the function
            # but do not touch the memory
            C = C.reshape((self.rffs.D, T))
        return np.sum(Ks * (C.T.dot(C)))

    def regularization_prime(self, Ks, C):
        T = Ks.shape[0]
        flatind = False
        if C.ndim == 1:
            # Support for flat alpha for scipy optimizers
            # This does not modify alpha outside of the function as we just assign a new view to it inside the function
            # but do not touch the memory
            C = C.reshape((self.rffs.D, T))
            flatind = True
        if flatind:
            return (C.dot(Ks)).flatten()
        else:
            return C.dot(Ks)

    def objective(self, Ls, v, w, Ks, C):
        return self.data_fitting(Ls, v, w, Ks, C) + self.lamb * self.regularization(Ks, C)

    def objective_prime(self, Ls, v, w, Ks, C):
        return self.data_fitting_prime(Ls, v, w, Ks, C) + self.lamb * self.regularization_prime(Ks, C)

    def objective_func(self, Ls, v, w, Ks):
        return functools.partial(self.objective, Ls, v, w, Ks)

    def objective_prime_func(self, Ls, v, w, Ks):
        return functools.partial(self.objective_prime, Ls, v, w, Ks)

    def fit(self, S, V, solver='L-BFGS-B', tol=1e-5, Ks=None):
        self.training_input = S
        self.training_output = V
        if Ks is None:
            Ks = self.kers.compute_K(S["xy_tuple"])
        C0 = np.random.normal(0, 1, V.get_T() * self.rffs.D)
        obj = self.objective_func(V.get_Ms(), V["x_flat"], V["y_flat"], Ks)
        grad = self.objective_prime_func(V.get_Ms(), V["x_flat"], V["y_flat"], Ks)
        record = []
        sol = optimize.minimize(fun=obj, x0=C0, jac=grad, tol=tol,
                                method=solver, callback=lambda X: record.append(obj(X)))
        self.C = sol["x"].reshape((self.rffs.D, V.get_T()))
        sol["record"] = record
        return sol

    def predict(self, Snew, Xnew):
        Ksnew = self.kers.compute_Knew(self.training_input["xy_tuple"], Snew["xy_tuple"])
        phis = self.rffs.eval(Xnew)
        return Ksnew.T.dot(self.C.T).dot(phis.T)







