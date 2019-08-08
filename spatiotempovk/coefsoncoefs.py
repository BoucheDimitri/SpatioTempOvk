import numpy as np
from smoothing import expandedridgesmoother as expridgesmoother
from sklearn.kernel_ridge import KernelRidge
from smoothing import parametrized_func as param_func
from ovkridge import ovkridge


class CoefsOnCoefs:

    def __init__(self, ker, funcdictin, muin, funcdictout, muout, lamb):
        self.funcdictin = funcdictin
        self.funcdictout = funcdictout
        self.muin = muin
        self.muout = muout
        self.lamb = lamb
        self.smootherin = expridgesmoother.ExpRidgeSmoother(self.funcdictin, self.muin)
        self.smootherout = expridgesmoother.ExpRidgeSmoother(self.funcdictout, self.muout)
        self.ker = ker
        self.kernelridge = None

    def fit(self, S, V):
        nin = len(S["xy_tuple"])
        x = [S["xy_tuple"][i][0] for i in range(nin)]
        y = [S["xy_tuple"][i][1] for i in range(nin)]
        win, basein = self.smootherin(x, y)
        nout = len(V["xy_tuple"])
        v = [S["xy_tuple"][i][0] for i in range(nout)]
        w = [S["xy_tuple"][i][1] for i in range(nout)]
        wout, baseout = self.smootherout(v, w)
        Dout = wout.shape[1]
        # self.kernelridges = [KernelRidge(alpha=self.lamb, kernel=self.ker)
        # self.kernelridge.fit(win, wout)
        self.kernelridge = ovkridge.SeparableOVKRidge(self.ker, np.eye(Dout), lamb=self.lamb)
        self.kernelridge.fit(win, wout)
        # TODO: Rewrite with independant kernel ridges

    def predict(self, S, Xnew):
        nin = len(S["xy_tuple"])
        x = [S["xy_tuple"][i][0] for i in range(nin)]
        y = [S["xy_tuple"][i][1] for i in range(nin)]
        win, basein = self.smootherin(x, y)
        wout = self.kernelridge.predict(win)
        outfuncs = [param_func.ParametrizedFunc(wout[i], self.funcdictout.features_basis()) for i in range(wout.shape[0])]
        outp = [outfunc(Xnew) for outfunc in outfuncs]
        return np.array(outp)
