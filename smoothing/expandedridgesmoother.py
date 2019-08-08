import numpy as np
import smoothing.smoother as smoother
import basisexpansion.expandedregs as expridge


class ExpRidgeSmoother(smoother.Smoother):

    def __init__(self, funcdict, mu):
        super(ExpRidgeSmoother, self).__init__()
        self.mu = mu
        self.funcdict = funcdict

    def __call__(self, X, Y):
        nprobs = len(X)
        if len(Y) != nprobs:
            raise ValueError("Number of problems for input ({}) and target ({}) do not match".format(nprobs, len(Y)))
        w = np.zeros((nprobs, self.funcdict.D))
        for t in range(nprobs):
            ridge = expridge.ExpandedRidge(self.mu, self.funcdict)
            ridge.fit(X[t], Y[t])
            w[t, :] = ridge.w.copy()
        return w, self.funcdict.features_basis()

