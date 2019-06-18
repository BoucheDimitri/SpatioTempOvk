import numpy as np
from functools import partial
import smoothing.smoother as smoother
import approxkernelridge.rffridge as rffridge


class RFFRidgeSmoother(smoother.Smoother):

    def __init__(self, rffeats, mu):
        super(RFFRidgeSmoother, self).__init__()
        self.mu = mu
        self.rffeats = rffeats

    def __call__(self, X, Y):
        nprobs = len(X)
        if len(Y) != nprobs:
            raise ValueError("Number of problems for input ({}) and target ({}) do not match".format(nprobs, len(Y)))
        w = np.zeros((nprobs, self.rffeats.D))
        for t in range(nprobs):
            ridge = rffridge.RFFRidge(self.mu, self.rffeats)
            ridge.fit(X[t], Y[t])
            w[t, :] = ridge.w.copy()
        return w, self.rffeats.features_basis()




