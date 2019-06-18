import numpy as np
from sklearn import kernel_ridge
from functools import partial
import smoothing.smoother as smoother


class RidgeSmoother(smoother.Smoother):

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






