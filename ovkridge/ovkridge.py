import spatiotempovk.kernels as kernels
from control import dlyap
import numpy as np
import operalib.ridge as ovkridge


class SeparableOVKRidge:

    def __init__(self, input_kernel, output_mat, lamb, normalize=False):
        if not isinstance(input_kernel, kernels.Kernel):
            self.k = kernels.Kernel(func=input_kernel, normalize=normalize)
        else:
            self.k = input_kernel
        self.B = output_mat
        self.lamb = lamb
        self.K = None
        self.alpha = None
        self.X = None

    def fit(self, X, Y, K=None):
        self.X = X
        if K is not None:
            self.K = K
        else:
            self.K = self.k.compute_K(X)
        n = X.shape[0]
        self.alpha = dlyap(-self.K/(self.lamb * n), self.B.T, Y/(self.lamb * n))

    def predict(self, Xnew):
        Knew = self.k.compute_Knew(self.X, Xnew)
        KoB = np.kron(Knew, self.B)
        return KoB.T.dot(self.alpha.flatten(order="C"))
        # return KoB.dot(self.alpha.flatten(order="F"))



import numpy as np
import matplotlib.pyplot as plt
import importlib
import scipy.optimize as optimize
import itertools
import pandas as pd
import os
import time
from functools import partial
import operalib.ridge as ovkridge
import syntheticdata.funcs1d as funcs1d
import functools
import sklearn.kernel_ridge as kernel_ridge
import pickle

import spatiotempovk.spatiotempdata as spatiotemp
import spatiotempovk.kernels as kernels
import spatiotempovk.losses as losses
import spatiotempovk.regularizers as regularizers
import spatiotempovk.regressors as regressors
import algebra.repeated_matrix as repmat
import smoothing.representer as repsmooth
import smoothing.parametrized_func as param_func
import solvers.gradientbased as gradientbased
import tsvalidation.sequentialval as seqval
import smoothing.fourierrandom as fourierrandom
import approxkernelridge.rffridge as rffridge
importlib.reload(repmat)
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(losses)
importlib.reload(regularizers)
importlib.reload(regressors)
importlib.reload(repsmooth)
importlib.reload(param_func)
importlib.reload(funcs1d)
importlib.reload(fourierrandom)
importlib.reload(rffridge)

# Plot parameters
plt.rcParams.update({"font.size": 30})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})


# Or load dataset
with open(os.getcwd() + "/dumps/datasets.pkl", "rb") as i:
    datain, dataout, dataintest, dataouttest = pickle.load(i)

sigma = 10
mu = 0.05
D = 40
d = 1

#
rffeats = rffridge.RandomFourierFeatures(sigma, D, d)

smoother = fourierrandom.RFFRidgeSmoother(rffeats, mu=0.05)

coefin, basisin = smoother(datain["x"], datain["y"])
coefout, basisout = smoother(dataout["x"], dataout["y"])


B = np.eye(D)
kerx = kernels.GaussianKernel(sigma=5)

test = SeparableOVKRidge(kerx, B, 0.01)

test.fit(coefin, coefout)

pred = test.predict(coefin[0].reshape((1, 40)))

parafunc = param_func.ParametrizedFunc(pred, basisout)

x0 = dataout["x"][0]
y0 = dataout["y"][0]

plt.plot(x0.flatten(), parafunc(x0))
plt.plot(x0.flatten(), y0)
