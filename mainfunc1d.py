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
importlib.reload(repmat)
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(losses)
importlib.reload(regularizers)
importlib.reload(regressors)
importlib.reload(repsmooth)
importlib.reload(param_func)
importlib.reload(funcs1d)

# Drawing function from normal(0, 1)
norm01 = functools.partial(np.random.normal, 0, 1)

# Determine (fixed) locations
nlocs = 50
locs = np.linspace(0, 1, nlocs).reshape((nlocs, 1))

# Build the data
Ntrain = 50
Ntest = 20
# Draw random Fourier functions
fouriers = funcs1d.random_fourier_func(norm01, nfreq=2, nsim=Ntrain + Ntest)
fouriers_prime = [four.prime() for four in fouriers]
datain = []
dataout = []
for n in range(Ntrain + Ntest):
    Yin = np.array([fouriers[n](x[0]) for x in locs])
    Yout = np.array([fouriers_prime[n](x[0]) for x in locs])
    datain.append((locs, Yin))
    dataout.append((locs, Yout))
# Store them in a spatio temp data instance
datain = spatiotemp.LocObsSet(datain)
dataout = spatiotemp.LocObsSet(dataout)
dataintest = datain.extract_subseq(Ntrain, Ntrain + Ntest)
dataouttest = dataout.extract_subseq(Ntrain, Ntrain + Ntest)
datain = datain.extract_subseq(0, Ntrain)
dataout = dataout.extract_subseq(0, Ntrain)


# Kernels
kernelx = kernels.GaussianKernel(sigma=0.3)
Kxin = kernelx.compute_K(locs)
kernely = kernels.GaussianKernel(sigma=0.5)
Kyin = kernely.compute_K(datain["y_flat"])
kers = kernels.ConvKernel(kernelx, kernely, Kxin, Kyin, sameloc=True)
Ks = kers.compute_K_from_mat(datain.Ms)

# Build regressor
l2 = losses.L2Loss()
lamb = 0.1
mu = 0.1
smoothreg = regularizers.TikhonovSpace()
globalreg = regularizers.TikhonovTime()
regressor = regressors.DiffLocObsOnFuncReg(l2, smoothreg, globalreg, mu, lamb, kernelx, kers)

# Test with gradient descent
Kxout = repmat.RepSymMatrix(Kxin, (Ntrain, Ntrain))
gd = gradientbased.GradientDescent(0.00001, 200, 1e-5, record=True)
obj = regressor.objective_func(dataout.Ms, dataout["y_flat"], Kxout, Ks)
grad = regressor.objective_grad_func(dataout.Ms, dataout["y_flat"], Kxout, Ks)
alpha0 = np.random.normal(0, 1, (Ntrain, Ntrain*nlocs))
sol = gd(obj, grad, alpha0)

# Fit regressor
Kxout = repmat.RepSymMatrix(Kxin, (Ntrain, Ntrain))
solu = regressor.fit(datain, dataout, Kx=Kxout, Ks=Ks)

pred = regressor.predict(datain.extract_subseq(0, 1), datain["x"][0])
plt.figure()
plt.plot(dataout["x"][0], pred.flatten(), label="predicted")
plt.plot(dataout["x"][0], dataout["y"][0], label="real")
plt.title("Example of fitting on training set (without regularization)")
plt.legend()

# See size of terms involved
regressor.data_fitting(dataout.Ms, dataout["y_flat"], Kxout, Ks, alpha0)
regressor.smoothreg(Kxout, Ks, alpha0)
regressor.globalreg(Kxout, Ks, alpha0)

# Plots
i = 3
plt.figure()
plt.scatter(dataout["x"][i], dataout["y"][i])

plt.figure()
plt.scatter(dataout["x"][i], pred[i])
