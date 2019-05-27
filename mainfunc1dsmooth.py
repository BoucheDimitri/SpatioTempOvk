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

# Smoothing
gausskerx = kernels.GaussianKernel(sigma=0.3)
Kx = gausskerx.compute_K(datain["x"][0])
ridge_smoother = repsmooth.RidgeSmoother(mu=0.001, kernel=gausskerx)
alphain, base = ridge_smoother(datain["x"], datain["y"])
alphaout, _ = ridge_smoother(datain["x"], dataout["y"])

# Test for smoothing
f0 = param_func.ParametrizedFunc(alphaout[0], base)
f0eval = [f0(x) for x in datain["x"][0]]
plt.plot(f0eval, label="smoothed")
plt.plot(dataout["y"][0], label="real")

# Operator valued kernel regression
reg = kernel_ridge.KernelRidge(alpha=0.1)
reg.fit(alphain, alphaout)
testalpha = reg.predict(alphain)

f0test = param_func.ParametrizedFunc(testalpha[0], base)
f0testeval = [f0test(x) for x in datain["x"][0]]
plt.plot(f0testeval, label="pred")
plt.legend()

# Out of sample test
# Smoothing
alphaintest, base = ridge_smoother(dataintest["x"], dataintest["y"])
alphaouttest = reg.predict(alphaintest)
f0test = param_func.ParametrizedFunc(alphaouttest[0], base)
f0testeval = [f0test(x) for x in datain["x"][0]]
plt.plot(f0testeval, label="pred")
plt.plot(dataouttest["y"][0], label="true")
plt.legend()