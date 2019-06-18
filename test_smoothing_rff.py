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
mu = 1
D = 40
d = 1

#
rffeats = rffridge.RandomFourierFeatures(sigma, D, d)
reg = rffridge.RFFRidge(mu, rffeats)
reg.fit(datain["x"][0], datain["y"][0])
test = reg.predict(datain["x"][0])
plt.plot(datain["x"][0].flatten(), datain["y"][0])
plt.plot(datain["x"][0].flatten(), test)

# basis = rffeats.features_basis()
# ypred = []
# for i in range(datain["x"][0].shape[0]):
#     evals = [reg.w[j] * basis[j](datain["x"][0][i])[0] for j in range(D)]
#     ypred.append(sum(evals))


smoother = fourierrandom.RFFRidgeSmoother(sigma, mu, D, d)

coef, basis = smoother(datain["x"], datain["y"])

reg = rffridge.RFFRidge(mu, smoother.rffeats)
reg.fit(datain["x"][0], datain["y"][0])
test = reg.predict(datain["x"][0])
plt.plot(datain["x"][0].flatten(), datain["y"][0])
plt.plot(datain["x"][0].flatten(), test)

ypred = []
for i in range(datain["x"][0].shape[0]):
    evals = [coef[0][j] * basis[j](datain["x"][0][i])[0] for j in range(D)]
    ypred.append(sum(evals))


parafunc = param_func.ParametrizedFunc(coef[0], basis)

x0 = datain["x"][0]

plt.plot(x0.flatten(), parafunc(x0))

test = np.array([basis[j](x0[0]) for j in range(len(basis))])
evalmat = parafunc.eval_matrix(x0)
tt = evalmat.dot(coef[0])


rffeats = rffridge.RandomFourierFeatures(sigma, D, d)
kertest = kernels.GaussianFuncKernel(5, rffeats, mu)

test = kertest.compute_K(datain["xy_tuple"])

testbis = kertest.compute_Knew(datain["xy_tuple"], dataintest["xy_tuple"])