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

# Plot parameters
plt.rcParams.update({"font.size": 30})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})

# Drawing function from normal(0, 1)
norm01 = functools.partial(np.random.normal, 0, 1)
uni01 = functools.partial(np.random.uniform, 0, 1)

# Determine (fixed) locations
nlocs = 50

# Build the data
Ntrain = 50
Ntest = 20
noisein = 0.25
noiseout = 2.5

# Draw random Fourier functions
# datain, dataout = funcs1d.fourier_dataset_varloc(Ntrain, noisein, noiseout, norm01, 2, nlocs, uni01)
test_locs = np.linspace(0, 1, 200).reshape((200, 1))
# dataintest, dataouttest = funcs1d.generate_fourier_dataset(Ntest, 0, 0, norm01, 2, test_locs)
# datain, dataout = spatiotemp.LocObsSet(datain), spatiotemp.LocObsSet(dataout)
# dataintest, dataouttest = spatiotemp.LocObsSet(dataintest), spatiotemp.LocObsSet(dataouttest)

# Or load dataset
with open(os.getcwd() + "/dumps/datasets.pkl", "rb") as i:
    datain, dataout, dataintest, dataouttest = pickle.load(i)

i=0
j=2
fig, ax = plt.subplots(nrows=2, ncols=2)
ax[0, 0].scatter(datain["x"][i].flatten(), datain["y"][i])
ax[0, 1].scatter(dataout["x"][i].flatten(), dataout["y"][i])
ax[0, 0].set_title("Function noisy evaluations")
ax[0, 1].set_title("Derivative noisy evaluations")
ax[1, 0].scatter(datain["x"][j].flatten(), datain["y"][j])
ax[1, 1].scatter(dataout["x"][j].flatten(), dataout["y"][j])


# Kernels
kernelx = kernels.GaussianKernel(sigma=0.075)
Kxin = kernelx.compute_K(datain["x_flat"])
rffeats = rffridge.RandomFourierFeatures(sigma=10, D=40, d=1)
kers = kernels.GaussianFuncKernel(5, rffeats, mu=0.05)
Ks = kers.compute_K(datain["xy_tuple"])
# kernely = kernels.GaussianKernel(sigma=0.5)
# Kyin = kernely.compute_K(datain["y_flat"])
# kers = kernels.ConvKernel(kernelx, kernely, Kxin, Kyin, sameloc=False)
# Ks = kers.compute_K_from_mat(datain.Ms)

# Build regressor
l2 = losses.L2Loss()
lamb = 0.001
mu = 0.01
smoothreg = regularizers.TikhonovSpace()
globalreg = regularizers.TikhonovTime()
regressor = regressors.DiffLocObsOnFuncReg(l2, smoothreg, globalreg, mu, lamb, kernelx, kers)

# # Test with gradient descent
# Kxout = repmat.RepSymMatrix(Kxin, (Ntrain, Ntrain))
# gd = gradientbased.GradientDescent(0.00001, 10, 1e-5, record=True)
# obj = regressor.objective_func(dataout.Ms, dataout["y_flat"], Kxout, Ks)
# grad = regressor.objective_grad_func(dataout.Ms, dataout["y_flat"], Kxout, Ks)
# alpha0 = np.random.normal(0, 1, (Ntrain, Ntrain*nlocs))
# sol = gd(obj, grad, alpha0)

# Fit regressor
# Kxout = repmat.RepSymMatrix(Kxin, (Ntrain, Ntrain))
Kxout = kernelx.compute_K(dataout["x_flat"])
solu = regressor.fit(datain, dataout, Kx=Kxout, Ks=Ks, tol=1e-3)

# Pickle regressor
with open(os.getcwd() + "/dumps/lamb0001_mu001_funker.pkl", "wb") as o:
    pickle.dump(regressor, o, pickle.HIGHEST_PROTOCOL)

# Or load dataset
with open(os.getcwd() + "/dumps/lamb001_mu001_convker.pkl", "rb") as i:
    regressor = pickle.load(i)

pred = regressor.predict(datain.extract_subseq(0, 1), datain["x"][0])
plt.figure()
plt.plot(dataout["x"][0], pred.flatten(), label="predicted")
plt.plot(dataout["x"][0], dataout["y"][0], label="real")
plt.title("Example of fitting on training set (without regularization)")
plt.legend()

# Prediction on test set
i = 1
predtest = regressor.predict(dataintest.extract_subseq(i, i+1), test_locs)
plt.figure()
plt.plot(test_locs, predtest.flatten(), label="predicted")
plt.plot(test_locs, dataouttest["y"][i], label="real")
# plt.title("Example on test set - Convolutional kernel - No regularization")
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
