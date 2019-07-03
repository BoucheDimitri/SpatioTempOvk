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
import spatiotempovk.approximate as approxsamponfunc
importlib.reload(repmat)
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(losses)
importlib.reload(regularizers)
importlib.reload(regressors)
importlib.reload(repsmooth)
importlib.reload(param_func)
importlib.reload(funcs1d)
importlib.reload(approxsamponfunc)

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
locs = np.linspace(0, 1, nlocs).reshape((nlocs, 1))
datain, dataout = funcs1d.generate_fourier_dataset(Ntrain, noisein, noiseout, norm01, 2, locs)
dataintest, dataouttest = funcs1d.generate_fourier_dataset(Ntest, 0, 0, norm01, 2, locs)
datain, dataout = spatiotemp.LocObsSet(datain), spatiotemp.LocObsSet(dataout)
dataintest, dataouttest = spatiotemp.LocObsSet(dataintest), spatiotemp.LocObsSet(dataouttest)

# # Or load dataset
# with open(os.getcwd() + "/dumps/datasets.pkl", "rb") as i:
#     datain, dataout, dataintest, dataouttest = pickle.load(i)

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
sigmarff=10
D = 100
kers = kernels.GaussianSameLoc(sigma=10)
Ks = kers.compute_K(datain["xy_tuple"])
# kernely = kernels.GaussianKernel(sigma=0.5)
# Kyin = kernely.compute_K(datain["y_flat"])
# kers = kernels.ConvKernel(kernelx, kernely, Kxin, Kyin, sameloc=False)
# Ks = kers.compute_K_from_mat(datain.Ms)

# Build regressor
l2 = losses.L2Loss()
lamb = 0.00005
reg = approxsamponfunc.RFFSampleinFuncout(D=D, sigmarff=sigmarff, loss=l2, lamb=lamb, d=1, kers=kers)


# # Test with gradient descent
# gd = gradientbased.GradientDescent(0.05, 50, 1e-5, record=True)
# obj = reg.objective_func(dataout.Ms, dataout["x_flat"], dataout["y_flat"], Ks)
# grad = reg.objective_prime_func(dataout.Ms, dataout["x_flat"], dataout["y_flat"], Ks)
# C0 = np.random.normal(0, 1, (D, Ntrain))
# sol = gd(obj, grad, C0)

# Fit regressor
solu = reg.fit(datain, dataout, Ks=Ks, tol=1e-4)
i = 1
pred = reg.predict(dataintest.extract_subseq(i, i+1), locs)

plt.figure()
plt.plot(locs.flatten(), pred.flatten(), label="predicted")
plt.plot(locs.flatten(), dataouttest["y"][i], label="real")
plt.title("Example of fitting on test set")
plt.legend()


# # Pickle regressor
# with open(os.getcwd() + "/dumps/lamb0001_mu001_funker.pkl", "wb") as o:
#     pickle.dump(regressor, o, pickle.HIGHEST_PROTOCOL)
#
# # Or load dataset
# with open(os.getcwd() + "/dumps/lamb001_mu001_convker.pkl", "rb") as i:
#     regressor = pickle.load(i)

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
