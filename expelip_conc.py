import numpy as np
import matplotlib.pyplot as plt
import importlib
import pandas as pd
import syntheticdata.funcs1d as funcs1d
import functools
import os

import spatiotempovk.spatiotempdata as spatiotemp
import spatiotempovk.kernels as kernels
import spatiotempovk.losses as losses
import spatiotempovk.regularizers as regularizers
import spatiotempovk.regressors as regressors
import algebra.repeated_matrix as repmat
import smoothing.representer as repsmooth
import smoothing.parametrized_func as param_func
import spatiotempovk.dictout as dictout
import approxkernelridge.rffridge as rffridge
import spatiotempovk.approximate as approxsamponfunc
import basisexpansion.funcdicts as funcdicts
import basisexpansion.expandedregs as expridge
import spatiotempovk.coefsoncoefs as coefsoncoefs
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
importlib.reload(dictout)
importlib.reload(expridge)
importlib.reload(funcdicts)
importlib.reload(coefsoncoefs)


def corrupt_data(xarray, varray, timevec, nmissingin, nmissingout, noisein, noiseout, seed=0):
    xlist_corrupt = []
    vlist_corrupt = []
    for i in range(xarray.shape[1]):
        nx = xarray.shape[0] - nmissingin
        nv = varray.shape[0] - nmissingout
        np.random.seed(seed)
        indsx = np.random.choice(xarray.shape[0], nx, replace=False)
        indsx.sort()
        np.random.seed(seed)
        indsv = np.random.choice(varray.shape[0], nv, replace=False)
        indsv.sort()
        np.random.seed(seed)
        noisex = np.random.normal(0, noisein, nx)
        np.random.seed(seed)
        noisev = np.random.normal(0, noiseout, nv)
        xlist_corrupt.append((timevec[indsx].reshape((nx, 1)), (xarray[indsx, i] + noisex).reshape((nx, 1))))
        vlist_corrupt.append((timevec[indsv].reshape((nv, 1)), (varray[indsv, i] + noisev).reshape((nv, 1))))
    return xlist_corrupt, vlist_corrupt





path = os.getcwd() + "/datalip/"
np.random.seed(0)
shuffle_inds = np.random.choice(32, 32, replace=False)
emg = pd.read_csv(path + "EMGmatlag.csv", header=None).values[:, shuffle_inds]
lipacc = pd.read_csv(path + "lipmatlag.csv", header=None).values[:, shuffle_inds]
timevec = pd.read_csv(path + "tfine.csv", header=None).values
# lipaccmean = lipacc.mean(axis=1).reshape((641, 1))
# lipaccstd = lipacc.std(axis=1).reshape((641, 1))
# emgmean = emg.mean(axis=1).reshape((641, 1))
# # centered_lipacc = (lipacc - lipaccmean)/lipaccstd
# centered_lipacc = (lipacc - lipaccmean)
# centered_emg = emg - emgmean

# with open(os.getcwd() + "/shuffle.pkl", "rb") as inp:
#     shuffle_inds = pickle.load(inp)

# Train/Test split
Ntrain = 25
Ntest = 32 - Ntrain

# Corrupt the training data
xtrainlist, vtrainlist = corrupt_data(emg[:, :Ntrain],
                                      lipacc[:, :Ntrain],
                                      timevec,
                                      nmissingin=200,
                                      nmissingout=200,
                                      noisein=0.03,
                                      noiseout=0.08)


# Put dat in spatio-temporal format
Xtrain = spatiotemp.LocObsSet(xtrainlist)
Vtrain = spatiotemp.LocObsSet(vtrainlist)
xtest = [(timevec, emg[:, i].reshape((641, 1))) for i in range(Ntrain, 32)]
vtest = [(timevec, lipacc[:, i].reshape((641, 1))) for i in range(Ntrain, 32)]
Xtest = spatiotemp.LocObsSet(xtest)
Vtest = spatiotemp.LocObsSet(vtest)

fig, ax = plt.subplots(ncols=2)
for i in range(32):
    ax[1].plot(lipacc[:, i])
    ax[0].plot(emg[:, i])

# Test for bandwidth parameter
# See if our input smoothing has the means to represent well the input functions
Dsmoothing = 300
sigmasmoothing = 45
musmoothing = 0.05
rffsx = rffridge.RandomFourierFeatures(sigmasmoothing, Dsmoothing, d=1)
testridge = rffridge.RFFRidge(musmoothing, rffsx)
i = 6
testridge.fit(Xtrain["x"][i], Xtrain["y"][i])
pred = testridge.predict(Xtrain["x"][i])
plt.figure()
plt.plot(Xtrain["x"][i], pred, label="predicted")
plt.plot(Xtrain["x"][i], Xtrain["y"][i], label="real")
plt.legend()


# Kernels
sigmarff = 45
D = 300
# kers = kernels.GaussianFuncKernel(sigma=3, rffeats=rffsx, mu=musmoothing)
kers = kernels.GaussianFuncKernel(sigma=3, funcdict=rffsx, mu=musmoothing)
Ks = kers.compute_K(Xtrain["xy_tuple"])
plt.figure()
plt.imshow(Ks)
rffs = rffridge.RandomFourierFeatures(sigmarff, D, d=1)
test = rffs.eval(Vtest["x"][0])
plt.imshow(test.dot(test.T))

# Test for bandwidth parameter
# To see if our output dictionary has the means to approximate well the output functions
muout = 0.1
testridge = rffridge.RFFRidge(muout, rffs)
i = 6
testridge.fit(Vtrain["x"][i], Vtrain["y"][i])
pred = testridge.predict(timevec.reshape((501, 1)))
plt.figure()
plt.plot(timevec, pred, label="predicted")
plt.plot(Vtrain["x"][i], Vtrain["y"][i], label="real")
plt.legend()

# Fit
# Build regressor
l2 = losses.L2Loss()
lamb = 0.001
reg = coefsoncoefs.CoefsOnCoefs(kernels.GaussianKernel(sigma=3), rffsx, musmoothing, rffs, muout, lamb)

# Fit regressor
reg.fit(Xtrain, Vtrain)

# Predict

pred = reg.predict(Xtest, timevec)
i = 4

# Pred on test set
plt.figure()
plt.plot(timevec.flatten(), pred[i, :], label="predicted")
plt.plot(timevec.flatten(), Vtest["y"][i], label="real")
plt.title("Example of fitting on test set")
plt.legend()

pred = reg.predict(Xtrain, timevec.reshape((501, 1)))
i = 7
# Pred on train set
plt.figure()
plt.plot(timevec.flatten(), pred[i, :], label="predicted")
plt.scatter(Vtrain["x"][i], Vtrain["y"][i], label="real")
plt.title("Example of fitting on test set")
plt.legend()
