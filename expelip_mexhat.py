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
import basisexpansion.expandedregs as expregs
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
importlib.reload(expregs)
importlib.reload(funcdicts)


def corrupt_data(xarray, varray, timevec, nmissingin, nmissingout, noisein, noiseout):
    xlist_corrupt = []
    vlist_corrupt = []
    for i in range(xarray.shape[1]):
        nx = xarray.shape[0] - nmissingin
        nv = varray.shape[0] - nmissingout
        indsx = np.random.choice(xarray.shape[0], nx, replace=False)
        indsx.sort()
        indsv = np.random.choice(varray.shape[0], nv, replace=False)
        indsv.sort()
        noisex = np.random.normal(0, noisein, nx)
        noisev = np.random.normal(0, noiseout, nv)
        xlist_corrupt.append((timevec[indsx].reshape((nx, 1)), (xarray[indsx, i] + noisex).reshape((nx, 1))))
        vlist_corrupt.append((timevec[indsv].reshape((nv, 1)), (varray[indsv, i] + noisev).reshape((nv, 1))))
    return xlist_corrupt, vlist_corrupt






# ################### LOAD THE DATA ####################################################################################
path = os.getcwd() + "/datalip/"
shuffle_inds = np.random.choice(32, 32, replace=False)
emg = pd.read_csv(path + "EMGmatlag.csv", header=None).values[:, shuffle_inds]
lipacc = pd.read_csv(path + "lipmatlag.csv", header=None).values[:, shuffle_inds]
timevec = pd.read_csv(path + "tfine.csv", header=None).values
lipaccmean = lipacc.mean(axis=1).reshape((641, 1))
lipaccstd = lipacc.std(axis=1).reshape((641, 1))
emgmean = emg.mean(axis=1).reshape((641, 1))
# centered_lipacc = (lipacc - lipaccmean)/lipaccstd
centered_lipacc = (lipacc - lipaccmean)
centered_emg = emg - emgmean


# Plot data
plt.figure()
for i in range(32):
    plt.plot(centered_lipacc[:, i])

# Plot data
plt.figure()
for i in range(32):
    plt.plot(lipacc[:, i])

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

# Plot examples of data
inds = [1, 3]
fig, ax = plt.subplots(2)
for i in inds:
    ax[0].scatter(xtrainlist[i][0], xtrainlist[i][1])
    ax[1].scatter(vtrainlist[i][0], vtrainlist[i][1])

# Put dat in spatio-temporal format
Xtrain = spatiotemp.LocObsSet(xtrainlist)
Vtrain = spatiotemp.LocObsSet(vtrainlist)
xtest = [(timevec.reshape((641, 1)), emg[:, i].reshape((641, 1))) for i in range(Ntrain, 32)]
vtest = [(timevec.reshape((641, 1)), lipacc[:, i].reshape((641, 1))) for i in range(Ntrain, 32)]
Xtest = spatiotemp.LocObsSet(xtest)
Vtest = spatiotemp.LocObsSet(vtest)


# plt.figure()
# for i in range(32):
#     plt.plot(emg[:, i])

# Test for bandwidth parameter
# See if our input smoothing has the means to represent well the input functions
musmoothing = 1
mexhats = funcdicts.MexHatDict((timevec[0], timevec[-1]), np.linspace(timevec[0], timevec[-1], 20), np.linspace(0.01, 0.1, 10))
# lspace = np.linspace(0, 0.69, 501)
# plt.figure()
# plt.plot(lspace, [mexhats.atom(0.3, 0.01, t) for t in lspace])
testridge = expregs.ExpandedRidge(musmoothing, mexhats)
i = 3
testridge.fit(Xtrain["x"][i], Xtrain["y"][i])
pred = testridge.predict(Xtrain["x"][i])
plt.figure()
plt.plot(Xtrain["x"][i], pred, label="predicted")
plt.plot(Xtrain["x"][i], Xtrain["y"][i], label="real")
plt.legend()


# Kernels
kers = kernels.GaussianFuncKernel(sigma=2, funcdict=mexhats, mu=musmoothing)
Ks = kers.compute_K(Xtrain["xy_tuple"])
plt.figure()
plt.imshow(Ks)


mexhatsout = funcdicts.MexHatDict((timevec[0], timevec[-1]), np.linspace(timevec[0], timevec[-1], 10), np.linspace(0.02, 0.1, 10))
# Test for bandwidth parameter
# To see if our output dictionary has the means to approximate well the output functions
testridge = expregs.ExpandedRidge(0.1, mexhatsout)
i = 4
testridge.fit(Vtrain["x"][i], Vtrain["y"][i])
pred = testridge.predict(timevec)
plt.figure()
plt.plot(timevec, pred, label="predicted")
plt.plot(Vtrain["x"][i], Vtrain["y"][i], label="real")
plt.legend()



# Fit
# Build regressor
l2 = losses.L2Loss()
lamb = 0.001
mu = 0.06
reg = dictout.FuncInDictOut(loss=l2, mu=mu, lamb=lamb, kers=kers, funcdic=mexhatsout)

# Fit regressor
solu = reg.fit(Xtrain, Vtrain, Ks=Ks, tol=1e-4)

# Predict

pred = reg.predict(Xtest,timevec)
i = 4

# Pred on test set
plt.figure()
plt.plot(timevec.flatten(), pred[i, :], label="predicted")
plt.plot(timevec.flatten(), Vtest["y"][i], label="real")
plt.title("Example of fitting on test set")
plt.legend()

pred = reg.predict(Xtrain, timevec)
i = 6
# Pred on train set
plt.figure()
plt.plot(timevec.flatten(), pred[i, :], label="predicted")
plt.scatter(Vtrain["x"][i], Vtrain["y"][i], label="real")
plt.title("Example of fitting on test set")
plt.legend()
