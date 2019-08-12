import numpy as np
import importlib
import pandas as pd
import os
import pickle

import spatiotempovk.spatiotempdata as spatiotemp
import spatiotempovk.kernels as kernels
import spatiotempovk.losses as losses
import spatiotempovk.regularizers as regularizers
import spatiotempovk.regressors as regressors
import algebra.repeated_matrix as repmat
import smoothing.representer as repsmooth
import smoothing.parametrized_func as param_func
import spatiotempovk.dictout as dictout
import spatiotempovk.approximate as approxsamponfunc
import basisexpansion.funcdicts as funcdicts
import basisexpansion.expandedregs as expregs
import spatiotempovk.coefsoncoefs as coefsoncoefs
importlib.reload(repmat)
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(losses)
importlib.reload(regularizers)
importlib.reload(regressors)
importlib.reload(repsmooth)
importlib.reload(param_func)
importlib.reload(approxsamponfunc)
importlib.reload(dictout)
importlib.reload(expregs)
importlib.reload(funcdicts)
importlib.reload(coefsoncoefs)


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


def mse_score(pred, true):
    return ((pred - true) ** 2).sum(axis=1).mean()



# ################### LOAD THE DATA ####################################################################################
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


# Kernels
musmoothing = 1
mexhats = funcdicts.MexHatDict((timevec[0], timevec[-1]), np.linspace(timevec[0], timevec[-1], 20), np.linspace(0.01, 0.1, 10))
kers = kernels.GaussianFuncKernel(sigma=2, funcdict=mexhats, mu=musmoothing)
Ks = kers.compute_K(Xtrain["xy_tuple"])

# output dict
mexhatsout = funcdicts.MexHatDict((timevec[0], timevec[-1]), np.linspace(timevec[0], timevec[-1], 10), np.linspace(0.02, 0.1, 10))
muout = 0.1

# lamb_grid = np.linspace(0.0001, 0.1, 50)
lamb_grid = np.linspace(0.001, 100, 200)
l2 = losses.L2Loss()

scores = np.zeros((len(lamb_grid)))
regressors = []

# TODO: Issue with the output from ParametrizedFunc, "eval takes 2 positional arguments, 3 were given"
for i in range(len(lamb_grid)):
    reg = coefsoncoefs.CoefsOnCoefs(kernels.GaussianKernel(sigma=3), mexhats, musmoothing, mexhatsout, muout, lamb_grid[i])
    reg.fit(Xtrain, Vtrain)
    pred = reg.predict(Xtest, timevec)
    scores[i] = mse_score(pred, np.squeeze(np.array(Vtest["y"]), axis=2))
    regressors.append(reg)
    print("lamb = " + str(lamb_grid[i]))

with open(os.getcwd() + "/tuning_mex_conc.pkl", "wb") as outp:
    pickle.dump((scores, regressors), outp)

#
# reg = coefsoncoefs.CoefsOnCoefs(kernels.GaussianKernel(sigma=3), mexhats, musmoothing, mexhatsout, muout, lamb_grid[0])
# reg.fit(Xtrain, Vtrain)
# pred = reg.predict(Xtest, timevec)




# with open(os.getcwd() + "/tuning_mex.pkl", "rb") as inp:
#     scores, regressors = pickle.load(inp)
#
#
# # pred = regressors[3][3].predict(Xtest, timevec.reshape((501, 1)))
# i = 3
# #
# # import matplotlib.pyplot as plt
# # # Pred on test set
# plt.figure()
# plt.plot(timevec.flatten(), pred[i, :], label="predicted")
# plt.plot(timevec.flatten(), Vtest["y"][i], label="real")
# plt.title("Example of fitting on test set")
# plt.legend()
#
# pred = regressors[0][11].predict(Xtrain, timevec.reshape((501, 1)))
# i = 5
# # Pred on train set
# plt.figure()
# plt.plot(timevec.flatten(), pred[i, :], label="predicted")
# plt.scatter(Vtrain["x"][i], Vtrain["y"][i], label="real")
# plt.title("Example of fitting on test set")
# plt.legend()
