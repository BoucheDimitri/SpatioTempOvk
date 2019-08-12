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
# kers = kernels.GaussianFuncKernel(sigma=2, funcdict=mexhats, mu=musmoothing)
kers = kernels.GaussianFuncIntKernel(sigma=0.5, funcdict=mexhats, mu=musmoothing, timevec=timevec)
Ks = kers.compute_K(Xtrain["xy_tuple"])

# output dict
mexhatsout = funcdicts.MexHatDict((timevec[0], timevec[-1]), np.linspace(timevec[0], timevec[-1], 10), np.linspace(0.02, 0.1, 10))

# mu_grid = np.linspace(0.001, 0.1, 20)
# lamb_grid = np.linspace(0.001, 0.1, 20)
mu_grid = np.linspace(0.001, 0.1, 30)
lamb_grid = np.linspace(0.0001, 0.1, 40)
l2 = losses.L2Loss()

scores = np.zeros((len(lamb_grid), len(mu_grid)))
regressors = []

for i in range(len(lamb_grid)):
    regressors.append([])
    for j in range(len(mu_grid)):
        reg = dictout.FuncInDictOut(loss=l2, mu=mu_grid[j], lamb=lamb_grid[i], kers=kers, funcdic=mexhatsout)
        solu = reg.fit(Xtrain, Vtrain, Ks=Ks, tol=1e-4)
        pred = reg.predict(Xtest, timevec)
        scores[i, j] = mse_score(pred, np.squeeze(np.array(Vtest["y"]), axis=2))
        regressors[i].append(reg)
        print("lamb = " + str(lamb_grid[i]) + " and mu = " + str(mu_grid[j]))

with open(os.getcwd() + "/tuning_mex_intker_lag.pkl", "wb") as outp:
    pickle.dump((mu_grid, lamb_grid, scores, regressors), outp)

with open(os.getcwd() + "/tuning/tuning_mex_intker_lag.pkl", "rb") as inp:
    mus, lambs, scores_mex_intker, regressors_mex_intker = pickle.load(inp)

with open(os.getcwd() + "/tuning/tuning_mex_funker_lag.pkl", "rb") as inp:
    mus, lambs, scores_mex_funker, regressors_mex_funker = pickle.load(inp)

with open(os.getcwd() + "/tuning/tuning_rff_funker.pkl", "rb") as inp:
    scores_rff_funker, regressors_rff_funker = pickle.load(inp)

with open(os.getcwd() + "/tuning/tuning_rff_intker.pkl", "rb") as inp:
    scores_rff_intker, regressors_rff_intker = pickle.load(inp)

with open(os.getcwd() + "/tuning/tuning_rff_conc.pkl", "rb") as inp:
    scores_rff_conc, regressors_rff_conc = pickle.load(inp)

np.unravel_index(scores_rff_funker.argmin(), scores.shape)

import matplotlib.pyplot as plt
pred = regressors[3][1].predict(Xtest, timevec)
i = 3

# Pred on test set
plt.figure()
plt.plot(timevec.flatten(), pred[i, :], label="predicted")
plt.plot(timevec.flatten(), Vtest["y"][i], label="real")
plt.title("Example of fitting on test set")
plt.legend()

pred = regressors[3][1].predict(Xtrain, timevec)
i = 2
# Pred on train set
plt.figure()
plt.plot(timevec.flatten(), pred[i, :], label="predicted")
plt.scatter(Vtrain["x"][i], Vtrain["y"][i], label="real")
plt.title("Example of fitting on test set")
plt.legend()
