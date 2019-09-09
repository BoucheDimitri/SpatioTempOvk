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

def mse_score(pred, true):
    return ((pred - true) ** 2).sum(axis=1).mean()


def ideal_smoothing_mse(mexsmoothing, musmoothing, Vtest, predcoeffs):
    N = Vtest.get_T()
    ideal_coeffs = []
    for i in range(N):
        testridge = expregs.ExpandedRidge(musmoothing, mexsmoothing)
        testridge.fit(Vtest["x"][i], Vtest["y"][i])
        ideal_coeffs.append(testridge.w)
    ideal_coeffs = np.array(ideal_coeffs)
    return ((ideal_coeffs - predcoeffs) ** 2).sum(axis=1).mean()

# Load the data
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

# Input noise params
noisein = 0
nmissingin = 0
nmissingout = 0
noiseout = 0.08

# Corrupt the training data
xtrainlist, vtrainlist = corrupt_data(emg[:, :Ntrain],
                                      lipacc[:, :Ntrain],
                                      timevec,
                                      nmissingin=nmissingin,
                                      nmissingout=nmissingout,
                                      noisein=noisein,
                                      noiseout=noiseout)

# Put dat in spatio-temporal format
Xtrain = spatiotemp.LocObsSet(xtrainlist)
Vtrain = spatiotemp.LocObsSet(vtrainlist)

# Testing data (fixed)
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
D = mexhatsout.D

# Loss
l2 = losses.L2Loss()

nmissinggrid = np.arange(0, 500, 25)

scores = []
scores_coeffs = []
regressors = []

musmoothingout = 0.1
mu = 0.0044
lamb = 0.0078

for i in range(len(nmissinggrid)):
    # Corrupt the training data
    xtrainlist, vtrainlist = corrupt_data(emg[:, :Ntrain],
                                          lipacc[:, :Ntrain],
                                          timevec,
                                          nmissingin=nmissingin,
                                          nmissingout=nmissinggrid[i],
                                          noisein=noisein,
                                          noiseout=noiseout)

    # Put dat in spatio-temporal format
    Xtrain = spatiotemp.LocObsSet(xtrainlist)
    Vtrain = spatiotemp.LocObsSet(vtrainlist)
    reg = dictout.FuncInDictOut(loss=l2, mu=mu, lamb=lamb, kers=kers, funcdic=mexhatsout)
    Ks = kers.compute_K(Xtrain["xy_tuple"])
    solu = reg.fit(Xtrain, Vtrain, Ks=Ks, tol=1e-4)
    predcoeffs = reg.predict_coeffs(Xtest)
    pred = reg.predict(Xtest, timevec)
    scores.append(mse_score(pred, np.squeeze(np.array(Vtest["y"]), axis=2)))
    scores_coeffs.append((1 / D) * ideal_smoothing_mse(mexhatsout, musmoothingout, Vtest, predcoeffs))
    regressors.append(reg)
    print(i)

with open(os.getcwd() + "/samples_exp_mex.pkl", "wb") as outp:
    pickle.dump((nmissinggrid, scores, scores_coeffs, regressors), outp)