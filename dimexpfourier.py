import numpy as np
import importlib
import pandas as pd
import os
import pickle

import spatiotempovk.spatiotempdata as spatiotemp
import spatiotempovk.kernels as kernels
import spatiotempovk.losses as losses
import spatiotempovk.dictout as dictout
import approxkernelridge.rffridge as rffridge
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(losses)
importlib.reload(dictout)
importlib.reload(rffridge)


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

def ideal_smoothing_mse(rffsmoothing, musmoothing, Vtest, predcoeffs):
    N = Vtest.get_T()
    ideal_coeffs = []
    for i in range(N):
        testridge = rffridge.RFFRidge(musmoothing, rffsmoothing)
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


# Input smoothing parameters
Dsmoothing = 300
sigmasmoothing = 45
musmoothing = 0.05
rffsx = rffridge.RandomFourierFeatures(sigmasmoothing, Dsmoothing, d=1)

# Kernels
sigmarff = 45
kers = kernels.GaussianFuncIntKernel(sigma=0.5, funcdict=rffsx, mu=musmoothing, timevec=timevec)
Ks = kers.compute_K(Xtrain["xy_tuple"])

# Dgrid = np.arange(1, 410, 10)
Dgrid = [5, 150, 300]

# Loss
l2 = losses.L2Loss()

# Regularization for the method
mu = 0.04537931034482759
lamb = 0.0001

# Regularization used for ideal output smoothing
musmoothingout = 0.05

# D = 100
# rffs = rffridge.RandomFourierFeatures(sigmarff, D, d=1)
# reg = dictout.FuncInDictOut(loss=l2, mu=mu, lamb=lamb, kers=kers, funcdic=rffs)
# solu = reg.fit(Xtrain, Vtrain, Ks=Ks, tol=1e-4)

regressors = []
scores_coeffs = []
scores = []

for i in range(len(Dgrid)):
    # if i == 0:
    #     rffs = rffridge.RandomFourierFeatures(sigmarff, Dgrid[i], d=1)
    # else:
    #     rffs.add_features(Dgrid[i] - Dgrid[i-1])
    rffs = rffridge.RandomFourierFeatures(sigmarff, Dgrid[i], d=1)
    reg = dictout.FuncInDictOut(loss=l2, mu=mu, lamb=lamb, kers=kers, funcdic=rffs)
    solu = reg.fit(Xtrain, Vtrain, Ks=Ks, tol=1e-4)
    predcoeffs = reg.predict_coeffs(Xtest)
    pred = reg.predict(Xtest, timevec)
    scores.append(mse_score(pred, np.squeeze(np.array(Vtest["y"]), axis=2)))
    scores_coeffs.append((1 / Dgrid[i]) * ideal_smoothing_mse(rffs, musmoothingout, Vtest, predcoeffs))
    regressors.append(reg)
    print(Dgrid[i])