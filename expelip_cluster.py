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

# plt.figure()
# for i in range(32):
#     plt.plot(emg[:, i])

# Test for bandwidth parameter
# See if our input smoothing has the means to represent well the input functions
Dsmoothing = 300
sigmasmoothing = 45
musmoothing = 0.1
rffsx = rffridge.RandomFourierFeatures(sigmasmoothing, Dsmoothing, d=1)

# Kernels
sigmarff = 45
D = 300
kers = kernels.GaussianFuncKernel(sigma=3, funcdict=rffsx, mu=musmoothing)
Ks = kers.compute_K(Xtrain["xy_tuple"])
rffs = rffridge.RandomFourierFeatures(sigmarff, D, d=1)


mu_grid = np.linspace(0.001, 0.1, 30)
lamb_grid = np.linspace(0.0001, 0.1, 40)
l2 = losses.L2Loss()

scores = np.zeros((len(lamb_grid), len(mu_grid)))
regressors = []

for i in range(len(lamb_grid)):
    regressors.append([])
    for j in range(len(mu_grid)):
        reg = dictout.FuncInDictOut(loss=l2, mu=mu_grid[j], lamb=lamb_grid[i], kers=kers, funcdic=rffs)
        solu = reg.fit(Xtrain, Vtrain, Ks=Ks, tol=1e-4)
        pred = reg.predict(Xtest, timevec)
        scores[i, j] = mse_score(pred, np.squeeze(np.array(Vtest["y"]), axis=2))
        regressors[i].append(reg)
        print("lamb = " + str(lamb_grid[i]) + " and mu = " + str(mu_grid[j]))

with open(os.getcwd() + "/tuning_rff_funker.pkl", "wb") as outp:
    pickle.dump((scores, regressors), outp)
#
# with open(os.getcwd() + "/tuning.pkl", "rb") as inp:
#     scores, regressors = pickle.load(inp)
#
# i=2
# j=3
# reg = regressors[i][j]
# pred = reg.predict(Xtest, timevec.reshape((501, 1)))
# i = 3
#
# import matplotlib.pyplot as plt
# # Pred on test set
# plt.figure()
# plt.plot(timevec.flatten(), pred[i, :], label="predicted")
# plt.plot(timevec.flatten(), Vtest["y"][i], label="real")
# plt.title("Example of fitting on test set")
# plt.legend()
#
# pred = reg.predict(Xtrain, timevec.reshape((501, 1)))
# i = 1
# # Pred on train set
# plt.figure()
# plt.plot(timevec.flatten(), pred[i, :], label="predicted")
# plt.scatter(Vtrain["x"][i], Vtrain["y"][i], label="real")
# plt.title("Example of fitting on test set")
# plt.legend()
#
