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


# Load the data
path = os.getcwd() + "/datalip/"
emg = pd.read_csv(path + "EMG.csv", header=None).values
lipacc = pd.read_csv(path + "LipAcc.csv", header=None).values

# Train/Test split
Ntrain = 25
Ntest = 32 - Ntrain

# Corrupt the training data
timevec = np.linspace(0, 0.69, 501)
xtrainlist, vtrainlist = corrupt_data(emg[:, :Ntrain],
                                      lipacc[:, :Ntrain],
                                      timevec,
                                      nmissingin=200,
                                      nmissingout=200,
                                      noisein=0.05,
                                      noiseout=0.1)

# Put dat in spatio-temporal format
Xtrain = spatiotemp.LocObsSet(xtrainlist)
Vtrain = spatiotemp.LocObsSet(vtrainlist)
xtest = [(timevec.reshape((501, 1)), emg[:, i].reshape((501, 1))) for i in range(Ntrain, 32)]
vtest = [(timevec.reshape((501, 1)), lipacc[:, i].reshape((501, 1))) for i in range(Ntrain, 32)]
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
kers = kernels.GaussianFuncKernel(sigma=3, rffeats=rffsx, mu=musmoothing)
Ks = kers.compute_K(Xtrain["xy_tuple"])
rffs = rffridge.RandomFourierFeatures(sigmarff, D, d=1)


mu_grid = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
lamb_grid = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
l2 = losses.L2Loss()

scores = np.zeros((len(lamb_grid), len(mu_grid)))
regressors = []

for i in range(len(lamb_grid)):
    regressors.append([])
    for j in range(len(mu_grid)):
        reg = dictout.FuncInDictOut(loss=l2, mu=mu_grid[j], lamb=lamb_grid[i], kers=kers, funcdic=rffs)
        solu = reg.fit(Xtrain, Vtrain, Ks=Ks, tol=1e-4)
        pred = reg.predict(Xtest, timevec.reshape((501, 1)))
        scores[i, j] = mse_score(pred, np.squeeze(np.array(Vtest["y"]), axis=2))
        regressors[i].append(reg)
        print("lamb = " + str(lamb_grid[i]) + " and mu = " + str(mu_grid[j]))

with open(os.getcwd() + "/tuning.pkl", "wb") as outp:
    pickle.dump((scores, regressors), outp)

# with open(os.getcwd() + "/tuning.pkl", "rb") as inp:
#     scores, regressors = pickle.load(inp)
