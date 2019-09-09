import numpy as np
import importlib
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

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

# Plot parameters
plt.rcParams.update({"font.size": 20})
plt.rcParams.update({"lines.linewidth": 3})
plt.rcParams.update({"lines.markersize": 10})



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


# Plot the data
fig, ax = plt.subplots(ncols=2)
for i in range(32):
    ax[1].plot(timevec.flatten(), lipacc[:, i])
    ax[0].plot(timevec.flatten(), emg[:, i])
ax[0].set_xlabel("seconds")
ax[1].set_xlabel("seconds")
ax[0].set_ylabel("Millivolts")
ax[1].set_ylabel("Meters/s$^2$")
ax[0].set_title("EMG Curves")
ax[1].set_title("Lip acceleration curves")

# Plot the data
fig, ax = plt.subplots(ncols=2)
for i in np.arange(1, 3):
    ax[0].scatter(Xtrain["x"][i], Xtrain["y"][i])
    ax[1].scatter(Vtrain["x"][i], Vtrain["y"][i])
ax[0].set_xlabel("seconds")
ax[1].set_xlabel("seconds")
ax[0].set_ylabel("Millivolts")
ax[1].set_ylabel("Meters/s$^2$")
ax[0].set_title("Noisy downsampled EMG")
ax[1].set_title("Noisy downsampled Lip acceleration")






with open(os.getcwd() + "/tuning/tuning_mex_intker_bis.pkl", "rb") as inp:
    mus, lambs, scores_mex_intker, regressors_mex_intker = pickle.load(inp)
print(np.unravel_index(scores_mex_intker.argmin(), scores_mex_intker.shape))

with open(os.getcwd() + "/tuning/tuning_mex_funker_bis.pkl", "rb") as inp:
    mus, lambs, scores_mex_funker, regressors_mex_funker = pickle.load(inp)
print(np.unravel_index(scores_mex_funker.argmin(), scores_mex_funker.shape))

with open(os.getcwd() + "/tuning/tuning_rff_funker_bis.pkl", "rb") as inp:
    scores_rff_funker, regressors_rff_funker = pickle.load(inp)
print(np.unravel_index(scores_rff_funker.argmin(), scores_rff_funker.shape))

with open(os.getcwd() + "/tuning/tuning_rff_intker_bis.pkl", "rb") as inp:
    scores_rff_intker, regressors_rff_intker = pickle.load(inp)
print(np.unravel_index(scores_rff_intker.argmin(), scores_rff_intker.shape))

with open(os.getcwd() + "/tuning/tuning_rff_conc_bis.pkl", "rb") as inp:
    scores_rff_conc, regressors_rff_conc = pickle.load(inp)
np.argmin(scores_rff_conc)

with open(os.getcwd() + "/tuning/tuning_mex_conc_bis.pkl", "rb") as inp:
    scores_mex_conc, regressors_mex_conc = pickle.load(inp)


mexintker = regressors_mex_intker[1][0]
mexfunker = regressors_mex_funker[33][10]
rffintker = regressors_rff_intker[0][10]
rfffunker = regressors_rff_funker[0][10]
rffconc = regressors_rff_conc[2]
mexconc = regressors_mex_conc[8]
regdict = {"FOD MEX Int": mexintker, "FOD MEX Coefs": mexfunker, "FOD RFF Int": rffintker,
           "FOD RFF Coefs": rfffunker, "COD RFF": rffconc, "COD MEX": mexconc}

preddict = {}
for key, item in regdict.items():
    preddict[key] = item.predict(Xtest, timevec)

Nexamples = 4
Nregressors = len(regdict)
fig, ax = plt.subplots(nrows=Nexamples, ncols=Nregressors, sharey="row", sharex="col")
start = 2
for i in range(start, start + Nexamples):
    j = 0
    for key, item in preddict.items():
        ax[i-start, j].plot(timevec.flatten(), item[i, :], label="predicted")
        ax[i-start, j].plot(timevec.flatten(), Vtest["y"][i], label="real")
        # if j == 0 and i == start:
        #     ax[i-start, j].legend()
        if i == start:
            ax[i-start, j].set_title(key)
        j += 1




# np.unravel_index(scores_rff_funker.argmin(), scores.shape)

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
