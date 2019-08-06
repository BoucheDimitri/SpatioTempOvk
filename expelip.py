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

# ################### LOAD THE DATA ####################################################################################
path = os.getcwd() + "/datalip/"
emg = pd.read_csv(path + "EMG.csv", header=None).values
lipacc = pd.read_csv(path + "LipAcc.csv", header=None).values

# Put the data in spatiotemp data format
timevec = np.linspace(0, 0.69, 501).reshape((501, 1))
emglist = [(timevec, emg[:, i].reshape((501, 1))) for i in range(32)]
lipacclist = [(timevec, lipacc[:, i].reshape((501, 1))) for i in range(32)]
X = spatiotemp.LocObsSet(emglist)
V = spatiotemp.LocObsSet(lipacclist)
Xtrain = X.extract_subseq(0, 22)
Vtrain = V.extract_subseq(0, 22)
Xtest = X.extract_subseq(22, 32)
Vtest = V.extract_subseq(22, 32)


# plt.figure()
# for i in range(32):
#     plt.plot(emg[:, i])


# Kernels
sigmarff = 60
D = 300
kers = kernels.GaussianSameLoc(sigma=10)
Ks = kers.compute_K(Xtrain["xy_tuple"])
plt.imshow(Ks)
rffs = rffridge.RandomFourierFeatures(sigmarff, D, d=1)

test = rffs.eval(Vtrain["x"][0])
plt.imshow(test.dot(test.T))



# Fit
# Build regressor
l2 = losses.L2Loss()
lamb = 0
mu = 0.01
reg = dictout.FuncInDictOut(loss=l2, mu=mu, lamb=lamb, kers=kers, funcdic=rffs)

# Fit regressor
solu = reg.fit(Xtrain, Vtrain, Ks=Ks, tol=1e-4)

# Predict
pred = reg.predict(Xtrain, timevec)
i = 3


plt.figure()
plt.plot(timevec.flatten(), pred[i, :], label="predicted")
plt.plot(timevec.flatten(), Vtrain["y"][i], label="real")
plt.title("Example of fitting on test set")
plt.legend()
