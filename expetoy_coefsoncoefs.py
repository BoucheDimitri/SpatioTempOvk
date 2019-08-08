import numpy as np
import matplotlib.pyplot as plt
import importlib
import pandas as pd
import syntheticdata.funcs1d as funcs1d
import pickle
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



# Or load dataset
with open(os.getcwd() + "/dumps/datasets.pkl", "rb") as i:
    Xtrain, Vtrain, Xtest, Vtest = pickle.load(i)



# plt.figure()
# for i in range(32):
#     plt.plot(emg[:, i])

# Test for bandwidth parameter
# See if our input smoothing has the means to represent well the input functions
Dsmoothing = 10
sigmasmoothing = 10
musmoothing = 0.1
rffsx = rffridge.RandomFourierFeatures(sigmasmoothing, Dsmoothing, d=1)
testridge = rffridge.RFFRidge(musmoothing, rffsx)
i = 0
testridge.fit(Xtrain["x"][i], Xtrain["y"][i])
pred = testridge.predict(Xtrain["x"][i])
plt.figure()
plt.plot(Xtrain["x"][i], pred, label="predicted")
plt.plot(Xtrain["x"][i], Xtrain["y"][i], label="real")
plt.legend()


# Kernels
sigmarff = 10
D = 10
rffs = rffridge.RandomFourierFeatures(sigmarff, D, d=1)
kers = kernels.GaussianFuncKernel(sigma=5, funcdict=rffs, mu=musmoothing)
# kers = kernels.GaussianSameLoc(sigma=10)
Ks = kers.compute_K(Xtrain["xy_tuple"])
plt.imshow(Ks)
test = rffs.eval(Vtrain["x"][0])
plt.imshow(test.dot(test.T))

# Test for bandwidth parameter
# To see if our output dictionary has the means to approximate well the output functions
muout = 0.1
testridge = rffridge.RFFRidge(muout, rffs)
i = 3
testridge.fit(Vtrain["x"][i], Vtrain["y"][i])
pred = testridge.predict(Vtest["x"][i])
plt.figure()
plt.plot(Vtest["x"][i], pred, label="predicted")
plt.plot(Vtrain["x"][i], Vtrain["y"][i], label="real")
plt.legend()

cc = coefsoncoefs.CoefsOnCoefs(kernels.GaussianKernel(sigma=5), rffsx, musmoothing, rffs, muout, 0)
cc.fit(Xtrain, Vtrain)
pred = cc.predict(Xtest, Vtest["x"][i])
plt.figure()
plt.plot(Vtest["x"][i], pred[0])
plt.plot(Vtest["x"][i], Vtest["y"][0])

# Fit
# Build regressor
l2 = losses.L2Loss()
lamb = 0.001
mu = 0.1
mexhatsout = funcdicts.MexHatDict((timevec[0], timevec[-1]), np.linspace(timevec[0], timevec[-1], 10), np.linspace(0.02, 0.1, 10))
reg = dictout.FuncInDictOut(loss=l2, mu=mu, lamb=lamb, kers=kers, funcdic=mexhatsout)

# Fit regressor
solu = reg.fit(Xtrain, Vtrain, Ks=Ks, tol=1e-4)

# Predict

pred = reg.predict(Xtest, timevec.reshape((501, 1)))
i = 5

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
