import numpy as np
import importlib
import os
import functools
import pickle

import spatiotempovk.spatiotempdata as spatiotemp
import spatiotempovk.kernels as kernels
import spatiotempovk.losses as losses
import spatiotempovk.regularizers as regularizers
import spatiotempovk.regressors as regressors
import algebra.repeated_matrix as repmat
import smoothing.representer as repsmooth
import smoothing.parametrized_func as param_func
import syntheticdata.funcs1d as funcs1d
importlib.reload(repmat)
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(losses)
importlib.reload(regularizers)
importlib.reload(regressors)
importlib.reload(repsmooth)
importlib.reload(param_func)
importlib.reload(funcs1d)

# Drawing function from normal(0, 1)
norm01 = functools.partial(np.random.normal, 0, 1)

# Determine (fixed) locations
nlocs = 50
locs = np.linspace(0, 1, nlocs).reshape((nlocs, 1))

# Build the data
Ntrain = 200
Ntest = 20
# Draw random Fourier functions
fouriers = funcs1d.random_fourier_func(norm01, nfreq=2, nsim=Ntrain + Ntest)
fouriers_prime = [four.prime() for four in fouriers]
datain = []
dataout = []
for n in range(Ntrain + Ntest):
    Yin = np.array([fouriers[n](x[0]) for x in locs])
    Yout = np.array([fouriers_prime[n](x[0]) for x in locs])
    datain.append((locs, Yin))
    dataout.append((locs, Yout))
# Store them in a spatio temp data instance
datain = spatiotemp.LocObsSet(datain)
dataout = spatiotemp.LocObsSet(dataout)
dataintest = datain.extract_subseq(Ntrain, Ntrain + Ntest)
dataouttest = dataout.extract_subseq(Ntrain, Ntrain + Ntest)
datain = datain.extract_subseq(0, Ntrain)
dataout = dataout.extract_subseq(0, Ntrain)


# Kernels
kernelx = kernels.GaussianKernel(sigma=0.3)
Kxin = kernelx.compute_K(locs)
kernely = kernels.GaussianKernel(sigma=0.5)
Kyin = kernely.compute_K(datain["y_flat"])
kers = kernels.ConvKernel(kernelx, kernely, Kxin, Kyin, sameloc=True)
Ks = kers.compute_K_from_mat(datain.Ms)

# Build regressor
l2 = losses.L2Loss()
lamb = 0
mu = 0
smoothreg = regularizers.TikhonovSpace()
globalreg = regularizers.TikhonovTime()
regressor = regressors.DiffLocObsOnFuncReg(l2, smoothreg, globalreg, mu, lamb, kernelx, kers)

# Fit regressor
Kxout = repmat.RepSymMatrix(Kxin, (Ntrain, Ntrain))
solu = regressor.fit(datain, dataout, Kx=Kxout, Ks=Ks)

# Predictions
predtrain = regressor.predict(datain.extract_subseq(0, 10), datain["x"][0])
predtest = regressor.predict(dataintest, datain["x"][0])

# Dump results
with open(os.getcwd() + "/output.pkl", "wb") as outfile:
    pickle.dump((predtrain, predtest, solu, Ks, Kxout), outfile)
