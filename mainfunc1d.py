import numpy as np
import matplotlib.pyplot as plt
import importlib
import scipy.optimize as optimize
import itertools
import functools

import spatiotempovk.spatiotempdata as spatiotemp
import spatiotempovk.kernels as kernels
import syntheticdata.funcs1d as funcs1d
import spatiotempovk.losses as losses
import spatiotempovk.regularizers as regularizers
import spatiotempovk.regressors as regressors
import algebra.repeated_matrix as repmat
import solvers.gradientbased as gradientbased
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(losses)
importlib.reload(regularizers)
importlib.reload(regressors)
importlib.reload(repmat)
importlib.reload(gradientbased)

# Drawing function from normal(0, 1)
norm01 = functools.partial(np.random.normal, 0, 1)

# Draw random polynomials of fixed degree
polys = funcs1d.random_polys(norm01, deg=20, nsim=100)
polysprime = [poly.prime() for poly in polys]

# Draw random fourier based functions with fixed number of frequencies
# fouriers = funcs1d.random_fourier_func(norm01, nsim=100)
# fouriers_prime = [four.prime() for four in fouriers]

# Determine (fixed) locations
nlocs = 50
locs = np.linspace(0, 1, nlocs).reshape((nlocs, 1))

# Build the data
N = 50
# Draw random Fourier functions
fouriers = funcs1d.random_fourier_func(norm01, nfreq=3, nsim=N)
fouriers_prime = [four.prime() for four in fouriers]
datain = []
dataout = []
for n in range(N):
    Yin = np.array([fouriers[n](x[0]) for x in locs])
    Yout = np.array([fouriers_prime[n](x[0]) for x in locs])
    datain.append((locs, Yin))
    dataout.append((locs, Yout))
# Store them in a spatio temp data instance
datain = spatiotemp.LocObsSet(datain)
dataout = spatiotemp.LocObsSet(dataout)

# Kernels
kernelx = kernels.GaussianKernel(sigma=0.3)
Kxin = kernelx.compute_K(locs)
kernely = kernels.GaussianKernel(sigma=1)
Kyin = kernely.compute_K(datain["y_flat"])
kers = kernels.ConvKernel(kernelx, kernely, Kxin, Kyin, sameloc=True)
Ks = kers.compute_K_from_mat(datain.Ms)

# Build regressor
l2 = losses.L2Loss()
lamb = 0.1
mu = 0.1
smoothreg = regularizers.TikhonovSpace()
globalreg = regularizers.TikhonovTime()
regressor = regressors.DiffLocObsOnFuncReg(l2, smoothreg, globalreg, mu, lamb, kernelx, kers)

#
Kxout = repmat.RepSymMatrix(Kxin, (N, N))
gd = gradientbased.GradientDescent(0.00001, 50, 1e-5, record=True)
obj = regressor.objective_func(dataout.Ms, dataout["y_flat"], Kxout, Ks)
grad = regressor.objective_grad_func(dataout.Ms, dataout["y_flat"], Kxout, Ks)
alpha0 = np.random.normal(0, 1, (N, N*nlocs))
sol = gd(obj, grad, alpha0)

# Fit regressor
Kxout = repmat.RepSymMatrix(Kxin, (N, N))
regressor.fit(datain, dataout, Kx=Kxout, Ks=Ks)

pred = regressor.predict(datain.extract_subseq(0, 10), datain["x"][0])

