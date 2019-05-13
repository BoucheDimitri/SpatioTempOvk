import numpy as np
import matplotlib.pyplot as plt
import importlib
import scipy.optimize as optimize

import spatiotempovk.spatiotempdata as spatiotemp
import spatiotempovk.kernels as kernels
import syntheticdata.argp2d as argp2d
import spatiotempovk.losses as losses
import spatiotempovk.regularizers as regularizers
import spatiotempovk.regressors as regressors
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(losses)
importlib.reload(regularizers)
importlib.reload(regressors)
importlib.reload(argp2d)

# Create synthetic data
argp = argp2d.draw_ar1_gp2d()
obs = argp2d.draw_observations(20, argp)

# Store this data in a SpatioTempData class instance
data = spatiotemp.SpatioTempData(obs)
Ms = data.get_Ms()
T = data.get_T()
barM = data.get_barM()

# Kernels for convolution
gausskerx = kernels.GaussianKernel(sigma=10)
gausskery = kernels.GaussianKernel(sigma=0.2)

# Compute kernel matrices
Kx = gausskerx.compute_K(data["x"])
Ky = gausskery.compute_K(data["y"])
convkers = kernels.ConvKernel(gausskerx, gausskery, Kx, Ky)

# Compute convolution kernel matrix
Ks = convkers.compute_K_from_mat(Ms)

# Define loss
loss = losses.L2Loss()

# Define regularizers and regularization params
spacereg = regularizers.TikhonovSpace()
timereg = regularizers.TikhonovTime()
mu = 1
lamb = 1

# Initialize alpha
alpha = np.random.normal(0, 1, (T, barM))

# Define regressor for problem
regressor = regressors.DiffSpatioTempRegressor(loss, spacereg, timereg, mu, lamb, gausskerx, convkers)

# Test for the regressor's functions
regressor.data_fitting(alpha, Ms, data["y"], Kx, Ks)
regressor.spacereg(alpha, Kx, Ks)
regressor.timereg(alpha, Kx, Ks)
regressor.data_fitting_prime(alpha, Ms, data["y"], Kx, Ks)
regressor.objective(alpha, Ms, data["y"], Kx, Ks)
regressor.objective_prime(alpha, Ms, data["y"], Kx, Ks)



# obj = learning.get_objective_func(loss_l2, Ms, y, Kx, Ks, mu, lamb)
# grad = learning.get_gradient_func(loss_l2_prime, Ms, y, Kx, Ks, mu, lamb)
#
#
# def gradient_descent(alpha0, obj, grad, nu, maxit, eps):
#     it = 0
#     gradnorms = []
#     objs = []
#     evalgrad = grad(alpha0)
#     evalobj = obj(alpha0)
#     gradnorm = np.linalg.norm(evalgrad)
#     gradnorms.append(gradnorm)
#     objs.append(evalobj)
#     alpha = alpha0.copy()
#     while (it < maxit) and (gradnorm > eps):
#         alpha -= nu * evalgrad
#         evalgrad = grad(alpha)
#         evalobj = obj(alpha)
#         gradnorm = np.linalg.norm(evalgrad)
#         gradnorms.append(gradnorm)
#         objs.append(evalobj)
#         it += 1
#         print(it)
#     return alpha, gradnorms, objs
#
# nu = 0.1
# maxit = 1000
# eps = 1e-3
# alpha, gn, ob = gradient_descent(alpha, obj, grad, nu, maxit, eps)
#
# sol = optimize.minimize(fun=obj, x0=alpha, jac=grad, method='L-BFGS-B')

