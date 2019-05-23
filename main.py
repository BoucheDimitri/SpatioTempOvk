import numpy as np
import matplotlib.pyplot as plt
import importlib
import scipy.optimize as optimize
import itertools
import functools

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
argp = argp2d.draw_ar1_gp2d(T=50)
obs = argp2d.draw_observations(20, argp)
# obs = argp2d.draw_observations_sameloc(20, argp)
nx, ny = argp[0].shape

# Store this data in a SpatioTempData class instance
data = spatiotemp.LocObsSet(obs)
Ms = data.get_Ms()
T = data.get_T()
barM = data.get_barM()

# Look at time series at a given location
test = np.array([data["y"][i][20] for i in range(data.get_T())])

# # Kernels for convolution
# gausskerx = kernels.GaussianKernel(sigma=10)
# gausskery = kernels.GaussianKernel(sigma=0.2)

# # Compute kernel matrices
# Kx = gausskerx.compute_K(data["x_flat"])
# Ky = gausskery.compute_K(data["y_flat"])
# convkers = kernels.ConvKernel(gausskerx, gausskery, Kx, Ky)
#
# # Compute convolution kernel matrix
# Ks = convkers.compute_K_from_mat(Ms)

# Define loss
loss = losses.L2Loss()

# Define regularizers and regularization params
spacereg = regularizers.TikhonovSpace()
timereg = regularizers.TikhonovTime()
mu = 0.001
lamb = 0.001

# Train/Test split
ntrain = 10
Strain_input = data.extract_subseq(0, 10)
Strain_output = data.extract_subseq(1, 11)
Strain = data.extract_subseq(0, 11)

# Kernels for convolution
gausskerx = kernels.GaussianKernel(sigma=10)
gausskery = kernels.GaussianKernel(sigma=0.2)

# NEW REGRESSION CLASS ########################################################"""

# Compute kernel matrices
Kxin = gausskerx.compute_K(Strain_input["x_flat"])
Ky = gausskery.compute_K(Strain_input["y_flat"])
convkers = kernels.ConvKernel(gausskerx, gausskery, Kxin, Ky)

# Compute convolution kernel matrix
Ks = convkers.compute_K(Strain_input["xy_tuple"])

Kx = gausskerx.compute_K(Strain_output["x_flat"])

cardV = sum(Strain_input.Ms)

alpha = np.random.normal(0, 1, (ntrain, cardV))

test_reg = regressors.DiffLocObsOnFuncReg(loss, spacereg, timereg, mu, lamb, gausskerx, convkers)

datafitting = functools.partial(test_reg.data_fitting, Strain_output.Ms, Strain_output["y_flat"], Kx, Ks)
datafitting_prime = functools.partial(test_reg.data_fitting_prime, Strain_output.Ms, Strain_output["y_flat"], Kx, Ks)

gamma=10
gradnorms=[np.linalg.norm(datafitting_prime(alpha))]
objs=[datafitting(alpha)]
for i in range(1000):
    grad = datafitting_prime(alpha)
    alpha -= 0.01*grad
    objs.append(datafitting(alpha))
    gradnorms.append(np.linalg.norm(grad))
    print(i)

regspace = functools.partial(test_reg.smoothreg, Kx, Ks)
regspace_prime = functools.partial(test_reg.smoothreg.prime, Kx, Ks)


gradnorms=[np.linalg.norm(regspace_prime(alpha))]
objs=[regspace(alpha)]
for i in range(1000):
    grad = regspace_prime(alpha)
    alpha -= 0.005*grad
    objs.append(regspace(alpha))
    gradnorms.append(np.linalg.norm(grad))
    print(i)

gradnorms=[np.linalg.norm(spacereg.prime(Kx, Ks, alpha))]
objs=[regspace(alpha)]
for i in range(1000):
    grad = spacereg.prime(Kx, Ks, alpha)
    alpha -= 0.01*grad
    objs.append(spacereg(Kx, Ks, alpha))
    gradnorms.append(np.linalg.norm(grad))
    print(i)



regglob = functools.partial(test_reg.globalreg, Kx, Ks)
regglob_prime = functools.partial(test_reg.globalreg.prime, Kx, Ks)

gradnorms=[np.linalg.norm(regglob_prime(alpha))]
objs=[regglob(alpha)]
for i in range(1000):
    grad = regglob_prime(alpha)
    alpha -= 0.0005*grad
    objs.append(regglob(alpha))
    gradnorms.append(np.linalg.norm(grad))
    print(i)




grad_func = test_reg.objective_grad_func(Strain_output.Ms, Strain_output["y_flat"], Kx, Ks)
obj_func = test_reg.objective_func(Strain_output.Ms, Strain_output["y_flat"], Kx, Ks)

gamma=0.01
gradnorms=[np.linalg.norm(grad_func(alpha))]
objs=[obj_func(alpha)]
for i in range(50):
    grad = grad_func(alpha)
    alpha -= 0.01*grad
    objs.append(obj_func(alpha))
    gradnorms.append(np.linalg.norm(grad))
    print(i)


# OLD REGRESSION CLASS ##############################################################
# Compute kernel matrices
Kx = gausskerx.compute_K(Strain["x_flat"])
Ky = gausskery.compute_K(Strain["y_flat"])
convkers = kernels.ConvKernel(gausskerx, gausskery, Kx, Ky)

# Compute convolution kernel matrix
Ks = convkers.compute_K(Strain["xy_tuple"])

cardV = sum(Strain.Ms)

alpha = np.random.normal(0, 1, (ntrain+1, cardV))
test_regbis = regressors.DiffSpatioTempRegressor(loss, spacereg, timereg, mu, lamb, gausskerx, convkers)

grad_func = test_regbis.objective_grad_func(Strain.Ms, Strain["y_flat"], Kx, Ks)
obj_func = test_regbis.objective_func(Strain.Ms, Strain["y_flat"], Kx, Ks)

gamma=0.01
gradnorms=[np.linalg.norm(grad_func(alpha))]
objs=[obj_func(alpha)]
for i in range(50):
    grad = grad_func(alpha)
    alpha -= 0.01*grad
    objs.append(obj_func(alpha))
    gradnorms.append(np.linalg.norm(grad))
    print(i)


gradnorms=[np.linalg.norm(spacereg.prime(Kx, Ks, alpha))]
objs=[spacereg(Kx, Ks, alpha)]
for i in range(500):
    grad = spacereg.prime(Kx, Ks, alpha)
    alpha -= 0.005*grad
    objs.append(spacereg(Kx, Ks, alpha))
    gradnorms.append(np.linalg.norm(grad))
    print(i)



# Initialize and train regressor
reg = regressors.DiffSpatioTempRegressor(loss, spacereg, timereg, mu, lamb, gausskerx, convkers)
reg.fit(Strain)

# Predict at new locations
Xnew = np.array(list(itertools.product(range(nx), range(ny))))
Ypred = reg.predict(Stest, Xnew)
Ypred = Ypred.reshape((50, 50))



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
# alpha, gn, ob = gradient_descent(alpha0, obj, grad, nu, maxit, eps)
# #
#
# # Initialize alpha
# alpha0 = np.zeros((T, barM))
# sol = optimize.minimize(fun=obj, x0=alpha0.flatten(), jac=grad, method='L-BFGS-B', tol=1e-5)
#
#
# #
# Strain = data.extract_subseq(0, 4)
# Stest = data.extract_subseq(4, 5)