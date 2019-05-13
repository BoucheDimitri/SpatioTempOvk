import numpy as np
import matplotlib.pyplot as plt
import importlib
import scipy.optimize as optimize
import itertools

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
obs = argp2d.draw_observations(200, argp)
nx, ny = argp[0].shape

# Store this data in a SpatioTempData class instance
data = spatiotemp.SpatioTempData(obs)
Ms = data.get_Ms()
T = data.get_T()
barM = data.get_barM()

# Kernels for convolution
gausskerx = kernels.GaussianKernel(sigma=10)
gausskery = kernels.GaussianKernel(sigma=0.2)

# Compute kernel matrices
Kx = gausskerx.compute_K(data["x_flat"])
Ky = gausskery.compute_K(data["y_flat"])
convkers = kernels.ConvKernel(gausskerx, gausskery, Kx, Ky)

# Compute convolution kernel matrix
Ks = convkers.compute_K_from_mat(Ms)

# Define loss
loss = losses.L2Loss()

# Define regularizers and regularization params
spacereg = regularizers.TikhonovSpace()
timereg = regularizers.TikhonovTime()
mu = 10000
lamb = 0.000001

# Train/Test split
Strain = data.extract_subseq(0, 4)
Stest = data.extract_subseq(3, 4)

# Initialize and train regressor
reg = regressors.DiffSpatioTempRegressor(loss, spacereg, timereg, mu, lamb, gausskerx, convkers)
reg.fit(Strain)

# Predict at new locations
Xnew = np.array(list(itertools.product(range(nx), range(ny))))
Ypred = reg.predict(Stest, Xnew)
Ypred = Ypred.reshape((50, 50))




def gradient_descent(alpha0, obj, grad, nu, maxit, eps):
    it = 0
    gradnorms = []
    objs = []
    evalgrad = grad(alpha0)
    evalobj = obj(alpha0)
    gradnorm = np.linalg.norm(evalgrad)
    gradnorms.append(gradnorm)
    objs.append(evalobj)
    alpha = alpha0.copy()
    while (it < maxit) and (gradnorm > eps):
        alpha -= nu * evalgrad
        evalgrad = grad(alpha)
        evalobj = obj(alpha)
        gradnorm = np.linalg.norm(evalgrad)
        gradnorms.append(gradnorm)
        objs.append(evalobj)
        it += 1
        print(it)
    return alpha, gradnorms, objs

nu = 0.1
maxit = 1000
eps = 1e-3
alpha, gn, ob = gradient_descent(alpha0, obj, grad, nu, maxit, eps)
#

# Initialize alpha
alpha0 = np.zeros((T, barM))
sol = optimize.minimize(fun=obj, x0=alpha0.flatten(), jac=grad, method='L-BFGS-B', tol=1e-5)


#
Strain = data.extract_subseq(0, 4)
Stest = data.extract_subseq(4, 5)