import numpy as np
import matplotlib.pyplot as plt
import importlib
import scipy.optimize as optimize

import spatiotempovk.spatiotempdata as spatiotemp
import spatiotempovk.kernels as kernels
import spatiotempovk.learning as learning
import syntheticdata.argp2d as argp2d
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(learning)
importlib.reload(argp2d)

argp = argp2d.draw_ar1_gp2d()
obs = argp2d.draw_observations(20, argp)

data = spatiotemp.SpatioTempData(obs)

gausskerx = kernels.GaussianKernel(sigma=10)
gausskery = kernels.GaussianKernel(sigma=0.2)

Kx = gausskerx.compute_K(data["x"])
Ky = gausskery.compute_K(data["y"])
convkers = kernels.ConvKernel(gausskerx, gausskery)

Ks = convkers.compute_K(data, Kx, Ky)

TbarM = Kx.shape[0]
T = Ks.shape[0]
# predfunc = learning.PredictFunc(np.random.normal(0, 1, (T, TbarM)), mu=1, lamb=1)


def loss_l2(y, x):
    return (y - x) ** 2


def loss_l2_prime(y, x):
    return - 2 * (y - x)

alpha = np.random.normal(0, 1, (Kx.shape[0] * Ks.shape[0]))

Ms = data.Ms
y = data["y"]
mu = 1
lamb = 1

obj = learning.get_objective_func(loss_l2, Ms, y, Kx, Ks, mu, lamb)
grad = learning.get_gradient_func(loss_l2_prime, Ms, y, Kx, Ks, mu, lamb)


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
alpha, gn, ob = gradient_descent(alpha, obj, grad, nu, maxit, eps)

sol = optimize.minimize(fun=obj, x0=alpha, jac=grad, method='L-BFGS-B')

