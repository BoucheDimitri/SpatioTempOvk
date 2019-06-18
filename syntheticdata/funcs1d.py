import numpy as np
import importlib
import matplotlib.pyplot as plt

import spatiotempovk.spatiotempdata as spatiotemp
import spatiotempovk.kernels as kernels
import spatiotempovk.losses as losses
import spatiotempovk.regularizers as regularizers
import spatiotempovk.regressors as regressors
import algebra.repeated_matrix as repmat
from smoothing import parametrized_func as para_func
importlib.reload(repmat)
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(losses)
importlib.reload(regularizers)
importlib.reload(regressors)
importlib.reload(para_func)


def random_polys(draw_coeff_func, draw_deg_func=None, deg=None, nsim=1):
    if draw_deg_func is None and deg is None:
        raise ValueError("Provide either draw_deg_func of deg")
    elif draw_deg_func is None and deg is not None:
        alphas = draw_coeff_func((nsim, deg))
        samples = [para_func.PolynomialBased(alphas[i]) for i in range(nsim)]
    else:
        samples = []
        degs = draw_deg_func(nsim)
        for i in range(nsim):
            alpha = draw_coeff_func(degs[i])
            samples.append(para_func.PolynomialBased(alpha))
    return samples


def plot_params_funcs(funcs, nplot, xlims, nlin=100):
    sp = np.linspace(xlims[0], xlims[1], nlin)
    fig, ax = plt.subplots()
    count = 0
    while count < nplot:
        ax.plot(sp, [funcs[count](x) for x in sp])
        count += 1


def random_fourier_func(draw_coeff_func, nfreq=3, interval = (0, 1), nsim=1):
    alphas = draw_coeff_func((nsim, 2 * nfreq))
    samples = [para_func.FourierBased(alphas[i], nfreq, interval) for i in range(nsim)]
    return samples


def fourier_dataset_varloc(N, noisein, noiseout, drawfunc, nfreq, M, locdrawfunc):
    # Draw random Fourier functions
    fouriers = random_fourier_func(drawfunc, nfreq=nfreq, nsim=N)
    fouriers_prime = [four.prime() for four in fouriers]
    datain = []
    dataout = []
    for n in range(N):
        locs = np.sort(locdrawfunc((M, 1)), axis=0)
        Yin = np.array([fouriers[n](x[0]) + np.random.normal(0, noisein) for x in locs])
        Yout = np.array([fouriers_prime[n](x[0]) + np.random.normal(0, noiseout) for x in locs])
        datain.append((locs, Yin))
        dataout.append((locs, Yout))
    return datain, dataout


def generate_fourier_dataset(N, noisein, noiseout, drawfunc, nfreq, locs):
    # Draw random Fourier functions
    fouriers = random_fourier_func(drawfunc, nfreq=nfreq, nsim=N)
    fouriers_prime = [four.prime() for four in fouriers]
    datain = []
    dataout = []
    for n in range(N):
        Yin = np.array([fouriers[n](x[0]) + np.random.normal(0, noisein) for x in locs])
        Yout = np.array([fouriers_prime[n](x[0]) + np.random.normal(0, noiseout) for x in locs])
        datain.append((locs, Yin))
        dataout.append((locs, Yout))
    return datain, dataout

#
# def generate_fourier_ar1_dataset(N, noise, drawfunc, nfreq, locs):

