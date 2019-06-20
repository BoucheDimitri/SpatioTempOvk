import numpy as np
import matplotlib.pyplot as plt
import importlib
import scipy.optimize as optimize
import itertools
import pandas as pd
import os
import time
from functools import partial
import operalib.ridge as ovkridge
import syntheticdata.funcs1d as funcs1d
import functools
import sklearn.kernel_ridge as kernel_ridge
import pickle

import spatiotempovk.spatiotempdata as spatiotemp
import spatiotempovk.kernels as kernels
import spatiotempovk.losses as losses
import spatiotempovk.regularizers as regularizers
import spatiotempovk.regressors as regressors
import algebra.repeated_matrix as repmat
import smoothing.representer as repsmooth
import smoothing.parametrized_func as param_func
import solvers.gradientbased as gradientbased
import tsvalidation.sequentialval as seqval
import approxkernelridge.rffridge as rffridge
importlib.reload(repmat)
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(losses)
importlib.reload(regularizers)
importlib.reload(regressors)
importlib.reload(repsmooth)
importlib.reload(param_func)
importlib.reload(funcs1d)

# Plot parameters
plt.rcParams.update({"font.size": 30})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})

# Drawing function from normal(0, 1)
norm01 = functools.partial(np.random.normal, 0, 1)
uni01 = functools.partial(np.random.uniform, 0, 1)

# Determine (fixed) locations
nlocs = 50

# Build the data
Ntrain = 50
Ntest = 20
noisein = 0.25
noiseout = 2.5

# Draw random Fourier functions
# datain, dataout = funcs1d.fourier_dataset_varloc(Ntrain, noisein, noiseout, norm01, 2, nlocs, uni01)
test_locs = np.linspace(0, 1, 200).reshape((200, 1))
# dataintest, dataouttest = funcs1d.generate_fourier_dataset(Ntest, 0, 0, norm01, 2, test_locs)
# datain, dataout = spatiotemp.LocObsSet(datain), spatiotemp.LocObsSet(dataout)
# dataintest, dataouttest = spatiotemp.LocObsSet(dataintest), spatiotemp.LocObsSet(dataouttest)

# Or load dataset
with open(os.getcwd() + "/dumps/datasets.pkl", "rb") as i:
    datain, dataout, dataintest, dataouttest = pickle.load(i)


