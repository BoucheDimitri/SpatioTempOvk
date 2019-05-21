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
import sklearn.kernel_ridge as kernel_ridge

import spatiotempovk.spatiotempdata as spatiotemp
import spatiotempovk.kernels as kernels
import spatiotempovk.losses as losses
import spatiotempovk.regularizers as regularizers
import spatiotempovk.regressors as regressors
import algebra.repeated_matrix as repmat
import smoothing.representer as repsmooth
import smoothing.parametrized_func as param_func
importlib.reload(repmat)
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(losses)
importlib.reload(regularizers)
importlib.reload(regressors)
importlib.reload(repsmooth)
importlib.reload(param_func)


# ############## LOAD CLIMATE DATA ##################################################################################

# Path to the data
path = os.getcwd() + "/data/NA-1990-2002-Monthly.csv"

# Read data
datapd = pd.read_csv(path)

# Transform date column to datetime
datapd["DATE"] = pd.to_datetime(datapd["YEAR"].astype(str) + "/" + datapd["MONTH"].astype(str),
                                yearfirst=True)

# Sort by DATE first, then LAT then LONG
datapd_sorted = datapd.sort_values(by=["DATE", "LAT", "LONG"])

# # Deseasonalize
# datapd_sorted["MONTH_LAT_LONG"] = [(datapd_sorted.iloc[o, 1], datapd_sorted.iloc[o, 2], datapd_sorted.iloc[o, 3]) for o in range(datapd_sorted.shape[0])]
# datapd_sorted["LAT_LONG"] = [(datapd_sorted.iloc[o, 2], datapd_sorted.iloc[o, 3]) for o in range(datapd_sorted.shape[0])]
# locs = datapd_sorted["LAT_LONG"].unique()
# for loc in locs:
#     for month in range(1, 13):
#         avg = datapd_sorted.loc[datapd_sorted["MONTH_LAT_LONG"] == (month, loc[0], loc[1]), "TMP"].mean()
#         # std = datapd_sorted.loc[datapd_sorted["MONTH_LAT_LONG"] == (month, loc[0], loc[1]), "TMP"].std()
#         datapd_sorted.loc[datapd_sorted["MONTH_LAT_LONG"] == (month, loc[0], loc[1]), "TMP"] -= avg
#         # datapd_sorted.loc[datapd_sorted["MONTH_LAT_LONG"] == (month, loc[0], loc[1]), "TMP"] *= (1 / std)

# Adding differentiation column
datapd_sorted["TMP_DIFF"] = datapd_sorted["TMP"].diff()


# Dates contained in the data
dates = pd.unique(datapd_sorted["DATE"])

# Extract data and put it in the right form for SpatioTempData
extract = [datapd_sorted[datapd_sorted.DATE == d] for d in dates]
# With first differencing
extract = [(subtab.loc[:, ["LAT", "LONG"]].values[1:], subtab.loc[:, ["TMP"]].diff().values[1:]) for subtab in extract]
# Origin series
# extract = [(subtab.loc[:, ["LAT", "LONG"]].values, subtab.loc[:, ["TMP"]].values) for subtab in extract]

# Create a SpatioTempData object from it
data = spatiotemp.SpatioTempData(extract)

# Train test data
Strain = data.extract_subseq(0, 100)
Slast = data.extract_subseq(99, 100)
Stest = data.extract_subseq(100, 101)
Ms = Strain.get_Ms()

temp = np.array([data["y"][i][10, 0] for i in range(data.get_T())])
temp_diff = temp[1:] - temp[0:data.get_T()-1]
temp1 = temp[:75]
temp2 = temp[75:]


tempX = temp[:155].reshape((155, 1))
tempY = temp[1:156].reshape((155, 1))

ind = tempX.flatten().argsort()

plt.figure()
plt.plot(tempX.flatten()[ind], tempY.flatten()[ind])


clf = kernel_ridge.KernelRidge(alpha=10, kernel="rbf", gamma=1)

Ypred, Yreal = sequential_validation(clf, tempX, tempY, 40)

tempXred = tempX.flatten()[40:]
ind = tempXred.argsort()
plt.figure()
plt.plot(tempXred[ind], Ypred[ind])

# temp_pred = []
# real = []
# for t in range(40, 150):
#     tempX = temp[t-40:t]
#     tempY = temp[t+1-40:t+1]
#     clf = kernel_ridge.KernelRidge(alpha=0.1, kernel="linear")
#     clf.fit(tempX.reshape((40, 1)), tempY)
#     pred = clf.predict(np.array([[temp[t]]]))
#     temp_pred.append(pred[0])
#     real.append(temp[t+1])
#     print(t)
#
#
# tempX = temp[:99]
# tempY = temp[1:100]
#
# gker = kernels.GaussianKernel(sigma=1)
# K = gker.compute_K(tempX)
#
# kr = kernel_ridge.KernelRidge(alpha=1, kernel="rbf", gamma=1)
# kr.fit(tempX.reshape((99, 1)), tempY)
# kr.predict(np.array([[temp[100]]]))
#

# ############################ TEST FOR SMOOTHING ######################################################################
#
# gausskerx = kernels.GaussianGeoKernel(sigma=700)
gausskerx = kernels.GaussianGeoKernel(sigma=700)
Kx = gausskerx.compute_K(Strain["x_flat"][:125])

ridge_smoother = repsmooth.RidgeSmoother(mu=1, kernel=gausskerx)

alpha, base = ridge_smoother(Strain["x"], Strain["y"])

alpha_test, _ = ridge_smoother(Stest["x"], Stest["y"])

alpha[:, 0]

X = alpha[:99, :]
Y = alpha[1:100, :]

gausskeralpha = kernels.GaussianKernel(sigma=100)
Kalpha = gausskeralpha.compute_K(alpha)


reg = ovkridge.OVKRidge()
reg.fit(X, Y)
pred = reg.predict(np.expand_dims(alpha[-1], axis=0))















def smoothed_test(x):
    return sum([alpha[0][i] * base[i](x) for i in range(alpha[0].shape[0])])


cm = plt.cm.get_cmap('RdYlBu')

plt.figure()
sc = plt.scatter(Strain["x"][0][:, 0], Strain["x"][0][:, 1], c=Strain["y"][0].flatten(), cmap=cm)
plt.colorbar(sc)

lower_coords = np.min(Strain["x"][0], axis=0)
upper_coords = np.max(Strain["x"][0], axis=0)
lat_list = np.linspace(lower_coords[0], upper_coords[0], 50)
long_list = np.linspace(lower_coords[1], upper_coords[1], 50)
lats, longs = np.meshgrid(lat_list, long_list)
z = np.zeros((50, 50))
for i in range(50):
    for j in range(50):
        z[i, j] = smoothed_test(np.array([lats[i, j], longs[i, j]]))
    print(i)

plt.figure()
ctf = plt.contourf(lats, longs, z, cmap=cm)
plt.colorbar(ctf)

plt.figure()
pred = [smoothed_test((Strain["x"][0][i, 0], Strain["x"][0][i, 1])) for i in range(125)]
sc = plt.scatter(Strain["x"][0][:, 0], Strain["x"][0][:, 1], c=pred, cmap=cm)
plt.colorbar(sc)

from sklearn import kernel_ridge

clf = kernel_ridge.KernelRidge(alpha=1, kernel=gausskerx)

clf.fit(Strain["x", 0], Strain["y"][0].flatten())

alpha = clf.dual_coef_
basis = []
for x in Strain["x", 0]:
    basis.append(partial(gausskerx, x))

predictor = param_func.ParametrizedFunc(alpha, basis)
pred_para = predictor(Strain["x", 0])

plt.figure()
sc = plt.scatter(Strain["x"][0][:, 0], Strain["x"][0][:, 1], c=pred_para, cmap=cm)
plt.colorbar(sc)