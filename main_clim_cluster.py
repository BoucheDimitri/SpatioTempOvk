import importlib
import pandas as pd
import os
import time
import pickle
import numpy as np
import functools

import spatiotempovk.spatiotempdata as spatiotemp
import spatiotempovk.kernels as kernels
import spatiotempovk.losses as losses
import spatiotempovk.regularizers as regularizers
import spatiotempovk.regressors as regressors
import algebra.repeated_matrix as repmat
importlib.reload(repmat)
importlib.reload(spatiotemp)
importlib.reload(kernels)
importlib.reload(losses)
importlib.reload(regularizers)
importlib.reload(regressors)

# Path to the data
path = os.getcwd() + "/data/NA-1990-2002-Monthly.csv"

# Read data
datapd = pd.read_csv(path)

# Transform date column to datetime
datapd["DATE"] = pd.to_datetime(datapd["YEAR"].astype(str) + "/" + datapd["MONTH"].astype(str),
                                yearfirst=True)

# Sort by DATE first, then LAT then LONG
datapd_sorted = datapd.sort_values(by=["DATE", "LAT", "LONG"])

# Deseasonalize
datapd_sorted["MONTH_LAT_LONG"] = [(datapd_sorted.iloc[o, 1], datapd_sorted.iloc[o, 2], datapd_sorted.iloc[o, 3]) for o in range(datapd_sorted.shape[0])]
datapd_sorted["LAT_LONG"] = [(datapd_sorted.iloc[o, 2], datapd_sorted.iloc[o, 3]) for o in range(datapd_sorted.shape[0])]
locs = datapd_sorted["LAT_LONG"].unique()
for loc in locs:
    for month in range(1, 13):
        avg = datapd_sorted.loc[datapd_sorted["MONTH_LAT_LONG"] == (month, loc[0], loc[1]), "TMP"].mean()
        datapd_sorted.loc[datapd_sorted["MONTH_LAT_LONG"] == (month, loc[0], loc[1]), "TMP"] -= avg


# Dates contained in the data
dates = pd.unique(datapd_sorted["DATE"])

# Extract data and put it in the right form for SpatioTempData
extract = [datapd_sorted[datapd_sorted.DATE == d] for d in dates]
extract = [(subtab.loc[:, ["LAT", "LONG"]].values, subtab.loc[:, ["TMP"]].values) for subtab in extract]

# Create a SpatioTempData object from it
data = spatiotemp.LocObsSet(extract)

# Train test data
ntrain = 10
Strain = data.extract_subseq(0, ntrain)
Slast = data.extract_subseq(ntrain - 1, ntrain)
Stest = data.extract_subseq(ntrain, ntrain + 1)
Strain_input = data.extract_subseq(0, ntrain - 1)
Strain_output = data.extract_subseq(1, ntrain)
Ms = Strain.get_Ms()


# ############# EXPLOITING SAME LOCATION #####################################################################
# Timer
start = time.time()

# Kernels
gausskerx = kernels.GaussianGeoKernel(sigma=1000)
gausskery = kernels.GaussianKernel(sigma=15)
Kx = gausskerx.compute_K(Strain_output["x_flat"])
# Ky = gausskery.compute_K(Strain["y_flat"])
# Kx = gausskerx.compute_K(Strain["x_flat"])
Ky = None
convkers = kernels.ConvKernel(gausskerx, gausskery, Kx, Ky, sameloc=False)

# Compute convolution kernel matrix
# Ks = convkers.compute_K_from_mat(Ms)
Ks = convkers.compute_K(Strain_input["xy_tuple"])

# Define loss
loss = losses.L2Loss()

# Define regularizers and regularization params
spacereg = regularizers.TikhonovSpace()
timereg = regularizers.TikhonovTime()
mu = 1
lamb = 1


#

MT = Kx.shape[0]
T = Ks.shape[0]
alpha = np.zeros((T, MT))

Strain.sameloc=False
test_reg0 = regressors.DiffSpatioTempRegressor(loss, spacereg, timereg, mu, lamb, gausskerx, convkers)

grad_func = test_reg0.objective_grad_func(Strain.Ms, Strain["y_flat"], Kx, Ks)
obj_func = test_reg0.objective_func(Strain.Ms, Strain["y_flat"], Kx, Ks)


gamma=0.01
gradnorms=[np.linalg.norm(grad_func(alpha))]
objs=[obj_func(alpha)]
for i in range(50):
    grad = grad_func(alpha)
    alpha -= 0.01*grad
    objs.append(obj_func(alpha))
    gradnorms.append(np.linalg.norm(grad))
    print(i)





test_reg = regressors.DiffLocObsOnFuncReg(loss, spacereg, timereg, mu, lamb, gausskerx, convkers)

datafitting = functools.partial(test_reg.data_fitting, Strain_output.Ms, Strain_output["y_flat"], Kx, Ks)
datafitting_prime = functools.partial(test_reg.data_fitting_prime, Strain_output.Ms, Strain_output["y_flat"], Kx, Ks)

datafitting = functools.partial(test_reg.data_fitting, Strain_input.Ms, Strain_output["y_flat"], repmat.RepSymMatrix(Kx, (ntrain-1, ntrain-1)), Ks)
datafitting_prime = functools.partial(test_reg.data_fitting_prime, Strain_input.Ms, Strain_output["y_flat"], repmat.RepSymMatrix(Kx, (ntrain-1, ntrain-1)), Ks)
regspace = functools.partial(test_reg.smoothreg, repmat.RepSymMatrix(Kx, (ntrain-1, ntrain-1)), Ks)
regspace_prime = functools.partial(test_reg.smoothreg.prime, repmat.RepSymMatrix(Kx, (ntrain-1, ntrain-1)), Ks)
regtime = functools.partial(test_reg.globalreg.prime, repmat.RepSymMatrix(Kx, (ntrain-1, ntrain-1)), Ks)
regtime_prime = functools.partial(test_reg.globalreg, repmat.RepSymMatrix(Kx, (ntrain-1, ntrain-1)), Ks)


import scipy.optimize as optimize

sol = optimize.minimize(datafitting, x0=alpha.flatten())

gamma=0.0001
gradnorms=[np.linalg.norm(datafitting_prime(alpha))]
objs=[datafitting(alpha)]
for i in range(50):
    grad = datafitting_prime(alpha)
    alpha -= 0.01*grad
    objs.append(datafitting(alpha))
    gradnorms.append(np.linalg.norm(grad))
    print(i)

gamma=0.01
gradnorms=[np.linalg.norm(regspace_prime(alpha))]
objs=[regspace(alpha)]
for i in range(50):
    grad = regspace_prime(alpha)
    alpha -= 0.01*grad
    objs.append(regspace(alpha))
    gradnorms.append(np.linalg.norm(grad))
    print(i)

gamma=0.01
gradnorms=[np.linalg.norm(regtime_prime(alpha))]
objs=[regtime(alpha)]
for i in range(50):
    grad = regtime_prime(alpha)
    alpha -= 0.01*grad
    objs.append(regtime(alpha))
    gradnorms.append(np.linalg.norm(grad))
    print(i)




grad_func = test_reg.objective_grad_func(Strain_input.Ms, Strain_output["y_flat"], repmat.RepSymMatrix(Kx, (ntrain-1, ntrain-1)), Ks)
obj_func = test_reg.objective_func(Strain_input.Ms, Strain_output["y_flat"], repmat.RepSymMatrix(Kx, (ntrain-1, ntrain-1)), Ks)

gamma=0.01
gradnorms=[np.linalg.norm(grad_func(alpha))]
objs=[obj_func(alpha)]
for i in range(50):
    grad = grad_func(alpha)
    alpha -= 0.01*grad
    objs.append(obj_func(alpha))
    gradnorms.append(np.linalg.norm(grad))
    print(i)




test_reg.data_fitting(Strain.Ms, Strain["y_flat"], repmat.RepSymMatrix(Kx, (ntrain, ntrain)), Ks, alpha)
test_reg.data_fitting_prime(Strain.Ms, Strain["y_flat"], repmat.RepSymMatrix(Kx, (ntrain, ntrain)), Ks, alpha.flatten())

test_reg.objective_prime(Strain.Ms, Strain["y_flat"], repmat.RepSymMatrix(Kx, (ntrain, ntrain)), Ks, alpha)

grad_func = test_reg.objective_grad_func(Strain.Ms, Strain["y_flat"], repmat.RepSymMatrix(Kx, (ntrain - 1, ntrain - 1)), Ks)

test_reg = regressors.DiffLocObsOnFuncReg(loss, spacereg, timereg, mu, lamb, gausskerx, convkers)

test_reg.fit(Strain_input, Strain_output, Ks=Ks, Kx=repmat.RepSymMatrix(Kx, (ntrain-1, ntrain-1)))

# Predict at new locations
# Xnew = np.array(list(itertools.product(range(nx), range(ny))))
Xnew = Strain["x_flat"][:125, :]
Ypred = reg.predict(Slast, Xnew)
Ytrainpred = reg.predict(Strain_input, Xnew)

end = time.time()
print(end - start)

with open(os.getcwd() + "pred_train.pkl", "wb") as outp:
    pickle.dump(Ytrainpred, outp)

end = time.time()
print(end - start)
