import numpy as np
import matplotlib.pyplot as plt
import importlib
import scipy.optimize as optimize
import itertools
import pandas as pd
import os

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

# Dates contained in the data
dates = pd.unique(datapd_sorted["DATE"])

# Extract data and put it in the right form for SpatioTempData
extract = [datapd_sorted[datapd_sorted.DATE == d] for d in dates]
extract = [(subtab.loc[:, ["LAT", "LONG"]].values, subtab.loc[:, ["TMP"]].values) for subtab in extract]

# Create a SpatioTempData object from it
data = spatiotemp.SpatioTempData(extract)

# Train test data
Strain = data.extract_subseq(0, 8)
Slast = data.extract_subseq(7, 8)
Stest = data.extract_subseq(8, 9)
Ms = Strain.get_Ms()


# ############# EXPLOITING SAME LOCATION #####################################################################
# Kernels
gausskerx = kernels.GaussianGeoKernel(sigma=1000)
gausskery = kernels.GaussianKernel(sigma=15)
Kx = gausskerx.compute_K(Strain["x_flat"][:125])
# Ky = gausskery.compute_K(Strain["y_flat"])
# Kx = gausskerx.compute_K(Strain["x_flat"])
Ky = None
convkers = kernels.ConvKernel(gausskerx, gausskery, Kx, Ky, sameloc=True)

# Compute convolution kernel matrix
# Ks = convkers.compute_K_from_mat(Ms)
Ks = convkers.compute_K(Strain["xy_tuple"])

# Define loss
loss = losses.L2Loss()

# Define regularizers and regularization params
spacereg = regularizers.TikhonovSpace()
timereg = regularizers.TikhonovTime()
mu = 1
lamb = 1

# Initialize and train regressor
reg = regressors.DiffSpatioTempRegressor(loss, spacereg, timereg, mu, lamb, gausskerx, convkers)
reg.fit(Strain, Ks=Ks, Kx=repmat.RepSymMatrix(Kx, (8, 8)))

# Predict at new locations
# Xnew = np.array(list(itertools.product(range(nx), range(ny))))
Xnew = Strain["x_flat"][:125, :]
Ypred = reg.predict(Slast, Xnew)


########## NOT EXPLOITING SAME LOCATION ################################################################################
# Kernels
gausskerx = kernels.GaussianGeoKernel(sigma=1000)
gausskery = kernels.GaussianKernel(sigma=15)
Ky = gausskery.compute_K(Strain["y_flat"])
Kx = gausskerx.compute_K(Strain["x_flat"])
convkers = kernels.ConvKernel(gausskerx, gausskery, Kx, Ky, sameloc=False)

# Compute convolution kernel matrix
Ks = convkers.compute_K_from_mat(Ms)

# Define loss
loss = losses.L2Loss()

# Define regularizers and regularization params
spacereg = regularizers.TikhonovSpace()
timereg = regularizers.TikhonovTime()
mu = 1
lamb = 1

# Initialize and train regressor
reg = regressors.DiffSpatioTempRegressor(loss, spacereg, timereg, mu, lamb, gausskerx, convkers)
# Force not same loc processing
Strain.sameloc = False
reg.fit(Strain, Ks=Ks, Kx=Kx)

# Predict at new locations
# Xnew = np.array(list(itertools.product(range(nx), range(ny))))
Xnew = Strain["x_flat"][:125, :]
Ypred = reg.predict(Slast, Xnew)







#
#
# Ms = datasmall.get_Ms()
# T = datasmall.get_T()
# barM = datasmall.get_barM()
#
# # Kernels for convolution
# gausskerx = kernels.GaussianGeoKernel(sigma=1000)
# gausskery = kernels.GaussianKernel(sigma=15)
#
# # Compute kernel matrices
# Kx = gausskerx.compute_K(datasmall["x_flat"][:125, :])
# Ky = gausskery.compute_K(datasmall["y_flat"])
# convkers = kernels.ConvKernel(gausskerx, gausskery, Kx, Ky, sameloc=True)
#
# Kx = repmat.RepSymMatrix(Kx, rep=(20, 20))
#
# # Compute convolution kernel matrix
# Ks = convkers.compute_K_from_mat(Ms)
#
# # Define loss
# loss = losses.L2Loss()
#
# # Define regularizers and regularization params
# spacereg = regularizers.TikhonovSpace()
# timereg = regularizers.TikhonovTime()
# mu = 1
# lamb = 1
#
# # Train/Test split
# Strain = data.extract_subseq(0, 4)
# Stest = data.extract_subseq(3, 4)
#
# # Initialize and train regressor
# reg = regressors.DiffSpatioTempRegressor(loss, spacereg, timereg, mu, lamb, gausskerx, convkers)
#
# reg.objective_func(datasmall.Ms, )
# reg.fit(datasmall, Kx=Kx, Ks=Ks)
#
# # Predict at new locations
# Xnew = np.array(list(itertools.product(range(nx), range(ny))))
# Ypred = reg.predict(Stest, Xnew)
# Ypred = Ypred.reshape((50, 50))
#
