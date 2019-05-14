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
