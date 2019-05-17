import numpy as np


def sequential_validation(predictor, X, Y, window):
    n = X.shape[0]
    Ypred = np.zeros((n-window, 1))
    for t in range(window, n):
        predictor.fit(X[t-window:t], Y[t-window:t])
        Ypred[t-window] = predictor.predict(X[t].reshape(1, -1))
    return np.array(Ypred), Y[window:]

