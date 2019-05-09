import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def generate_gaussian2d(beta, sigma, n, T, mu0, sigma0):
    obs = []
    obs.append(np.random.normal(mu0, sigma0, (n, n)))
    filt = beta * np.ones((3, 3))
    filt *= (1 / np.sum(filt))
    filt[1, 1] = 1
    for t in range(1, T):
        means2d = ndimage.filters.convolve(obs[t-1], filt)
        x = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x[i, j] = np.random.normal(means2d[i, j], sigma)
        obs.append(x)
    return obs


test = generate_gaussian2d(0.7, 0.1, 100, 10, 0, 1)

plt.figure()
plt.imshow(test[0])

plt.figure()
plt.imshow(test[1])

plt.figure()
plt.imshow(test[9])

plt.figure()
plt.imshow(test[8])