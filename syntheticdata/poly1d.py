from functools import partial
import numpy as np
import importlib

from smoothing import parametrized_func as para_func
importlib.reload(para_func)


def random_polys(draw_coeff_func, draw_deg_func=None, deg=None, nsim=1):
    if draw_deg_func is None and deg is None:
        raise ValueError("Provide either draw_deg_func of deg")
    elif draw_deg_func is None and deg is not None:
        alphas = draw_coeff_func((nsim, deg))
        samples = [para_func.Polynomial(alphas[i]) for i in range(nsim)]
    else:
        samples = []
        degs = draw_deg_func(nsim)
        for i in range(nsim):
            alpha = draw_coeff_func(degs[i])
            samples.append(para_func.Polynomial(alpha))
    return samples



norm01 = partial(np.random.normal, 0, 1)
randint110 = partial(np.random.randint, 1, 10)

rand_polys = random_polys(norm01, randint110, nsim=100)