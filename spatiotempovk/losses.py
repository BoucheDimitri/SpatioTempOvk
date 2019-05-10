import numpy as np


class DiffLoss:

    def __init__(self):
        pass

    def __call__(self, y, x):
        pass

    def prime(self, y, x):
        pass

    def prox(self, y, x):
        pass


class NonDiffLoss:

    def __init__(self):
        pass

    def __call__(self, y, x):
        pass

    def prox(self, y, x):
        pass


class L2Loss(DiffLoss):

    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, y, x):
        return np.linalg.norm(y - x) ** 2

    def prime(self, y, x):
        return -2 * (y - x)


