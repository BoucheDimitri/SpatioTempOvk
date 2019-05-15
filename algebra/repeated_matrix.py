import numpy as np


class RepSymMatrix:

    def __init__(self, genmat, rep, slice=False):
        self.genmat = genmat
        self.rep = rep
        self.shape = (genmat.shape[0] * rep[0], genmat.shape[0] * rep[1])
        self.slice = slice

    @classmethod
    def new_instance(cls, genmat, rep, slice):
        return cls(genmat, rep, slice)

    def __getitem__(self, item):
        if isinstance(item, int):
            return np.tile(self.genmat[item % self.genmat.shape[0]], self.rep[1])
        else:
            raise ValueError("Only integer indexing is supported")

    def transpose(self):
        return RepSymMatrix.new_instance(self.genmat, rep=self.rep[::-1], slice=self.slice)

    def dot_dim1(self, other):
        if self.shape[1] != other.shape[0]:
            raise ValueError("shape {} and {} not aligned".format(self.shape, other.shape))
        else:
            if self.slice:
                res = np.zeros((self.genmat.shape[0],))
                for n in range(self.rep[1]):
                    res += self.genmat.dot(other[n * self.genmat.shape[0]: (n + 1) * self.genmat.shape[0]])
                return np.tile(res, self.rep[0])
            else:
                res = np.tile(self.genmat, self.rep[1]).dot(other)
                return np.tile(res, self.rep[0])

    def dot(self, other):
        if other.ndim == 1:
            return self.dot_dim1(other)
        elif other.ndim == 2:
            if self.shape[1] != other.shape[0]:
                raise ValueError("shape {} and {} not aligned".format(self.shape, other.shape))
            else:
                res = np.zeros((self.shape[0], other.shape[1]))
                for d in range(other.shape[1]):
                    res[:, d] = self.dot_dim1(other[:, d])
                return res



