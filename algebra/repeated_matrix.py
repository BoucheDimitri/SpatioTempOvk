import numpy as np


class RepSymMatrix:

    """
    Class for matrices that are bloc repetitions of a given small matrix
    The utility of this class is mostly to avoid storing the whole matrix in memory at anytime
    For that purpose we implement a dot operation and a reduced __getitem__ method

    Parameters
    ----------
    genmat: numpy.ndarray
        The small matrix which is repeated, must be a symetric matrix
    rep: tuple
        Number of repetitions of genmat along each axis
    slice: bool
        Should operations be divided in small products involving only genmat on the left for the dot products

    Attributes
    ----------
    genmat: numpy.ndarray
        The small matrix which is repeated
    rep: tuple
        Number of repetitions of genmat along each axis
    shape: tuple
        The virtual shape of the matrix if it was constructed completely
    slice: bool
        Should operations be divided in small products involving only genmat on the left for the dot products
    """

    def __init__(self, genmat, rep, slice=True):
        self.genmat = genmat
        self.rep = rep
        self.shape = (genmat.shape[0] * rep[0], genmat.shape[0] * rep[1])
        self.slice = slice

    @classmethod
    def new_instance(cls, genmat, rep, slice):
        return cls(genmat, rep, slice)

    def __getitem__(self, item):
        """
        Indexing operation, only support rows extraction, thus item should be integer

        Parameters
        ----------
        item: int
            index, should be int

        Returns
        -------
        row: np.ndarray
            row corresponding to the index item
        """
        if isinstance(item, int):
            return np.tile(self.genmat[item % self.genmat.shape[0]], self.rep[1])
        else:
            raise ValueError("Only integer indexing is supported")

    def transpose(self):
        """
        Reimplementation of the transpose method

        Returns
        -------
        transposed_mat: algebra.repeated_matrix.RepSymMatrix
            A new class instance corresponding to the virtual transpose of the repeated matrix
        """
        return RepSymMatrix.new_instance(self.genmat, rep=self.rep[::-1], slice=self.slice)

    def dot_dim1(self, other):
        """
        Dot product with vector

        Parameters
        ----------
        other: numpy.ndarray
            We must have other.ndim = 1

        Returns
        -------
        dot_result: numpy.ndarray
            The result of the dot product
        """
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
        """
        Dot product with vector or matrix

        Parameters
        ----------
        other: numpy.ndarray
            We must have other.ndim <= 2

        Returns
        -------
        dot_result: numpy.ndarray
            The result of the dot product
        """
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



