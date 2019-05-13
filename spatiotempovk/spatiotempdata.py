import numpy as np


class SpatioTempData:
    """
    Class for spatio temporal data, mostly to reimplement the __getitem__ magic method

    Parameters
    ----------
    S: list
        list of tuples (X_t, Y_t), len(S)=T, X_t.shape=(nobs_t, space_dim) and Y_t.shape=(nobs_t, output_dim)

    Attributes
    ----------
    T: int
        Number of time periods in the data
    X: list
        Locations data, list of numpy.ndarray X_t with X_t.shape=(nobs_t, space_dim)
    Y: list
        Measurements data, list numpy.ndarray Y_t with Y_t.shape=(nobs_t, output_dim)
    """
    def __init__(self, S):
        self.T = len(S)
        self.X = [S[t][0] for t in range(self.T)]
        self.Y = [S[t][1] for t in range(self.T)]
        self.Ms = [S[t][0].shape[0] for t in range(self.T)]
        # Test of correspondance between number of measurements and number of locations
        Msy = [S[t][1].shape[0] for t in range(self.T)]
        for t in range(self.T):
            if self.Ms[t] != Msy[t]:
                raise Exception('Numbers of locations and measurements do not correspond at time: {}'.format(t))

    def __getitem__(self, key):
        # First key corresponds to the choice of data, "x" for space, "y" for measurement, : for both
        # Concatenate all time observations if no other key is specified
        if not isinstance(key, tuple):
            if key == "x":
                return np.concatenate([self.X[t] for t in range(self.T)])
            elif key == "y":
                return np.concatenate([self.Y[t] for t in range(self.T)])
            else:
                return np.concatenate([self.X[t] for t in range(self.T)]), np.concatenate([self.Y[t] for t in range(self.T)])
        # Second key corresponds to time, return data at all locations for a given time t
        # First key keeps its role (either "x", "y" or :)
        elif len(key) == 2:
            if key[0] == "x":
                return self.X[key[1]]
            elif key[0] == "y":
                return self.Y[key[1]]
            else:
                return self.X[key[1]], self.Y[key[1]]
        # Third key corresponds to location
        # First and second key keeps their role
        elif len(key) == 3:
            if key[0] == "x":
                return self.X[key[1]][key[2]]
            elif key[0] == "y":
                return self.Y[key[1]][key[2]]
            else:
                return self.X[key[1]][key[2]], self.Y[key[1]][key[2]]

    @classmethod
    def new_instance(cls, S):
        """
        Class method to create new instance from within class


        Parameters
        ----------
        S: list
            list of tuples (X_t, Y_t), len(S)=T, X_t.shape=(nobs_t, space_dim) and Y_t.shape=(nobs_t, output_dim)

        Returns
        -------

        data: SpatioTempData
            SpatioTempData new instance
        """
        return cls(S)

    def get_original_data(self, t0=None, t1=None):
        """
        Get back data in the original form from the data, possility to extract subsequence using time bounds

        Parameters
        ----------
            t0: int
                lower time bound
            t1: int
                upper time bound

        Returns
        -------
            data: list
                list of tuples (X_t, Y_t), len(S)=t1 - t0, X_t.shape=(nobs_t, space_dim) and Y_t.shape=(nobs_t, output_dim)
        """
        if t0 is None and t1 is None:
            return [(self.X[t], self.Y[t]) for t in range(self.T)]
        elif t0 is not None and t1 is None:
            return [(self.X[t], self.Y[t]) for t in range(t0, self.T)]
        elif t0 is None and t1 is not None:
            return [(self.X[t], self.Y[t]) for t in range(t1)]
        else:
            return [(self.X[t], self.Y[t]) for t in range(t0, t1)]

    def extract_subseq(self, t0, t1):
        """
        Extract subsequence as SpatioTempObject using time bounds

        Parameters
        ----------
            t0: int
                lower time bound
            t1: int
                upper time bound

        Returns
        -------
            subseq: SpatioTempData
                sub instance of the object within bounds t0 and t1
        """
        return SpatioTempData.new_instance(self.get_original_data(t0=t0, t1=t1))

    def flat_index(self, t, m):
        """
        Get flatten index corresponding to a given time and location

        Parameters
        ----------
        t: int
            time index
        m: int
            location index

        Returns
        -------
        flatindex: int
            the flatten index
        """

        if m <= self.Ms[t]:
            return sum(self.Ms[:t]) + m
        else:
            raise IndexError('Location out of bound for time {}'.format(t))

    def get_Ms(self):
        return self.Ms.copy()

    def get_barM(self):
        return sum(self.Ms)

    def get_T(self):
        return len(self.Ms)

