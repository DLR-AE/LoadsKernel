import numpy as np


class MatrixInterpolation:
    """
    This is a simple class for linear matrix interpolation.
    The intention is to create an interpolation method for AIC matrices, which is similar to but faster than
    interp1d from scipy.
    A test cases using unsteady AIC matrices of the Allegra configuration resulted in numerically equal results
    and a 5-10x faster computation time compared to scipy. Extrapolation is supported.

    Inputs
       x: 1-D array of n_samples
    data: 3-D array of matrices to interpolate, Notation [n_samples, n_rows, n_columns]

    Output
          2-D array of interpolated matrix values
    """

    def __init__(self, x, data):
        self.x = np.array(x)
        self.n_samples = len(self.x)
        self.data = np.array(data)
        self.gradients = np.zeros_like(self.data)

        self.calc_gradients()

    def __call__(self, i):
        return self.interpolate(i)

    def calc_gradients(self):
        # calculate the gradients between all samples
        for m in range(self.n_samples - 1):
            self.gradients[m, :, :] = (self.data[m + 1, :, :] - self.data[m, :, :]) / (self.x[m + 1] - self.x[m])
        # repeat the last set of gradients to make sure there are right-sided gradients available in case of extrapolation
        self.gradients[-1, :, :] = self.gradients[-2, :, :]

    def interpolate(self, i):
        # find the nearest neighbor
        pos = np.abs(self.x - i).argmin()
        # calculate the delta
        delta = i - self.x[pos]
        # when the delta is negative, then take the left sided gradients, except when we are at the lower bound
        if delta < 0.0 and pos > 0:
            pos_gradients = pos - 1
        else:
            pos_gradients = pos
        # perform linear interpolation and return result
        return self.data[pos, :, :] + self.gradients[pos_gradients, :, :] * delta
