import numpy as np
import scipy.sparse as sp


def read_csv(filename, sparse_output=False):
    # use numpy to load comma separated data
    data = np.loadtxt(filename, comments='#', delimiter=',')
    # optionally, convert to sparse
    if sparse_output:
        data = sp.csc_matrix(data)
    return data
