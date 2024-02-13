import numpy as np


def insert_coo(sparsematrix, submatrix, idx1, idx2):
    """
    For sparse matrices, "fancy indexing" is not supported / not implemented as of 2017
    In case of a coo matrix, row and column based indexing is supported and used below.
    Observation: For sparse matrices where data is continously inserted, the COO format
    becomes slower and slower the bigger the matrix becomes. This is possibly due to the
    resizing of the numpy arrays.
    """
    row, col = np.meshgrid(idx1, idx2, indexing='ij')
    sparsematrix.row = np.concatenate((sparsematrix.row, row.reshape(-1)))
    sparsematrix.col = np.concatenate((sparsematrix.col, col.reshape(-1)))
    sparsematrix.data = np.concatenate((sparsematrix.data, submatrix.reshape(-1)))
    return sparsematrix


def insert_lil(sparsematrix, submatrix, idx1, idx2):
    """
    In contrast to COO, the LIL format is based on lists, which are faster to expand.
    Since my first implementation in 2014, the interface to the LIL matrix has improved and
    allows the direct assignment of data to matrix items, accessed by their index.
    """
    row, col = np.meshgrid(idx1, idx2, indexing='ij')
    sparsematrix[row, col] = submatrix
    return sparsematrix
