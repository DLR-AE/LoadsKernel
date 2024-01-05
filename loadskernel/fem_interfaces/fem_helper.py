import numpy as np
import matplotlib.pyplot as plt


def calc_MAC(X, Y, plot=True):
    MAC = np.zeros((X.shape[1], Y.shape[1]),)
    for jj in range(Y.shape[1]):
        for ii in range(X.shape[1]):
            q1 = np.dot(X[:, ii].conj().T, X[:, ii])
            q2 = np.dot(Y[:, jj].conj().T, Y[:, jj])
            q3 = np.dot(X[:, ii].conj().T, Y[:, jj])
            MAC[ii, jj] = np.real(np.abs(q3) ** 2 / q1 / q2)

    if plot:
        plt.figure()
        plt.pcolor(MAC, cmap='hot_r')
        plt.colorbar()
        plt.grid('on')

        return MAC, plt
    return MAC


def force_matrix_symmetry(matrix):
    return (matrix + matrix.T) / 2.0


def check_matrix_symmetry(matrix):
    # Check if a matrix is symmetric by forcing symmetry and then compare with the original matrix.
    matrix_sym = force_matrix_symmetry(matrix)
    result = (matrix != matrix_sym).nnz == 0
    return result
