import numpy as np
import matplotlib.pyplot as plt


def calc_MAC(X, Y, plot=False):
    """
    This is function vectorizes the calculation of the classical modal assurance criterion (MAC), see for examle
    equation 4a and 4b in [1]. X and Y are the two eigenvectors which will be compared.
    [1] Allemang, R. J., “The Modal Assurance Criterion – Twenty Years of Use and Abuse,” Sound and Vibration, pp. 14–21, Aug.
    2003.
    """
    # Create an empty matrix
    MAC = np.zeros((X.shape[1], Y.shape[1]))
    # Pre-compute the terms in the nominator and denominator
    q1 = np.diag(X.conj().T.dot(X))
    q2 = np.diag(Y.conj().T.dot(Y))
    q3 = X.conj().T.dot(Y)
    # Loop over the number of modes of the second eigenvector and calculate the MAC values per column.
    # This loop is necessary in case the size of the the eigenvectors differs.
    for jj in range(Y.shape[1]):
        MAC[:, jj] = np.real(np.abs(q3[:, jj]) ** 2.0 / q1 / q2[jj])
    # Optionally, vizualize the results
    if plot:
        plt.figure()
        plt.pcolor(MAC, cmap='hot_r')
        plt.colorbar()
        plt.grid('on')

        return MAC, plt
    return MAC

def calc_MACXP(lam_1, my_1, lam_2, my_2, plot=False):
    """
    This is function vectorizes the calculation of the extendet modal assurance criterion (MACXP), see for examle
    equation 28 in [1].

    Nomenklature:
    eigenvalues = lam_1,2
    eigenvectors = my_1,2

    [1] Vacher, P., Jacquier, B., and Bucharles, A., “Extensions of the MAC criterion to complex mode”, in Proceedings of the
    international conference on noise and vibration engineering., Leuven, Belgium, 2010.
    """
    # Create an empty matrix
    MAC = np.zeros((my_1.shape[1], my_2.shape[1]))
    # Pre-compute the terms in the nominator and denominator
    nominator = np.abs(my_1.conj().T.dot(my_2)) / np.abs(np.conj(lam_1) + lam_2) \
        + np.abs(my_1.T.dot(my_2)) / np.abs(lam_1 + lam_2)
    denom1 = np.abs(my_1.conj().T.dot(my_1)) / (2.0 * np.abs(lam_1.real)) \
        + np.abs(my_1.T.dot(my_1)) / (2.0 * np.abs(lam_1))
    denom2 = np.abs(my_2.conj().T.dot(my_2)) / (2.0 * np.abs(lam_2.real)) \
        + np.abs(my_2.T.dot(my_2)) / (2.0 * np.abs(lam_2))
    # Loop over the number of modes of the second eigenvector and calculate the MAC values per column.
    # This loop is necessary in case the size of the the eigenvectors differs.

    MAC = nominator ** 2.0 / denom1 / denom2
    # Optionally, vizualize the results
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
