import logging
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
    # Optionally, visualize the results
    if plot:
        plt.figure()
        plt.pcolor(MAC, cmap='hot_r')
        plt.colorbar()
        plt.grid('on')
        plt.title('MAC')

        return MAC, plt
    return MAC

def calc_MACXP(lam_1, my_1, lam_2, my_2, plot=False):
    """
    This is function vectorizes the calculation of the pole-weighted modal assurance criterion (MACXP), see equation 28 in [1].
    Notation: eigenvalues = lam_1,2 and eigenvectors = my_1,2

    [1] Vacher, P., Jacquier, B., and Bucharles, A., “Extensions of the MAC criterion to complex mode”, in Proceedings of the
    international conference on noise and vibration engineering., Leuven, Belgium, 2010.
    """
    # For the MACXP criterion, the number of modes has to be the same
    assert len(lam_1) == len(lam_2), 'Number of eigenvalues not equal: {} vs. {}'.format(len(lam_1), len(lam_2))
    assert my_1.shape[0] == my_1.shape[1], 'Size of eigenvector 1 not square: {}'.format(my_1.shape)
    assert my_2.shape[0] == my_2.shape[1], 'Size of eigenvector 2 not square: {}'.format(my_2.shape)

    # Helper functions
    def nominator(lam_1, my_1, lam_2, my_2):
        return np.abs(my_1.T.dot(my_2)) / np.tile(np.abs(lam_1 + lam_2),(len(lam_1),1))

    def denom(lam, my):
        return np.diag(my.conj().T.dot(my) / (2.0*np.abs(np.real(lam))) + np.abs(my.T.dot(my)) / (2.0*np.abs(lam)))

    # Pre-compute the terms in the nominator and denominator
    nomin1 = nominator(lam_1.conj(), my_1.conj(), lam_2, my_2)
    nomin2 = nominator(lam_1, my_1, lam_2, my_2)
    denom1 = denom(lam_1, my_1)
    denom2 = denom(lam_2, my_2)
    # Assemble everything
    MACXP = (nomin1 + nomin2) ** 2.0 / (denom1 * denom2)
    # The results should be completely real, but an imaginary part that is numerically zero remains and is discarded here.
    MACXP = np.abs(MACXP)
    # Optionally, visualize the results
    if plot:
        plt.figure()
        plt.pcolor(MACXP, cmap='hot_r')
        plt.colorbar()
        plt.grid('on')
        plt.title('MACXP')
        return MACXP, plt
    return MACXP

def calc_PCC(lam_1, lam_2, plot=False):
    """
    This is the most simple pole correlation criterion I could think of. It calculated a value between 0.0 and 1.0 where
    1.0 indicates identical poles. The cirterion works also for complex eigenvalues.
    """
    PCC = np.zeros((len(lam_1), len(lam_2)))
    for jj in range(len(lam_2)):
        # Calculate the delta with respect to all other poles.
        delta = np.abs(lam_1 - lam_2[jj])
        # Scale the values to the range of 0.0 to 1.0 and store in matrix.
        PCC[:,jj] =  1.0 - delta / delta.max()
    # Optionally, visualize the results
    if plot:
        plt.figure()
        plt.pcolor(PCC, cmap='hot_r')
        plt.colorbar()
        plt.grid('on')
        plt.title('PCC')
        return PCC, plt
    return PCC

def force_matrix_symmetry(matrix):
    return (matrix + matrix.T) / 2.0


def check_matrix_symmetry(matrix):
    # Check if a matrix is symmetric by forcing symmetry and then compare with the original matrix.
    matrix_sym = force_matrix_symmetry(matrix)
    result = (matrix != matrix_sym).nnz == 0
    return result
