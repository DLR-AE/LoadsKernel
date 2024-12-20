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
        PCC[:, jj] = 1.0 - delta / delta.max()
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
