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
        plot_correlation_matrix(MAC, 'MAC')
    return MAC


def calc_PCC(lam1, lam2, plot=False):
    """
    This is the most simple pole correlation criterion I could think of. It calculates a value between 0.0 and 1.0 where
    1.0 indicates identical poles. What is important is that the criterion works for complex eigenvalues.
    """
    PCC = np.zeros((len(lam1), len(lam2)))
    for jj in range(len(lam2)):
        # Calculate the delta with respect to all other poles.
        delta = np.abs(lam1 - lam2[jj])
        # Scale the values to the range of 0.0 to 1.0 and store in matrix.
        PCC[:, jj] = 1.0 - delta / delta.max()
    # Optionally, visualize the results
    if plot:
        plot_correlation_matrix(PCC, 'PCC')
    return PCC


def calc_HDM(lam1, lam2, plot=False):
    """
    Hyperbolic distance metric. Proposed by [1], implementation and application shown in [2], section 6.2.5.

    [1] Luspay, T., Péni, T., Gőzse, I., Szabó, Z., and Vanek, B., “Model reduction for LPV systems based on approximate modal
    decomposition”, International Journal for Numerical Methods in Engineering, vol. 113, no. 6, pp. 891–909, 2018,
    https://doi.org/10.1002/nme.5692.
    [2] Jelicic, G., “System Identification of Parameter-Varying Aeroelastic Systems using Real-Time Operational Modal
    Analysis”, Deutsches Zentrum für Luft- und Raumfahrt  e. V., 2022, https://doi.org/10.57676/P9QV-CK92.
    """
    HDM = np.zeros((len(lam1), len(lam2)))
    # Map to unit circle.
    fs = 2.56 * np.max(np.abs(np.concatenate((lam1, lam1))) / 2.0 / np.pi)
    z1 = np.exp(lam1 / fs)
    z2 = np.exp(lam2 / fs)
    # Mirror unstable poles about unit circle.
    pos1 = lam1.real > 0.0
    pos2 = lam2.real > 0.0
    z1[pos1] = 1.0 / z1[pos1].conj()
    z2[pos2] = 1.0 / z2[pos2].conj()
    # Implementation of equations (4) and (A4) in [1] per row
    for jj in range(len(lam2)):
        HDM[:, jj] = 1.0 - np.abs((z1 - z2[jj]) / (1.0 - z1 * z2[jj].conj()))
    # Optionally, visualize the results
    if plot:
        plot_correlation_matrix(HDM, 'HDM')
    return HDM


def plot_correlation_matrix(matrix, name='Correlation Matrix'):
    plt.figure()
    plt.pcolor(matrix, cmap='hot_r')
    plt.colorbar()
    plt.grid('on')
    plt.title(name)
    plt.show()


def force_matrix_symmetry(matrix):
    return (matrix + matrix.T) / 2.0


def check_matrix_symmetry(matrix):
    # Check if a matrix is symmetric by forcing symmetry and then compare with the original matrix.
    matrix_sym = force_matrix_symmetry(matrix)
    result = (matrix != matrix_sym).nnz == 0
    return result
