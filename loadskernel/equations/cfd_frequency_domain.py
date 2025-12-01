import copy
import logging
import numpy as np

from scipy import linalg
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from scipy.stats import norm

from loadskernel import solution_tools
from loadskernel.equations.common import Common
from loadskernel.fem_interfaces import fem_helper
from loadskernel.interpolate import MatrixInterpolation
from loadskernel.equations.mona_frequency_domain import GustExcitation


class KMethod(GustExcitation):

    def build_AIC_interpolators(self):
        Qhh = []
        for Qjj_unsteady, k_red in zip(self.aero['Qjj_unsteady'], self.aero['k_red']):
            Qhh.append(self.PHIlh.T.dot(self.aerogrid['Nmat'].T.dot(self.aerogrid['Amat'].dot(Qjj_unsteady).dot(
                self.Djh_1 + complex(0, 1) * k_red / (self.macgrid['c_ref'] / 2.0) * self.Djh_2))))
        self.Qhh_interp = interp1d(self.aero['k_red'], Qhh, kind='cubic', axis=0, fill_value="extrapolate")

    def build_systems(self):
        self.A = np.zeros((self.n_modes, self.n_modes, self.n_freqs), dtype='complex128')  # [Antwort, Anregung, Frequenz]
        self.B = np.zeros((self.n_modes, self.n_modes, self.n_freqs), dtype='complex128')  # [Antwort, Anregung, Frequenz]
        for i_f in range(self.n_freqs):
            self.A[:, :, i_f], self.B[:, :, i_f] = self.system(self.k_reds[i_f])

    def system(self, k_red):
        rho = self.atmo['rho']
        Qhh = self.Qhh_interp(k_red)
        # Schwochow equation (7.10)
        A = -self.Mhh - rho / 2.0 * (self.macgrid['c_ref'] / 2.0 / k_red) ** 2.0 * Qhh
        B = -self.Khh
        return A, B

    def calc_eigenvalues(self):
        eigenvalues = []
        eigenvectors = []
        freqs = []
        damping = []
        Vtas = []
        eigenvalue, eigenvector = linalg.eig(self.A[:, :, 0], self.B[:, :, 0])
        # sorting
        idx_pos = np.where(eigenvalue.real >= 0.0)[0]  # nur oszillierende Eigenbewegungen
        idx_sort = np.argsort(1.0 / eigenvalue.real[idx_pos] ** 0.5)  # sort result by eigenvalue

        eigenvalue = eigenvalue[idx_pos][idx_sort]
        eigenvector = eigenvector[:, idx_pos][:, idx_sort]
        # store results
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)

        # calculate frequencies and damping ratios
        freqs.append(1.0 / eigenvalue.real ** 0.5 / 2.0 / np.pi)
        damping.append(eigenvalue.imag / eigenvalue.real)
        Vtas.append(self.macgrid['c_ref'] / 2.0 / self.k_reds[0] / eigenvalue.real ** 0.5)

        for i_f in range(1, self.n_freqs):
            eigenvalue, eigenvector = linalg.eig(self.A[:, :, i_f], self.B[:, :, i_f])
            # sorting
            idx_pos = np.where(eigenvalue.real >= 0.0)[0]
            MAC = fem_helper.calc_MAC(eigenvectors[-1], eigenvector[:, idx_pos], plot=False)
            idx_sort = [MAC[x, :].argmax() for x in range(MAC.shape[0])]

            eigenvalue = eigenvalue[idx_pos][idx_sort]
            eigenvector = eigenvector[:, idx_pos][:, idx_sort]
            # store results
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
            freqs.append(1.0 / eigenvalue.real ** 0.5 / 2.0 / np.pi)
            damping.append(eigenvalue.imag / eigenvalue.real)
            Vtas.append(self.macgrid['c_ref'] / 2.0 / self.k_reds[i_f] / eigenvalue.real ** 0.5)

        self.freqs = np.array(freqs)
        self.damping = np.array(damping)
        self.Vtas = np.array(Vtas)


class KEMethod(KMethod):

    def system(self, k_red):
        rho = self.atmo['rho']
        Qhh = self.Qhh_interp(k_red)
        # Nastran equation (2-120)
        A = self.Khh
        B = (k_red / self.macgrid['c_ref'] * 2.0) ** 2.0 * self.Mhh + rho / 2.0 * Qhh
        return A, B

    def calc_eigenvalues(self):
        eigenvalues = []
        eigenvectors = []
        freqs = []
        damping = []
        Vtas = []
        eigenvalue, eigenvector = linalg.eig(self.A[:, :, 0], self.B[:, :, 0])
        # sorting
        idx_pos = range(self.n_modes)
        V = ((eigenvalue.real ** 2 + eigenvalue.imag ** 2) / eigenvalue.real) ** 0.5
        freq = self.k_reds[0] * V / np.pi / self.macgrid['c_ref']
        idx_sort = np.argsort(freq)

        eigenvalue = eigenvalue[idx_pos][idx_sort]
        eigenvector = eigenvector[:, idx_pos][:, idx_sort]
        # store results
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        freqs.append(freq[idx_sort])
        Vtas.append(V[idx_sort])
        damping.append(np.imag(V[idx_sort] ** 2 / eigenvalue))

        for i_f in range(1, self.n_freqs):
            eigenvalue, eigenvector = linalg.eig(self.A[:, :, i_f], self.B[:, :, i_f])
            # sorting
            if i_f >= 2:
                pes = eigenvalues[-1] + (self.k_reds[i_f] - self.k_reds[i_f - 1]) * (eigenvalues[-1] - eigenvalues[-2]) \
                    / (self.k_reds[i_f - 1] - self.k_reds[i_f - 2])
                idx_sort = []
                for pe in pes:
                    idx_sort.append(((pe.real - eigenvalue.real) ** 2 + (pe.imag - eigenvalue.imag) ** 2).argmin())

            eigenvalue = eigenvalue[idx_pos][idx_sort]
            eigenvector = eigenvector[:, idx_pos][:, idx_sort]
            # store results
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
            V = ((eigenvalue.real ** 2 + eigenvalue.imag ** 2) / eigenvalue.real) ** 0.5
            freq = self.k_reds[i_f] * V / np.pi / self.macgrid['c_ref']
            freqs.append(freq)
            Vtas.append(V)
            damping.append(-eigenvalue.imag / eigenvalue.real)

        self.freqs = np.array(freqs)
        self.damping = np.array(damping)
        self.Vtas = np.array(Vtas)


class PKMethodSchwochow(KMethod):
    """
    This PK-Method uses a formulation proposed by Schwochow [1].
    Summary: The aerodynamic forces are split in a velocity and a deformation dependent part and added to the damping and
    stiffness term in the futter equation respectively. In this way, the aerodynamic damping and stiffness are treated
    seperately and in a more physical way. According to Schwochow, this leads to a better approximation of the damping in the
    flutter solution.

    [1] Schwochow, J., “Die aeroelastische Stabilitätsanalyse - Ein praxisnaher Ansatz Intervalltheoretischen Betrachtung von
    Modellierungsunsicherheiten am Flugzeug zur”, Dissertation, Universität Kassel, Kassel, 2012.
    """

    def setup_frequence_parameters(self):
        self.n_modes_rbm = 5
        self.n_modes_f = self.model['mass'][self.trimcase['mass']]['n_modes'][()]
        self.n_modes = self.n_modes_f + self.n_modes_rbm

        self.states = ["y'", "z'", "$\\Phi'$", "$\\Theta'$", "$\\Psi'$", ]  # noqa: W605
        for i_mode in range(1, self.n_modes_f + 1):
            self.states += ['Uf' + str(i_mode)]
        self.states += ["v'", "w'", "p'", "q'", "r'"]
        for i_mode in range(1, self.n_modes_f + 1):
            self.states += ['$\\mathrm{{ \\dot Uf{} }}$'.format(str(i_mode))]  # noqa: W605

        self.Vvec = self.simcase['flutter_para']['Vtas']

    def eval_equations(self):
        self.setup_frequence_parameters()

        logging.info('building systems')
        self.build_AIC_interpolators()
        logging.info('starting p-k iterations to match k_red with Vtas and omega')
        # Compute initial guess at k_red=0.0 and first flight speed
        self.Vtas = self.Vvec[0]
        eigenvalue, eigenvector = linalg.eig(self.system(k_red=0.0))
        bandbreite = eigenvalue.__abs__().max() - eigenvalue.__abs__().min()
        # No zero eigenvalues
        idx_pos = np.where(eigenvalue.__abs__() / bandbreite >= 1e-3)[0]
        # Sort initial results by eigenvalue
        idx_sort = np.argsort(np.abs(eigenvalue.imag[idx_pos]))
        eigenvalues0 = eigenvalue[idx_pos][idx_sort]
        eigenvectors0 = eigenvector[:, idx_pos][:, idx_sort]
        k0 = eigenvalues0.imag * self.macgrid['c_ref'] / 2.0 / self.Vtas

        eigenvalues = []
        eigenvectors = []
        freqs = []
        damping = []
        Vtas = []
        # Loop over modes
        for i_mode in range(len(eigenvalues0)):
            logging.debug('Mode {}'.format(i_mode + 1))
            eigenvalues_per_mode = []
            eigenvectors_per_mode = []
            k_old = copy.deepcopy(k0[i_mode])
            eigenvalues_old = copy.deepcopy(eigenvalues0)
            eigenvectors_old = copy.deepcopy(eigenvectors0)
            # Loop over Vtas
            for _, V in enumerate(self.Vvec):
                self.Vtas = V
                e = 1.0
                n_iter = 0
                # Iteration to match k_red with Vtas and omega of the mode under investigation
                while e >= 1e-3:
                    eigenvalues_new, eigenvectors_new = self.calc_eigenvalues(self.system(k_old),
                                                                              eigenvalues_old, eigenvectors_old)
                    # Small switch since this function is reused by class PKMethodRodden
                    if self.simcase['flutter_para']['method'] in ['pk', 'pk_schwochow']:
                        # For this implementaion, the reduced frequency may become negative.
                        k_now = eigenvalues_new[i_mode].imag * self.macgrid['c_ref'] / 2.0 / self.Vtas
                    elif self.simcase['flutter_para']['method'] in ['pk_rodden']:
                        # Allow only positive reduced frequencies in the implementation following Rodden.
                        k_now = np.abs(eigenvalues_new[i_mode].imag) * self.macgrid['c_ref'] / 2.0 / self.Vtas
                    # Use relaxation for improved convergence, which helps in some cases to avoid oscillations of the
                    # iterative solution.
                    k_new = k_old + 0.8 * (k_now - k_old)
                    e = np.abs(k_new - k_old)
                    k_old = k_new
                    n_iter += 1
                    # If no convergence is achieved, stop and issue a warning. Typically, the iteration converges in less than
                    # ten loops, so 50 should be more than enough and prevents excessive calculation times.
                    if n_iter > 50:
                        logging.warning('No convergence for mode {} at Vtas={:.2f} with k_red={:.5f} and e={:.5f}'.format(
                            i_mode + 1, V, k_new, e))
                        break
                eigenvalues_old = eigenvalues_new
                eigenvectors_old = eigenvectors_new
                eigenvalues_per_mode.append(eigenvalues_new[i_mode])
                eigenvectors_per_mode.append(eigenvectors_new[:, i_mode])
            # Store results
            eigenvalues_per_mode = np.array(eigenvalues_per_mode)
            eigenvalues.append(eigenvalues_per_mode)
            eigenvectors.append(np.array(eigenvectors_per_mode).T)
            freqs.append(eigenvalues_per_mode.imag / 2.0 / np.pi)
            damping.append(eigenvalues_per_mode.real / np.abs(eigenvalues_per_mode))
            Vtas.append(self.Vvec)

        response = {'eigenvalues': np.array(eigenvalues).T,
                    'eigenvectors': np.array(eigenvectors).T,
                    'freqs': np.array(freqs).T,
                    'damping': np.array(damping).T,
                    'Vtas': np.array(Vtas).T,
                    'states': self.states,
                    }
        return response

    def calc_eigenvalues(self, A, eigenvalues_old, eigenvectors_old):
        # Find all eigenvalues and eigenvectors
        eigenvalue, eigenvector = linalg.eig(A)
        # To match the modes with the previous step, use a correlation cirterion as specified in the JCL.
        if 'tracking' not in self.simcase['flutter_para']:
            # Set a default.
            tracking_method = 'MAC'
        else:
            tracking_method = self.simcase['flutter_para']['tracking']
        # Calculate the correlation bewteen the old and current modes.
        if tracking_method == 'MAC':
            # Most simple, use only the modal assurance criterion (MAC).
            correlation = fem_helper.calc_MAC(eigenvectors_old, eigenvector)
        elif tracking_method == 'MAC*PCC':
            # Combining MAC and pole correlation cirterion (PCC) for improved handling of complex conjugate pairs.
            correlation = fem_helper.calc_MAC(eigenvectors_old, eigenvector) * fem_helper.calc_PCC(eigenvalues_old, eigenvalue)
        elif tracking_method == 'MAC*HDM':
            # Combining MAC and hyperboloic distance metric (HDM) for improved handling of complex conjugate pairs.
            correlation = fem_helper.calc_MAC(eigenvectors_old, eigenvector) * fem_helper.calc_HDM(eigenvalues_old, eigenvalue)
        # Based on the correlation matrix, find the best match and apply to the modes.
        idx_pos = self.get_best_match(correlation)
        eigenvalues = eigenvalue[idx_pos]
        eigenvectors = eigenvector[:, idx_pos]
        return eigenvalues, eigenvectors

    def get_best_match(self, MAC):
        """
        It is important that no pole is dropped or selected twice. The solution is to keep record of the matches that are
        still available so that, if the best match is already taken, the second best match is selected.
        """
        possible_matches = [True] * MAC.shape[1]
        possible_idx = np.arange(MAC.shape[1])
        idx_pos = []
        for x in range(MAC.shape[0]):
            # the highest MAC value indicates the best match
            best_match = MAC[x, possible_matches].argmax()
            # reconstruct the corresponding index
            idx_pos.append(possible_idx[possible_matches][best_match])
            # remove the best match from the list of candidates
            possible_matches[possible_idx[possible_matches][best_match]] = False
        return idx_pos

    def calc_Qhh_1(self, Qjj_unsteady):
        return self.PHIlh.T.dot(self.aerogrid['Nmat'].T.dot(self.aerogrid['Amat'].dot(Qjj_unsteady).dot(self.Djh_1)))

    def calc_Qhh_2(self, Qjj_unsteady):
        return self.PHIlh.T.dot(self.aerogrid['Nmat'].T.dot(self.aerogrid['Amat'].dot(Qjj_unsteady).dot(self.Djh_2)))

    def build_AIC_interpolators(self):
        Qhh_1 = []
        Qhh_2 = []
        # Mirror the AIC matrices with respect to the real axis to allow negative reduced frequencies.
        Qjj_mirrored = np.concatenate((np.flip(self.aero['Qjj_unsteady'].conj(), axis=0), self.aero['Qjj_unsteady']), axis=0)
        k_mirrored = np.concatenate((np.flip(-self.aero['k_red']), self.aero['k_red']))
        # Do some pre-multiplications first, then the interpolation
        for Qjj in Qjj_mirrored:
            Qhh_1.append(self.calc_Qhh_1(Qjj))
            Qhh_2.append(self.calc_Qhh_2(Qjj))
        self.Qhh_1_interp = MatrixInterpolation(k_mirrored, Qhh_1)
        self.Qhh_2_interp = MatrixInterpolation(k_mirrored, Qhh_2)

    def system(self, k_red):
        rho = self.atmo['rho']

        Qhh_1 = self.Qhh_1_interp(k_red)
        Qhh_2 = self.Qhh_2_interp(k_red)
        Mhh_inv = np.linalg.inv(self.Mhh)

        upper_part = np.concatenate((np.zeros((self.n_modes, self.n_modes), dtype='complex128'),
                                     np.eye(self.n_modes, dtype='complex128')), axis=1)
        lower_part = np.concatenate((-Mhh_inv.dot(self.Khh - rho / 2.0 * self.Vtas ** 2.0 * Qhh_1),
                                     -Mhh_inv.dot(self.Dhh - rho / 2.0 * self.Vtas * Qhh_2)), axis=1)
        A = np.concatenate((upper_part, lower_part))
        return A


class PKMethodRodden(PKMethodSchwochow):
    """
    This PK-Method uses a formulation as implemented in Nastran by Rodden and Johnson [2], Section 2.6, Equation (2-131).
    Summary: The matrix of the aerodynamic forces Qhh includes both a velocity and a deformation dependent part. The real and
    the imaginary parts are then added to the damping and stiffness term in the futter equation respectively.

    [2] Rodden, W., and Johnson, E., MSC.Nastran Version 68 Aeroelastic Analysis User’s Guide. MSC.Software Corporation, 2010.
    """

    def build_AIC_interpolators(self):
        # Same formulation as in K-Method, but with custom, linear matrix interpolation
        Qhh = []
        for Qjj_unsteady, k_red in zip(self.aero['Qjj_unsteady'], self.aero['k_red']):
            Qhh.append(self.PHIlh.T.dot(self.aerogrid['Nmat'].T.dot(self.aerogrid['Amat'].dot(Qjj_unsteady).dot(
                self.Djh_1 + complex(0, 1) * k_red / (self.macgrid['c_ref'] / 2.0) * self.Djh_2))))
        self.Qhh_interp = MatrixInterpolation(self.aero['k_red'], Qhh)

    def system(self, k_red):
        rho = self.atmo['rho']

        # Make sure that k_red is not zero due to the division by k_red. In addition, limit k_red to the smallest
        # frequency the AIC matrices were calculated for.
        k_red = np.max([k_red, np.min(self.aero['k_red'])])

        Qhh = self.Qhh_interp(k_red)
        Mhh_inv = np.linalg.inv(self.Mhh)

        upper_part = np.concatenate((np.zeros((self.n_modes, self.n_modes), dtype='complex128'),
                                     np.eye(self.n_modes, dtype='complex128')), axis=1)
        lower_part = np.concatenate((-Mhh_inv.dot(self.Khh - rho / 2 * self.Vtas ** 2.0 * Qhh.real),
                                     -Mhh_inv.dot(self.Dhh - rho / 4 * self.Vtas * self.macgrid['c_ref'] / k_red * Qhh.imag)),
                                    axis=1)
        A = np.concatenate((upper_part, lower_part))
        return A
