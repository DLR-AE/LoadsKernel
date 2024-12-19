import copy
import logging
import numpy as np

from scipy import linalg

from loadskernel.equations.frequency_domain import PKMethodSchwochow


class StateSpaceAnalysis(PKMethodSchwochow):

    def eval_equations(self):
        self.setup_frequence_parameters()

        logging.info('building systems')
        self.build_AICs()
        eigenvalue, eigenvector = linalg.eig(self.system(self.Vvec[0]))

        bandbreite = eigenvalue.__abs__().max() - eigenvalue.__abs__().min()
        idx_pos = np.where(eigenvalue.__abs__() / bandbreite >= 1e-3)[0]  # no zero eigenvalues
        idx_sort = np.argsort(np.abs(eigenvalue.imag[idx_pos]))  # sort result by eigenvalue
        eigenvalues0 = eigenvalue[idx_pos][idx_sort]
        eigenvectors0 = eigenvector[:, idx_pos][:, idx_sort]

        eigenvalues = []
        eigenvectors = []
        freqs = []
        damping = []
        Vtas = []
        eigenvalues_old = copy.deepcopy(eigenvalues0)
        eigenvectors_old = copy.deepcopy(eigenvectors0)
        # loop over Vtas
        for _, V in enumerate(self.Vvec):
            Vtas_i = V
            eigenvalues_i, eigenvectors_i = self.calc_eigenvalues(self.system(Vtas_i), eigenvalues_old, eigenvectors_old)

            # store
            eigenvalues.append(eigenvalues_i)
            eigenvectors.append(eigenvectors_i)
            freqs.append(eigenvalues_i.imag / 2.0 / np.pi)
            damping.append(eigenvalues_i.real / np.abs(eigenvalues_i))
            Vtas.append([Vtas_i] * len(eigenvalues_i))

            eigenvalues_old = eigenvalues_i
            eigenvectors_old = eigenvectors_i

        response = {'eigenvalues': np.array(eigenvalues),
                    'eigenvectors': np.array(eigenvectors),
                    'freqs': np.array(freqs),
                    'damping': np.array(damping),
                    'Vtas': np.array(Vtas),
                    'states': self.states,
                    }
        return response

    def calc_Qhh_1(self, Qjj):
        return self.PHIlh.T.dot(self.aerogrid['Nmat'].T.dot(self.aerogrid['Amat'].dot(Qjj).dot(self.Djh_1)))

    def calc_Qhh_2(self, Qjj):
        return self.PHIlh.T.dot(self.aerogrid['Nmat'].T.dot(self.aerogrid['Amat'].dot(Qjj).dot(self.Djh_2)))

    def build_AICs(self):
        # do some pre-multiplications first, then the interpolation
        if self.jcl.aero['method'] in ['mona_steady']:
            self.Qhh_1 = self.calc_Qhh_1(self.aero['Qjj'])
            self.Qhh_2 = self.calc_Qhh_2(self.aero['Qjj'])
#         elif self.jcl.aero['method'] in ['mona_unsteady']:
#             ABCD = self.aero['ABCD']
#             for k_red in self.aero['k_red']:
#                 D = np.zeros((self.aerogrid['n'], self.aerogrid['n']), dtype='complex')
#                 j = 1j # imaginary number
#                 for i_pole, beta in zip(np.arange(0,self.aero['n_poles']), self.aero['betas']):
#                     D += ABCD[3+i_pole, :, :] * j*k_red / (j*k_red + beta)
#                 Qjj_unsteady = ABCD[0, :, :] + ABCD[1, :, :]*j*k_red + ABCD[2, :, :]*(j*k_red)**2 + D
#                 Qhh_1.append(self.calc_Qhh_1(Qjj_unsteady))
#                 Qhh_2.append(self.calc_Qhh_2(Qjj_unsteady))

    def system(self, Vtas):
        rho = self.atmo['rho']
        Mhh_inv = np.linalg.inv(self.Mhh)

        upper_part = np.concatenate((np.zeros((self.n_modes, self.n_modes), dtype='float'),
                                     np.eye(self.n_modes, dtype='float')), axis=1)
        lower_part = np.concatenate((-Mhh_inv.dot(self.Khh - rho / 2.0 * Vtas ** 2.0 * self.Qhh_1),
                                     -Mhh_inv.dot(self.Dhh - rho / 2.0 * Vtas * self.Qhh_2)), axis=1)
        A = np.concatenate((upper_part, lower_part))
        return A


class JacobiAnalysis(PKMethodSchwochow):

    def __init__(self, response):
        self.response = response

    def eval_equations(self):
        dxyz = self.response['X'][6:9]
        Vtas = sum(dxyz ** 2) ** 0.5

        eigenvalue, eigenvector = linalg.eig(self.system())

        bandbreite = eigenvalue.__abs__().max() - eigenvalue.__abs__().min()
        idx_pos = np.where(eigenvalue.__abs__() / bandbreite >= 1e-3)[0]  # no zero eigenvalues
        idx_sort = np.argsort(np.abs(eigenvalue.imag[idx_pos]))  # sort result by eigenvalue
        eigenvalues = eigenvalue[idx_pos][idx_sort]
        eigenvectors = eigenvector[:, idx_pos][:, idx_sort]

        # store
        self.response['eigenvalues'] = np.array(eigenvalues, ndmin=2)
        self.response['eigenvectors'] = np.array(eigenvectors, ndmin=3)
        self.response['freqs'] = np.array(eigenvalues.imag / 2.0 / np.pi, ndmin=2)
        self.response['damping'] = np.array(eigenvalues.real / np.abs(eigenvalues), ndmin=2)
        self.response['Vtas'] = np.array([Vtas] * len(eigenvalues), ndmin=2)

    def system(self):
        A = self.response['A']
        return A
