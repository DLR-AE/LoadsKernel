import logging
import numpy as np

from scipy.interpolate import interp1d

from loadskernel.interpolate import MatrixInterpolation
from loadskernel.equations.mona_frequency_domain import KMethod as MonaKMethod
from loadskernel.equations.mona_frequency_domain import KEMethod as MonaKEMethod
from loadskernel.equations.mona_frequency_domain import PKMethodRodden as MonaPKMethodRodden


class KMethod(MonaKMethod):

    def build_AIC_interpolators(self):
        Qhh = []
        for i_k, _ in enumerate(self.GAFs['k_red']):
            Qhh.append(self.PHIkh.T.dot(self.GAFs['Qhk'][:, :, i_k]) / self.GAFs['q_dyn'])
        self.Qhh_interp = interp1d(self.GAFs['k_red'], Qhh, kind='cubic', axis=0, fill_value="extrapolate")

    def setup_frequence_parameters(self):
        self.n_modes = self.model['mass'][self.trimcase['mass']]['n_modes'][()] + 5
        self.k_reds = self.simcase['flutter_para']['k_red']
        self.n_freqs = len(self.k_reds)

        if self.k_reds.max() > np.max(self.GAFs['k_red']):
            logging.warning('Required reduced frequency = %0.3f but GAFs given only up to %0.3f',
                            self.k_reds.max(), np.max(self.GAFs['k_red']))


class KEMethod(MonaKEMethod, KMethod):
    """
    The CFD-based KE-Method uses the combined formulations of the CFD-based K-Method (see above)
    and the Mona-based KE-Method (imported as MonaKMethod). This is achieved by inheriting twice.
    """


class PKMethodRodden(MonaPKMethodRodden):

    def build_AIC_interpolators(self):
        # Same formulation as in K-Method, but with custom, linear matrix interpolation
        Qhh = []
        for i_k, _ in enumerate(self.GAFs['k_red']):
            Qhh.append(self.PHIkh.T.dot(self.GAFs['Qhk'][:, :, i_k]) / self.GAFs['q_dyn'])
        self.Qhh_interp = MatrixInterpolation(self.GAFs['k_red'], Qhh)

    def system(self, k_red):
        rho = self.atmo['rho']
        # Make sure that k_red is not zero due to the division by k_red. If k_red=0.0, set to a small value.
        # This line is the only difference to the mona-based PKMethodRodden, because GAFs from CFD are also
        # calculated for k_red=0.0.
        k_red = np.max([k_red, 0.001])

        Qhh = self.Qhh_interp(k_red)
        Mhh_inv = np.linalg.inv(self.Mhh)

        upper_part = np.concatenate((np.zeros((self.n_modes, self.n_modes)),
                                     np.eye(self.n_modes)), axis=1)
        lower_part = np.concatenate((-Mhh_inv.dot(self.Khh - rho / 2 * self.Vtas ** 2.0 * Qhh.real),
                                     -Mhh_inv.dot(self.Dhh - rho / 4 * self.Vtas * self.macgrid['c_ref'] / k_red * Qhh.imag)),
                                    axis=1)
        A = np.concatenate((upper_part, lower_part))
        return A
