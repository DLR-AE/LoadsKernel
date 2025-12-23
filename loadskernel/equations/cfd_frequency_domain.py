import logging
import numpy as np

from scipy.interpolate import interp1d
from scipy.fftpack import fft

from loadskernel.interpolate import MatrixInterpolation
from loadskernel.equations.mona_frequency_domain import KMethod as MonaKMethod
from loadskernel.equations.mona_frequency_domain import KEMethod as MonaKEMethod
from loadskernel.equations.mona_frequency_domain import PKMethodRodden as MonaPKMethodRodden
from loadskernel.equations.mona_frequency_domain import GustExcitation as MonaGustExcitation


class GustExcitation(MonaGustExcitation):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize additional attributes to avoid defining them outside __init__
        # Interpolators
        self.Qhh_interp = None
        self.Qhk_interp = None
        self.Qgustk_interp = None

    def build_AIC_interpolators(self):
        # Similar as in the fultter solutions, but re-scale the Qxx matrices with dynamic pressure q_dyn to obtain forces
        # in SI units. Also, reorder matrices (freq, aero panels, modes) because the interpolator works along the first axis.
        Qhh = self.q_dyn * np.moveaxis(self.GAFs['Qhh'], -1, 0)
        Qhk = self.q_dyn * np.moveaxis(self.GAFs['Qhk'], -1, 0)
        Qgusth = self.q_dyn * np.moveaxis(self.GAFs['Qgusth'], -1, 0)
        Qgustk = self.q_dyn * np.moveaxis(self.GAFs['Qgustk'], -1, 0)
        self.Qhh_interp = MatrixInterpolation(self.GAFs['k_red'], Qhh)
        self.Qhk_interp = MatrixInterpolation(self.GAFs['k_red'], Qhk)
        self.Qgusth_interp = MatrixInterpolation(self.GAFs['k_red'], Qgusth)
        self.Qgustk_interp = MatrixInterpolation(self.GAFs['k_red'], Qgustk)

    def transfer_function(self, f):
        omega = 2.0 * np.pi * f
        Qhh = self.Qhh_interp(self.f2k(f))
        TF = np.linalg.inv(-self.Mhh * omega ** 2 + complex(0, 1) * omega * self.Dhh + self.Khh - Qhh)
        return TF

    def calc_gust_excitation(self, freqs, t):
        gust_f = fft(self.one_m_cosine_gust(t))
        Ph_fourier = np.zeros((self.n_modes, len(freqs)), dtype='complex128')
        Pk_fourier = np.zeros((self.aerogrid['n'] * 6, len(freqs)), dtype='complex128')
        for i, f in enumerate(freqs):
            Qgusth = self.Qgusth_interp(self.f2k(f))
            Qgustk = self.Qgustk_interp(self.f2k(f))
            Pk_fourier[:, i] = Qgustk.dot(gust_f[i])
            Ph_fourier[:, i] = Qgusth.dot(gust_f[i])
        return Ph_fourier, Pk_fourier

    def one_m_cosine_gust(self, t):
        # This is "only" the gust signal; unlike with panel methods, there is no relationship with the aicraft geometry here.
        # The effect of the aircraft penetrating into the gust was already captured during the GAF computations.
        tw = self.simcase['gust_gradient'] * 2.0 / self.Vtas
        T1 = self.simcase['gust_para']['T1']
        gust = self.Vtas * self.WG_TAS * 0.5 * (1 - np.cos(2.0 * np.pi * (t - T1) / tw))
        gust[np.where(t > tw)] = 0.0
        gust[np.where(t < T1)] = 0.0
        gust[np.where(t > tw + T1)] = 0.0
        return gust

    def calc_aero_response(self, freqs, Uh, dUh_dt):
        # Because the motion is included in the GAF computations, matrix Qhk is multiplied "only" by the generalized
        # deformations Uh.
        Ph_fourier = np.zeros((self.n_modes, len(freqs)), dtype='complex128')
        Pk_fourier = np.zeros((self.aerogrid['n'] * 6, len(freqs)), dtype='complex128')
        for i, f in enumerate(freqs):
            Qhk = self.Qhk_interp(self.f2k(f))
            Qhh = self.Qhh_interp(self.f2k(f))
            Ph_fourier[:, i] = Qhh.dot(Uh[:, i])
            Pk_fourier[:, i] = Qhk.dot(Uh[:, i])
        return Ph_fourier, Pk_fourier


class KMethod(MonaKMethod):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initialize frequently used attributes to satisfy linters/static analyzers
        self.n_freqs = None
        self.n_modes = None
        self.k_reds = None

    def build_AIC_interpolators(self):
        # Move k_red to axis 0, then create interpolator
        Qhh = np.moveaxis(self.GAFs['Qhh'], -1, 0)
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
        Qhh = np.moveaxis(self.GAFs['Qhh'], -1, 0)
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
