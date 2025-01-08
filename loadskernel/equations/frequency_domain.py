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


class GustExcitation(Common):

    def eval_equations(self):
        self.setup_frequence_parameters()

        logging.info('building transfer functions')
        self.build_AIC_interpolators()  # unsteady
        positiv_TFs = self.build_transfer_functions(self.positiv_fftfreqs)
        TFs = np.zeros((self.n_modes, self.n_modes, self.n_freqs), dtype='complex128')
        for i_mode in range(self.n_modes):
            TFs[:, i_mode, :] = self.mirror_fouriersamples_even(positiv_TFs[:, i_mode, :])

        logging.info('calculating gust excitation (in physical coordinates, this may take considerable time and memory)')
        Ph_gust_fourier, Pk_gust_fourier = self.calc_gust_excitation(self.positiv_fftfreqs, self.t)
        Ph_gust_fourier = self.mirror_fouriersamples_even(Ph_gust_fourier)
        Pk_gust_fourier = self.mirror_fouriersamples_even(Pk_gust_fourier)

        logging.info('calculating responses')
        Uh_fourier = TFs * Ph_gust_fourier  # [Antwort, Anregung, Frequenz]
        Uh = ifft(np.array((Uh_fourier) * (1j * self.fftomega) ** 0).sum(axis=1))
        dUh_dt = ifft(np.array((Uh_fourier) * (1j * self.fftomega) ** 1).sum(axis=1))
        d2Uh_dt2 = ifft(np.array((Uh_fourier) * (1j * self.fftomega) ** 2).sum(axis=1))

        logging.info('reconstructing aerodynamic forces (in physical coordinates, this may take considerable time and memory)')
        Ph_aero_fourier, Pk_aero_fourier = self.calc_aero_response(
            self.positiv_fftfreqs,
            np.array((Uh_fourier) * (1j * self.fftomega) ** 0).sum(axis=1)[:, :self.n_freqs // 2 + 1],
            np.array((Uh_fourier) * (1j * self.fftomega) ** 1).sum(axis=1)[:, :self.n_freqs // 2 + 1],)
        Ph_aero_fourier = self.mirror_fouriersamples_even(Ph_aero_fourier)
        Pk_aero_fourier = self.mirror_fouriersamples_even(Pk_aero_fourier)
        Pk_aero = np.real(ifft(Pk_gust_fourier) + ifft(Pk_aero_fourier))[:, self.t_output]
        Pk_gust = np.real(ifft(Pk_gust_fourier))[:, self.t_output]

        # split h-set into b- and f-set
        # remember that the x-component was omitted
        Ucg = np.concatenate((np.zeros((len(self.t_output), 1)), Uh[:5, self.t_output].T.real - Uh[:5, 0].real), axis=1)
        dUcg_dt = np.concatenate((np.zeros((len(self.t_output), 1)),
                                  dUh_dt[:5, self.t_output].T.real - dUh_dt[:5, 0].real), axis=1)
        d2Ucg_dt2 = np.concatenate((np.zeros((len(self.t_output), 1)),
                                    d2Uh_dt2[:5, self.t_output].T.real - d2Uh_dt2[:5, 0].real), axis=1)
        Uf = Uh[5:, self.t_output].T.real - Uh[5:, 0].real
        dUf_dt = dUh_dt[5:, self.t_output].T.real - dUh_dt[5:, 0].real
        d2Uf_dt2 = d2Uh_dt2[5:, self.t_output].T.real - d2Uh_dt2[5:, 0].real

        g_cg = np.zeros((len(self.t_output), 3))
        commands = np.zeros((len(self.t_output), self.solution.n_inputs))

        #  x, y, z, Phi, Theta, Psi,  u, v, w, p, q, r in DIN 9300 body fixed system for flight physics
        X = np.concatenate((Ucg * np.array([-1., 1., -1., -1., 1., -1.]),
                            dUcg_dt * np.array([-1., 1., -1., -1., 1., -1.]),
                            Uf,  # modal deformations
                            dUf_dt,  # modal velocities
                            commands,
                            ), axis=1)
        response = {'X': X,
                    't': np.array([self.t[self.t_output]]).T,
                    'Pk_aero': Pk_aero.T,
                    'Pk_gust': Pk_gust.T,
                    'Pk_unsteady': Pk_aero.T * 0.0,
                    'dUcg_dt': dUcg_dt,
                    'd2Ucg_dt2': d2Ucg_dt2,
                    'Uf': Uf,
                    'dUf_dt': dUf_dt,
                    'd2Uf_dt2': d2Uf_dt2,
                    'g_cg': g_cg,
                    }
        return response

    def setup_frequence_parameters(self):
        self.n_modes = self.model['mass'][self.trimcase['mass']]['n_modes'][()] + 5
        self.Vtas, self.q_dyn = self.recover_Vtas(self.X0)
        # Number of sample points
        if self.simcase['gust']:
            t_factor = 10.0  # increase resolution (df) by extending simulation time
        else:
            t_factor = 1.0
        dt = self.simcase['dt']
        self.fmax = 1 / dt
        self.n_freqs = int(t_factor * self.simcase['t_final'] * self.fmax)
        if self.n_freqs % 2 != 0:  # n_freq is odd
            self.n_freqs += 1  # make even
        # sample spacing
        self.t = np.linspace(0.0, self.n_freqs * dt, self.n_freqs)
        # indices of time samples to returned for post-processing
        self.t_output = np.where(self.t <= self.simcase['t_final'])[0]
        # samples only from zero up to the Nyquist frequency
        self.freqs = np.linspace(0.0, self.fmax / 2.0, self.n_freqs // 2)
        # whole frequency space including negative frequencies
        fftfreqs = fftfreq(self.n_freqs, dt)
        self.fftomega = 2.0 * np.pi * fftfreqs
        # positive only frequencies where we need to calculate the TFs and excitations
        self.positiv_fftfreqs = np.abs(fftfreqs[:self.n_freqs // 2 + 1])
        self.positiv_fftomega = 2.0 * np.pi * self.positiv_fftfreqs

        logging.info('Frequency domain solution with tfinal = {}x{} s, nfreq = {}, fmax={} Hz and df = {} Hz'.format(
            t_factor, self.simcase['t_final'], self.n_freqs // 2, self.fmax / 2.0, self.fmax / self.n_freqs))
        if self.f2k(self.freqs.max()) > np.max(self.aero['k_red']):
            logging.warning('Required reduced frequency = {:0.3} but AICs given only up to {:0.3}'.format(
                self.f2k(self.freqs.max()), np.max(self.aero['k_red'])))

    def mirror_fouriersamples_even(self, fouriersamples):
        mirrored_fourier = np.zeros((fouriersamples.shape[0], self.n_freqs), dtype='complex128')
        mirrored_fourier[:, :self.n_freqs // 2] = fouriersamples[:, :-1]
        mirrored_fourier[:, self.n_freqs // 2:] = np.flip(fouriersamples[:, 1:], axis=1).conj()
        return mirrored_fourier

    def build_transfer_functions(self, freqs):
        TFs = np.zeros((self.n_modes, self.n_modes, len(freqs)), dtype='complex128')  # [Antwort, Anregung, Frequenz]
        for i, f in enumerate(freqs):
            TFs[:, :, i] = self.transfer_function(f)
        return TFs

    def transfer_function(self, f):
        omega = 2.0 * np.pi * f
        Qhh_1 = self.Qhh_1_interp(self.f2k(f))
        Qhh_2 = self.Qhh_2_interp(self.f2k(f))
        TF = np.linalg.inv(-self.Mhh * omega ** 2 + complex(0, 1) * omega * (self.Dhh - Qhh_2) + self.Khh - Qhh_1)
        return TF

    def build_AIC_interpolators(self):
        # interpolation of physical AIC
        self.Qjj_interp = MatrixInterpolation(self.aero['k_red'], self.aero['Qjj_unsteady'])
        # do some pre-multiplications first, then the interpolation
        Qhh_1 = []
        Qhh_2 = []
        for Qjj_unsteady in self.aero['Qjj_unsteady']:
            Qhh_1.append(self.q_dyn * self.PHIlh.T.dot(self.aerogrid['Nmat'].T.dot(
                self.aerogrid['Amat'].dot(Qjj_unsteady).dot(self.Djh_1))))
            Qhh_2.append(self.q_dyn * self.PHIlh.T.dot(self.aerogrid['Nmat'].T.dot(
                self.aerogrid['Amat'].dot(Qjj_unsteady).dot(self.Djh_2 / self.Vtas))))
        self.Qhh_1_interp = MatrixInterpolation(self.aero['k_red'], Qhh_1)
        self.Qhh_2_interp = MatrixInterpolation(self.aero['k_red'], Qhh_2)

    def calc_aero_response(self, freqs, Uh, dUh_dt):
        # Notation: [n_panels, timesteps]
        wj_fourier = self.Djh_1.dot(Uh) + self.Djh_2.dot(dUh_dt) / self.Vtas
        Ph_fourier, Pk_fourier = self.calc_P_fourier(freqs, wj_fourier)
        return Ph_fourier, Pk_fourier

    def calc_gust_excitation(self, freqs, t):
        # Notation: [n_panels, timesteps]
        # Eventuell muss wj_gust_f noch skaliert werden mit 2.0/N * np.abs(wj_gust_f[:,0:N//2])
        wj_gust_f = fft(self.wj_gust(t))
        Ph_fourier, Pk_fourier = self.calc_P_fourier(freqs, wj_gust_f)
        return Ph_fourier, Pk_fourier

    def calc_P_fourier(self, freqs, wj):
        Ph_fourier = np.zeros((self.n_modes, len(freqs)), dtype='complex128')
        Pk_fourier = np.zeros((self.aerogrid['n'] * 6, len(freqs)), dtype='complex128')
        for i, f in enumerate(freqs):
            # The interpolation of Qjj is computationally very expensive, especially for a large number of frequencies.
            Qjj = self.Qjj_interp(self.f2k(f))
            Pk_fourier[:, i] = self.q_dyn * self.PHIlk.T.dot(self.aerogrid['Nmat'].T.dot(
                self.aerogrid['Amat'].dot(Qjj.dot(wj[:, i]))))
            Ph_fourier[:, i] = self.PHIkh.T.dot(Pk_fourier[:, i])
        return Ph_fourier, Pk_fourier

    def wj_gust(self, t):
        ac_position = np.array([t * self.Vtas] * self.aerogrid['n'])
        panel_offset = np.array([self.aerogrid['offset_j'][:, 0]] * len(t)).T
        s_gust = ac_position - panel_offset - self.s0
        # downwash der 1-cos Boe auf ein jedes Panel berechnen
        wj_gust = self.WG_TAS * 0.5 * (1 - np.cos(np.pi * s_gust / self.simcase['gust_gradient']))
        wj_gust[np.where(s_gust <= 0.0)] = 0.0
        wj_gust[np.where(s_gust > 2 * self.simcase['gust_gradient'])] = 0.0
        # Ausrichtung der Boe fehlt noch
        gust_direction_vector = np.sum(self.aerogrid['N'] * np.dot(np.array([0, 0, 1]), solution_tools.calc_drehmatrix(
            self.simcase['gust_orientation'] / 180.0 * np.pi, 0.0, 0.0)), axis=1)
        wj = wj_gust * np.array([gust_direction_vector] * len(t)).T
        return wj

    def f2k(self, f):
        return 2.0 * np.pi * f * self.macgrid['c_ref'] / 2.0 / self.Vtas

    def k2f(self, k_red):
        return k_red * self.Vtas / np.pi / self.macgrid['c_ref']


class TurbulenceExcitation(GustExcitation):
    """
    So far, the calculation procedure is identical to the normal 1-cos gust excitation.
    All functionalities are inherited from the GustExcitation class, only the excitation itself is replaced.
    """

    def calc_sigma(self, n_samples):
        """
        Calculate sigma so that U_sigma,ref has the probability to occur once per n_samples.
        Formula as given in: https://de.wikipedia.org/wiki/Normalverteilung, compare with values of 'z sigma' in the first
        column of the table.
        Explanations on using the percent point function (ppf) from scipy:https://stackoverflow.com/questions/60699836/
        how-to-use-norm-ppf
        """
        p = 1.0 - 1.0 / n_samples
        sigma = norm.ppf((p + 1.0) / 2.0, loc=0.0, scale=1.0)
        return sigma

    def calc_psd_vonKarman(self, freqs):
        # calculate turbulence excitation by von Karman power spectral density according to CS-25.341 b)
        L = 762.0 / self.Vtas  # normalized turbulence scale [s], 2500.0 ft = 762.0 m
        psd_karman = 2.0 * L * (1.0 + 8.0 / 3.0 * (1.339 * L * 2.0 * np.pi * freqs) ** 2.0) \
            / (1.0 + (1.339 * L * 2.0 * np.pi * freqs) ** 2.0) ** (11.0 / 6.0)
        # set psd to zero for f=0.0 to achieve y_mean = 0.0
        if freqs[0] == 0.0:
            psd_karman[0] = 0.0
        # Calculate the RMS value for cross-checking. Exclude first frequency with f=0.0 from the integral.
        logging.info('RMS of PSD input (should approach 1.0): {:.4f}'.format(np.trapezoid(psd_karman[1:], freqs[1:]) ** 0.5))
        return psd_karman

    def calc_gust_excitation(self, freqs, t):
        # Calculate turbulence excitation by von Karman power spectral density according to CS-25.341(b)
        # For calculation of limit loads in the time domain, scale u_sigma such that it has the probability to occur once
        # during the simulation time.
        sigma = self.calc_sigma(self.n_freqs)
        u_sigma = self.u_sigma / sigma  # turbulence gust intensity [m/s]
        logging.info("Using RMS turbulence intensity u_sigma = {:.4f} m/s, sigma = {:.4f}.".format(u_sigma, sigma))

        psd_karman = self.calc_psd_vonKarman(freqs)
        # Apply a scaling in the frequency domain to achieve the correct amplitude in the time domain.
        # Then, CS-25 wants us to take the square root.
        psd_scaled = u_sigma * (psd_karman * len(freqs) * self.fmax) ** 0.5
        # generate a random phase
        # create a new seed for the random numbers, so that when using multiprocessing, every worker gets a new seed
        np.random.seed()
        random_phases = np.random.random_sample(len(freqs)) * 2.0 * np.pi

        # apply to all panels with phase delay according to geometrical position
        # time delay of every panel in [s]
        time_delay = self.aerogrid['offset_j'][:, 0] / self.Vtas
        # phase delay of every panel and frequency in [rad]
        phase_delay = -np.tile(time_delay, (len(freqs), 1)).T * 2.0 * np.pi * freqs
        # Ausrichtung der Boe fehlt noch
        gust_direction_vector = np.sum(self.aerogrid['N'] * np.dot(np.array([0, 0, 1]), solution_tools.calc_drehmatrix(
            self.simcase['gust_orientation'] / 180.0 * np.pi, 0.0, 0.0)), axis=1)
        # Notation: [n_panels, n_freq]
        wj_gust_f = psd_scaled * np.exp(1j * (random_phases + phase_delay)) * gust_direction_vector[:, None] / self.Vtas
        Ph_fourier, Pk_fourier = self.calc_P_fourier(freqs, wj_gust_f)

        return Ph_fourier, Pk_fourier

        """
        Checks:
        psd_karman = 2.0*L * (1.0+8.0/3.0*(1.339*L*2.0*np.pi*freqs)**2.0)/(1.0+(1.339*L*2.0*np.pi*freqs)**2.0)**(11.0/6.0)
        psd_dryden = 2.0*L * (1.0+3.0*(L*2.0*np.pi*freqs)**2.0)          /(1.0+(L*2.0*np.pi*freqs)**2.0)**(2.0)
        from matplotlib import pyplot as plt
        plt.figure()
        plt.loglog(freqs, psd_karman, label='von Karman')
        plt.loglog(freqs, psd_dryden, label='Dryden')
        plt.ylabel('PSD $[m^2/s^2/Hz]$ ??')
        plt.xlabel('Frequency [Hz]')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

        print('RMS of PSD input (must be close to u_sigma): {:.4f}'.format(np.trapezoid(psd_karman, freqs)**0.5))
        print('RMS in freq. dom: {:.4f}'.format(np.trapezoid(psd_scaled, freqs)**0.5))

        w_turb_f = psd_scaled*np.exp(1j*random_phases)
        print('The amplitude may not change after adding a random phase: {}'.format(np.allclose(np.abs(w_turb_f), psd_scaled)))

        # calculate mean and rms values
        fouriersamples_mirrored = self.mirror_fouriersamples_even(w_turb_f[None, :])

        print('--- Time signals ---')
        y = ifft(fouriersamples_mirrored)
        print('Mean: {:.4f}'.format(np.mean(y)))
        print('RMS: {:.4f}'.format(np.mean(np.abs(y)**2.0)**0.5))
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(self.t, y[0, :])
        plt.ylabel('$U_{turbulence} [m/s]$')
        plt.xlabel('Time [s]')
        plt.grid(True)
        plt.show()


        # compare frequency content
        y_fourier = fft(y)
        from matplotlib import pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
        ax1.scatter(self.fftomega, fouriersamples_mirrored[0, :].real, marker='s', label='original')
        ax1.scatter(self.fftomega, y_fourier[0, :].real, marker='.', label='freq -> time -> freq, no delay')
        ax2.scatter(self.fftomega, fouriersamples_mirrored[0, :].imag, marker='s', label='original')
        ax2.scatter(self.fftomega, y_fourier[0, :].imag, marker='.', label='freq -> time -> freq, no delay')
        ax1.set_ylabel('real')
        ax2.set_ylabel('imag')
        ax2.set_xlabel('freq')
        ax1.legend(loc='upper right')
        ax1.grid(True)
        ax2.grid(True)
        """


class LimitTurbulence(TurbulenceExcitation):

    def eval_equations(self):
        self.setup_frequence_parameters()

        logging.info('building transfer functions')
        self.build_AIC_interpolators()  # unsteady
        # Notation: [Antwort, Anregung, Frequenz]
        positiv_TFs = self.build_transfer_functions(self.positiv_fftfreqs)

        logging.info('calculating gust excitation (in physical coordinates, this may take considerable time and memory)')
        Ph_gust, Pk_gust = self.calc_white_noise_excitation(self.positiv_fftfreqs)

        psd_karman = self.calc_psd_vonKarman(self.positiv_fftfreqs)

        logging.info('calculating responses')
        Hdisp = np.sum(positiv_TFs * Ph_gust, axis=1)

        logging.info('reconstructing aerodynamic forces (in physical coordinates, this may take considerable time and memory)')
        # Aerodynamic forces due to the elastic reaction of the aircraft
        wj_aero = self.Djh_1.dot(Hdisp) + self.Djh_2.dot(Hdisp * (1j * self.positiv_fftomega) ** 1) / self.Vtas
        _, Pk_aero = self.calc_P_fourier(self.positiv_fftfreqs, wj_aero)
        Haero = self.PHIstrc_mon.T.dot(self.PHIk_strc.T.dot(Pk_aero))
        # Aerodynamic forces due to the gust / turbulence
        Hgust = self.PHIstrc_mon.T.dot(self.PHIk_strc.T.dot(Pk_gust))
        # Inertial forces due to the elastic reation of the aircraft
        Hiner = self.PHIstrc_mon.T.dot(-self.Mgg.dot(
            self.mass['PHIh_strc'].T).dot(Hdisp * (1j * self.positiv_fftomega) ** 2))

        # Force Summation Method: P = Pext + Piner
        H = Haero + Hgust + Hiner

        A = np.trapezoid(np.real(H * H.conj()) * psd_karman, self.positiv_fftfreqs) ** 0.5
        Pmon = self.u_sigma * A

        logging.info('calculating correlations')
        # Using H[:,None, :] * H.conj()[None, :, :] to calculate all coefficients at once would be nice but requires much
        # memory. Looping over all rows is more memory efficient and even slightly faster.
        # Once the integral is done, the matrix is much smaller.
        correlations = np.zeros((self.mongrid['n'] * 6, self.mongrid['n'] * 6))
        for i_row in range(6 * self.mongrid['n']):
            correlations[i_row, :] = np.trapezoid(np.real(H[i_row, :].conj() * H) * psd_karman, self.positiv_fftfreqs)
        correlations /= (A * A[:, None])

        response = {'Pmon_turb': np.expand_dims(Pmon, axis=0),
                    'correlations': correlations,
                    }
        return response

    def calc_white_noise_excitation(self, freqs):
        # white noise with constant amplitude for all frequencies
        white_noise = np.ones_like(freqs)
        # apply to all panels with phase delay according to geometrical position
        # time delay of every panel in [s]
        time_delay = self.aerogrid['offset_j'][:, 0] / self.Vtas
        # phase delay of every panel and frequency in [rad]
        phase_delay = -np.tile(time_delay, (len(freqs), 1)).T * 2.0 * np.pi * freqs
        # Ausrichtung der Boe fehlt noch
        gust_direction_vector = np.sum(self.aerogrid['N'] * np.dot(np.array([0, 0, 1]), solution_tools.calc_drehmatrix(
            self.simcase['gust_orientation'] / 180.0 * np.pi, 0.0, 0.0)), axis=1)
        # Notation: [n_panels, n_freq]
        wj_gust_f = white_noise * np.exp(1j * (phase_delay)) * gust_direction_vector[:, None] / self.Vtas
        Ph_fourier, Pk_fourier = self.calc_P_fourier(freqs, wj_gust_f)

        return Ph_fourier, Pk_fourier


class KMethod(GustExcitation):

    def eval_equations(self):
        self.setup_frequence_parameters()

        logging.info('building systems')
        self.build_AIC_interpolators()  # unsteady
        self.build_systems()
        logging.info('calculating eigenvalues')
        self.calc_eigenvalues()

        response = {'freqs': self.freqs,
                    'damping': self.damping,
                    'Vtas': self.Vtas,
                    }
        return response

    def setup_frequence_parameters(self):
        self.n_modes = self.model['mass'][self.trimcase['mass']]['n_modes'][()] + 5
        self.k_reds = self.simcase['flutter_para']['k_red']
        self.n_freqs = len(self.k_reds)

        if self.k_reds.max() > np.max(self.aero['k_red']):
            logging.warning('Required reduced frequency = {:0.3} but AICs given only up to {:0.3}'.format(
                self.k_reds.max(), np.max(self.aero['k_red'])))

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

        self.states = ["y'", "z'", "$\Phi'$", "$\Theta'$", "$\Psi'$", ]  # noqa: W605
        for i_mode in range(1, self.n_modes_f + 1):
            self.states += ['Uf' + str(i_mode)]
        self.states += ["v'", "w'", "p'", "q'", "r'"]
        for i_mode in range(1, self.n_modes_f + 1):
            self.states += ['$\mathrm{{ \dot Uf{} }}$'.format(str(i_mode))]  # noqa: W605

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
