'''
Created on Aug 5, 2019

@author: voss_ar
'''
import numpy as np
from scipy import linalg
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d 
import logging
from matplotlib import pyplot as plt

from loadskernel.trim_tools import * 
from loadskernel.model_equations.common import Common

class GustExcitation(Common):
    
    def eval_equations(self):
        self.Vtas, self.q_dyn = self.recover_Vtas(self.X0)
        # Number of sample points
        dt = self.simcase['dt']
        f = 1/dt
        self.n_freqs = int(self.simcase['t_final']*f)
        if self.n_freqs % 2 != 0: # n_freq is odd
            self.n_freqs += 1 # make even
        # sample spacing
        t = np.linspace(0.0, self.n_freqs*dt, self.n_freqs)
        freqs = np.linspace(0.0, f/2.0, self.n_freqs//2) # samples only from zero up to the Nyquist frequency
        fftfreqs = fftfreq(self.n_freqs, dt) # whole frequency space including negative frequencies
        fftomega = 2.0*np.pi*fftfreqs
        fourierfreqs = np.abs(fftfreqs[:self.n_freqs//2+1]) # positive only frequencies where we need to calculate the TFs and excitations
        

        if self.k_red(freqs.max()) > np.max(self.model.aero['k_red']):
            logging.warning('Required reduced frequency = {:0.3} but AICs given only up to {:0.3}'.format(self.k_red(freqs.max()), np.max(self.model.aero['k_red'])))
         
        self.build_AIC_interpolators() # unsteady
        fouriersamples = self.build_transfer_functions(fourierfreqs)
        TFs = np.zeros((self.n_modes, self.n_modes, self.n_freqs), dtype='complex128')
        for i_mode in range(self.n_modes):
            TFs[:,i_mode,:] = self.mirror_fouriersamples_even(fouriersamples[:,i_mode,:])
        logging.info('Transfer functions finished.')
        
        Pf_fouriersamples, Pk_fouriersamples = self.calc_gust_excitation(fourierfreqs, t)
        Pf_fourier = self.mirror_fouriersamples_even(Pf_fouriersamples)
        Pk_fourier = self.mirror_fouriersamples_even(Pk_fouriersamples)
        logging.info('Gust excitation finished.')

        Uf_fourier = TFs[:,:,:] * Pf_fourier # [Antwort, Anregung, Frequenz]
        Uf       = ifft( np.array((Uf_fourier)*(1j*fftomega)**0).sum(axis=1) )
        dUf_dt   = ifft( np.array((Uf_fourier)*(1j*fftomega)**1).sum(axis=1) )
        d2Uf_dt2 = ifft( np.array((Uf_fourier)*(1j*fftomega)**2).sum(axis=1) )
        Pk_aero  = ifft( Pk_fourier )
        g_cg = np.zeros((3,self.n_freqs))
        commands = np.zeros((self.trim.n_inputs, self.n_freqs))
        # set rigid body accelerations to zero as they are included in flexible modes
        dUcg_dt   = np.zeros((6,self.n_freqs))
        d2Ucg_dt2 = np.zeros((6,self.n_freqs))
        X = np.concatenate((Uf[:6,:],      # x, y, z, Phi, Theta, Psi
                            dUf_dt[:6,:],  # u, v, w, p, q, r
                            Uf[:,:],      # modal deformations
                            dUf_dt[:,:],  # modal velocities
                            commands,
                            ))
        response = {'X':X.T,
                    't': np.array([t]).T,
                    'Pk_aero': Pk_aero.T,
                    'dUcg_dt': dUcg_dt.T,
                    'd2Ucg_dt2': d2Ucg_dt2.T,
                    'Uf': Uf.T,
                    'dUf_dt': dUf_dt.T,
                    'd2Uf_dt2': d2Uf_dt2.T,
                    'g_cg': g_cg.T,
                    }
        return response  
        
        
#         plt.figure(2)
#         #plt.plot(freqs, 2.0/self.n_freqs * np.abs(Pfs_f[:,0:self.n_freqs//2]).T, label='Pf')
#         plt.plot(freqs, 2.0/self.n_freqs * np.abs(Pf_fourier[:,:self.n_freqs//2]).T, label='Pf')
#         plt.xlabel('f [Hz]')
#         plt.ylabel('Amplitude')
#         plt.yscale('log')
#         plt.legend()
#         plt.grid()
#         
#         plt.figure(1)
#         #plt.plot(t, uf.T, label='x freq. dom.')
#         plt.plot(t, uf.T, label='x freq. dom.')
#         plt.xlabel('t [s]')
#         plt.ylabel('Amplitude')
#         plt.grid()
#         plt.legend()
#         plt.show()


        

    def mirror_fouriersamples_even(self, fouriersamples):
        mirrored_fourier = np.zeros((fouriersamples.shape[0], self.n_freqs), dtype='complex128')
        mirrored_fourier[:,:self.n_freqs//2] = fouriersamples[:,:-1]
        mirrored_fourier[:,self.n_freqs//2:] = np.flip(fouriersamples[:,1:], axis=1).conj()
        return mirrored_fourier
            
    def build_transfer_functions(self, freqs):
        TFs = np.zeros((self.n_modes, self.n_modes, len(freqs)), dtype='complex128') # [Antwort, Anregung, Frequenz]
        for i_f in range(len(freqs)):
            TFs[:,:,i_f] = self.transfer_function(freqs[i_f], n=0)
        return TFs


    def transfer_function(self, f, n=0):
        omega = 2.0*np.pi*f
        Qff_1 = self.Qff_1_interp(self.k_red(f))
        Qff_2 = self.Qff_2_interp(self.k_red(f))
        TF = np.linalg.inv(-self.Mff*omega**2 + 1j*omega*(self.Dff + self.q_dyn*Qff_2) + self.Kff + self.q_dyn*Qff_1)*(1j*omega)**n
        return TF

    def build_AIC_interpolators(self):
        # interpolation of physical AIC
        self.Qjj_interp = interp1d( self.model.aero['k_red'], self.model.aero['Qjj_unsteady'][self.i_aero], axis=0, fill_value="extrapolate")
        # do some pre-multiplications first, then the interpolation
        Qff_1 = []; Qff_2 = []; Qfj = []
        for Qjj_unsteady in self.model.aero['Qjj_unsteady'][self.i_aero]:
            Qff_1.append( self.PHI_1.T.dot(self.model.aerogrid['Amat'].dot(Qjj_unsteady)).dot(self.PHI_1) )
            Qff_2.append( self.PHI_2.T.dot(self.model.aerogrid['Amat'].dot(Qjj_unsteady)).dot(self.PHI_2) )
            Qfj.append( self.PHIlf.T.dot(self.model.aerogrid['Nmat'].T.dot(self.model.aerogrid['Amat'].dot(Qjj_unsteady))) )
        self.Qff_1_interp = interp1d( self.model.aero['k_red'], Qff_1, axis=0, fill_value="extrapolate")
        self.Qff_2_interp = interp1d( self.model.aero['k_red'], Qff_2, axis=0, fill_value="extrapolate")
        self.Qfj_interp = interp1d( self.model.aero['k_red'], Qfj, axis=0, fill_value="extrapolate")    
    
    def calc_gust_excitation(self, freqs, t):
        # Notation: [n_panels, timesteps]
        wj_gust_f = fft(self.wj_gust(t)) # Eventuell muss wj_gust_f noch skaliert werden mit 2.0/N * np.abs(wj_gust_f[:,0:N//2])
        Pf_fourier = np.zeros((self.n_modes, len(freqs)), dtype='complex128')
        Pk_fourier = np.zeros((self.model.aerogrid['n']*6, len(freqs)), dtype='complex128')
        for i_f in range(len(freqs)):
            Pf_fourier[:,i_f] = self.calc_Pf_fourier(freqs[i_f], wj_gust_f[:,i_f])
            Pk_fourier[:,i_f] = self.calc_Pk_fourier(freqs[i_f], wj_gust_f[:,i_f])
        return Pf_fourier, Pk_fourier
    
    def calc_Pf_fourier(self,f, wj):
        Qfj = self.Qfj_interp(self.k_red(f))
        Pf = self.q_dyn * Qfj.dot(wj)
        return Pf
    
    def calc_Pk_fourier(self,f, wj):
        Qjj = self.Qjj_interp(self.k_red(f))
        Pk = self.q_dyn * self.model.Dlk.T.dot(self.model.aerogrid['Nmat'].T.dot(self.model.aerogrid['Amat'].dot(Qjj.dot(wj))))
        return Pk
    
    def wj_gust(self, t):
        ac_position = np.array([t * self.Vtas]*self.model.aerogrid['n'])
        panel_offset = np.array([self.model.aerogrid['offset_j'][:,0]]*t.__len__()).T
        s_gust = (ac_position - panel_offset - self.s0)
        # downwash der 1-cos Boe auf ein jedes Panel berechnen
        wj_gust = self.WG_TAS * 0.5 * (1-np.cos(np.pi * s_gust / self.simcase['gust_gradient']))
        wj_gust[np.where(s_gust <= 0.0)] = 0.0
        wj_gust[np.where(s_gust > 2*self.simcase['gust_gradient'])] = 0.0
        #wj_gust = self.WG_TAS * 0.5 * (1-np.cos(np.pi * s_gust / self.simcase['gust_gradient']))
        # Ausrichtung der Boe fehlt noch
        gust_direction_vector = np.sum(self.model.aerogrid['N'] * np.dot(np.array([0,0,1]), calc_drehmatrix( self.simcase['gust_orientation']/180.0*np.pi, 0.0, 0.0 )), axis=1)
        wj = wj_gust *  np.array([gust_direction_vector]*t.__len__()).T
        return wj
    
    def k_red(self, f):
        return 2.0*np.pi * f * self.model.macgrid['c_ref']/2.0 / self.Vtas
