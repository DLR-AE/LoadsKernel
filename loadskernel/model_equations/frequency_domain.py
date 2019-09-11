'''
Created on Aug 5, 2019

@author: voss_ar
'''
import numpy as np
from scipy import linalg
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d 
import logging, itertools, copy

from loadskernel.trim_tools import * 
from loadskernel.build_mass import BuildMass
from loadskernel.model_equations.common import Common

class GustExcitation(Common):
    
    def eval_equations(self):
        self.setup_frequence_parameters()
        
        logging.info('building transfer functions') 
        self.build_AIC_interpolators() # unsteady
        positiv_TFs = self.build_transfer_functions(self.positiv_fftfreqs)
        TFs = np.zeros((self.n_modes, self.n_modes, self.n_freqs), dtype='complex128')
        for i_mode in range(self.n_modes):
            TFs[:,i_mode,:] = self.mirror_fouriersamples_even(positiv_TFs[:,i_mode,:])
        
        logging.info('calculating gust excitation')
        Ph_gust_fourier, Pk_gust_fourier = self.calc_gust_excitation(self.positiv_fftfreqs, self.t)
        Ph_gust_fourier = self.mirror_fouriersamples_even(Ph_gust_fourier)
        Pk_gust_fourier = self.mirror_fouriersamples_even(Pk_gust_fourier)
        
        logging.info('calculating responses')
        Uh_fourier = TFs * Ph_gust_fourier # [Antwort, Anregung, Frequenz]
        Uh       = ifft( np.array((Uh_fourier)*(1j*self.fftomega)**0).sum(axis=1) )
        dUh_dt   = ifft( np.array((Uh_fourier)*(1j*self.fftomega)**1).sum(axis=1) )
        d2Uh_dt2 = ifft( np.array((Uh_fourier)*(1j*self.fftomega)**2).sum(axis=1) )
        
        logging.info('reconstructing aerodynamic forces in physical coordinates')
        Ph_aero_fourier, Pk_aero_fourier = self.calc_aero_response(self.positiv_fftfreqs,
                                                                   np.array((Uh_fourier)*(1j*self.fftomega)**0).sum(axis=1)[:,:self.n_freqs//2+1], 
                                                                   np.array((Uh_fourier)*(1j*self.fftomega)**1).sum(axis=1)[:,:self.n_freqs//2+1], )
        Ph_aero_fourier = self.mirror_fouriersamples_even(Ph_aero_fourier)
        Pk_aero_fourier = self.mirror_fouriersamples_even(Pk_aero_fourier)
        Pk_aero  = np.real(ifft( Pk_gust_fourier ) + ifft( Pk_aero_fourier ))[:,self.t_output]
        Pk_gust  = np.real(ifft( Pk_gust_fourier ))[:,self.t_output]
        
        # split h-set into b- and f-set
        # remember that the x-component was omitted
        Ucg       = np.concatenate((np.zeros((len(self.t_output),1)), Uh[:5,self.t_output].T.real - Uh[:5,0].real), axis=1)
        dUcg_dt   = np.concatenate((np.zeros((len(self.t_output),1)), dUh_dt[:5,self.t_output].T.real - dUh_dt[:5,0].real), axis=1)
        d2Ucg_dt2 = np.concatenate((np.zeros((len(self.t_output),1)), d2Uh_dt2[:5,self.t_output].T.real - d2Uh_dt2[:5,0].real), axis=1)
        Uf        = Uh[5:,self.t_output].T.real - Uh[5:,0].real
        dUf_dt    = dUh_dt[5:,self.t_output].T.real - dUh_dt[5:,0].real
        d2Uf_dt2  = d2Uh_dt2[5:,self.t_output].T.real - d2Uh_dt2[5:,0].real
        
        g_cg = np.zeros((len(self.t_output), 3))
        commands = np.zeros((len(self.t_output), self.trim.n_inputs))
        
        X = np.concatenate((Ucg * np.array([-1.,1.,-1.,-1.,1.,-1.]),     # in DIN 9300 body fixed system for flight physics,  x, y, z, Phi, Theta, Psi
                            dUcg_dt * np.array([-1.,1.,-1.,-1.,1.,-1.]), # in DIN 9300 body fixed system for flight physics,  u, v, w, p, q, r
                            Uf,      # modal deformations
                            dUf_dt,  # modal velocities
                            commands,
                            ), axis=1)
        response = {'X':X,
                    't': np.array([self.t[self.t_output]]).T,
                    'Pk_aero': Pk_aero.T,
                    'Pk_gust': Pk_gust.T,
                    'Pk_unsteady':Pk_aero.T*0.0,
                    'dUcg_dt': dUcg_dt,
                    'd2Ucg_dt2': d2Ucg_dt2,
                    'Uf': Uf,
                    'dUf_dt': dUf_dt,
                    'd2Uf_dt2': d2Uf_dt2,
                    'g_cg': g_cg,
                    }
        return response  
    
    def setup_frequence_parameters(self):
        self.n_modes = self.model.mass['n_modes'][self.i_mass] + 5
        self.Vtas, self.q_dyn = self.recover_Vtas(self.X0)
        # Number of sample points
        t_factor = 10.0 # increase resolution (df) by extending simulation time
        dt = self.simcase['dt']
        fmax = 1/dt
        self.n_freqs = int(t_factor*self.simcase['t_final']*fmax)
        if self.n_freqs % 2 != 0: # n_freq is odd
            self.n_freqs += 1 # make even
        # sample spacing
        self.t = np.linspace(0.0, self.n_freqs*dt, self.n_freqs)
        self.t_output = np.where(self.t<=self.simcase['t_final'])[0] # indices of time samples to returned for post-processing
        self.freqs = np.linspace(0.0, fmax/2.0, self.n_freqs//2) # samples only from zero up to the Nyquist frequency
        fftfreqs = fftfreq(self.n_freqs, dt) # whole frequency space including negative frequencies
        self.fftomega = 2.0*np.pi*fftfreqs
        self.positiv_fftfreqs = np.abs(fftfreqs[:self.n_freqs//2+1]) # positive only frequencies where we need to calculate the TFs and excitations
        
        logging.info('Frequency domain solution with tfinal = {}x{} s, nfreq = {}, fmax={} Hz and df = {} Hz'.format(t_factor, self.simcase['t_final'], self.n_freqs//2, fmax/2.0, fmax/self.n_freqs) )
        if self.f2k(self.freqs.max()) > np.max(self.model.aero['k_red']):
            logging.warning('Required reduced frequency = {:0.3} but AICs given only up to {:0.3}'.format(self.f2k(self.freqs.max()), np.max(self.model.aero['k_red'])))

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
        Qhh_1 = self.Qhh_1_interp(self.f2k(f))
        Qhh_2 = self.Qhh_2_interp(self.f2k(f))
        TF = np.linalg.inv(-self.Mhh*omega**2 + np.complex(0,1)*omega*(self.Dhh - Qhh_2) + self.Khh - Qhh_1)
        return TF

    def build_AIC_interpolators(self):
        # interpolation of physical AIC
        self.Qjj_interp = interp1d( self.model.aero['k_red'], self.model.aero['Qjj_unsteady'][self.i_aero], axis=0, fill_value="extrapolate")
        # do some pre-multiplications first, then the interpolation
        Qhh_1 = []; Qhh_2 = []
        for Qjj_unsteady in self.model.aero['Qjj_unsteady'][self.i_aero]:
            Qhh_1.append(self.q_dyn * self.PHIlh.T.dot(self.model.aerogrid['Nmat'].T.dot(self.model.aerogrid['Amat'].dot(Qjj_unsteady).dot(self.Djh_1))) )
            Qhh_2.append(self.q_dyn * self.PHIlh.T.dot(self.model.aerogrid['Nmat'].T.dot(self.model.aerogrid['Amat'].dot(Qjj_unsteady).dot(self.Djh_2 / self.Vtas ))) )
        self.Qhh_1_interp = interp1d( self.model.aero['k_red'], Qhh_1, axis=0, fill_value="extrapolate")
        self.Qhh_2_interp = interp1d( self.model.aero['k_red'], Qhh_2, axis=0, fill_value="extrapolate")
    
    def calc_aero_response(self, freqs, Uh, dUh_dt):
        # Notation: [n_panels, timesteps]
        wj_fourier = self.Djh_1.dot(Uh) + self.Djh_2.dot(dUh_dt) / self.Vtas
        Ph_fourier = np.zeros((self.n_modes, len(freqs)), dtype='complex128')
        Pk_fourier = np.zeros((self.model.aerogrid['n']*6, len(freqs)), dtype='complex128')
        for i_f in range(len(freqs)):
            Ph_fourier[:,i_f], Pk_fourier[:,i_f] = self.calc_P_fourier(freqs[i_f], wj_fourier[:,i_f])
        return Ph_fourier, Pk_fourier
    
    def calc_gust_excitation(self, freqs, t):
        # Notation: [n_panels, timesteps]
        wj_gust_f = fft(self.wj_gust(t)) # Eventuell muss wj_gust_f noch skaliert werden mit 2.0/N * np.abs(wj_gust_f[:,0:N//2])
        Ph_fourier = np.zeros((self.n_modes, len(freqs)), dtype='complex128')
        Pk_fourier = np.zeros((self.model.aerogrid['n']*6, len(freqs)), dtype='complex128')
        for i_f in range(len(freqs)):
            Ph_fourier[:,i_f], Pk_fourier[:,i_f] = self.calc_P_fourier(freqs[i_f], wj_gust_f[:,i_f])
        return Ph_fourier, Pk_fourier
    
    def calc_P_fourier(self,f, wj):
        Qjj = self.Qjj_interp(self.f2k(f))
        Pk = self.q_dyn * self.model.PHIlk.T.dot(self.model.aerogrid['Nmat'].T.dot(self.model.aerogrid['Amat'].dot(Qjj.dot(wj))))
        Ph = self.PHIkh.T.dot(Pk)
        return Ph, Pk
    
    def wj_gust(self, t):
        ac_position = np.array([t * self.Vtas]*self.model.aerogrid['n'])
        panel_offset = np.array([self.model.aerogrid['offset_j'][:,0]]*t.__len__()).T
        s_gust = (ac_position - panel_offset - self.s0)
        # downwash der 1-cos Boe auf ein jedes Panel berechnen
        wj_gust = self.WG_TAS * 0.5 * (1-np.cos(np.pi * s_gust / self.simcase['gust_gradient']))
        wj_gust[np.where(s_gust <= 0.0)] = 0.0
        wj_gust[np.where(s_gust > 2*self.simcase['gust_gradient'])] = 0.0
        # Ausrichtung der Boe fehlt noch
        gust_direction_vector = np.sum(self.model.aerogrid['N'] * np.dot(np.array([0,0,1]), calc_drehmatrix( self.simcase['gust_orientation']/180.0*np.pi, 0.0, 0.0 )), axis=1)
        wj = wj_gust *  np.array([gust_direction_vector]*t.__len__()).T
        return wj
    
    def f2k(self, f):
        return 2.0*np.pi * f * self.model.macgrid['c_ref']/2.0 / self.Vtas
    
    def k2f(self, k_red):
        return k_red * self.Vtas / np.pi / self.model.macgrid['c_ref']

class KMethod(GustExcitation):
    
    def eval_equations(self):
        self.setup_frequence_parameters()
        
        logging.info('building systems') 
        self.build_AIC_interpolators() # unsteady
        self.build_systems()
        logging.info('calculating eigenvalues')
        self.calc_eigenvalues()
        
        response = {'freqs':self.freqs,
                    'damping':self.damping,
                    'Vtas':self.Vtas,
                   }
        return response  

                        
    def setup_frequence_parameters(self):
        self.n_modes = self.model.mass['n_modes'][self.i_mass] + 5
        self.k_reds = self.simcase['flutter_para']['k_red']
        self.n_freqs = len(self.k_reds)
                
        if self.k_reds.max() > np.max(self.model.aero['k_red']):
            logging.warning('Required reduced frequency = {:0.3} but AICs given only up to {:0.3}'.format(self.k_reds.max(), np.max(self.model.aero['k_red'])))
    
    def build_AIC_interpolators(self):
        Qhh = []
        for Qjj_unsteady, k_red in zip(self.model.aero['Qjj_unsteady'][self.i_aero], self.model.aero['k_red']):
            Qhh.append(self.PHIlh.T.dot(self.model.aerogrid['Nmat'].T.dot(self.model.aerogrid['Amat'].dot(Qjj_unsteady).dot(self.Djh_1 + np.complex(0,1)*k_red/(self.model.macgrid['c_ref']/2.0)*self.Djh_2))) )
        self.Qhh_interp = interp1d( self.model.aero['k_red'], Qhh, kind='cubic', axis=0, fill_value="extrapolate")
        
    def build_systems(self):
        self.A = np.zeros((self.n_modes, self.n_modes, self.n_freqs), dtype='complex128') # [Antwort, Anregung, Frequenz]
        self.B = np.zeros((self.n_modes, self.n_modes, self.n_freqs), dtype='complex128') # [Antwort, Anregung, Frequenz]
        for i_f in range(self.n_freqs):
            self.A[:,:,i_f], self.B[:,:,i_f] = self.system(self.k_reds[i_f])
    
    def system(self, k_red):
        rho = self.model.atmo['rho'][self.i_atmo]
        Qhh = self.Qhh_interp(k_red)
        # Schwochow equation (7.10)
        A = -self.Mhh - rho/2.0*(self.model.macgrid['c_ref']/2.0/k_red)**2.0*Qhh
        B = -self.Khh
        return A, B
    
    def calc_eigenvalues(self):
        eigenvalues = []; eigenvectors = []; freqs = []; damping = []; Vtas = []
        eigenvalue, eigenvector = linalg.eig(self.A[:,:,0], self.B[:,:,0])
        # sorting
        idx_pos = np.where(eigenvalue.real >= 0.0)[0]  # nur oszillierende Eigenbewegungen
        idx_sort = np.argsort(1.0/eigenvalue.real[idx_pos]**0.5)  # sort result by eigenvalue
        
        eigenvalue = eigenvalue[idx_pos][idx_sort]
        eigenvector = eigenvector[:, idx_pos][:, idx_sort]
        # store results
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        
        # calculate frequencies and damping ratios
        freqs.append(1.0/eigenvalue.real**0.5 / 2.0/np.pi)  
        damping.append(eigenvalue.imag/eigenvalue.real) 
        Vtas.append(self.model.macgrid['c_ref']/2.0/self.k_reds[0]/eigenvalue.real**0.5)
        
        for i_f in range(1, self.n_freqs):
            eigenvalue, eigenvector = linalg.eig(self.A[:,:,i_f], self.B[:,:,i_f])
            # sorting
            idx_pos = np.where(eigenvalue.real >= 0.0)[0]
            MAC = BuildMass.calc_MAC(BuildMass, eigenvectors[-1], eigenvector[:, idx_pos], plot=False)
            idx_sort = [MAC[x, :].argmax() for x in range(MAC.shape[0])]
            
            eigenvalue = eigenvalue[idx_pos][idx_sort]
            eigenvector = eigenvector[:, idx_pos][:, idx_sort]
            # store results
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
            freqs.append(1.0/eigenvalue.real**0.5 / 2.0/np.pi)
            damping.append(eigenvalue.imag/eigenvalue.real) 
            Vtas.append(self.model.macgrid['c_ref']/2.0/self.k_reds[i_f]/eigenvalue.real**0.5)

        self.freqs = np.array(freqs)
        self.damping = np.array(damping)
        self.Vtas = np.array(Vtas)
    
    
class KEMethod(KMethod):
    
    def system(self, k_red):
        rho = self.model.atmo['rho'][self.i_atmo]
        Qhh = self.Qhh_interp(k_red)
        # Nastran equation (2-120)
        A = self.Khh
        B = (k_red/self.model.macgrid['c_ref']*2.0)**2.0 * self.Mhh + rho/2.0 * Qhh
        return A, B
    
    def calc_eigenvalues(self):
        eigenvalues = []; eigenvectors = []; freqs = []; damping = []; Vtas = []
        eigenvalue, eigenvector = linalg.eig(self.A[:,:,0], self.B[:,:,0])
        # sorting
        idx_pos = range(self.n_modes)
        V = ((eigenvalue.real**2 + eigenvalue.imag**2) / eigenvalue.real)**0.5
        freq = self.k_reds[0]*V/np.pi/self.model.macgrid['c_ref']
        idx_sort = np.argsort(freq)
        
        eigenvalue = eigenvalue[idx_pos][idx_sort]
        eigenvector = eigenvector[:, idx_pos][:, idx_sort]
        # store results
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        freqs.append(freq[idx_sort])
        Vtas.append(V[idx_sort])
        damping.append(np.imag(V[idx_sort]**2 / eigenvalue))
        
        for i_f in range(1, self.n_freqs):
            eigenvalue, eigenvector = linalg.eig(self.A[:,:,i_f], self.B[:,:,i_f])
            # sorting
            if i_f >=2:
                pes = eigenvalues[-1] + (self.k_reds[i_f]-self.k_reds[i_f-1])*(eigenvalues[-1] - eigenvalues[-2])/(self.k_reds[i_f-1]-self.k_reds[i_f-2])
                idx_sort = []
                for pe in pes:
                    idx_sort.append(((pe.real-eigenvalue.real)**2 + (pe.imag-eigenvalue.imag)**2).argmin())
            
            eigenvalue = eigenvalue[idx_pos][idx_sort]
            eigenvector = eigenvector[:, idx_pos][:, idx_sort]
            # store results
            eigenvalues.append(eigenvalue)
            eigenvectors.append(eigenvector)
            V = ((eigenvalue.real**2 + eigenvalue.imag**2) / eigenvalue.real)**0.5
            freq = self.k_reds[i_f]*V/np.pi/self.model.macgrid['c_ref']
            freqs.append(freq)
            Vtas.append(V)
            damping.append(-eigenvalue.imag/eigenvalue.real) 
            
        self.freqs = np.array(freqs)
        self.damping = np.array(damping)
        self.Vtas = np.array(Vtas)

class PKMethod(KMethod):
    
    def setup_frequence_parameters(self):
        self.n_modes = self.model.mass['n_modes'][self.i_mass] + 5
        self.Vvec = self.simcase['flutter_para']['Vtas']        
        
    def eval_equations(self):
        self.setup_frequence_parameters()
        
        logging.info('building systems') 
        self.build_AIC_interpolators() # unsteady
        logging.info('starting iterations for {} modes to match k_red with Vtas and omega'.format(self.n_modes)) 
        # compute initial guess at k_red=0.0 and first flight speed
        self.Vtas = self.Vvec[0]
        eigenvalue, eigenvector = linalg.eig(self.system(k_red=0.0))
        idx_pos = np.where(eigenvalue.imag > 0.0)[0]  # nur oszillierende Eigenbewegungen
        idx_sort = np.argsort(np.abs(eigenvalue.imag[idx_pos]))  # sort result by eigenvalue
        eigenvalues0 = eigenvalue[idx_pos][idx_sort]
        eigenvectors0 = eigenvector[:, idx_pos][:, idx_sort]
        k0 = eigenvalues0.imag*self.model.macgrid['c_ref']/2.0/self.Vtas
    
        eigenvalues = []; eigenvectors = []; freqs = []; damping = []; Vtas = []
        # loop over modes
        for i_mode in range(self.n_modes):
            logging.debug('Mode {}'.format(i_mode+1))
            eigenvalues_i = []; eigenvectors_i = []
            k_old = copy.deepcopy(k0[i_mode])
            eigenvectors_old = copy.deepcopy(eigenvectors0)
            # loop over Vtas
            for i_V in range(len(self.Vvec)):
                self.Vtas = self.Vvec[i_V]
                e = 1.0; n_iter = 0
                # iteration to match k_red with Vtas and omega of the mode under investigation
                while e >= 1e-4:
                    eigenvalues_new, eigenvectors_new = self.calc_eigenvalues(self.system(k_old), eigenvectors_old)
                    k_new = eigenvalues_new[i_mode].imag*self.model.macgrid['c_ref']/2.0/self.Vtas
                    e = np.abs(k_new - k_old)
                    k_old = k_new
                    n_iter += 1
                    if n_iter > 100:
                        logging.warning('PK-Iteration did NOT converge for mode {} at Vtas={} with k_red={}. The residual k_red is e={}'.format(i_mode+1, self.Vvec[i_V], k_new, e))
                        break
                eigenvectors_old = eigenvectors_new
                eigenvalues_i.append(eigenvalues_new[i_mode])
            # store 
            eigenvalues_i = np.array(eigenvalues_i)
            eigenvalues.append(eigenvalues_i)
            freqs.append(eigenvalues_i.imag /2.0/np.pi)
            #damping.append(eigenvalues_i.real / np.abs(eigenvalues_i))
            damping.append(2.0 * eigenvalues_i.real / eigenvalues_i.imag)
            Vtas.append(self.Vvec)
            
        response = {'freqs':np.array(freqs).T,
                    'damping':np.array(damping).T,
                    'Vtas':np.array(Vtas).T,
                   }
        return response    
            
    def calc_eigenvalues(self, A, eigenvector_old):
        eigenvalue, eigenvector = linalg.eig(A)
        idx_pos = np.where(eigenvalue.imag >= 0.0)[0]  # nur oszillierende Eigenbewegungen
        #idx_sort = np.argsort(np.abs(eigenvalue.imag[idx_pos]))  # sort result by eigenvalue
        MAC = BuildMass.calc_MAC(BuildMass, eigenvector_old, eigenvector[:, idx_pos], plot=False)
        idx_sort = [MAC[x, :].argmax() for x in range(MAC.shape[0])]
        eigenvalues = eigenvalue[idx_pos][idx_sort]
        eigenvectors = eigenvector[:, idx_pos][:, idx_sort]
        return eigenvalues, eigenvectors

    def build_AIC_interpolators(self):
        # do some pre-multiplications first, then the interpolation
        Qhh_1 = []; Qhh_2 = []
        for Qjj_unsteady in self.model.aero['Qjj_unsteady'][self.i_aero]:
            Qhh_1.append(self.PHIlh.T.dot(self.model.aerogrid['Nmat'].T.dot(self.model.aerogrid['Amat'].dot(Qjj_unsteady).dot(self.Djh_1))) )
            Qhh_2.append(self.PHIlh.T.dot(self.model.aerogrid['Nmat'].T.dot(self.model.aerogrid['Amat'].dot(Qjj_unsteady).dot(self.Djh_2))) )
        self.Qhh_1_interp = interp1d( self.model.aero['k_red'], Qhh_1, kind='slinear', axis=0, fill_value="extrapolate")
        self.Qhh_2_interp = interp1d( self.model.aero['k_red'], Qhh_2, kind='slinear', axis=0, fill_value="extrapolate")
    
    def system(self, k_red):
        rho = self.model.atmo['rho'][self.i_atmo]
        
        Qhh_1 = self.Qhh_1_interp(k_red)
        Qhh_2 = self.Qhh_2_interp(k_red)
        Mhh_inv = np.linalg.inv(self.Mhh)
        
        upper_part = np.concatenate((np.zeros((self.n_modes, self.n_modes), dtype='complex128'), np.eye(self.n_modes, dtype='complex128')))
        lower_part = np.concatenate(( -Mhh_inv.dot(self.Khh - rho/2.0*self.Vtas**2.0*Qhh_1), -Mhh_inv.dot(self.Dhh - rho/2.0*self.Vtas*Qhh_2)))
        A = np.concatenate((upper_part, lower_part), axis=1)
        
        return A 
