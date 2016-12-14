
import logging, copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
from scipy.optimize import curve_fit
from scipy import signal

class analysis:
    def __init__(self, jcl, model, responses):
        self.jcl = jcl
        self.model = model
        self.responses = responses
    
    
    def analyse_states(self, filename_pdf):
        pp = PdfPages(filename_pdf)
        
        # function to be fitted  
        def oscillating_function(t,A,decay,f,phi,offset):
            return A * np.exp(-decay*t) * (np.cos(2*np.pi*f*t+phi)) + offset #+np.sin(omega*t+phi))+offset
        def exp_function(t,A,gamma):
            return A * np.exp(-gamma*t)
        
        # bound of parameters for fitting
        bounds_gr=([0.0, -np.inf],
                   [+np.inf, +np.inf])
        bounds_ls=([-np.inf, -np.inf],
                   [0.0, +np.inf])
        
        Fs_all = []
        Gammas_all = []
            
        for i in range(len(self.responses)):
            # get states and state description
            rbm_states = [2, 4]
            rbm_desc = [ 'z', 'theta']
            modes = self.jcl.mass['modes'][self.jcl.mass['key'].index(self.jcl.trimcase[i]['mass'])]
            flex_desc = ['flex mode '+str(mode) for mode in modes]
            flex_states = range(12,12+len(modes))
            states = rbm_states + flex_states 
            desc = rbm_desc + flex_desc

            # set up plotting and logging
            fig1 = plt.figure()
            ax1 = fig1.gca()
            fig2 = plt.figure()
            ax2 = fig2.gca()
            colors = iter(plt.cm.jet(np.linspace(0,1,len(states))))
            markers = itertools.cycle(('+', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'x', 'D', ))
            logging.info('Identified properties of states for {}:'.format(self.jcl.trimcase[i]['desc']))
            logging.info('--------------------------------------------------------------------------------------')
            logging.info('                     F [Hz]     gamma_gr   (fitted)     gamma_ls   (fitted)' )
            Fs = []
            Gammas = []
            
            # loop over states
            for j in range(len(states)):
                marker = next(markers)
                color = next(colors)
                
                start = np.where(self.responses[i]['t'] >= 0.0)[0][0] # use only part of signal
                t = self.responses[i]['t'].reshape(-1)[start:]
                state = signal.detrend(self.responses[i]['X'][start:,states[j]], type='linear',)
                n = len(t)
                
                # fft
                fourier = np.fft.fft(state)
                dt = self.jcl.simcase[i]['dt']
                df = 1/(n*dt)
                m = int(n/2) 
                freqs = [0+x*df for x in range(m)]
                amps = 2.0*np.abs(fourier[1:m])
                ax1.plot(freqs[1:], amps, marker=marker, c=color, label=desc[j])

                # fitting
                ax2.plot(t, state, marker=marker, markevery=10, c=color, linewidth=2.0, label=desc[j])
                extrema_gr = signal.argrelextrema(state, np.greater)[0]
                extrema_ls = signal.argrelextrema(state, np.less)[0]

                if extrema_gr.size >= 3 and extrema_ls.size >= 3 and (state[extrema_gr].max() - state[extrema_gr].min()) > 0.001:#self.jcl.simcase[i]['disturbance']:
                    #ax2.plot(t[extrema_gr], state[extrema_gr], 'rs')
                    #ax2.plot(t[extrema_ls], state[extrema_ls], 'ro')

                    # Logarithmic decrement: log_dec = 1/n * ln(u(ti)/u(ti+n)) mit n = 1,2,...
                    # a) erster bis letzter Wert
                    log_dec_gr = 1.0/len(extrema_gr) * np.log(np.abs(state[extrema_gr[0]]/state[extrema_gr[-1]]))
                    log_dec_ls = 1.0/len(extrema_ls) * np.log(np.abs(state[extrema_ls[0]]/state[extrema_ls[-1]]))
                    # b) schrittweise auswerten, dann Mittelwert bilden
                    # log_dec_gr =  np.mean([np.log(np.abs(state[extrema_gr[x]]/state[extrema_gr[x+1]])) for x in range(len(extrema_gr)-1)])
                    # log_dec_ls =  np.mean([np.log(np.abs(state[extrema_ls[x]]/state[extrema_ls[x+1]])) for x in range(len(extrema_ls)-1)])
                    
                    # Damping: log_dec = omega0 * D * T, gamma = omega0 * D
                    T = np.mean(np.concatenate((np.diff(t[extrema_gr]), np.diff(t[extrema_ls]))))
                    F = 1.0/T
                    gamma_gr = log_dec_gr / T
                    gamma_ls = log_dec_ls / T
                    
                    # Einschwindungsvorgaenge sorgen fuer eine leicht unterschiedliche Daempfung. Diese wird durch ein Fitting durch die Extrema besser erfasst.
                    # Ausserdem ist das Fitting eine gute Visualisierung.
                    p0 = [0.0, np.mean([gamma_gr, gamma_ls])]
                    popt_gr, pcov = curve_fit(exp_function, t[extrema_gr], state[extrema_gr], p0=p0, sigma=None, bounds=bounds_gr)
                    ax2.plot(t, exp_function(t, popt_gr[0], popt_gr[1]), 'k--', linewidth=1.0)
                    popt_ls, pcov = curve_fit(exp_function, t[extrema_ls], state[extrema_ls], p0=p0, sigma=None, bounds=bounds_ls )
                    ax2.plot(t, exp_function(t, popt_ls[0], popt_ls[1]), 'k--', linewidth=1.0)
                    
                    tmp = '{:>20} {:< 10.4g} {:< 10.4g} ({:< 10.4g}) {:< 10.4g} ({:< 10.4g})'.format(desc[j], F, gamma_gr, popt_gr[1], gamma_ls, popt_ls[1] )
                    Gamma = np.mean([popt_gr[1], popt_ls[1]])
                    
#                     # oscillating function
#                      bounds=([0.0    , -np.inf, 0.0    , -np.inf, -np.inf],
#                              [+np.inf, +np.inf, +np.inf, +np.inf, +np.inf])
#                     p0 = [1.0, gamma, F, 0.0, 0.0]
#                     popt, pcov = curve_fit(oscillating_function, t, state, p0=p0, sigma=None, bounds=bounds )
#                     ax2.plot(t, oscillating_function(t, popt[0], popt[1], popt[2], popt[3], popt[4]), 'k--', linewidth=1.0)
#                     tmp = '{:>20} {:< 10.4g} {:< 10.4g} {:< 10.4g}'.format(desc[j],popt[0], popt[2], popt[1] )
                
                else:
                    tmp = '{:>20} No oscillation detected.'.format(desc[j])    
                    F = np.NaN
                    Gamma = np.NaN
                    
                logging.info(tmp)
                Fs.append(F)
                Gammas.append(Gamma)
            
            # outer loop: finalize plotting and logging      
            Fs_all.append(Fs)
            Gammas_all.append(Gammas)      
            logging.info('--------------------------------------------------------------------------------------')    
            ax1.set_title(self.jcl.trimcase[i]['desc'])
            ax1.set_xlabel('f [Hz]')
            ax1.set_ylabel('Amplitude')
            ax2.set_xlabel('t [s]')
            ax2.set_ylabel('Amplitude of State')
            ax2.set_title(self.jcl.trimcase[i]['desc'])
            ax1.grid(b=True, which='both', axis='both')
            ax2.grid(b=True, which='both', axis='both')
            ax1.legend(loc='best', fontsize=10)
            ax2.legend(loc='best', fontsize=10)
            pp.savefig(fig1)
            pp.savefig(fig2)
            plt.close(fig1)
            plt.close(fig2)
        
        # plot frequences
        fig = plt.figure()
        ax = fig.gca()
        Fs_all = np.array(Fs_all)
        Ma = np.array([trimcase['Ma'] for trimcase in self.jcl.trimcase])
        colors = iter(plt.cm.jet(np.linspace(0,1,Fs_all.shape[1])))
        markers = itertools.cycle(('+', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'x', 'D', ))
        for j in range(Fs_all.shape[1]): 
            pos = ~np.isnan(Fs_all[:,j])
            marker = next(markers)
            color = next(colors)
            if pos.any():
                ax.plot(Ma[pos], Fs_all[pos,j], marker=marker, c=color, linewidth=2.0, label=desc[j] )
        ax.set_xlabel('Ma')
        ax.set_ylabel('f [Hz]')
        ax.grid(b=True, which='both', axis='both')
        ax.legend(loc='best', fontsize=10)
        pp.savefig(fig)
        
        # plot damping
        fig = plt.figure()
        ax = fig.gca()
        Gammas_all = np.array(Gammas_all)
        Ma = np.array([trimcase['Ma'] for trimcase in self.jcl.trimcase])
        colors = iter(plt.cm.jet(np.linspace(0,1,Fs_all.shape[1])))
        markers = itertools.cycle(('+', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'x', 'D', ))
        for j in range(Gammas_all.shape[1]): 
            pos = ~np.isnan(Gammas_all[:,j])
            marker = next(markers)
            color = next(colors)
            if pos.any():
                ax.plot(Ma[pos], Gammas_all[pos,j], marker=marker, c=color, linewidth=2.0, label=desc[j] )
        ax.set_xlabel('Ma')
        ax.set_ylabel('Gamma (damping)')
        ax.grid(b=True, which='both', axis='both')
        ax.legend(loc='best', fontsize=10)
        pp.savefig(fig)
        plt.close(fig)
        
        pp.close()
        #plt.show()
        
    
        
        
        
    