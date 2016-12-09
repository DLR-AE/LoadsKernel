
import logging, copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
from scipy.optimize import curve_fit

class analysis:
    def __init__(self, jcl, model, responses):
        self.jcl = jcl
        self.model = model
        self.responses = responses
    
    
    def analyse_states(self, filename_pdf):
        pp = PdfPages(filename_pdf)
        
        # function to be fitted  
        def exp_function(t,A,decay,f,phi,offset):
            return A*np.exp(-decay*t)*(np.cos(2*np.pi*f*t+phi))+offset#+np.sin(omega*t+phi))+offset
        
        # bound of parameters for fitting
        bounds=([0.0    , -np.inf, 0.0    , -np.inf, -np.inf],
                [+np.inf, +np.inf, +np.inf, +np.inf, +np.inf])
        
        for i in range(len(self.responses)):
            # get states and state description
            rbm_states = [2, 4]
            rbm_desc = [ 'z', 'theta']
            modes = self.jcl.mass['modes'][self.jcl.mass['key'].index(self.jcl.trimcase[i]['mass'])]
            flex_desc = ['flex mode '+str(mode) for mode in modes]
            flex_states = range(12,12+len(modes))
            states = rbm_states + flex_states 
            desc = rbm_desc + flex_desc
            desc_estimate = copy.deepcopy(desc)
            
            # set up plotting and logging
            plt.figure()
            colors = iter(plt.cm.jet(np.linspace(0,1,len(states))))
            markers = itertools.cycle(('+', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'x', 'D', ))
            logging.info('Identified properties of states for {}:'.format(self.jcl.trimcase[i]['desc']))
            logging.info('--------------------------------------------------------------------------------------')
            logging.info('                     A          f [Hz]     decay')

            # loop over states
            for j in range(len(states)):
                plt.plot(self.responses[i]['t'], self.responses[i]['X'][:,states[j]], marker=next(markers), c=next(colors), linewidth=2.0)
                try:
                    start = int(len(self.responses[i]['t']) *0.5)
                    popt, pcov = curve_fit(exp_function, self.responses[i]['t'].reshape(-1)[start:], self.responses[i]['X'][start:,states[j]], p0=None, sigma=None, bounds=bounds )
                    tmp = '{:>20} {:< 10.4g} {:< 10.4g} {:< 10.4g}'.format(desc[j],popt[0], popt[2], popt[1] )
                    desc_estimate.insert(desc_estimate.index(desc[j])+1,'estimate')
                    plt.plot(self.responses[i]['t'][start:], exp_function(self.responses[i]['t'][start:], popt[0], popt[1], popt[2], popt[3], popt[4]), 'k--', linewidth=1.0)
                except:
                    tmp = '{:>20} No match.'.format(desc[j])
                logging.info(tmp)
                
            # finalize plotting and logging    
            logging.info('--------------------------------------------------------------------------------------')    
            plt.title(self.jcl.trimcase[i]['desc'])
            plt.grid('on')
            plt.legend(desc_estimate, loc='best', fontsize=8)
            pp.savefig()
            plt.close()
        pp.close()
    
        
        
        
    