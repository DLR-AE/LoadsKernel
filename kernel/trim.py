# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:35:22 2014

@author: voss_ar
"""
import numpy as np
import scipy.optimize as so

class trim:
    def __init__(self,model, trimcase):
        self.model = model
        self.trimcase = trimcase
            
    def set_trimcond(self):
        i_atmo = self.model.atmo['key'].index(self.trimcase['altitude'])
        n_modes = len(self.model.jcl.mass['modes'])
        
        
        if self.trimcase['manoeuver'] in ['PU', 'PD']:
            # inputs
            self.trimcond_X = np.array([
                ['x',        'fix',  0.0,],
                ['y',        'fix',  0.0,],
                ['z',        'fix',  -self.model.atmo['h'][i_atmo],],
                ['phi',      'fix',  0.0,],
                ['theta',    'free', 3.0/180*np.pi,],
                ['psi',      'fix',  0.0,],
                ['u',        'fix',  self.trimcase['Ma'] * self.model.atmo['a'][i_atmo],],
                ['v',        'fix',  0.0,],
                ['w',        'fix',  0.0,],
                ['p',        'fix',  0.0,],
                ['q',        'fix', 0.0,],
                ['r',        'fix',  0.0,],
                ])
            for i_mode in range(n_modes):
                self.trimcond_X = np.vstack((self.trimcond_X ,  ['Uf'+str(i_mode), 'free', 0.0]))
            for i_mode in range(n_modes):
                self.trimcond_X = np.vstack((self.trimcond_X ,  ['dUf_dt'+str(i_mode), 'free', 0.0]))
                
            self.trimcond_X = np.vstack((self.trimcond_X , ['AIL-S1',   'free', -10.0/180*np.pi,],  ['AIL-S2',   'fix',  0.0,], ['AIL-S3',   'fix',  0.0,], ['AIL-S4',   'fix',  0.0,]))
                
            # outputs
            self.trimcond_Y = np.array([ 
                ['u',        'fix',  self.trimcase['Ma'] * self.model.atmo['a'][i_atmo],],
                ['v',        'fix',  0.0,],
                ['w',        'fix',  0.0,],
                ['p',        'fix',  0.0,],
                ['q',        'fix', 0.0,],
                ['r',        'fix',  0.0,],
                ['du',       'free',  0.0,],
                ['dv',       'fix',  0.0,],
                ['dw',       'free', 0.0,],
                ['dp',       'fix',  0.0,],
                ['dq',       'target', 0.0,],
                ['dr',       'fix',  0.0,],
                ])
                
            for i_mode in range(n_modes):
                self.trimcond_Y = np.vstack((self.trimcond_Y ,  ['dUf_dt'+str(i_mode), 'target', 0.0]))
            for i_mode in range(n_modes):
                self.trimcond_Y = np.vstack((self.trimcond_Y ,  ['d2Uf_d2t'+str(i_mode), 'target', 0.0]))
            self.trimcond_Y = np.vstack((self.trimcond_Y , ['Nz',       'target',  self.trimcase['Nz'],]))

            #alpha0 = 11.5987/180*np.pi                        
            #ails10 = -10.1889/180*np.pi
            #ails20 = -6.2139/180*np.pi  
        else:
           print 'Trimconditions not implemented for trimcase: ' + self.trimcase['manoeuver']
           
    
    def exec_trim(self):
        # The purpose of HYBRD is to find a zero of a system of N non-
        # linear functions in N variables by a modification of the Powell
        # hybrid method.  The user must provide a subroutine which calcu-
        # lates the functions.  The Jacobian is then calculated by a for-
        # ward-difference approximation.
        # http://www.math.utah.edu/software/minpack/minpack/hybrd.html
    
                
        import model_equations # Warum muss der import hier stehen??
        model_equations = model_equations.rigid(self.model, self.trimcase, self.trimcond_X, self.trimcond_Y)
        X_free_0 = np.array(self.trimcond_X[:,2], dtype='float')[np.where((self.trimcond_X[:,1] == 'free'))[0]]
        
        # bypass
        if False:
            print 'running bypass...'
            self.response = model_equations.eval_equations( X_free_0, 'full_output')
        else:
            print 'running trim for ' + str(len(X_free_0)) + ' variables...'
            X_free, info, status, msg= so.fsolve(model_equations.eval_equations, X_free_0, args=('trim'), full_output=True)
            print msg
            print 'function evaluations: ' + str(info['nfev'])
            
            # if trim was successful, then do one last evaluation with the final parameters.
            if status == 1:
                print 'Trimsolution: '
                for i_X in range(len(X_free)):
                    print self.trimcond_X[:,0][np.where((self.trimcond_X[:,1] == 'free'))[0]][i_X] + ': %.4f' % float(X_free[i_X])
                    
                self.response = model_equations.eval_equations(X_free, 'full_output')
                
                # store response
            

        
        
        
        
        
        
        