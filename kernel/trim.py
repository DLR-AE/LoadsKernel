# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:35:22 2014

@author: voss_ar
"""
import numpy as np
import scipy.optimize as so

class trim:
    def __init__(self, model, jcl, trimcase, simcase):
        self.model = model
        self.jcl = jcl
        self.trimcase = trimcase
        self.simcase = simcase     
        
    def set_trimcond(self):
        # init
        i_atmo = self.model.atmo['key'].index(self.trimcase['altitude'])
        i_mass = self.model.mass['key'].index(self.trimcase['mass'])
        n_modes = self.model.mass['n_modes'][i_mass]
        u = self.trimcase['Ma'] * self.model.atmo['a'][i_atmo]
        z = -self.model.atmo['h'][i_atmo]
        #q = (self.trimcase['Nz'] - 1.0)*9.81/u
        
        # inputs
        self.trimcond_X = np.array([
            ['x',        'fix',  0.0,],
            ['y',        'fix',  0.0,],
            ['z',        'fix',  z  ,],
            ['phi',      'fix',  0.0,],
            ['theta',    'free', 1.0/180*np.pi,],
            ['psi',      'fix',  0.0,],
            ['u',        'fix',  u,  ],
            ['v',        'fix',  0.0,],
            ['w',        'fix',  0.0,],
            ['p',        'fix',  self.trimcase['p'],],
            ['q',        'fix',  self.trimcase['q'],],
            ['r',        'fix',  self.trimcase['r'],],
            ])
        for i_mode in range(n_modes):
            self.trimcond_X = np.vstack((self.trimcond_X ,  ['Uf'+str(i_mode), 'free', 0.0]))
        for i_mode in range(n_modes):
            self.trimcond_X = np.vstack((self.trimcond_X ,  ['dUf_dt'+str(i_mode), 'free', 0.0]))
            
        self.trimcond_X = np.vstack((self.trimcond_X , ['command_xi',   'free', 0.0,],  ['command_eta',   'free',  0.0,], ['command_zeta',   'free',  0.0,]))
            
        # outputs
        self.trimcond_Y = np.array([ 
            ['u',        'fix',    u,  ],
            ['v',        'fix',    0.0,],
            ['w',        'fix',    0.0,],
            ['p',        'fix',    self.trimcase['p'],],
            ['q',        'fix',    self.trimcase['q'],],
            ['r',        'fix',    self.trimcase['r'],],
            ['du',       'free',   0.0,],
            ['dv',       'free',   0.0,],
            ['dw',       'free',   0.0,],
            ['dp',       'target', self.trimcase['pdot'],],
            ['dq',       'target', self.trimcase['qdot'],],
            ['dr',       'target', self.trimcase['rdot'],],
            ])
            
        for i_mode in range(n_modes):
            self.trimcond_Y = np.vstack((self.trimcond_Y ,  ['dUf_dt'+str(i_mode), 'target', 0.0]))
        for i_mode in range(n_modes):
            self.trimcond_Y = np.vstack((self.trimcond_Y ,  ['d2Uf_d2t'+str(i_mode), 'target', 0.0]))
        self.trimcond_Y = np.vstack((self.trimcond_Y , ['dcommand_xi',   'fix', 0.0,],  ['dcommand_eta',   'fix',  0.0,], ['dcommand_zeta',   'fix',  0.0,]))

        
        self.trimcond_Y = np.vstack((self.trimcond_Y , ['Nz',       'target',  self.trimcase['Nz'],]))

        
    
    def exec_trim(self):
        # The purpose of HYBRD is to find a zero of a system of N non-
        # linear functions in N variables by a modification of the Powell
        # hybrid method.  The user must provide a subroutine which calcu-
        # lates the functions.  The Jacobian is then calculated by a for-
        # ward-difference approximation.
        # http://www.math.utah.edu/software/minpack/minpack/hybrd.html
    
                
        import model_equations # Warum muss der import hier stehen??
        equations = model_equations.nastran(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y)

        X_free_0 = np.array(self.trimcond_X[:,2], dtype='float')[np.where((self.trimcond_X[:,1] == 'free'))[0]]
        
        bypass = False
        if bypass:
            print 'running bypass...'
            self.response = equations.eval_equations(X_free_0, time=0.0, type='trim_full_output')
        else:
            print 'running trim for ' + str(len(X_free_0)) + ' variables...'
            X_free, info, status, msg= so.fsolve(equations.eval_equations, X_free_0, args=(0.0, 'trim'), full_output=True)
            print msg
            print 'function evaluations: ' + str(info['nfev'])
            
            # if trim was successful, then do one last evaluation with the final parameters.
            if status == 1:
                print ''
                print 'Trimsolution: '
                print '--------------------' 
                for i_X in range(len(X_free)):
                    print self.trimcond_X[:,0][np.where((self.trimcond_X[:,1] == 'free'))[0]][i_X] + ': %.4f' % float(X_free[i_X])
                    
                self.response = equations.eval_equations(X_free, time=0.0, type='trim_full_output')
            else:
                self.response = 'Failure: ' + msg
                # store response
            

        
    def exec_sim(self):
        import model_equations 
        equations = model_equations.nastran(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y, self.simcase)
        
        X0 = self.response['X'] 
        dt = self.simcase['dt']
        t_final = self.simcase['t_final']
        print 'running time simulation for ' + str(t_final) + ' sec...'
        print 'Progress:'
        from scipy.integrate import ode
        integrator = ode(equations.ode_arg_sorter).set_integrator('vode', method='adams') # non-stiff: 'adams', stiff: 'bdf'
        integrator.set_initial_value(X0, 0.0)
        X_t = []
        t = []
        while integrator.successful() and integrator.t < t_final:  
            integrator.integrate(integrator.t+dt)
            X_t.append(integrator.y)
            t.append(integrator.t)
            print str(integrator.t) + ' sec - ' + str(equations.counter) + ' function evaluations'
            
        if integrator.successful():
            print 'Simulation finished.'
            print 'running (again) with full outputs at selected time steps...' 
            for i_step in np.arange(0,len(t)):
                response_step = equations.eval_equations(X_t[i_step], t[i_step], type='sim_full_output')
                for key in self.response.keys():
                    self.response[key] = np.vstack((self.response[key],response_step[key]))


        else:
            self.response['t'] = 'Failure'
            
            
            
        # B
#        from scipy.integrate import odeint
#        X_t, info = odeint(equations.eval_equations, X0, t, args=('sim',), full_output=True, h0=0.001)
#        print info['message']
#        print 'time steps: ' + str(t)
#        print 'time step evaluatins: ' + str(info['nfe'])
#        if info['message'] == 'Integration successful.':
#            for i_step in np.arange(1,len(X_t)):
#                response_step = equations.eval_equations(X_t[i_step], t[i_step], type='sim_full_output')
#                for key in self.response.keys():
#                    self.response[key] = np.vstack((self.response[key],response_step[key]))
#            self.response['t'] = t
#            self.response['t_cur'] = info['tcur']
#
#        else:
#            self.response['t'] = 'Failure: ' + info['message']
            
        
                
            
    
        
        
