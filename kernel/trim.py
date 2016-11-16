# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:35:22 2014

@author: voss_ar
"""
import numpy as np
import scipy.optimize as so
import logging, sys

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
        
        # ---------------
        # --- default --- 
        # ---------------
        # Bedingung: free parameters in trimcond_X == target parameters in trimcond_Y
        # inputs
        self.trimcond_X = np.array([
            ['x',        'fix',  0.0,],
            ['y',        'fix',  0.0,],
            ['z',        'fix',  z  ,],
            ['phi',      'fix',  0.0,],
            ['theta',    'free', 0.0,],
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
        
        
        # ------------------
        # --- pitch only --- 
        # ------------------
        if self.trimcase['manoeuver'] == 'pitch':
            logging.info('setting trim conditions to "pitch"')
            # inputs
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_xi'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            # outputs
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dp'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dr'))[0][0],1] = 'free'
            
        # -----------------------------------
        # --- pitch and roll only, no yaw --- 
        # -----------------------------------
        elif self.trimcase['manoeuver'] == 'pitch&roll':
            logging.info('setting trim conditions to "pitch&roll"')
            # inputs
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            # outputs
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dr'))[0][0],1] = 'free'
        
        # ---------------------
        # --- level landing --- 
        # ---------------------
        elif self.trimcase['manoeuver'] in ['L1wheel', 'L2wheel']:
            logging.info('setting trim conditions to "level landing"')
            # inputs
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'w'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'w'))[0][0],2] = self.trimcase['w']
            # outputs
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dr'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dw'))[0][0],1] = 'fix'
            
        # -----------------------
        # --- 3 wheel landing --- 
        # -----------------------
        elif self.trimcase['manoeuver'] in ['L3wheel']:
            logging.info('setting trim conditions to "3 wheel landing"')
            # inputs
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'w'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'w'))[0][0],2] = self.trimcase['w']
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'theta'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'theta'))[0][0],2] = self.trimcase['theta']
            # outputs
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dr'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dw'))[0][0],1] = 'fix'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'Nz'))[0][0],1] = 'free'
            
        # ------------------
        # --- segelflug --- 
        # -----------------
        # Sinken (w) wird erlaubt, damit die Geschwindigkeit konstant bleibt (du = 0.0)
        # Eigentlich muesste Vtas konstant sein, ist aber momentan nicht als trimcond vorgesehen... Das wird auch schwierig, da die Machzahl vorgegeben ist.
        elif self.trimcase['manoeuver'] == 'segelflug':
            logging.info('setting trim conditions to "segelflug"')
            # inputs
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'w'))[0][0],1] = 'free'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'phi'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'phi'))[0][0],2] = self.trimcase['phi']
#             self.trimcond_X[np.where((self.trimcond_X[:,0] == 'psi'))[0][0],1] = 'fix'
#             self.trimcond_X[np.where((self.trimcond_X[:,0] == 'psi'))[0][0],2] = self.trimcase['psi']
            # outputs
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'w'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'du'))[0][0],1] = 'target'
       
        # -------------------------
        # --- pratt, alpha only --- 
        # -------------------------
        elif self.trimcase['manoeuver'] == 'pratt':
            logging.info('setting trim conditions to "pratt"')
            # inputs
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_xi'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_eta'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            # outputs
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dp'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dq'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dr'))[0][0],1] = 'free'
        
        # --------------
        # --- bypass --- 
        # --------------
        # Die Steuerkommandos xi, eta und zeta werden vorgegeben und die resultierenden Beschleunigungen sind frei. 
        elif self.trimcase['manoeuver'] == 'bypass':
            logging.info('setting trim conditions to "bypass"')
            # inputs
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'phi'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'phi'))[0][0],2] = self.trimcase['phi']
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'theta'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'theta'))[0][0],2] = self.trimcase['theta']
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'psi'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'psi'))[0][0],2] = self.trimcase['psi']
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_xi'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_xi'))[0][0],2] = self.trimcase['command_xi']
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_eta'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_eta'))[0][0],2] = self.trimcase['command_eta']
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_zeta'))[0][0],2] = self.trimcase['command_zeta']
            # outputs
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'Nz'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dp'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dq'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dr'))[0][0],1] = 'free'
        else:
            logging.info('setting trim conditions to "default"')
        
    
    def exec_trim(self):
        # The purpose of HYBRD is to find a zero of a system of N non-
        # linear functions in N variables by a modification of the Powell
        # hybrid method.  The user must provide a subroutine which calcu-
        # lates the functions.  The Jacobian is then calculated by a for-
        # ward-difference approximation.
        # http://www.math.utah.edu/software/minpack/minpack/hybrd.html
    
                
        import model_equations # Warum muss der import hier stehen??
        
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid']:
            equations = model_equations.steady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
        X_free_0 = np.array(self.trimcond_X[:,2], dtype='float')[np.where((self.trimcond_X[:,1] == 'free'))[0]]
        
        if self.trimcase['manoeuver'] == 'bypass':
            logging.info('running bypass...')
            self.response = equations.eval_equations(X_free_0, time=0.0, type='trim_full_output')
        else:
            logging.info('running trim for ' + str(len(X_free_0)) + ' variables...')
            X_free, info, status, msg= so.fsolve(equations.eval_equations, X_free_0, args=(0.0, 'trim'), full_output=True)
            logging.info(msg)
            logging.info('function evaluations: ' + str(info['nfev']))
            
            # if trim was successful, then do one last evaluation with the final parameters.
            if status == 1:
                logging.info('')
                logging.info('Trimsolution: ')
                logging.info('--------------------')
                for i_X in range(len(X_free)):
                    logging.info(self.trimcond_X[:,0][np.where((self.trimcond_X[:,1] == 'free'))[0]][i_X] + ': %.4f' % float(X_free[i_X]))
                    
                self.response = equations.eval_equations(X_free, time=0.0, type='trim_full_output')
            else:
                self.response = None
                logging.warning('Failure: ' + msg)
                # store response
            

        
    def exec_sim(self):
        import model_equations 
        if self.jcl.aero['method'] in [ 'mona_steady', 'hybrid'] and not self.simcase['landinggear']:
            equations = model_equations.steady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y, self.simcase)
        elif self.simcase['landinggear'] and self.jcl.landinggear['method'] == 'generic':
            logging.info('adding 2 x {} states for landing gear'.format(self.model.lggrid['n']))
            lg_states_X = []
            lg_states_Y = []
            for i in range(self.model.lggrid['n']):
                lg_states_X.append(self.response['p1'][i] - self.jcl.landinggear['para'][i]['stroke_length'] - self.jcl.landinggear['para'][i]['fitting_length'])
                lg_states_Y.append(self.response['dp1'][i])
            for i in range(self.model.lggrid['n']):
                lg_states_X.append(self.response['dp1'][i])
                lg_states_Y.append(self.response['ddp1'][i])
            self.response['X'] = np.hstack((self.response['X'], lg_states_X ))
            self.response['Y'] = np.hstack((self.response['Y'], lg_states_Y ))
            equations = model_equations.landing(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y, self.simcase)
        elif self.jcl.aero['method'] in [ 'mona_unsteady']:
            # initialize lag states with zero and extend steady response vectors X and Y
            logging.info('adding {} x {} unsteady lag states to the system'.format(self.model.aerogrid['n'],self.model.aero['n_poles']))
            lag_states = np.zeros((self.model.aerogrid['n'] * self.model.aero['n_poles'])) 
            self.response['X'] = np.hstack((self.response['X'], lag_states ))
            self.response['Y'] = np.hstack((self.response['Y'], lag_states ))
            equations = model_equations.unsteady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y, self.simcase)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))

        X0 = self.response['X']
        dt = self.simcase['dt']
        t_final = self.simcase['t_final']
        logging.info('running time simulation for ' + str(t_final) + ' sec...')
        #print 'Progress:'
        from scipy.integrate import ode
        integrator = ode(equations.ode_arg_sorter).set_integrator('vode', method='adams', nsteps=2000, rtol=1e-5, atol=1e-5, max_step=1e-4) # non-stiff: 'adams', stiff: 'bdf'
        #integrator = ode(equations.ode_arg_sorter).set_integrator('vode', method='adams', nsteps=2000, rtol=1e-2, atol=1e-8, max_step=5e-4) # non-stiff: 'adams', stiff: 'bdf'
#         integrator = ode(equations.ode_arg_sorter).set_integrator('dopri5', nsteps=2000, rtol=1e-2, atol=1e-8, max_step=1e-4)
        integrator.set_initial_value(X0, 0.0)
        X_t = []
        t = []
        while integrator.successful() and integrator.t < t_final:  
            integrator.integrate(integrator.t+dt)
            X_t.append(integrator.y)
            t.append(integrator.t)
            #print str(integrator.t) + ' sec - ' + str(equations.counter) + ' function evaluations'
            
        if integrator.successful():
            logging.info('Simulation finished.')
            logging.info('running (again) with full outputs at selected time steps...')
            for i_step in np.arange(0,len(t)):
                response_step = equations.eval_equations(X_t[i_step], t[i_step], type='sim_full_output')
                for key in self.response.keys():
                    self.response[key] = np.vstack((self.response[key],response_step[key]))

        else:
            self.response['t'] = None
            logging.error('Integration failed! Exit.')
            sys.exit()
            