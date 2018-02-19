# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:35:22 2014

@author: voss_ar
"""
import numpy as np
import scipy.optimize as so
import logging, sys, copy
import io_functions

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
        Vtas = self.trimcase['Ma'] * self.model.atmo['a'][i_atmo]
        u = Vtas
        z = -self.model.atmo['h'][i_atmo]
        #q = (self.trimcase['Nz'] - 1.0)*9.81/u
        
        # ---------------
        # --- default --- 
        # ---------------
        # Bedingung: free parameters in trimcond_X == target parameters in trimcond_Y
        # inputs
        self.trimcond_X = np.array([
            ['x',        'fix',    0.0,],
            ['y',        'fix',    0.0,],
            ['z',        'fix',    z  ,],
            ['phi',      'fix',    0.0,],
            ['theta',    'free',   0.0,], # dependent on u and w if dz = 0
            ['psi',      'fix',    0.0,],
            ['u',        'free',   u,  ],
            ['v',        'fix',    0.0,],
            ['w',        'free',   0.0,],
            ['p',        'fix',    self.trimcase['p'],],
            ['q',        'fix',    self.trimcase['q'],],
            ['r',        'fix',    self.trimcase['r'],],
            ])
        for i_mode in range(n_modes):
            self.trimcond_X = np.vstack((self.trimcond_X ,  ['Uf'+str(i_mode), 'free', 0.0]))
        for i_mode in range(n_modes):
            self.trimcond_X = np.vstack((self.trimcond_X ,  ['dUf_dt'+str(i_mode), 'free', 0.0]))
            
        self.trimcond_X = np.vstack((self.trimcond_X , ['command_xi',   'free', 0.0,],  ['command_eta',   'free',  0.0,], ['command_zeta',   'free',  0.0,]))
            
        # outputs
        self.trimcond_Y = np.array([ 
            ['dx',       'target',   Vtas,], # dx = Vtas if dz = 0
            ['dy',       'free',   0.0,],
            ['dz',       'target', 0.0,],
            ['dphi',     'free',   0.0,],
            ['dtheta',   'free',   0.0,],
            ['dpsi',     'free',   0.0,],
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
        self.trimcond_Y = np.vstack((self.trimcond_Y ,
                                     ['dcommand_xi',    'fix',  0.0,],
                                     ['dcommand_eta',   'fix',  0.0,],
                                     ['dcommand_zeta',  'fix',  0.0,],
                                   ))

        self.trimcond_Y = np.vstack((self.trimcond_Y , 
                                     ['Nz',       'target',    self.trimcase['Nz']],
                                     ['Vtas',     'free',  Vtas,],
                                   ))
        
        
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

            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dx'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dz'))[0][0],2] = self.trimcase['dz']
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'Vtas'))[0][0],1] = 'target'
            # outputs
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dr'))[0][0],1] = 'free'
             
        # -----------------------
        # --- 3 wheel landing --- 
        # -----------------------
        elif self.trimcase['manoeuver'] in ['L3wheel']:
            logging.info('setting trim conditions to "3 wheel landing"')
            # inputs
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dx'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dz'))[0][0],2] = self.trimcase['dz']
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'Vtas'))[0][0],1] = 'target'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'theta'))[0][0],1] = 'target'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'theta'))[0][0],2] = self.trimcase['theta']
            # outputs
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dr'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'Nz'))[0][0],1] = 'free'
            
        # ------------------
        # --- segelflug --- 
        # -----------------
        # Sinken (w) wird erlaubt, damit die Geschwindigkeit konstant bleibt (du = 0.0)
        # Eigentlich muesste Vtas konstant sein, ist aber momentan nicht als trimcond vorgesehen... Das wird auch schwierig, da die Machzahl vorgegeben ist.
        elif self.trimcase['manoeuver'] == 'segelflug':
            logging.info('setting trim conditions to "segelflug"')
            # inputs 
            # without changes

            # outputs
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dx'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dz'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'du'))[0][0],1] = 'target'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dw'))[0][0],1] = 'target'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'Nz'))[0][0],1] = 'free'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'Vtas'))[0][0],1] = 'target'
       
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
    
    def calc_jacobian(self):
        import model_equations # Warum muss der import hier stehen??
#         equations = model_equations.steady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y)
#         X0 = np.array(self.trimcond_X[:,2], dtype='float')
        if self.jcl.aero['method'] in [ 'mona_steady', 'hybrid']:
            equations = model_equations.steady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y)
            X0 = np.array(self.trimcond_X[:,2], dtype='float')
            n_poles     = 0
        elif self.jcl.aero['method'] in [ 'mona_unsteady']:
            # initialize lag states with zero and extend steady response vectors X and Y
            logging.info('adding {} x {} unsteady lag states to the system'.format(self.model.aerogrid['n'],self.model.aero['n_poles']))
            lag_states = np.zeros((self.model.aerogrid['n'] * self.model.aero['n_poles'])) 
            X0 = np.hstack((np.array(self.trimcond_X[:,2], dtype='float'), lag_states ))
            equations = model_equations.unsteady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y, self.simcase)
            n_poles     = self.model.aero['n_poles']
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
        def approx_jacobian(X0,func,epsilon,dt):
            """Approximate the Jacobian matrix of callable function func
               * Parameters
                 x       - The state vector at which the Jacobian matrix is desired
                 func    - A vector-valued function of the form f(x,*args)
                 epsilon - The peturbation used to determine the partial derivatives
                 *args   - Additional arguments passed to func
            """
            X0 = np.asfarray(X0)
            jac = np.zeros([len(func(*(X0,0.0, 'sim'))),len(X0)])
            dX = np.zeros(len(X0))
            for i in range(len(X0)):
                f0 = func(*(X0,0.0, 'sim'))
                dX[i] = epsilon
                fi = func(*(X0+dX,0.0+dt, 'sim'))
                jac[:,i] = (fi - f0)/epsilon
                dX[i] = 0.0
            return jac
        
        logging.info('Calculating jacobian for ' + str(len(X0)) + ' variables...')
        jac = approx_jacobian(X0=X0, func=equations.equations, epsilon=0.001, dt=1.0) # epsilon sollte klein sein, dt sollte 1.0s sein
        X = self.response['X']
        Y = self.response['Y']
        self.response.clear()
        self.response['X'] = X
        self.response['Y'] = Y
        self.response['X0'] = X0 # Linearisierungspunkt
        self.response['Y0'] = equations.equations(X0, t=0.0, type='trim')
        self.response['jac'] = jac
        
        n_j         = self.model.aerogrid['n']
        i_mass      = self.model.mass['key'].index(self.trimcase['mass'])
        n_modes     = self.model.mass['n_modes'][i_mass] 
        # States need to be reordered into ABCD matrices!
        # X = [ rbm,  flex,  command_cs,  lag_states ]
        # Y = [drbm, dflex, dcommand_cs, dlag_states, Nz]
        # [Y] = [A B] * [X]
        #       [C D]   
        
        idx_A = range(0,12+n_modes*2) + range(12+n_modes*2+3, 12+n_modes*2+3+n_j*n_poles)
        idx_B = range(12+n_modes*2,12+n_modes*2+3)
        idx_C = range(12+n_modes*2+3, 12+n_modes*2+3+1)
        self.response['A'] = jac[idx_A,:][:,idx_A] # aircraft itself
        self.response['B'] = jac[idx_A,:][:,idx_B] # reaction of aircraft on external excitation
        self.response['C'] = jac[idx_C,:][:,idx_A] # sensors
        self.response['D'] = jac[idx_C,:][:,idx_B] # reaction of sensors on external excitation
            
    def calc_derivatives(self):
        import model_equations # Warum muss der import hier stehen??
        
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid']:
            equations = model_equations.steady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
        A = self.jcl.general['A_ref'] #sum(self.model.aerogrid['A'][:])
        delta = 0.01   
            
        X0 = np.array(self.trimcond_X[:,2], dtype='float')
        response0 = equations.equations(X0, 0.0, 'sim_full_output')
        logging.info('Calculating derivatives for ' + str(len(X0)) + ' variables...')
        logging.info('MAC_ref = {}'.format(self.jcl.general['MAC_ref']))
        logging.info('A_ref = {}'.format(self.jcl.general['A_ref']))
        logging.info('b_ref = {}'.format(self.jcl.general['b_ref']))
        logging.info('c_ref = {}'.format(self.jcl.general['c_ref']))
        logging.info('')
        logging.info('Derivatives given in body axis (aft-right-up):')
        logging.info('--------------------------------------------------------------------------------------')
        logging.info('                     Cx         Cy         Cz         Cmx        Cmx        Cmz')
        for i in range(len(X0)):
            Xi = copy.deepcopy(X0)
            Xi[i] += delta
            response = equations.equations(Xi, 0.0, 'sim_full_output')
            Pmac_c = (response['Pmac']-response0['Pmac'])/response['q_dyn']/A/delta
            tmp = '{:>20} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g}'.format( self.trimcond_X[i,0], Pmac_c[0], Pmac_c[1], Pmac_c[2], Pmac_c[3]/self.model.macgrid['b_ref'], Pmac_c[4]/self.model.macgrid['c_ref'], Pmac_c[5]/self.model.macgrid['b_ref'] )
            logging.info(tmp)
        logging.info('--------------------------------------------------------------------------------------')
                            
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
        elif self.jcl.aero['method'] in [ 'nonlin_steady']:
            equations = model_equations.nonlin_steady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y)
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
            
            # no errors, check trim status for success
            if status == 1:
                # if trim was successful, then do one last evaluation with the final parameters.
                self.response = equations.eval_equations(X_free, time=0.0, type='trim_full_output')
            else:
                self.response = None
                logging.warning('Trim failed for subcase {}. The Trim solver reports: {}'.format(self.trimcase['subcase'] + msg))
                return
            

        
    def exec_sim(self):
        import model_equations 
        if self.jcl.aero['method'] in [ 'mona_steady', 'hybrid'] and not self.simcase['landinggear']:
            equations = model_equations.steady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y, self.simcase, X0=self.response['X'])
        elif self.jcl.aero['method'] in [ 'nonlin_steady']:
            equations = model_equations.nonlin_steady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y, self.simcase, X0=self.response['X'])
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
            self.response['Y'] = np.hstack((self.response['Y'][:-2], lg_states_Y, self.response['Y'][-2:] ))
            equations = model_equations.landing(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y, self.simcase, X0=self.response['X'])
        elif self.jcl.aero['method'] in [ 'mona_unsteady']:
            if 'disturbance' in self.simcase.keys():
                logging.info('adding disturbance of {} to state(s) '.format(self.simcase['disturbance']))
                self.response['X'][11+self.simcase['disturbance_mode']] += self.simcase['disturbance']
            # Initialize lag states with zero and extend steady response vectors X and Y
            # Distinguish between pyhsical rfa on panel level and generalized rfa. This influences the number of lag states.
            if self.jcl.aero.has_key('method_rfa') and self.jcl.aero['method_rfa'] == 'halfgeneralized':
                n_modes = self.model.mass['n_modes'][self.model.mass['key'].index(self.trimcase['mass'])]
                logging.info('adding {} x {} unsteady lag states to the system'.format(2 * n_modes,self.model.aero['n_poles']))
                lag_states = np.zeros((2 * n_modes * self.model.aero['n_poles'])) 
            else:
                logging.info('adding {} x {} unsteady lag states to the system'.format(self.model.aerogrid['n'],self.model.aero['n_poles']))
                lag_states = np.zeros((self.model.aerogrid['n'] * self.model.aero['n_poles'])) 
            self.response['X'] = np.hstack((self.response['X'], lag_states ))
            self.response['Y'] = np.hstack((self.response['Y'], lag_states ))
            equations = model_equations.unsteady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y, self.simcase, X0=self.response['X'])
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))

        X0 = self.response['X']
        dt = self.simcase['dt']
        t_final = self.simcase['t_final']
        logging.info('running time simulation for ' + str(t_final) + ' sec...')
        #print 'Progress:'
        from scipy.integrate import ode
        integrator = ode(equations.ode_arg_sorter).set_integrator('vode', method='adams', nsteps=2000, rtol=1e-6, atol=1e-6, max_step=5e-4) # non-stiff: 'adams', stiff: 'bdf'
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
            self.response = None
            logging.error('Integration failed! Exit.')
            #sys.exit()
            
            
    def iterative_trim(self):
        import model_equations # Warum muss der import hier stehen??
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid']:
            equations = model_equations.steady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y)
        elif self.jcl.aero['method'] in [ 'cfd_steady']:
            equations = model_equations.cfd_steady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y)
            io_functions.specific_functions.copy_para_file(io_functions.specific_functions(),self.jcl, self.trimcase)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
        # remove modes from trimcond_Y and _Y
        i_mass = self.model.mass['key'].index(self.trimcase['mass'])
        n_modes = self.model.mass['n_modes'][i_mass]
        
        for i_mode in range(n_modes):
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'Uf'+str(i_mode)))[0][0],1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:,0] == 'dUf_dt'+str(i_mode)))[0][0],1] = 'fix'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'dUf_dt'+str(i_mode)))[0][0],1] = 'fix'
            self.trimcond_Y[np.where((self.trimcond_Y[:,0] == 'd2Uf_d2t'+str(i_mode)))[0][0],1] = 'fix'
        #X0 = np.copy(self.response['X'])
        #X_free_0 = X0[np.where((self.trimcond_X[:,1] == 'free'))[0]] # start trim with solution from normal trim
        X_free_0 = np.array(self.trimcond_X[:,2], dtype='float')[np.where((self.trimcond_X[:,1] == 'free'))[0]] # start trim from scratch
        
        if self.trimcase['manoeuver'] == 'bypass':
            logging.info('running bypass...')
            self.response = equations.eval_equations(X_free_0, time=0.0, type='trim_full_output')
        else:
            logging.info('running trim for ' + str(len(X_free_0)) + ' variables...')
            try:
                X_free, info, status, msg= so.fsolve(equations.eval_equations_iteratively, X_free_0, args=(0.0, 'trim'), full_output=True, epsfcn=1.0e-3, xtol=1.0e-3 )
            except model_equations.TauError as e:
                self.response = None
                logging.warning('Trim failed for subcase {} due to TauError: {}'.format(self.trimcase['subcase'], e))
                return
            except model_equations.ConvergenceError as e:
                self.response = None
                logging.warning('Trim failed for subcase {} due to ConvergenceError: {}'.format(self.trimcase['subcase'], e))
                return
            else:
                logging.info(msg)
                logging.info('function evaluations: ' + str(info['nfev']))
            
                # no errors, check trim status for success
                if status == 1:
                    # if trim was successful, then do one last evaluation with the final parameters.
                    self.response = equations.eval_equations_iteratively(X_free, time=0.0, type='trim_full_output')
                else:
                    self.response = None
                    logging.warning('Trim failed for subcase {}. The Trim solver reports: {}'.format(self.trimcase['subcase'] + msg))
                    return
       