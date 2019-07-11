# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:35:22 2014

@author: voss_ar
"""
import numpy as np
import scipy.optimize as so
import logging, copy
from scipy.integrate import ode
from loadskernel.integrate import RungeKutta4, ExplicitEuler
import loadskernel.io_functions.specific_functions as specific_io

class Trim:
    def __init__(self, model, jcl, trimcase, simcase):
        self.model = model
        self.jcl = jcl
        self.trimcase = trimcase
        self.simcase = simcase    
        
        self.response = None
        self.successful = None
         
        self.states = None
        self.inputs = None
        self.state_derivatives = None
        self.input_derivatives = None
        self.outputs = None
        
        self.trimcond_X = None
        self.trimcond_Y = None

        self.n_states = None
        self.n_inputs = None
        self.n_state_derivatives = None
        self.n_input_derivatives = None
        self.n_outputs = None
        self.n_lg_states = None
        self.n_lag_states = None
        
        self.idx_states = None
        self.idx_inputs = None
        self.idx_state_derivatives = None
        self.idx_input_derivatives = None
        self.idx_outputs = None
        self.idx_lg_states = None
        self.idx_lg_derivatives = None
        self.idx_lag_states = None
        self.idx_lag_derivatives = None
        
    def set_trimcond(self):
        # init
        i_atmo = self.model.atmo['key'].index(self.trimcase['altitude'])
        i_mass = self.model.mass['key'].index(self.trimcase['mass'])
        n_modes = self.model.mass['n_modes'][i_mass]
        vtas = self.trimcase['Ma'] * self.model.atmo['a'][i_atmo]
        theta = 1.0/180.0*np.pi # starting with a small angle of attack increases the performance and convergence of the CFD solution
        u = vtas*np.cos(theta)
        w = vtas*np.sin(theta)
        z = -self.model.atmo['h'][i_atmo]
        
        # ---------------
        # --- default --- 
        # ---------------
        # Bedingung: free parameters in trimcond_X == target parameters in trimcond_Y
        # inputs
        self.states = np.array([
            ['x',        'fix',    0.0,],
            ['y',        'fix',    0.0,],
            ['z',        'fix',    z  ,],
            ['phi',      'fix',    0.0,],
            ['theta',    'free',   theta,], # dependent on u and w if dz = 0
            ['psi',      'fix',    0.0,],
            ['u',        'free',   u,  ],
            ['v',        'fix',    0.0,],
            ['w',        'free',   w,],
            ['p',        'fix',    self.trimcase['p'],],
            ['q',        'fix',    self.trimcase['q'],],
            ['r',        'fix',    self.trimcase['r'],],
            ], dtype='|S16')
        for i_mode in range(n_modes):
            self.states = np.vstack((self.states ,  ['Uf'+str(i_mode), 'free', 0.0]))
        for i_mode in range(n_modes):
            self.states = np.vstack((self.states ,  ['dUf_dt'+str(i_mode), 'free', 0.0]))
        
        self.inputs = np.array([
            ['command_xi',   'free', 0.0,],  
            ['command_eta',  'free',  0.0,], 
            ['command_zeta', 'free',  0.0,],
            ['thrust',  'fix', 0.0]
            ], dtype='|S16')
        
        # outputs
        self.state_derivatives = np.array([ 
            ['dx',       'target',   vtas,], # dx = vtas if dz = 0
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
            ], dtype='|S16')
            
        for i_mode in range(n_modes):
            self.state_derivatives = np.vstack((self.state_derivatives ,  ['dUf_dt'+str(i_mode), 'target', 0.0]))
        for i_mode in range(n_modes):
            self.state_derivatives = np.vstack((self.state_derivatives ,  ['d2Uf_d2t'+str(i_mode), 'target', 0.0]))
        
        self.input_derivatives = np.array([
            ['dcommand_xi',    'fix',  0.0,],
            ['dcommand_eta',   'fix',  0.0,],
            ['dcommand_zeta',  'fix',  0.0,],
            ['dthrust',        'fix',  0.0,],
            ], dtype='|S16')

        self.outputs = np.array([
            ['Nz',       'target',  self.trimcase['Nz']],
            ['Vtas',     'free',    vtas,],
            ['beta',     'free',    0.0]
            ], dtype='|S16')
        
        # ------------------
        # --- pitch only --- 
        # ------------------
        if self.trimcase['maneuver'] == 'pitch':
            logging.info('setting trim conditions to "pitch"')
            # inputs
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dp'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dr'))[0][0],1] = 'free'
            
        # -----------------------------------
        # --- pitch and roll only, no yaw --- 
        # -----------------------------------
        elif self.trimcase['maneuver'] == 'pitch&roll':
            logging.info('setting trim conditions to "pitch&roll"')
            # inputs
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dr'))[0][0],1] = 'free'
            
        # -----------------------------------
        # --- pitch and yaw only, no roll --- 
        # -----------------------------------
        elif self.trimcase['maneuver'] == 'pitch&yaw':
            logging.info('setting trim conditions to "pitch&yaw"')
            # inputs
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],1] = 'fix'
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dp'))[0][0],1] = 'free'
        
        # ---------------------
        # --- level landing --- 
        # ---------------------
        elif self.trimcase['maneuver'] in ['L1wheel', 'L2wheel']:
            logging.info('setting trim conditions to "level landing"')
            # inputs
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],1] = 'fix'

            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dx'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dz'))[0][0],2] = self.trimcase['dz']
            self.outputs[np.where((self.outputs[:,0] == 'Vtas'))[0][0],1] = 'target'
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dr'))[0][0],1] = 'free'
             
        # -----------------------
        # --- 3 wheel landing --- 
        # -----------------------
        elif self.trimcase['maneuver'] in ['L3wheel']:
            logging.info('setting trim conditions to "3 wheel landing"')
            # inputs
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dx'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dz'))[0][0],2] = self.trimcase['dz']
            self.states[np.where((self.states[:,0] == 'theta'))[0][0],1] = 'target'
            self.states[np.where((self.states[:,0] == 'theta'))[0][0],2] = self.trimcase['theta']
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dr'))[0][0],1] = 'free'
            self.outputs[np.where((self.outputs[:,0] == 'Nz'))[0][0],1] = 'free'
            self.outputs[np.where((self.outputs[:,0] == 'Vtas'))[0][0],1] = 'target'
            
        # ------------------
        # --- segelflug --- 
        # -----------------
        # Sinken (w) wird erlaubt, damit die Geschwindigkeit konstant bleibt (du = 0.0)
        elif self.trimcase['maneuver'] == 'segelflug':
            logging.info('setting trim conditions to "segelflug"')
            # inputs 
            # without changes
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dx'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dz'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'du'))[0][0],1] = 'target'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dw'))[0][0],1] = 'target'
            self.outputs[np.where((self.outputs[:,0] == 'Nz'))[0][0],1] = 'free'
            self.outputs[np.where((self.outputs[:,0] == 'Vtas'))[0][0],1] = 'target'
       
        # -------------------------
        # --- pratt, alpha only --- 
        # -------------------------
        elif self.trimcase['maneuver'] == 'pratt':
            logging.info('setting trim conditions to "pratt"')
            # inputs
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_eta'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dp'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dq'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dr'))[0][0],1] = 'free'
        
        # ----------------
        # --- CS fixed --- 
        # ----------------
        elif self.trimcase['maneuver'] == 'Xi&Zeta-fixed':
            logging.info('setting trim conditions to "Xi&Zeta-fixed"')
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],2] = self.trimcase['command_xi']
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],2] = self.trimcase['command_zeta']
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dp'))[0][0],1] = 'free'          
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dr'))[0][0],1] = 'free'

        elif self.trimcase['maneuver'] == 'CS-fixed':
            logging.info('setting trim conditions to "CS-fixed"')
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_eta'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],2] = self.trimcase['command_xi']
            self.inputs[np.where((self.inputs[:,0] == 'command_eta'))[0][0],2] = self.trimcase['command_eta']
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],2] = self.trimcase['command_zeta']
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dp'))[0][0],1] = 'free'   
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dq'))[0][0],1] = 'free'        
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dr'))[0][0],1] = 'free'
            
        elif self.trimcase['maneuver'] == 'CS&Acc-fixed':
            logging.info('setting trim conditions to "CS&Acc-fixed"')
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_eta'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],2] = self.trimcase['command_xi']
            self.inputs[np.where((self.inputs[:,0] == 'command_eta'))[0][0],2] = self.trimcase['command_eta']
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],2] = self.trimcase['command_zeta']
            self.states[np.where((self.states[:,0] == 'p'))[0][0],1] = 'free'   
            self.states[np.where((self.states[:,0] == 'q'))[0][0],1] = 'free'        
            self.states[np.where((self.states[:,0] == 'r'))[0][0],1] = 'free'
        
        # ----------------
        # --- sideslip --- 
        # ----------------
        elif self.trimcase['maneuver'] == 'sideslip':
            logging.info('setting trim conditions to "sideslip"')
            # fixed roll and yaw control
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],2] = self.trimcase['command_xi']
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],2] = self.trimcase['command_zeta']
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dp'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dr'))[0][0],1] = 'free'
            
            # set sideslip condition
            self.states[np.where((self.states[:,0] == 'psi'))[0][0],1] = 'free'
            self.states[np.where((self.states[:,0] == 'v'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dy'))[0][0],1] = 'target'
            self.outputs[np.where((self.outputs[:,0] == 'beta'))[0][0],1] = 'target'
            self.outputs[np.where((self.outputs[:,0] == 'beta'))[0][0],2] = self.trimcase['beta']
        
        # --------------
        # --- bypass --- 
        # --------------
        # Die Steuerkommandos xi, eta und zeta werden vorgegeben und die resultierenden Beschleunigungen sind frei. 
        elif self.trimcase['maneuver'] == 'bypass':
            logging.info('setting trim conditions to "bypass"')
            theta = self.trimcase['theta']
            u = vtas*np.cos(theta)
            w = vtas*np.sin(theta)
            # inputs
            self.states[np.where((self.states[:,0] == 'phi'))[0][0],1] = 'fix'
            self.states[np.where((self.states[:,0] == 'phi'))[0][0],2] = self.trimcase['phi']
            self.states[np.where((self.states[:,0] == 'theta'))[0][0],1] = 'fix'
            self.states[np.where((self.states[:,0] == 'theta'))[0][0],2] = theta
            self.states[np.where((self.states[:,0] == 'u'))[0][0],2] = u
            self.states[np.where((self.states[:,0] == 'w'))[0][0],2] = w
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],2] = self.trimcase['command_xi']
            self.inputs[np.where((self.inputs[:,0] == 'command_eta'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_eta'))[0][0],2] = self.trimcase['command_eta']
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],2] = self.trimcase['command_zeta']
            # outputs
            self.outputs[np.where((self.outputs[:,0] == 'Nz'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dp'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dq'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dr'))[0][0],1] = 'free'
        
        else:
            logging.info('setting trim conditions to "default"')
            
        if hasattr(self.jcl, 'engine'):
            logging.info('setting trim conditions to include thrust')
            # inputs
            self.inputs[np.where((self.inputs[:,0] == 'thrust'))[0][0],1] = 'free'
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'du'))[0][0],1] = 'target'
            
        
        # append inputs to X vector...
        self.trimcond_X = np.vstack((self.states , self.inputs))
        self.n_states   = self.states.__len__()
        self.n_inputs   = self.inputs.__len__()
        self.idx_states = range(0,self.n_states)
        self.idx_inputs = range(self.n_states, self.n_states+self.n_inputs)
        
        # ... and input derivatives and outputs to Y vector
        self.trimcond_Y = np.vstack((self.state_derivatives, self.input_derivatives, self.outputs))
        self.n_state_derivatives    = self.state_derivatives.__len__()
        self.n_input_derivatives    = self.input_derivatives.__len__()
        self.n_outputs              = self.outputs.__len__()
        self.idx_state_derivatives  = range(0,self.n_state_derivatives)        
        self.idx_input_derivatives  = range(self.n_state_derivatives, self.n_state_derivatives+self.n_input_derivatives)
        self.idx_outputs            = range(self.n_state_derivatives+self.n_input_derivatives, self.n_state_derivatives+self.n_input_derivatives+self.n_outputs)
        
    def calc_jacobian(self):
        import model_equations # Warum muss der import hier stehen??
#         equations = model_equations.Steady(self.model, self.jcl, self.trimcase, self.trimcond_X, self.trimcond_Y)
#         X0 = np.array(self.trimcond_X[:,2], dtype='float')
        if self.jcl.aero['method'] in [ 'mona_steady', 'hybrid']:
            equations = model_equations.Steady(self)
#             X0 = np.array(self.trimcond_X[:,2], dtype='float')
            X0 = self.response['X']
            self.idx_lag_states = []
        elif self.jcl.aero['method'] in [ 'mona_unsteady']:
            # initialize lag states with zero and extend steady response vectors X and Y
            if self.jcl.aero.has_key('method_rfa') and self.jcl.aero['method_rfa'] == 'generalized':
                logging.error('Generalized RFA not yet implemented.')
            elif self.jcl.aero.has_key('method_rfa') and self.jcl.aero['method_rfa'] == 'halfgeneralized':
                n_modes = self.model.mass['n_modes'][self.model.mass['key'].index(self.trimcase['mass'])]
                logging.info('adding {} x {} unsteady lag states to the system'.format(2 * n_modes,self.model.aero['n_poles']))
                lag_states = np.zeros((2 * n_modes * self.model.aero['n_poles'])) 
                n_poles     = self.model.aero['n_poles']
                n_j         = 2 * n_modes
            else:
                logging.info('adding {} x {} unsteady lag states to the system'.format(self.model.aerogrid['n'],self.model.aero['n_poles']))
                lag_states = np.zeros((self.model.aerogrid['n'] * self.model.aero['n_poles'])) 
                n_poles     = self.model.aero['n_poles']
                n_j         = self.model.aerogrid['n']
#             X0 = np.hstack((np.array(self.trimcond_X[:,2], dtype='float'), lag_states ))
            X0 = np.hstack((self.response['X'], lag_states ))
             # add lag states to system
            self.n_lag_states = lag_states.__len__()
            self.idx_lag_states         = range(self.n_states+self.n_inputs, self.n_states+self.n_inputs+self.n_lag_states)
            self.idx_lag_derivatives    = range(self.n_state_derivatives+self.n_input_derivatives, self.n_state_derivatives+self.n_input_derivatives+self.n_lag_states)
            self.idx_outputs            = range(self.n_state_derivatives+self.n_input_derivatives+self.n_lag_states, self.n_state_derivatives+self.n_input_derivatives+self.n_lag_states+self.n_outputs)

            equations = model_equations.Unsteady(self)
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
#         X = self.response['X']
#         Y = self.response['Y']
        self.response.clear()
#         self.response['X'] = X
#         self.response['Y'] = Y
        self.response = {}
        self.response['X0'] = X0 # Linearisierungspunkt
        self.response['Y0'] = equations.equations(X0, t=0.0, modus='trim')
        self.response['jac'] = jac
        self.response['states'] = self.states[:,0]
        self.response['state_derivativess'] = self.state_derivatives[:,0]
        self.response['inputs'] = self.inputs[:,0]
        self.response['outputs'] = self.outputs[:,0]
        
        
        i_mass      = self.model.mass['key'].index(self.trimcase['mass'])
        n_modes     = self.model.mass['n_modes'][i_mass] 
        # States need to be reordered into ABCD matrices!
        # X = [ rbm,  flex,  command_cs,  lag_states ]
        # Y = [drbm, dflex, dcommand_cs, dlag_states, outputs]
        # [Y] = [A B] * [X]
        #       [C D]   
        idx_A = self.idx_states+self.idx_lag_states
        idx_B = self.idx_inputs
        idx_C = self.idx_outputs
        self.response['A'] = jac[idx_A,:][:,idx_A] # aircraft itself
        self.response['B'] = jac[idx_A,:][:,idx_B] # reaction of aircraft on external excitation
        self.response['C'] = jac[idx_C,:][:,idx_A] # sensors
        self.response['D'] = jac[idx_C,:][:,idx_B] # reaction of sensors on external excitation
        self.response['idx_A'] = idx_A
        self.response['idx_B'] = idx_B
        self.response['idx_C'] = idx_C
            
    def calc_derivatives(self):
        import model_equations # Warum muss der import hier stehen??
        
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid']:
            equations = model_equations.Steady(self)
        elif self.jcl.aero['method'] in [ 'nonlin_steady']:
            equations = model_equations.NonlinSteady(self)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
        A = self.jcl.general['A_ref'] #sum(self.model.aerogrid['A'][:])
        delta = 0.01   
            
        X0 = np.array(self.trimcond_X[:,2], dtype='float')
        response0 = equations.equations(X0, 0.0, 'trim_full_output')
        logging.info('Calculating derivatives for ' + str(len(X0)) + ' variables...')
        logging.info('MAC_ref = {}'.format(self.jcl.general['MAC_ref']))
        logging.info('A_ref = {}'.format(self.jcl.general['A_ref']))
        logging.info('b_ref = {}'.format(self.jcl.general['b_ref']))
        logging.info('c_ref = {}'.format(self.jcl.general['c_ref']))
        logging.info('')
        logging.info('Derivatives given in body axis (aft-right-up):')
        logging.info('--------------------------------------------------------------------------------------')
        logging.info('                     Cx         Cy         Cz         Cmx        Cmy        Cmz')
        for i in range(len(X0)):
            Xi = copy.deepcopy(X0)
            Xi[i] += delta
            response = equations.equations(Xi, 0.0, 'trim_full_output')
            Pmac_c = (response['Pmac']-response0['Pmac'])/response['q_dyn']/A/delta
            tmp = '{:>20} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g} {:< 10.4g}'.format( self.trimcond_X[i,0], Pmac_c[0], Pmac_c[1], Pmac_c[2], Pmac_c[3]/self.model.macgrid['b_ref'], Pmac_c[4]/self.model.macgrid['c_ref'], Pmac_c[5]/self.model.macgrid['b_ref'] )
            logging.info(tmp)
        logging.info('--------------------------------------------------------------------------------------')
                            
    def exec_trim(self):
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid', 'nonlin_steady']:
            self.direct_trim()
        elif self.jcl.aero['method'] in [ 'cfd_steady']:
            self.iterative_trim()
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
    def direct_trim(self):
        # The purpose of HYBRD is to find a zero of a system of N non-
        # linear functions in N variables by a modification of the Powell
        # hybrid method.  The user must provide a subroutine which calcu-
        # lates the functions.  The Jacobian is then calculated by a for-
        # ward-difference approximation.
        # http://www.math.utah.edu/software/minpack/minpack/hybrd.html
    
                
        import model_equations # Warum muss der import hier stehen??
        
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid'] and not hasattr(self.jcl, 'landinggear'):
            equations = model_equations.Steady(self)
        elif self.jcl.aero['method'] in [ 'nonlin_steady']:
            equations = model_equations.NonlinSteady(self)
        elif self.simcase['landinggear'] and self.jcl.landinggear['method'] == 'generic':
            equations = model_equations.Landing(self)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))
        
        X_free_0 = np.array(self.trimcond_X[:,2], dtype='float')[np.where((self.trimcond_X[:,1] == 'free'))[0]]
        
        if self.trimcase['maneuver'] == 'bypass':
            logging.info('running bypass...')
            self.response = equations.eval_equations(X_free_0, time=0.0, modus='trim_full_output')
            self.successful = True
        else:
            logging.info('running trim for ' + str(len(X_free_0)) + ' variables...')
            X_free, info, status, msg= so.fsolve(equations.eval_equations, X_free_0, args=(0.0, 'trim'), full_output=True)
            logging.info(msg)
            logging.info('function evaluations: ' + str(info['nfev']))
            
            # no errors, check trim status for success
            if status == 1:
                # if trim was successful, then do one last evaluation with the final parameters.
                self.response = equations.eval_equations(X_free, time=0.0, modus='trim_full_output')
                self.successful = True
            else:
                self.response = {}
                self.successful = False
                logging.warning('Trim failed for subcase {}. The Trim solver reports: {}'.format(self.trimcase['subcase'], msg))
                return

    def exec_sim(self):
        import model_equations 
        if self.jcl.aero['method'] in [ 'mona_steady', 'hybrid'] and not hasattr(self.jcl, 'landinggear'):
            equations = model_equations.Steady(self, X0=self.response['X'], simcase=self.simcase)
        elif self.jcl.aero['method'] in [ 'nonlin_steady']:
            equations = model_equations.NonlinSteady(self, X0=self.response['X'], simcase=self.simcase)
        elif self.simcase['landinggear'] and self.jcl.landinggear['method'] == 'generic':
            logging.info('adding 2 x {} states for landing gear'.format(self.model.extragrid['n']))
            lg_states = []
            lg_derivatives = []
            for i in range(self.model.extragrid['n']):
                lg_states.append(self.response['p1'][i] - self.jcl.landinggear['para'][i]['stroke_length'] - self.jcl.landinggear['para'][i]['fitting_length'])
                lg_derivatives.append(self.response['dp1'][i])
            for i in range(self.model.extragrid['n']):
                lg_states.append(self.response['dp1'][i])
                lg_derivatives.append(self.response['ddp1'][i])
            # add lag states to system
            self.response['X'] = np.hstack((self.response['X'], lg_states ))
            self.response['Y'] = np.hstack((self.response['Y'][self.idx_state_derivatives + self.idx_input_derivatives], lg_derivatives, self.response['Y'][self.idx_outputs] ))
            self.n_lg_states = lg_states.__len__()
            self.idx_lg_states         = range(self.n_states+self.n_inputs, self.n_states+self.n_inputs+self.n_lg_states)
            self.idx_lg_derivatives    = range(self.n_state_derivatives+self.n_input_derivatives, self.n_state_derivatives+self.n_input_derivatives+self.n_lg_states)
            self.idx_outputs            = range(self.n_state_derivatives+self.n_input_derivatives+self.n_lg_states, self.n_state_derivatives+self.n_input_derivatives+self.n_lg_states+self.n_outputs)
            equations = model_equations.Landing(self, X0=self.response['X'], simcase=self.simcase)
        elif self.jcl.aero['method'] in [ 'mona_unsteady']:
            if 'disturbance' in self.simcase.keys():
                logging.info('adding disturbance of {} to state(s) '.format(self.simcase['disturbance']))
                self.response['X'][11+self.simcase['disturbance_mode']] += self.simcase['disturbance']
            # Initialize lag states with zero and extend steady response vectors X and Y
            # Distinguish between pyhsical rfa on panel level and generalized rfa. This influences the number of lag states.
            if self.jcl.aero.has_key('method_rfa') and self.jcl.aero['method_rfa'] == 'generalized':
                logging.error('Generalized RFA not yet implemented.')
            elif self.jcl.aero.has_key('method_rfa') and self.jcl.aero['method_rfa'] == 'halfgeneralized':
                n_modes = self.model.mass['n_modes'][self.model.mass['key'].index(self.trimcase['mass'])]
                logging.info('adding {} x {} unsteady lag states to the system'.format(2 * n_modes,self.model.aero['n_poles']))
                lag_states = np.zeros((2 * n_modes * self.model.aero['n_poles'])) 
            else:
                logging.info('adding {} x {} unsteady lag states to the system'.format(self.model.aerogrid['n'],self.model.aero['n_poles']))
                lag_states = np.zeros((self.model.aerogrid['n'] * self.model.aero['n_poles'])) 
            # add lag states to system
            self.response['X'] = np.hstack((self.response['X'], lag_states ))
            self.response['Y'] = np.hstack((self.response['Y'][self.idx_state_derivatives + self.idx_input_derivatives], lag_states, self.response['Y'][self.idx_outputs] ))
            self.n_lag_states = lag_states.__len__()
            self.idx_lag_states         = range(self.n_states+self.n_inputs, self.n_states+self.n_inputs+self.n_lag_states)
            self.idx_lag_derivatives    = range(self.n_state_derivatives+self.n_input_derivatives, self.n_state_derivatives+self.n_input_derivatives+self.n_lag_states)
            self.idx_outputs            = range(self.n_state_derivatives+self.n_input_derivatives+self.n_lag_states, self.n_state_derivatives+self.n_input_derivatives+self.n_lag_states+self.n_outputs)
            equations = model_equations.Unsteady(self, X0=self.response['X'], simcase=self.simcase)
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))

        X0 = self.response['X']
        dt = self.simcase['dt']
        t_final = self.simcase['t_final']
        logging.info('running time simulation for ' + str(t_final) + ' sec...')
#         integrator = RungeKutta4(equations.ode_arg_sorter).set_integrator(stepwidth=1e-4)
        integrator = ode(equations.ode_arg_sorter).set_integrator('vode', method='adams', nsteps=2000, rtol=1e-4, atol=1e-4, max_step=5e-4) # non-stiff: 'adams', stiff: 'bdf'
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
            #equations = model_equations.Unsteady(self, X0=self.response['X'], simcase=self.simcase)
            equations.eval_equations(self.response['X'], 0.0, modus='sim_full_output')
            for i_step in np.arange(0,len(t)):
                response_step = equations.eval_equations(X_t[i_step], t[i_step], modus='sim_full_output')
                for key in self.response.keys():
                    self.response[key] = np.vstack((self.response[key],response_step[key]))
                self.successful = True

        else:
            self.response = {}
            self.successful = False
            logging.warning('Integration failed!')
            return
            
            
    def iterative_trim(self):
        import model_equations # Warum muss der import hier stehen??
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid']:
            equations = model_equations.Steady(self)
        elif self.jcl.aero['method'] in [ 'cfd_steady']:
            equations = model_equations.CfdSteady(self)
            specific_io.check_para_path(self.jcl)
            specific_io.copy_para_file(self.jcl, self.trimcase)
            specific_io.check_tau_folders(self.jcl)
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
        X_free_0 = np.array(self.trimcond_X[:,2], dtype='float')[np.where((self.trimcond_X[:,1] == 'free'))[0]] # start trim from scratch
        
        if self.trimcase['maneuver'] == 'bypass':
            logging.info('running bypass...')
            self.response = equations.eval_equations(X_free_0, time=0.0, modus='trim_full_output')
        else:
            logging.info('running trim for ' + str(len(X_free_0)) + ' variables...')
            try:
                X_free, info, status, msg= so.fsolve(equations.eval_equations_iteratively, X_free_0, args=(0.0, 'trim'), full_output=True, epsfcn=1.0e-3, xtol=1.0e-3 )
            except model_equations.TauError as e:
                self.response = {}
                self.successful = False
                logging.warning('Trim failed for subcase {} due to TauError: {}'.format(self.trimcase['subcase'], e))
                return
            except model_equations.ConvergenceError as e:
                self.response = {}
                self.successful = False
                logging.warning('Trim failed for subcase {} due to ConvergenceError: {}'.format(self.trimcase['subcase'], e))
                return
            else:
                logging.info(msg)
                logging.info('function evaluations: ' + str(info['nfev']))
            
                # no errors, check trim status for success
                if status == 1:
                    # if trim was successful, then do one last evaluation with the final parameters.
                    self.response = equations.eval_equations_iteratively(X_free, time=0.0, modus='trim_full_output')
                    self.successful = True
                else:
                    self.response = {}
                    self.successful = False
                    logging.warning('Trim failed for subcase {}. The Trim solver reports: {}'.format(self.trimcase['subcase'], msg))
                    return
       