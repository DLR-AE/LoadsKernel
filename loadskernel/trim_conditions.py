import numpy as np
import logging

class TrimConditions:
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
        self.lg_states = None
        self.lg_derivatives = None
        self.lag_states = None
        
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
        # set states, derivatives, inputs and output parameters according to requested maneuver
        self.set_defaults()
        self.set_maneuver()
        self.add_engine()
        # append inputs to X vector...
        self.trimcond_X = np.vstack((self.states , self.inputs))
        self.n_states   = self.states.__len__()
        self.n_inputs   = self.inputs.__len__()
        self.idx_states = list(range(0,self.n_states))
        self.idx_inputs = list(range(self.n_states, self.n_states+self.n_inputs))
        
        # ... and input derivatives and outputs to Y vector
        self.trimcond_Y = np.vstack((self.state_derivatives, self.input_derivatives, self.outputs))
        self.n_state_derivatives    = self.state_derivatives.__len__()
        self.n_input_derivatives    = self.input_derivatives.__len__()
        self.n_outputs              = self.outputs.__len__()
        self.idx_state_derivatives  = list(range(0,self.n_state_derivatives))
        self.idx_input_derivatives  = list(range(self.n_state_derivatives, self.n_state_derivatives+self.n_input_derivatives))
        self.idx_outputs            = list(range(self.n_state_derivatives+self.n_input_derivatives, self.n_state_derivatives+self.n_input_derivatives+self.n_outputs))

    def set_defaults(self):
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
            ], dtype='<U18')
        for i_mode in range(n_modes):
            self.states = np.vstack((self.states ,  ['Uf'+str(i_mode), 'free', 0.0]))
        for i_mode in range(n_modes):
            self.states = np.vstack((self.states ,  ['dUf_dt'+str(i_mode), 'free', 0.0]))
        
        self.inputs = np.array([
            ['command_xi',   'free', 0.0,],  
            ['command_eta',  'free',  0.0,], 
            ['command_zeta', 'free',  0.0,],
            ['thrust',  'fix', 0.0]
            ], dtype='<U18')
        
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
            ], dtype='<U18')
            
        for i_mode in range(n_modes):
            self.state_derivatives = np.vstack((self.state_derivatives ,  ['dUf_dt'+str(i_mode), 'target', 0.0]))
        for i_mode in range(n_modes):
            self.state_derivatives = np.vstack((self.state_derivatives ,  ['d2Uf_d2t'+str(i_mode), 'target', 0.0]))
        
        self.input_derivatives = np.array([
            ['dcommand_xi',    'fix',  0.0,],
            ['dcommand_eta',   'fix',  0.0,],
            ['dcommand_zeta',  'fix',  0.0,],
            ['dthrust',        'fix',  0.0,],
            ], dtype='<U18')

        self.outputs = np.array([
            ['Nz',       'target',  self.trimcase['Nz']],
            ['Vtas',     'free',    vtas,],
            ['beta',     'free',    0.0]
            ], dtype='<U18')
    
    def set_maneuver(self):
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
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],1] = 'fix'

            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dx'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dz'))[0][0],2] = self.trimcase['dz']
            self.outputs[np.where((self.outputs[:,0] == 'Vtas'))[0][0],1] = 'target'
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dp'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dr'))[0][0],1] = 'free'
             
        # -----------------------
        # --- 3 wheel landing --- 
        # -----------------------
        elif self.trimcase['maneuver'] in ['L3wheel']:
            logging.info('setting trim conditions to "3 wheel landing"')
            # inputs
            self.inputs[np.where((self.inputs[:,0] == 'command_xi'))[0][0],1] = 'fix'
            self.inputs[np.where((self.inputs[:,0] == 'command_zeta'))[0][0],1] = 'fix'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dx'))[0][0],1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dz'))[0][0],2] = self.trimcase['dz']
            self.states[np.where((self.states[:,0] == 'theta'))[0][0],1] = 'target'
            self.states[np.where((self.states[:,0] == 'theta'))[0][0],2] = self.trimcase['theta']
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:,0] == 'dp'))[0][0],1] = 'free'
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
            i_atmo = self.model.atmo['key'].index(self.trimcase['altitude'])
            vtas = self.trimcase['Ma'] * self.model.atmo['a'][i_atmo]
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

    def add_engine(self):
        if hasattr(self.jcl, 'engine'):
            if 'thrust' in self.trimcase and self.trimcase['thrust'] in ['free', 'balanced']:
                logging.info('setting trim conditions to balanced thrust')
                # inputs
                self.inputs[np.where((self.inputs[:,0] == 'thrust'))[0][0],1] = 'free'
                # outputs
                self.state_derivatives[np.where((self.state_derivatives[:,0] == 'du'))[0][0],1] = 'target'
            elif 'thrust' in self.trimcase:
                logging.info('setting trim conditions to {} [N] thrust per engine'.format(self.trimcase['thrust']))
                # inputs
                self.inputs[np.where((self.inputs[:,0] == 'thrust'))[0][0],2] = self.trimcase['thrust']

    def add_landinggear(self):
        self.lg_states = []
        self.lg_derivatives = []
        
        if self.jcl.landinggear['method'] in ['generic']:
            logging.info('adding 2 x {} states for landing gear'.format(self.model.extragrid['n']))
            for i in range(self.model.extragrid['n']):
                self.lg_states.append(self.response['p1'][i] - self.jcl.landinggear['para'][i]['stroke_length'] - self.jcl.landinggear['para'][i]['fitting_length'])
                self.lg_derivatives.append(self.response['dp1'][i])
            for i in range(self.model.extragrid['n']):
                self.lg_states.append(self.response['dp1'][i])
                self.lg_derivatives.append(self.response['ddp1'][i])
        elif self.jcl.landinggear['method'] in ['skid']:  
            pass
        # update response with landing gear states
        self.response['X'] = np.hstack((self.response['X'], self.lg_states ))
        self.response['Y'] = np.hstack((self.response['Y'][self.idx_state_derivatives + self.idx_input_derivatives], self.lg_derivatives, self.response['Y'][self.idx_outputs] ))
        # set indices
        self.n_lg_states = self.lg_states.__len__()
        self.idx_lg_states         = list(range(self.n_states+self.n_inputs, self.n_states+self.n_inputs+self.n_lg_states))
        self.idx_lg_derivatives    = list(range(self.n_state_derivatives+self.n_input_derivatives, self.n_state_derivatives+self.n_input_derivatives+self.n_lg_states))
        self.idx_outputs            = list(range(self.n_state_derivatives+self.n_input_derivatives+self.n_lg_states, self.n_state_derivatives+self.n_input_derivatives+self.n_lg_states+self.n_outputs))
    
    def add_lagstates(self):
        # Initialize lag states with zero and extend steady response vectors X and Y
        # Distinguish between pyhsical rfa on panel level and generalized rfa. This influences the number of lag states.
        if 'method_rfa' in self.jcl.aero and self.jcl.aero['method_rfa'] == 'generalized':
            logging.error('Generalized RFA not yet implemented.')
        elif 'method_rfa' in self.jcl.aero and self.jcl.aero['method_rfa'] == 'halfgeneralized':
            n_modes = self.model.mass['n_modes'][self.model.mass['key'].index(self.trimcase['mass'])]
            logging.info('adding {} x {} unsteady lag states to the system'.format(2 * n_modes,self.model.aero['n_poles']))
            self.lag_states = np.zeros((2 * n_modes * self.model.aero['n_poles'])) 
        else:
            logging.info('adding {} x {} unsteady lag states to the system'.format(self.model.aerogrid['n'],self.model.aero['n_poles']))
            self.lag_states = np.zeros((self.model.aerogrid['n'] * self.model.aero['n_poles'])) 
        # update response with lag states
        self.response['X'] = np.hstack((self.response['X'], self.lag_states ))
        self.response['Y'] = np.hstack((self.response['Y'][self.idx_state_derivatives + self.idx_input_derivatives], self.lag_states, self.response['Y'][self.idx_outputs] ))
        # set indices
        self.n_lag_states = self.lag_states.__len__()
        self.idx_lag_states         = list(range(self.n_states+self.n_inputs, self.n_states+self.n_inputs+self.n_lag_states))
        self.idx_lag_derivatives    = list(range(self.n_state_derivatives+self.n_input_derivatives, self.n_state_derivatives+self.n_input_derivatives+self.n_lag_states))
        self.idx_outputs            = list(range(self.n_state_derivatives+self.n_input_derivatives+self.n_lag_states, self.n_state_derivatives+self.n_input_derivatives+self.n_lag_states+self.n_outputs))
            
        
        
            