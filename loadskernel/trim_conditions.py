import numpy as np
import logging
from loadskernel.io_functions.data_handling import load_hdf5_dict
from loadskernel.cfd_interfaces.mpi_helper import setup_mpi


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

        # Initialize MPI interface
        self.have_mpi, self.comm, self.status, self.myid = setup_mpi()

    def set_trimcond(self):
        # set states, derivatives, inputs and output parameters according to requested maneuver
        self.set_defaults()
        self.set_maneuver()
        self.add_engine()
        self.add_stabilizer_setting()
        self.add_flap_setting()
        # append inputs to X vector...
        self.trimcond_X = np.vstack((self.states, self.inputs))
        self.n_states = len(self.states)
        self.n_inputs = len(self.inputs)
        self.idx_states = list(range(0, self.n_states))
        self.idx_inputs = list(
            range(self.n_states, self.n_states + self.n_inputs))

        # ... and input derivatives and outputs to Y vector
        self.trimcond_Y = np.vstack((self.state_derivatives, self.input_derivatives, self.outputs))
        self.n_state_derivatives = len(self.state_derivatives)
        self.n_input_derivatives = len(self.input_derivatives)
        self.n_outputs = len(self.outputs)
        self.idx_state_derivatives = list(range(0, self.n_state_derivatives))
        self.idx_input_derivatives = list(range(
            self.n_state_derivatives, self.n_state_derivatives + self.n_input_derivatives))
        self.idx_outputs = list(range(self.n_state_derivatives + self.n_input_derivatives, self.n_state_derivatives
                                      + self.n_input_derivatives + self.n_outputs))

    def set_defaults(self):
        """
        The trim condition is a mix of strings and numeric values. This is possible with np.array of 'object' type.
        However, object-arrays have some disadvantages, such as lack of compatibility with mathematical operations
        such as np.cross() or HDF5 files, but present the trim conditions in a nice and compact form. For the model
        equations, the numerical values are converted into np.arrays of 'float' type.
        (This is much better than the old methods of converting the numerical values to 'string' type arrays.)

        Requirement for a determined trim condition: 'free' parameters in trimcond_X == 'target' parameters in trimcond_Y
        """
        # init
        self.atmo = load_hdf5_dict(self.model['atmo'][self.trimcase['altitude']])
        self.n_modes = self.model['mass'][self.trimcase['mass']]['n_modes'][()]
        vtas = self.trimcase['Ma'] * self.atmo['a']
        # starting with a small angle of attack increases the performance and convergence of the CFD solution
        theta = 0.0 / 180.0 * np.pi
        u = vtas * np.cos(theta)
        w = vtas * np.sin(theta)
        z = -self.atmo['h']

        # Default trim conditions in case maneuver = ''.
        # right hand side
        self.states = np.array(
            [['x', 'fix', 0.0],
             ['y', 'fix', 0.0],
             ['z', 'fix', z],
             ['phi', 'fix', 0.0],
             ['theta', 'free', theta],  # dependent on u and w if dz = 0
             ['psi', 'fix', 0.0],
             ['u', 'free', u],
             ['v', 'fix', 0.0],
             ['w', 'free', w],
             ['p', 'fix', self.trimcase['p']],
             ['q', 'fix', self.trimcase['q']],
             ['r', 'fix', self.trimcase['r']],
             ], dtype=object)
        for i_mode in range(1, self.n_modes + 1):
            self.states = np.vstack((self.states, np.array(
                ['Uf' + str(i_mode), 'free', 0.0], dtype=object)))
        for i_mode in range(1, self.n_modes + 1):
            self.states = np.vstack((self.states, np.array(
                ['dUf_dt' + str(i_mode), 'fix', 0.0], dtype=object)))

        self.inputs = np.array(
            [['command_xi', 'free', 0.0],
             ['command_eta', 'free', 0.0],
             ['command_zeta', 'free', 0.0],
             ['thrust', 'fix', 0.0],
             ['stabilizer', 'fix', 0.0],
             ['flap_setting', 'fix', 0.0],
             ], dtype=object)

        # left hand side
        self.state_derivatives = np.array(
            [['dx', 'target', vtas],  # dx = vtas if dz = 0
             ['dy', 'free', 0.0],
             ['dz', 'target', 0.0],
             ['dphi', 'free', 0.0],
             ['dtheta', 'free', 0.0],
             ['dpsi', 'free', 0.0],
             ['du', 'free', 0.0],
             ['dv', 'free', 0.0],
             ['dw', 'free', 0.0],
             ['dp', 'target', self.trimcase['pdot']],
             ['dq', 'target', self.trimcase['qdot']],
             ['dr', 'target', self.trimcase['rdot']],
             ], dtype=object)

        for i_mode in range(1, self.n_modes + 1):
            self.state_derivatives = np.vstack((self.state_derivatives,
                                                np.array(['dUf_dt' + str(i_mode), 'fix', 0.0], dtype=object)))
        for i_mode in range(1, self.n_modes + 1):
            self.state_derivatives = np.vstack((self.state_derivatives,
                                                np.array(['d2Uf_d2t' + str(i_mode), 'target', 0.0], dtype=object)))

        self.input_derivatives = np.array(
            [['dcommand_xi', 'fix', 0.0],
             ['dcommand_eta', 'fix', 0.0],
             ['dcommand_zeta', 'fix', 0.0],
             ['dthrust', 'fix', 0.0],
             ['dstabilizer', 'fix', 0.0],
             ['dflap_setting', 'fix', 0.0],
             ], dtype=object)

        self.outputs = np.array(
            [['Nz', 'target', self.trimcase['Nz']],
             ['Vtas', 'free', vtas, ],
             ['beta', 'free', 0.0]
             ], dtype=object)

    def set_maneuver(self):
        """
        Long list of if-statements can be replace by match-case in the future (with python 3.10), see
        https://docs.python.org/3.10/whatsnew/3.10.html#pep-634-structural-pattern-matching
        Until then, the Flake8 warning C901 'TrimConditions.set_maneuver' is too complex (16) is squelched because I see no
        other way to do the selection.
        """

        # Trim about pitch axis only
        if self.trimcase['maneuver'] in ['pitch', 'elevator']:
            logging.info('Setting trim conditions to "pitch"')
            # inputs
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 1] = 'fix'

            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dp'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dr'))[0][0], 1] = 'free'

        # Trim about pitch axis only, but with stabilizer
        elif self.trimcase['maneuver'] in ['stabilizer']:
            logging.info('Setting trim conditions to "stabilizer"')
            # inputs
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_eta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'stabilizer'))[0][0], 1] = 'free'
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dp'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dr'))[0][0], 1] = 'free'

        # Trim about pitch and roll axis, no yaw
        elif self.trimcase['maneuver'] == 'pitch&roll':
            logging.info('Setting trim conditions to "pitch&roll"')
            # inputs
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 1] = 'fix'
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dr'))[0][0], 1] = 'free'

        # Trim about pitch and yaw axis, no roll
        elif self.trimcase['maneuver'] == 'pitch&yaw':
            logging.info('Setting trim conditions to "pitch&yaw"')
            # inputs
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 1] = 'fix'
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dp'))[0][0], 1] = 'free'

        # Trim conditin for level landing at a given sink rate 'dz'
        elif self.trimcase['maneuver'] in ['L1wheel', 'L2wheel']:
            logging.info('Setting trim conditions to "level landing"')
            # inputs
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 1] = 'fix'

            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dx'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dz'))[0][0], 2] = self.trimcase['dz']
            self.outputs[np.where((self.outputs[:, 0] == 'Vtas'))[0][0], 1] = 'target'
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dp'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dr'))[0][0], 1] = 'free'

        # Trim condition with prescribed angle theta, e.g. for 3 wheel landing
        elif self.trimcase['maneuver'] in ['L3wheel']:
            logging.info('Setting trim conditions to "3 wheel landing"')
            # inputs
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 1] = 'fix'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dx'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dz'))[0][0], 2] = self.trimcase['dz']
            self.states[np.where((self.states[:, 0] == 'theta'))[0][0], 1] = 'target'
            self.states[np.where((self.states[:, 0] == 'theta'))[0][0], 2] = self.trimcase['theta']
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dp'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dr'))[0][0], 1] = 'free'
            self.outputs[np.where((self.outputs[:, 0] == 'Nz'))[0][0], 1] = 'free'
            self.outputs[np.where((self.outputs[:, 0] == 'Vtas'))[0][0], 1] = 'target'

        # Trim condition for a glider. Sink rate 'w' is allowed so that the velocity remains constant (du=0.0)
        elif self.trimcase['maneuver'] == 'segelflug':
            logging.info('Setting trim conditions to "segelflug"')
            # inputs
            # without changes
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dx'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dz'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'du'))[0][0], 1] = 'target'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dw'))[0][0], 1] = 'target'
            self.outputs[np.where((self.outputs[:, 0] == 'Nz'))[0][0], 1] = 'free'
            self.outputs[np.where((self.outputs[:, 0] == 'Vtas'))[0][0], 1] = 'target'

        # Trim conditions with alpha only, e.g. for Pratt formula
        elif self.trimcase['maneuver'] == 'pratt':
            logging.info('Setting trim conditions to "pratt"')
            # inputs
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_eta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 1] = 'fix'
            # outputs
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dp'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dq'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dr'))[0][0], 1] = 'free'

        # Trim condition with prescribed control surface deflections Xi and Zeta
        elif self.trimcase['maneuver'] == 'Xi&Zeta-fixed':
            logging.info('Setting trim conditions to "Xi&Zeta-fixed"')
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 2] = self.trimcase['command_xi']
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 2] = self.trimcase['command_zeta']
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dp'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dr'))[0][0], 1] = 'free'

        # Trim condition with prescribed control surface deflections, accelerations free
        elif self.trimcase['maneuver'] == 'CS-fixed':
            logging.info('Setting trim conditions to "CS-fixed"')
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_eta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 2] = self.trimcase['command_xi']
            self.inputs[np.where((self.inputs[:, 0] == 'command_eta'))[0][0], 2] = self.trimcase['command_eta']
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 2] = self.trimcase['command_zeta']
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dp'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dq'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dr'))[0][0], 1] = 'free'

        # Trim condition with prescribed control surface deflections, roll, pitch and yaw rates free
        elif self.trimcase['maneuver'] == 'CS&Acc-fixed':
            logging.info('Setting trim conditions to "CS&Acc-fixed"')
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_eta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 2] = self.trimcase['command_xi']
            self.inputs[np.where((self.inputs[:, 0] == 'command_eta'))[0][0], 2] = self.trimcase['command_eta']
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 2] = self.trimcase['command_zeta']
            self.states[np.where((self.states[:, 0] == 'p'))[0][0], 1] = 'free'
            self.states[np.where((self.states[:, 0] == 'q'))[0][0], 1] = 'free'
            self.states[np.where((self.states[:, 0] == 'r'))[0][0], 1] = 'free'

        # Trim condition that allows sideslip
        elif self.trimcase['maneuver'] == 'sideslip':
            logging.info('Setting trim conditions to "sideslip"')
            # fixed roll and yaw control
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 2] = self.trimcase['command_xi']
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 2] = self.trimcase['command_zeta']
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dp'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dr'))[0][0], 1] = 'free'

            # set sideslip condition
            self.states[np.where((self.states[:, 0] == 'psi'))[0][0], 1] = 'free'
            self.states[np.where((self.states[:, 0] == 'v'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dy'))[0][0], 1] = 'target'
            self.outputs[np.where((self.outputs[:, 0] == 'beta'))[0][0], 1] = 'target'
            self.outputs[np.where((self.outputs[:, 0] == 'beta'))[0][0], 2] = self.trimcase['beta']

        # Trim condition for a coordinated sideslip at a given angle beta
        elif self.trimcase['maneuver'] == 'coordinated_sideslip':
            logging.info('Setting trim conditions to "sideslip"')

            # set sideslip condition
            self.states[np.where((self.states[:, 0] == 'psi'))[0][0], 1] = 'free'
            self.states[np.where((self.states[:, 0] == 'v'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dy'))[0][0], 1] = 'target'
            self.outputs[np.where((self.outputs[:, 0] == 'beta'))[0][0], 1] = 'target'
            self.outputs[np.where((self.outputs[:, 0] == 'beta'))[0][0], 2] = self.trimcase['beta']

        # Trim condition for no trim / bypass with prescribed euler angles and control surface deflections.
        # Used for bypass analyses and for debugging.
        elif self.trimcase['maneuver'] in ['bypass', 'derivatives']:
            logging.info('Setting trim conditions to "bypass"')
            vtas = self.trimcase['Ma'] * self.atmo['a']
            theta = self.trimcase['theta']
            u = vtas * np.cos(theta)
            w = vtas * np.sin(theta)
            # inputs
            self.states[np.where((self.states[:, 0] == 'phi'))[0][0], 1] = 'fix'
            self.states[np.where((self.states[:, 0] == 'phi'))[0][0], 2] = self.trimcase['phi']
            self.states[np.where((self.states[:, 0] == 'theta'))[0][0], 1] = 'fix'
            self.states[np.where((self.states[:, 0] == 'theta'))[0][0], 2] = theta
            self.states[np.where((self.states[:, 0] == 'u'))[0][0], 2] = u
            self.states[np.where((self.states[:, 0] == 'w'))[0][0], 2] = w
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 2] = self.trimcase['command_xi']
            self.inputs[np.where((self.inputs[:, 0] == 'command_eta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_eta'))[0][0], 2] = self.trimcase['command_eta']
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 2] = self.trimcase['command_zeta']
            # outputs
            self.outputs[np.where((self.outputs[:, 0] == 'Nz'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dp'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dq'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dr'))[0][0], 1] = 'free'

        # Trim condtions for a windtunnel-like setting.
        # All remaining trim conditions are given, only the structural flexibility is calculated.
        elif self.trimcase['maneuver'] in ['windtunnel']:
            logging.info('Setting trim conditions to "windtunnel"')
            vtas = self.trimcase['Ma'] * self.atmo['a']
            theta = self.trimcase['theta']
            u = vtas * np.cos(theta)
            w = vtas * np.sin(theta)
            # inputs
            self.states[np.where((self.states[:, 0] == 'phi'))[0][0], 1] = 'fix'
            self.states[np.where((self.states[:, 0] == 'phi'))[0][0], 2] = self.trimcase['phi']
            self.states[np.where((self.states[:, 0] == 'theta'))[0][0], 1] = 'fix'
            self.states[np.where((self.states[:, 0] == 'theta'))[0][0], 2] = theta
            self.states[np.where((self.states[:, 0] == 'u'))[0][0], 1] = 'fix'
            self.states[np.where((self.states[:, 0] == 'u'))[0][0], 2] = u
            self.states[np.where((self.states[:, 0] == 'w'))[0][0], 1] = 'fix'
            self.states[np.where((self.states[:, 0] == 'w'))[0][0], 2] = w
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_xi'))[0][0], 2] = self.trimcase['command_xi']
            self.inputs[np.where((self.inputs[:, 0] == 'command_eta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_eta'))[0][0], 2] = self.trimcase['command_eta']
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 1] = 'fix'
            self.inputs[np.where((self.inputs[:, 0] == 'command_zeta'))[0][0], 2] = self.trimcase['command_zeta']
            # outputs
            self.outputs[np.where((self.outputs[:, 0] == 'Nz'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dx'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dz'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dp'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dq'))[0][0], 1] = 'free'
            self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'dr'))[0][0], 1] = 'free'

        else:
            logging.info('Setting trim conditions to "default"')

    def add_stabilizer_setting(self):
        if 'stabilizer' in self.trimcase:
            self.inputs[np.where((self.inputs[:, 0] == 'stabilizer'))[0][0], 2] = self.trimcase['stabilizer']

    def add_flap_setting(self):
        if 'flap_setting' in self.trimcase:
            self.inputs[np.where((self.inputs[:, 0] == 'flap_setting'))[0][0], 2] = self.trimcase['flap_setting']

    def add_engine(self):
        if hasattr(self.jcl, 'engine'):
            if 'thrust' in self.trimcase and self.trimcase['thrust'] in ['free', 'balanced']:
                logging.info('Setting trim conditions to "balanced thrust"')
                # inputs
                self.inputs[np.where((self.inputs[:, 0] == 'thrust'))[0][0], 1] = 'free'
                # outputs
                self.state_derivatives[np.where((self.state_derivatives[:, 0] == 'du'))[0][0], 1] = 'target'
            elif 'thrust' in self.trimcase:
                logging.info('Setting trim conditions to {} [N] thrust per engine'.format(self.trimcase['thrust']))
                # inputs
                self.inputs[np.where((self.inputs[:, 0] == 'thrust'))[0][0], 2] = self.trimcase['thrust']

    def add_landinggear(self):
        self.lg_states = []
        self.lg_derivatives = []

        if self.jcl.landinggear['method'] in ['generic']:
            logging.info('Adding 2 x {} states for landing gear'.format(self.model.extragrid['n']))
            for i in range(self.model.extragrid['n']):
                self.lg_states.append(self.response['p1'][i] - self.jcl.landinggear['para'][i]['stroke_length']
                                      - self.jcl.landinggear['para'][i]['fitting_length'])
                self.lg_derivatives.append(self.response['dp1'][i])
            for i in range(self.model.extragrid['n']):
                self.lg_states.append(self.response['dp1'][i])
                self.lg_derivatives.append(self.response['ddp1'][i])
        elif self.jcl.landinggear['method'] in ['skid']:
            pass

        # expand 1d list to 2d array
        self.lg_states = np.expand_dims(self.lg_states, axis=0)
        self.lg_derivatives = np.expand_dims(self.lg_derivatives, axis=0)
        # update response with landing gear states
        self.response['X'] = np.append(self.response['X'], self.lg_states, axis=1)
        self.response['Y'] = np.hstack((self.response['Y'][:, self.idx_state_derivatives + self.idx_input_derivatives],
                                        self.lg_derivatives, self.response['Y'][:, self.idx_outputs]))
        # set indices
        self.n_lg_states = self.lg_states.shape[1]
        self.idx_lg_states = list(range(self.n_states + self.n_inputs, self.n_states + self.n_inputs + self.n_lg_states))
        self.idx_lg_derivatives = list(range(self.n_state_derivatives + self.n_input_derivatives, self.n_state_derivatives
                                             + self.n_input_derivatives + self.n_lg_states))
        self.idx_outputs = list(range(self.n_state_derivatives + self.n_input_derivatives + self.n_lg_states,
                                      self.n_state_derivatives + self.n_input_derivatives + self.n_lg_states + self.n_outputs))

    def add_lagstates(self):
        # Initialize lag states with zero and extend steady response vectors X and Y
        # Distinguish between pyhsical rfa on panel level and generalized rfa. This influences the number of lag states.
        n_poles = self.model['aero'][self.trimcase['aero']]['n_poles'][()]
        if 'method_rfa' in self.jcl.aero and self.jcl.aero['method_rfa'] == 'generalized':
            logging.error('Generalized RFA not yet implemented.')
        elif 'method_rfa' in self.jcl.aero and self.jcl.aero['method_rfa'] == 'halfgeneralized':
            logging.info('Adding {} x {} unsteady lag states to the system'.format(2 * self.n_modes, n_poles))
            self.lag_states = np.zeros((1, 2 * self.n_modes * n_poles))
        else:
            logging.info('Adding {} x {} unsteady lag states to the system'.format(self.model['aerogrid']['n'][()], n_poles))
            self.lag_states = np.zeros((1, self.model['aerogrid']['n'][()] * n_poles))
        # update response with lag states
        self.response['X'] = np.append(self.response['X'], self.lag_states, axis=1)
        self.response['Y'] = np.hstack((self.response['Y'][:, self.idx_state_derivatives + self.idx_input_derivatives],
                                        self.lag_states, self.response['Y'][:, self.idx_outputs]))
        # set indices
        self.n_lag_states = self.lag_states.shape[1]
        self.idx_lag_states = list(range(self.n_states + self.n_inputs, self.n_states + self.n_inputs + self.n_lag_states))
        self.idx_lag_derivatives = list(range(self.n_state_derivatives + self.n_input_derivatives, self.n_state_derivatives
                                              + self.n_input_derivatives + self.n_lag_states))
        self.idx_outputs = list(range(self.n_state_derivatives + self.n_input_derivatives + self.n_lag_states,
                                      self.n_state_derivatives + self.n_input_derivatives + self.n_lag_states
                                      + self.n_outputs))

    def set_modal_states_fix(self):
        # remove modes from trimcond_Y and _Y
        for i_mode in range(1, self.n_modes + 1):
            self.trimcond_X[np.where((self.trimcond_X[:, 0] == 'Uf' + str(i_mode)))[0][0], 1] = 'fix'
            self.trimcond_X[np.where((self.trimcond_X[:, 0] == 'dUf_dt' + str(i_mode)))[0][0], 1] = 'fix'
            self.trimcond_Y[np.where((self.trimcond_Y[:, 0] == 'dUf_dt' + str(i_mode)))[0][0], 1] = 'fix'
            self.trimcond_Y[np.where((self.trimcond_Y[:, 0] == 'd2Uf_d2t' + str(i_mode)))[0][0], 1] = 'fix'
