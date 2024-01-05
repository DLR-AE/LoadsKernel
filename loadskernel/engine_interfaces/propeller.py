# encoding=utf8
import copy
import logging
import sys
import yaml
import numpy as np

from panelaero import VLM

from loadskernel import equations
from loadskernel import spline_functions
from loadskernel import spline_rules
from loadskernel.solution_tools import calc_drehmatrix
from loadskernel import build_aero_functions

try:
    from pyPropMat.pyPropMat import Prop
except ImportError:
    pass


class PyPropMat4Loads(object):
    """
    This class calculates the aerodynamic forces and moments of a propeller following equations 1 to 4 in [1].
    The the aerodynamic derivatives are calculated using PyPropMat provided by Christopher Koch, AE-SIM from [2].

    Inputs:
    - filename of the input-file for PyPropMat (*.yaml)
    - flight_condition_dict (dict): dictionary with flight data
        {V (float): true air speed
        Ma (float): flight mach number
        Rho (float): air density
        N_rpm (float): propeller revolutions in rpm}

    Sign conventions:
    - The derivatives are given in a forward-right-down coordinate system.
    - Theta and q are defined a a positive rotation about the y-axis.
    - Psi and r are defined a a positive rotation about the z-axis.

    [1] Rodden, W., and Rose, T. Propeller/nacelle whirl flutter addition to MSC/nastran, in
    Proceedings of the 1989 MSC World User's Conference, 1989.

    [2] https://phabricator.ae.go.dlr.de/diffusion/180/
    """

    def __init__(self, filename):
        # Check if pyPropMat was imported successfully, see try/except statement in the import section.
        if "pyPropMat" in sys.modules:
            self.prop = Prop(filename)
        else:
            logging.error('pyPropMat was/could NOT be imported!')
        self.prop_coeffs = None

    def calc_loads(self, parameter_dict):
        """
        Convert signs from structural coordinates (aft-right-up) to propeller coordinates (forward-right-down):
        positive alpha = wind from below = positive theta
        positive beta = wind from the right side = negative psi
        positive structural q = pitch-up = positive propeller q
        positive structural r = yaw to the left = negative proller r

        Convert signs propeller coordinates (forward-right-down) to structural coordinates (aft-right-up):
        switch Fz and Mz
        """
        diameter = parameter_dict['diameter']
        Vtas = parameter_dict['Vtas']
        q_dyn = parameter_dict['q_dyn']
        theta = parameter_dict['alpha']
        psi = -parameter_dict['beta']
        q = parameter_dict['pqr'][1]
        r = -parameter_dict['pqr'][2]

        # Calculate the area of the propeller disk with S = pi * r^2
        S = np.pi * (0.5 * diameter) ** 2.0

        # Calculate the coefficients for the current operational point. It is sufficient to calculate them only once
        # at the first run.
        if self.prop_coeffs is None:
            flight_condition_dict = {'V': Vtas,
                                     'Ma': parameter_dict['Ma'],
                                     'Rho': parameter_dict['rho'],
                                     'N_rpm': parameter_dict['RPM']}
            self.prop_coeffs = self.prop.get_propeller_coefficients(flight_condition_dict, include_lag=False)

        # Get the aerodynamic coefficients
        Cz_theta = self.prop_coeffs['z_theta']
        Cz_psi = self.prop_coeffs['z_psi']
        Cz_q = self.prop_coeffs['z_q']
        Cz_r = self.prop_coeffs['z_r']
        Cy_theta = self.prop_coeffs['y_theta']
        Cy_psi = self.prop_coeffs['y_psi']
        Cy_q = self.prop_coeffs['y_q']
        Cy_r = self.prop_coeffs['y_r']
        Cm_theta = self.prop_coeffs['m_theta']
        Cm_psi = self.prop_coeffs['m_psi']
        Cm_q = self.prop_coeffs['m_q']
        Cm_r = self.prop_coeffs['m_r']
        Cn_theta = self.prop_coeffs['n_theta']
        Cn_psi = self.prop_coeffs['n_psi']
        Cn_q = self.prop_coeffs['n_q']
        Cn_r = self.prop_coeffs['n_r']

        # initialize empty force vector
        P_prop = np.zeros(6)
        # Side force Fy, equation 3 in [1]
        P_prop[1] += q_dyn * S * (Cy_theta * theta + Cy_psi * psi + Cy_q * q * diameter / (2.0 * Vtas)
                                  + Cy_r * r * diameter / (2.0 * Vtas))
        # Lift force Fz, equation 1 in [1], convert signs (see above)
        P_prop[2] -= q_dyn * S * (Cz_theta * theta + Cz_psi * psi + Cz_q * q * diameter / (2.0 * Vtas)
                                  + Cz_r * r * diameter / (2.0 * Vtas))
        # Pitching moment My, equation 2 in [1]
        P_prop[4] += q_dyn * S * diameter * (Cm_theta * theta + Cm_psi * psi + Cm_q * q * diameter / (2.0 * Vtas)
                                             + Cm_r * r * diameter / (2.0 * Vtas))
        # Yawing moment Mz, equation 4 in [1], convert signs (see above)
        P_prop[5] -= q_dyn * S * diameter * (Cn_theta * theta + Cn_psi * psi + Cn_q * q * diameter / (2.0 * Vtas)
                                             + Cn_r * r * diameter / (2.0 * Vtas))

        # Check: An onflow from below (pos. alpha = neg. Theta) creates an upward lift (pos. Fz / P_prop[2])
        return P_prop


def read_propeller_input(filename):
    # construct an aerodynamic grid in the y-direction (like a starboard wing)
    caerocards = []
    EID = 1000
    logging.info('Read propeller input from: %s' % filename)
    with open(filename, 'r') as ymlfile:
        data = yaml.safe_load(ymlfile)
    n_span = len(data['tabulatedData']) - 1
    n_chord = 8
    for i_strip in range(n_span):
        # read first line of CAERO card
        caerocard = {'EID': EID,
                     'CP': 0,
                     'n_span': 1,  # n_boxes
                     'n_chord': n_chord,  # n_boxes
                     }
        # read second line of CAERO card
        caerocard['X1'] = np.array([-0.5 * data['tabulatedData'][i_strip][1], data['tabulatedData'][i_strip][0], 0.0])
        caerocard['X2'] = np.array([+0.5 * data['tabulatedData'][i_strip][1], data['tabulatedData'][i_strip][0], 0.0])
        caerocard['X3'] = np.array([+0.5 * data['tabulatedData'][i_strip + 1][1], data['tabulatedData'][i_strip + 1][0], 0.0])
        caerocard['X4'] = np.array([-0.5 * data['tabulatedData'][i_strip + 1][1], data['tabulatedData'][i_strip + 1][0], 0.0])
        # read the pitch angel, given in the third column
        if len(data['tabulatedData'][i_strip]) == 3:
            caerocard['cam_rad'] = data['tabulatedData'][i_strip][2] / 180.0 * np.pi
        else:
            caerocard['cam_rad'] = 0.0

        caerocards.append(caerocard)
        EID += 1000

    # from CAERO cards, construct corner points... '
    # then, combine four corner points to one panel
    grid_ID = 0  # the file number is used to set a range of grid IDs
    caero_grid = {'ID': [], 'offset': []}
    caero_panels = {"ID": [], 'CP': [], 'CD': [], "cornerpoints": []}
    cam_rad = []
    for caerocard in caerocards:
        # calculate LE, Root and Tip vectors [x,y,z]^T
        LE = caerocard['X4'] - caerocard['X1']
        Root = caerocard['X2'] - caerocard['X1']
        Tip = caerocard['X3'] - caerocard['X4']

        # assume equidistant spacing
        d_chord = np.linspace(0.0, 1.0, caerocard['n_chord'] + 1)
        # assume equidistant spacing
        d_span = np.linspace(0.0, 1.0, caerocard['n_span'] + 1)

        # build matrix of corner points
        # index based on n_divisions
        grids_map = np.zeros((caerocard['n_chord'] + 1, caerocard['n_span'] + 1), dtype='int')
        for i_strip in range(caerocard['n_span'] + 1):
            for i_row in range(caerocard['n_chord'] + 1):
                offset = caerocard['X1'] + LE * d_span[i_strip] + (Root * (1.0 - d_span[i_strip])
                                                                   + Tip * d_span[i_strip]) * d_chord[i_row]
                caero_grid['ID'].append(grid_ID)
                caero_grid['offset'].append(offset)
                grids_map[i_row, i_strip] = grid_ID
                grid_ID += 1
        # build panels from cornerpoints
        # index based on n_boxes
        panel_ID = caerocard['EID']
        for i_strip in range(caerocard['n_span']):
            for i_row in range(caerocard['n_chord']):
                caero_panels['ID'].append(panel_ID)
                caero_panels['CP'].append(caerocard['CP'])  # applying CP of CAERO card to all grids
                caero_panels['CD'].append(caerocard['CP'])
                caero_panels['cornerpoints'].append([grids_map[i_row, i_strip], grids_map[i_row + 1, i_strip],
                                                     grids_map[i_row + 1, i_strip + 1], grids_map[i_row, i_strip + 1]])
                cam_rad.append(caerocard['cam_rad'])
                panel_ID += 1
    caero_panels['ID'] = np.array(caero_panels['ID'])
    caero_panels['CP'] = np.array(caero_panels['CP'])
    caero_panels['CD'] = np.array(caero_panels['CD'])
    caero_panels['cornerpoints'] = np.array(caero_panels['cornerpoints'])
    caero_grid['ID'] = np.array(caero_grid['ID'])
    caero_grid['offset'] = np.array(caero_grid['offset'])
    cam_rad = np.array(cam_rad)
    return caero_grid, caero_panels, cam_rad


class VLM4PropModel(object):

    def __init__(self, filename, coord, atmo):
        self.filename = filename
        self.coord = coord
        self.atmo = atmo

    def build_aerogrid(self):
        self.aerogrid = build_aero_functions.build_aerogrid(self.filename, method_caero='VLM4Prop')
        logging.info('The aerodynamic propeller model consists of {} panels.'.format(self.aerogrid['n']))

    def build_pacgrid(self):
        A = np.sum(self.aerogrid['A'])
        b_ref = self.aerogrid['offset_P3'][:, 1].max()
        mean_aero_choord = A / b_ref
        self.pacgrid = {'ID': np.array([0]),
                        'offset': np.array([[0.0, 0.0, 0.0]]),
                        'set': np.array([[0, 1, 2, 3, 4, 5]]),
                        'CD': np.array([0]),
                        'CP': np.array([0]),
                        'coord_desc': 'bodyfixed',
                        'A_ref': A,
                        'b_ref': b_ref,
                        'c_ref': mean_aero_choord,
                        }

        rules = spline_rules.rules_aeropanel(self.aerogrid)
        self.PHIjk = spline_functions.spline_rb(self.aerogrid, '_k', self.aerogrid, '_j',
                                                rules, self.coord, sparse_output=True)
        self.PHIlk = spline_functions.spline_rb(self.aerogrid, '_k', self.aerogrid, '_l',
                                                rules, self.coord, sparse_output=True)

        rules = spline_rules.rules_point(self.pacgrid, self.aerogrid)
        self.Dkx1 = spline_functions.spline_rb(self.pacgrid, '', self.aerogrid, '_k',
                                               rules, self.coord, sparse_output=False)
        self.Djx1 = self.PHIjk.dot(self.Dkx1)

    def build_AICs_steady(self, Ma):
        self.aero = {}
        self.aero['Gamma_jj'], __ = VLM.calc_Gammas(aerogrid=copy.deepcopy(self.aerogrid), Ma=Ma, xz_symmetry=True)


class VLM4PropLoads(object):
    """
    This class calculates the aerodynamic forces and moments of a propeller using the VLM.
    The propeller blade is modeled as a "starboard wing" in the xy-plane and rotates about the z-axis.
    Limitations:
    - This is a quasi-steady approach.
    - Modeling of the wake as a horse shoe vortex (straight wake, like for a wing, no spiral shape) and
      no interference with the wake of the preceding blade(s).
    - Modeling of only one blade and the onflow is rotated --> No interaction between neighboring blades.
    - Only overall Mach number correction possible, no variation in radial direction (influence negligible up to about Ma=0.3)
    - Currently, only a positive rotation about the z-axis (clockwise) feasible. A counter-clockwise rotation would
      lead to an onflow of the lifting surface from the back, compromising the VLM assumptions.

    Sign conventions:
    - The forces Pmac are given in the propeller coordinate system (up-right-forward).
    - The forces P_prop are given in the aircraft coordinate system (aft-right-up).
    - Alpha positive = w positive, wind from below
    - Beta  positive = v positive, wind from the right side
    """

    def __init__(self, model):

        self.model = model

        # only for compatibility with Common equations class
        self.Djx1 = self.model.Djx1

        # rotation matrix from propeller into aircraft system
        self.Tprop2aircraft = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

    def calc_loads(self, parameter_dict):
        # update the operational point with the input from the parameter_dict
        self.Vtas = parameter_dict['Vtas']
        self.alpha = parameter_dict['alpha']
        self.beta = parameter_dict['beta']
        self.RPM = parameter_dict['RPM']
        self.t = parameter_dict['t']
        # set some propeller-related parameters given in the parameter_dict
        self.n_blades = parameter_dict['n_blades']
        # set the initial blade position at t=0.0, which is horizontal
        self.phi_blades = np.linspace(0.0, 2.0 * np.pi, int(self.n_blades), endpoint=False)
        # based on the RPM and the time, calculate the current position of the blades
        self.phi_blades += self.RPM / 60.0 * self.t * 2.0 * np.pi
        # calculate the rotation vector of the propeller in the propeller coordinate system
        self.rot_vec = self.Tprop2aircraft.T.dot(parameter_dict['rotation_vector'])
        # select Gamma matrices based on mach number
        self.i_aero = 0
        # calculate the forces Pmac at the propeller hub
        Pmac = self.calc_Pmac()
        # initialize empty force vector and convert into aircraft-fixed coordinate system
        P_prop = np.zeros(6)
        P_prop[:3] = self.Tprop2aircraft.dot(Pmac[:3])
        P_prop[3:] = self.Tprop2aircraft.dot(Pmac[3:])
        return P_prop

    def rbm_nonlin(self, dUmac_dt):
        wj = np.ones(self.model.aerogrid['n']) * np.sum(dUmac_dt[:3] ** 2.0) ** 0.5
        Pk = equations.common.Common.calc_Pk_nonlin(self, dUmac_dt, wj)
        return Pk

    def camber_twist_nonlin(self, dUmac_dt):
        Ujx1 = np.dot(self.Djx1, dUmac_dt)
        Vtas_local = (Ujx1[self.model.aerogrid['set_j'][:, 0]] ** 2.0
                      + Ujx1[self.model.aerogrid['set_j'][:, 1]] ** 2.0
                      + Ujx1[self.model.aerogrid['set_j'][:, 2]] ** 2.0) ** 0.5
        wj = np.sin(self.model.aerogrid['cam_rad']) * Vtas_local * -1.0
        Pk = equations.common.Common.calc_Pk_nonlin(self, dUmac_dt, wj)
        return Pk

    def blade2body(self, phi):
        Tblade2body = np.zeros((6, 6))
        Tblade2body[0:3, 0:3] = calc_drehmatrix(0.0, 0.0, phi)
        Tblade2body[3:6, 3:6] = calc_drehmatrix(0.0, 0.0, phi)
        return Tblade2body, Tblade2body.T

    def calc_Pmac(self):
        # calculate the motion of the propeller
        dUmac_dt_aircraft = np.array([-self.Vtas * np.sin(self.alpha),
                                      self.Vtas * np.sin(self.beta),
                                      self.Vtas * np.cos(self.alpha) * np.cos(self.beta), 0.0, 0.0, 0.0])
        dUmac_dt_rpm = np.concatenate(([0.0, 0.0, 0.0], self.rot_vec * self.RPM / 60.0 * 2.0 * np.pi))
        # loop over all blades
        self.Pk = []
        self.Pmac = []
        self.dUmac_dt = []
        logging.debug('                                     Fx       Fy       Fz       Mx       My       Mz')
        for i_blade in range(self.n_blades):
            # calculate the motion seen by the blade
            phi_i = self.phi_blades[i_blade]
            Tblade2body, Tbody2blade = self.blade2body(phi_i)
            dUmac_dt_blade = Tbody2blade.dot(dUmac_dt_aircraft) + dUmac_dt_rpm
            self.dUmac_dt.append(dUmac_dt_blade)
            # calculate the forces on the blade
            Pk_rbm = self.rbm_nonlin(dUmac_dt_blade)
            Pk_cam = self.camber_twist_nonlin(dUmac_dt_blade)
            self.Pk.append(Pk_rbm + Pk_cam)
            # sum the forces at the origin and rotate back into aircraft system
            self.Pmac.append(Tblade2body.dot(self.model.Dkx1.T.dot(Pk_rbm + Pk_cam)))
            logging.debug('Forces from blade {} at {:>5.1f} : {:> 8.1f} {:> 8.1f} {:> 8.1f} {:> 8.1f} {:> 8.1f} \
                {:> 8.1f}'.format(i_blade, phi_i / np.pi * 180.0,
                                  self.Pmac[i_blade][0], self.Pmac[i_blade][1], self.Pmac[i_blade][2],
                                  self.Pmac[i_blade][3], self.Pmac[i_blade][4], self.Pmac[i_blade][5],))
        # calculate the sum over all blades
        Pmac_sum = np.sum(self.Pmac, axis=0)
        logging.debug('                         Sum : {:> 8.1f} {:> 8.1f} {:> 8.1f} {:> 8.1f} {:> 8.1f} {:> 8.1f}'.format(
            Pmac_sum[0], Pmac_sum[1], Pmac_sum[2], Pmac_sum[3], Pmac_sum[4], Pmac_sum[5]))
        return Pmac_sum


class PropellerPrecessionLoads(object):

    def __init__(self):
        pass

    def precession_moments(self, parameter_dict):
        # get relevant parameters
        I = parameter_dict['rotation_inertia']
        rot_vec = parameter_dict['rotation_vector']
        RPM = parameter_dict['RPM']
        pqr = parameter_dict['pqr']
        # initialize empty force vector
        P_prop = np.zeros(6)
        # calculate angular velocity vector rad/s
        omega = rot_vec * RPM / 60.0 * 2.0 * np.pi
        # calculate Mxyz
        P_prop[3:] = np.cross(omega, pqr) * I
        return P_prop
