'''
Created on Jul 4, 2022

@author: voss_ar
'''
import logging, sys
import numpy as np
try:
    from pyPropMat.pyPropMat import Prop
except:
    pass

class PropellerAeroLoads(object):
    """
    This class calculates the aerodynamic forces and moments of a propeller following equations 1 to 4 in [1].
    The the aerodynamic derivatives are calculated using pyPropMat provided by Christopher Koch, AE-SIM from [2]. 
    
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
    
    [1] Rodden, W., and Rose, T., “Propeller/nacelle whirl flutter addition to MSC/nastran,” in 
    Proceedings of the 1989 MSC World User’s Conference, 1989.
    
    [2] https://phabricator.ae.go.dlr.de/diffusion/180/
    """
    
    def __init__(self, filename):
        # Check if pyPropMat was imported successfully, see try/except statement in the import section.
        if "pyPropMat" in sys.modules:
            self.prop = Prop(filename)
        else:
            logging.error('pyPropMat was/could NOT be imported!'.format(self.jcl.aero['method']))
        self.prop_coeffs = None
    
    def calc_loads(self, parameter_dict):
        """
        Convert signs from structural coordinates (aft-right-up) to propeller coordinates (forward-right-down):
        positive alpha = wind from below = negative theta
        positive beta = wind from the right side = positive psi
        positive structural q = pitch-up = positive propeller q
        positive structural r = yaw to the left = negative proller r
        
        Convert signs propeller coordinates (forward-right-down) to structural coordinates (aft-right-up):
        switch Fz and Mz
        """
        diameter    =  parameter_dict['diameter']
        Vtas        =  parameter_dict['Vtas']
        q_dyn       =  parameter_dict['q_dyn']
        theta       = -parameter_dict['alpha']
        psi         =  parameter_dict['beta']
        q           =  parameter_dict['pqr'][1]
        r           = -parameter_dict['pqr'][2]
        
        # Calculate the area of the propeller disk with S = pi * r^2
        S = np.pi*(0.5*diameter)**2.0
        
        # Calculate the coefficients for the current operational point. It is sufficient to calculate them only once 
        # at the first run.
        if self.prop_coeffs == None:
            flight_condition_dict = {'V': Vtas, 
                                     'Ma': parameter_dict['Ma'], 
                                     'Rho': parameter_dict['rho'], 
                                     'N_rpm': parameter_dict['RPM']}
            self.prop_coeffs = self.prop.get_propeller_coefficients(flight_condition_dict, include_lag=False)
        
        # Get the aerodynamic coefficients
        Cz_theta    = self.prop_coeffs['z_theta']
        Cz_psi      = self.prop_coeffs['z_psi']
        Cz_q        = self.prop_coeffs['z_q']
        Cz_r        = self.prop_coeffs['z_r']
        Cy_theta    = self.prop_coeffs['y_theta']
        Cy_psi      = self.prop_coeffs['y_psi']
        Cy_q        = self.prop_coeffs['y_q']
        Cy_r        = self.prop_coeffs['y_r']
        Cm_theta    = self.prop_coeffs['m_theta']
        Cm_psi      = self.prop_coeffs['m_psi']
        Cm_q        = self.prop_coeffs['m_q']
        Cm_r        = self.prop_coeffs['m_r']
        Cn_theta    = self.prop_coeffs['n_theta']
        Cn_psi      = self.prop_coeffs['n_psi']
        Cn_q        = self.prop_coeffs['n_q']
        Cn_r        = self.prop_coeffs['n_r']
    
        # initialize empty force vector
        P_prop = np.zeros(6)
        # Side force Fy, equation 3 in [1]
        P_prop[1] += q_dyn*S * (Cy_theta*theta + Cy_psi*psi + Cy_q*q*diameter/(2.0*Vtas) + Cy_r*r*diameter/(2.0*Vtas))
        # Lift force Fz, equation 1 in [1], convert signs (see above)
        P_prop[2] -= q_dyn*S * (Cz_theta*theta + Cz_psi*psi + Cz_q*q*diameter/(2.0*Vtas) + Cz_r*r*diameter/(2.0*Vtas))
        # Pitching moment My, equation 2 in [1]
        P_prop[4] += q_dyn*S*diameter * (Cm_theta*theta + Cm_psi*psi + Cm_q*q*diameter/(2.0*Vtas) + Cm_r*r*diameter/(2.0*Vtas))
        # Yawing moment Mz, equation 4 in [1], convert signs (see above)
        P_prop[5] -= q_dyn*S*diameter * (Cn_theta*theta + Cn_psi*psi + Cn_q*q*diameter/(2.0*Vtas) + Cn_r*r*diameter/(2.0*Vtas))
        
        # Check: An onflow from below (pos. alpha = neg. Theta) creates an downward lift (neg. Fz / P_prop[2])
        return P_prop
    
class PropellerPrecessionLoads(object):
    
    def __init__(self):
        pass
    
    def precession_moments(self, parameter_dict):
        # get relevant parameters
        I       = parameter_dict['rotation_inertia']
        rot_vec = parameter_dict['rotation_vector']
        RPM     = parameter_dict['RPM']
        pqr     = parameter_dict['pqr']
        # initialize empty force vector
        P_prop  = np.zeros(6)
        # calculate angular velocity vector rad/s
        omega = rot_vec * RPM / 60.0 * 2.0 * np.pi 
        # calculate Mxyz
        P_prop[3:] = np.cross(omega, pqr) * I 
        return P_prop
    