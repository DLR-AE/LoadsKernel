import numpy as np


class EngineLoads(object):

    def __init__(self):
        pass

    def torque_moments(self, parameter_dict):
        """
        This function calculates the torque from a given power and RPM setting.
        """
        rot_vec = parameter_dict['rotation_vector']
        RPM = parameter_dict['RPM']
        power = parameter_dict['power']
        # initialize empty force vector
        P_engine = np.zeros(6)
        # calculate angular velocity rad/s
        omega = RPM / 60.0 * 2.0 * np.pi
        # calculate Mxyz
        P_engine[3:] = -rot_vec * power / omega
        return P_engine

    def thrust_forces(self, parameter_dict, thrust):
        """
        This is the most thrust model. The requested thrust (from the trim) is aligned with the thrust orientation vector.
        """
        thrust_vector = parameter_dict['thrust_vector']
        # initialize empty force vector
        P_engine = np.zeros(6)
        # calculate Fxyz components
        P_engine[:3] = thrust_vector * thrust
        return P_engine
