'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import logging

class Efcs:
    def __init__(self):
        self.keys = ['ELEV1', 'ELEV2',]
        self.Ux2_0 = np.array([0.0, 0.0])
        self.Ux2_lower = np.array([-30.0, -30.0,])/180*np.pi
        self.Ux2_upper = np.array([ 30.0,  30.0,])/180*np.pi
                
    def cs_mapping(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_ELEV1 = self.Ux2_0[0]
        delta_ELEV2 = self.Ux2_0[1]
        
        # eta - Nickachse
        delta_ELEV1 -= command_eta
        delta_ELEV2 -= command_eta
        
        Ux2 = np.array([delta_ELEV1, delta_ELEV2,])
        
        violation_lower = Ux2 < self.Ux2_lower
        if np.any(violation_lower):
            logging.warning( 'Commanded CS deflection not possible, violation of lower Ux2 bounds!')
            Ux2[violation_lower] = self.Ux2_lower[violation_lower]
            
        violation_upper = Ux2 > self.Ux2_upper
        if np.any(violation_upper):
            logging.warning( 'Commanded CS deflection not possible, violation of upper Ux2 bounds!')
            Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2