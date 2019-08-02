'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import logging


class Efcs:
    def __init__(self):
        self.keys = ['ELEV-R1', 'ELEV-R2', 'ELEV-L1', 'ELEV-L2']
        self.Ux2_0 = np.array([0.0, 0.0, 0.0, 0.0])
        self.Ux2_lower = np.array([-30.0, -30.0, -30.0, -30.0])/180*np.pi
        self.Ux2_upper = np.array([ 30.0,  30.0, 30.0, 30.0])/180*np.pi
        
        self.alpha_lower = -10.0/180*np.pi
        self.alpha_upper =  10.0/180*np.pi
                
    def cs_mapping(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_ELEVR1 = self.Ux2_0[0]
        delta_ELEVR2 = self.Ux2_0[1]
        delta_ELEVL1 = self.Ux2_0[2]
        delta_ELEVL2 = self.Ux2_0[3]
        
        # eta - Nickachse
        delta_ELEVR1 -= command_eta
        delta_ELEVR2 -= command_eta
        delta_ELEVL1 -= command_eta
        delta_ELEVL2 -= command_eta

        Ux2 = np.array([delta_ELEVR1, delta_ELEVR2, delta_ELEVL1, delta_ELEVL2])
        
        violation_lower = Ux2 < self.Ux2_lower
        if np.any(violation_lower):
            logging.warning( 'Commanded CS deflection not possible, violation of lower Ux2 bounds!')
            Ux2[violation_lower] = self.Ux2_lower[violation_lower]
            
        violation_upper = Ux2 > self.Ux2_upper
        if np.any(violation_upper):
            logging.warning( 'Commanded CS deflection not possible, violation of upper Ux2 bounds!')
            Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2
    
    def alpha_protetcion(self, alpha):
        if alpha < self.alpha_lower:
            logging.warning( 'Commanded alpha not possible, violation of lower alpha bounds!')
            alpha = self.alpha_lower
        if alpha > self.alpha_upper:
            logging.warning( 'Commanded alpha not possible, violation of upper alpha bounds!')
            alpha = self.alpha_upper
        return alpha