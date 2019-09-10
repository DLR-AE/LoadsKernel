'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import logging

class Efcs:
    def __init__(self):
        self.keys = ['ELEV_R', 'ELEV_L',]
        self.Ux2_0 = np.array([0.0, 0.0])
                
    def cs_mapping(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_ELEV_L = self.Ux2_0[0]
        delta_ELEV_R = self.Ux2_0[1]
        
        # eta - Nickachse
        delta_ELEV_L -= command_eta
        delta_ELEV_R -= command_eta
        
        Ux2 = np.array([delta_ELEV_R, delta_ELEV_L,])
        
        return Ux2