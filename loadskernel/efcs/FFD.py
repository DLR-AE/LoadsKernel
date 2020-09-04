'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import logging

class Efcs:
    def __init__(self):
        self.keys = ['R11FLP', 'R12AIL', 'L11FLP', 'L12AIL', 'R31RUD', 'L31RUD', 'STAB', 'L-STAB',]
        self.Ux2_0 = np.array([0.0, 0.0])
                
    def cs_mapping(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_STAB = self.Ux2_0[0]
        delta_LSTAB = self.Ux2_0[1]
        
        # eta - Nickachse
        delta_STAB -= command_eta
        delta_LSTAB -= command_eta
        
        Ux2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, delta_STAB, delta_LSTAB])
        
        return Ux2