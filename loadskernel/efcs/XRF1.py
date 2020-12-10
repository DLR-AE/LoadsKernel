'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import logging, copy

class Efcs:
    def __init__(self):
        self.keys = ['L11FLP', #0
                     'L12FLP', #1
                     'L13AIL', #2
                     'L14FLP', #3
                     'R11FLP', #4
                     'R12FLP', #5
                     'R13AIL', #6
                     'R14FLP', #7
                     'L21ELV', #8
                     'R21ELV', #9
                     'L31RUB', #10
                     ]
        self.Ux2_0 = np.array([0.0]*11)
        self.Ux2_lower = np.array([-30.0]*11)/180*np.pi
        self.Ux2_upper = np.array([ 30.0]*11)/180*np.pi
        
        self.alpha_lower = -10.0/180*np.pi
        self.alpha_upper =  10.0/180*np.pi
                
    def cs_mapping(self, commands):
        
        command_xi = commands[0] 
        command_eta = commands[1]
        command_zeta = commands[2]
        
        # Ausgangsposition
        Ux2 = copy.deepcopy(self.Ux2_0)            
        
        # xi - Rollachse
        Ux2[6] -= command_xi
        Ux2[2]  += command_xi
        
        # eta - Nickachse
        Ux2[8] -= command_eta
        Ux2[9] -= command_eta
        
        # zeta - Gierachse
        Ux2[10] -= command_zeta
        
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