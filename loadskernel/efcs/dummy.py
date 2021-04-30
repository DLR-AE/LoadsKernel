'''
Created on Aug 2, 2019

@author: voss_ar
'''
import numpy as np
import logging

class Efcs:
    def __init__(self):
        self.keys = ['dummy']
        self.Ux2 = np.array([0.0])
                
    def cs_mapping(self, commands):
        """
        Do nothing in particular, this is just a dummy EFCS.
        command_xi = commands[0] 
        command_eta = commands[1]
        command_zeta = commands[2]
        ...
        """
        
        
        return self.Ux2