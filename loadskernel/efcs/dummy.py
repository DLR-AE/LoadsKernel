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
                
    def cs_mapping(self, command_xi, command_eta, command_zeta):
        return self.Ux2