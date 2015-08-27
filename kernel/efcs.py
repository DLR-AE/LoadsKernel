# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:36:41 2015

@author: voss_ar
"""
import numpy as np

class mephisto:
    def __init__(self):
        self.keys = ['AIL-S1', 'AIL-S2', 'AIL-S3', 'AIL-S4']
        self.Ux2_0 = np.array([0.0, 0.0, 0.0, 0.0])
        self.Ux2_lower = np.array([-5.0, -5.0,-5.0,-5.0])/180*np.pi
        self.Ux2_upper = np.array([ 5.0,  5.0, 5.0, 5.0])/180*np.pi
                
    def efcs(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_AILS1 = self.Ux2_0[0]
        delta_AILS2 = self.Ux2_0[1]
        delta_AILS3 = self.Ux2_0[2]
        delta_AILS4 = self.Ux2_0[3]
        
        # xi - Rollachse
        delta_AILS1 -= command_xi
        delta_AILS2 -= command_xi
        delta_AILS3 += command_xi
        delta_AILS4 += command_xi
        
        # eta - Nickachse
        delta_AILS1 -= command_eta
        delta_AILS2 -= command_eta
        delta_AILS3 -= command_eta
        delta_AILS4 -= command_eta
        
        # zeta - Gierachse
        #delta_AILS1 -= command_zeta
        #delta_AILS2 -= command_zeta
        #delta_AILS3 -= command_zeta
        #delta_AILS4 -= command_zeta
        
        Ux2 = np.array([delta_AILS1, delta_AILS2, delta_AILS3, delta_AILS4])
        
        violation_lower = Ux2 < self.Ux2_lower
        if np.any(violation_lower):
            print 'Warning: commanded CS deflection not possible, violation of lower Ux2 bounds!'
            Ux2[violation_lower] = self.Ux2_lower[violation_lower]
            
        violation_upper = Ux2 > self.Ux2_upper
        if np.any(violation_upper):
            print 'Warning: commanded CS deflection not possible, violation of upper Ux2 bounds!'
            Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2
