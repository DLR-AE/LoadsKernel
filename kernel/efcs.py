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
        
        self.alpha_lower = -5.0/180*np.pi
        self.alpha_upper = 10.0/180*np.pi
        
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
        
    def alpha_protetcion(self, alpha):
        if alpha < self.alpha_lower:
            print 'Warning: commanded alpha not possible, violation of lower alpha bounds!'
            alpha = self.alpha_lower
        if alpha > self.alpha_upper:
            print 'Warning: commanded alpha not possible, violation of upper alpha bounds!'
            alpha = self.alpha_upper
        return alpha
        

class allegra:
    def __init__(self):
        self.keys = ['ELEV1', 'ELEV2', 'RUDDER']
        self.Ux2_0 = np.array([0.0, 0.0, 0.0])
        self.Ux2_lower = np.array([-30.0, -30.0, -30.0])/180*np.pi
        self.Ux2_upper = np.array([ 30.0,  30.0, 30.0])/180*np.pi
        
        self.alpha_lower = -4.0/180*np.pi
        self.alpha_upper = 6.0/180*np.pi
                
    def efcs(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_ELEV1 = self.Ux2_0[0]
        delta_ELEV2 = self.Ux2_0[1]
        delta_RUDDER = self.Ux2_0[2]              
        
        # eta - Nickachse
        delta_ELEV1 -= command_eta
        delta_ELEV2 -= command_eta
        
        # zeta - Gierachse
        delta_RUDDER = command_zeta
        
        Ux2 = np.array([delta_ELEV1, delta_ELEV2, delta_RUDDER])
        
        violation_lower = Ux2 < self.Ux2_lower
        if np.any(violation_lower):
            print 'Warning: commanded CS deflection not possible, violation of lower Ux2 bounds!'
            Ux2[violation_lower] = self.Ux2_lower[violation_lower]
            
        violation_upper = Ux2 > self.Ux2_upper
        if np.any(violation_upper):
            print 'Warning: commanded CS deflection not possible, violation of upper Ux2 bounds!'
            Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2

    def alpha_protetcion(self, alpha):
        if alpha < self.alpha_lower:
            print 'Warning: commanded alpha not possible, violation of lower alpha bounds!'
            alpha = self.alpha_lower
        if alpha > self.alpha_upper:
            print 'Warning: commanded alpha not possible, violation of upper alpha bounds!'
            alpha = self.alpha_upper
        return alpha
        
class discus2c:
    def __init__(self):
        self.keys = ['AIL_Rin', 'AIL_Rout', 'AIL_Lin', 'AIL_Lout', 'ELEV_R', 'ELEV_L', 'RUDD' ]
        self.Ux2_0     = np.array([0.0,     0.0,   0.0,   0.0,   0.0,   0.0,   0.0])
        self.Ux2_lower = np.array([-30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0])/180*np.pi
        self.Ux2_upper = np.array([ 30.0,  30.0,  30.0,  30.0,  30.0,  30.0,  30.0])/180*np.pi
        
        self.alpha_lower = -10.0/180*np.pi
        self.alpha_upper =  10.0/180*np.pi
                
    def efcs(self, command_xi, command_eta, command_zeta):

        # Ausgangsposition
        delta_AIL_Rin  = self.Ux2_0[0]
        delta_AIL_Rout = self.Ux2_0[1]
        delta_AIL_Lin  = self.Ux2_0[2]
        delta_AIL_Lout = self.Ux2_0[3]
        delta_ELEV_R   = self.Ux2_0[4]
        delta_ELEV_L   = self.Ux2_0[5]
        delta_RUDD     = self.Ux2_0[6]
        
        # xi - Rollachse
        delta_AIL_Rin  -= command_xi
        delta_AIL_Rout -= command_xi
        delta_AIL_Lin  += command_xi
        delta_AIL_Lout += command_xi
        
        # eta - Nickachse
        delta_ELEV_R -= command_eta
        delta_ELEV_L -= command_eta
        
        # zeta - Gierachse
        delta_RUDD = command_zeta
        
        Ux2 = np.array([delta_AIL_Rin, delta_AIL_Rout, delta_AIL_Lin, delta_AIL_Lout, delta_ELEV_R, delta_ELEV_L, delta_RUDD])
        
        violation_lower = Ux2 < self.Ux2_lower
        if np.any(violation_lower):
            print 'Warning: commanded CS deflection not possible, violation of lower Ux2 bounds!'
            Ux2[violation_lower] = self.Ux2_lower[violation_lower]
            
        violation_upper = Ux2 > self.Ux2_upper
        if np.any(violation_upper):
            print 'Warning: commanded CS deflection not possible, violation of upper Ux2 bounds!'
            Ux2[violation_upper] = self.Ux2_upper[violation_upper]
            
        return Ux2

    def alpha_protetcion(self, alpha):
        if alpha < self.alpha_lower:
            print 'Warning: commanded alpha not possible, violation of lower alpha bounds!'
            alpha = self.alpha_lower
        if alpha > self.alpha_upper:
            print 'Warning: commanded alpha not possible, violation of upper alpha bounds!'
            alpha = self.alpha_upper
        return alpha
