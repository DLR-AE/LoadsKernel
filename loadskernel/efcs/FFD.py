import numpy as np

class Efcs:
    def __init__(self):
        self.keys = ['ELEVL1', 'ELEVL2', 'ELEVR1', 'ELEVR2', 'RUDDL1', 'RUDDR1']
        self.Ux2_0 = np.array([0.0]*6)
                
    def cs_mapping(self, commands):
        
        command_xi = commands[0] 
        command_eta = commands[1]
        command_zeta = commands[2]

        # Ausgangsposition
        delta_ELEVL1 = self.Ux2_0[0]
        delta_ELEVL2 = self.Ux2_0[1]
        delta_ELEVR1 = self.Ux2_0[2]
        delta_ELEVR2 = self.Ux2_0[3]
        delta_RUDDL1 = self.Ux2_0[4]
        delta_RUDDR1 = self.Ux2_0[5]
        
        # eta - Nickachse
        delta_ELEVL1 -= command_eta
        delta_ELEVL2 -= command_eta
        delta_ELEVR1 -= command_eta
        delta_ELEVR2 -= command_eta
        
        # xi - Rollachse
        delta_ELEVL1 += command_xi 
        delta_ELEVL2 += command_xi
        delta_ELEVR1 -= command_xi
        delta_ELEVR2 -= command_xi
        
        # zeta - Gierachse
        delta_RUDDL1 += command_zeta
        delta_RUDDR1 += command_zeta
        
        Ux2 = np.array([delta_ELEVL1, delta_ELEVL2, delta_ELEVR1, delta_ELEVR2, delta_RUDDL1, delta_RUDDR1])
        
        return Ux2