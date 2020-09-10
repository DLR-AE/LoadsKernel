
import numpy as np
import copy

class Efcs:
    def __init__(self):
        self.keys = ['L13AIL01', 'R13AIL01', 'L12ELV01', 'R21ELV01', 'L31RUD01']
        self.Ux2_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                
    def cs_mapping(self, commands):

        command_xi = commands[0] 
        command_eta = commands[1]
        command_zeta = commands[2]

        # Ausgangsposition
        Ux2 = copy.deepcopy(self.Ux2_0)
        
        # xi - Rollachse
        Ux2[1] -= command_xi # Kommando postiv, Rollen nach rechts, rechtes Querruder nach oben
        Ux2[0]  += command_xi
        
        # eta - Nickachse
        Ux2[2] -= command_eta # Kommando positiv, Nicken nach oben, Höhenruder nach oben
        Ux2[3] -= command_eta
        
        # zeta - Gierachse
        Ux2[4] -= command_zeta # Kommando negativ, Gieren nach rechts, Seitenruder nach rechts
                
        return Ux2