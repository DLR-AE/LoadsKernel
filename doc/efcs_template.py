'''
This is a template for an EFCS. For each aircraft, an EFCS must be written which maps the pilot 
commands to control surface deflections Ux2. This is because every aircraft has different control 
surfaces (e.g. one or two elevators, multiple ailerons, etc.)  
'''
import numpy as np

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