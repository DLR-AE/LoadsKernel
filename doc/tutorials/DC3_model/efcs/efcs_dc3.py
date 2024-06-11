import copy
import numpy as np


class Efcs():

    def __init__(self):
        self.keys = ['RUD', 'ELE-LFT', 'ELE-RIG', 'AIL-LFT', 'AIL-RIG']
        self.Ux2 = np.zeros(5)

    def cs_mapping(self, commands):
        command_xi = commands[0]
        command_eta = commands[1]
        command_zeta = commands[2]

        Ux2 = copy.deepcopy(self.Ux2)
        # positive xi (stick to the right) => roll to the right
        Ux2[3] += command_xi
        Ux2[4] -= command_xi

        # positive eta (stick pulled) => nose up
        Ux2[1] -= command_eta
        Ux2[2] -= command_eta

        # negative zeta (right pedal pushed) => yaw to the right
        Ux2[0] -= command_zeta
        return Ux2
