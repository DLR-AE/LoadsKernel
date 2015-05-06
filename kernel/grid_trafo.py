# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:12:08 2015

@author: voss_ar
"""

import numpy as np

def grid_trafo(grid, coord, dest_coord):
    
    for i_point in range(len(grid['ID'])):
        pos_coord = coord['ID'].index(grid['CP'][i_point])
        grid['offset'][i_point] = np.dot(coord['dircos'][pos_coord],grid['offset'][i_point])+coord['offset'][pos_coord]
        grid['CP'][i_point] = dest_coord
    
    