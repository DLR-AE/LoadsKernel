# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:12:08 2015

@author: voss_ar
"""

import numpy as np

def grid_trafo(grid, coord, dest_coord):
    
    for i_point in range(len(grid['ID'])):
        pos_coord = coord['ID'].index(grid['CP'][i_point])
        pos_coord_dest = coord['ID'].index(dest_coord)
        offset_tmp = np.dot(coord['dircos'][pos_coord],grid['offset'][i_point])+coord['offset'][pos_coord]
        offset = np.dot(coord['dircos'][pos_coord_dest].T,offset_tmp)+coord['offset'][pos_coord_dest]
        grid['offset'][i_point] = offset
        grid['CP'][i_point] = dest_coord
    
def force_trafo(grid, coord, forcevector):
    # Especially with monitoring stations, coordinate system CP and CD might differ. 
    # It is assumed the force and moments vector is in the coordinate system defined with CP.
    forcevector_local = np.zeros(np.shape(forcevector))

    for i_station in range(grid['n']):
        i_coord_source = coord['ID'].index(grid['CP'][i_station])
        i_coord_dest = coord['ID'].index(grid['CD'][i_station])

        dircos_source = np.zeros((6,6))
        dircos_source[0:3,0:3] = coord['dircos'][i_coord_source]
        dircos_source[3:6,3:6] = coord['dircos'][i_coord_source]
        dircos_dest = np.zeros((6,6))
        dircos_dest[0:3,0:3] = coord['dircos'][i_coord_dest]
        dircos_dest[3:6,3:6] = coord['dircos'][i_coord_dest]

        forcevector_local[grid['set'][i_station]] = dircos_dest.T.dot(dircos_source.dot(forcevector[grid['set'][i_station]]))
        
    return forcevector_local
