# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:12:08 2015

@author: voss_ar
"""

import numpy as np
import scipy.sparse as sp
from loadskernel import spline_functions

def grid_trafo(grid, coord, dest_coord):
    
    for i_point in range(len(grid['ID'])):
        pos_coord = coord['ID'].index(grid['CP'][i_point])
        pos_coord_dest = coord['ID'].index(dest_coord)
        offset_tmp = np.dot(coord['dircos'][pos_coord],grid['offset'][i_point])+coord['offset'][pos_coord]
        offset = np.dot(coord['dircos'][pos_coord_dest].T,offset_tmp)+coord['offset'][pos_coord_dest]
        grid['offset'][i_point] = offset
        grid['CP'][i_point] = dest_coord
        grid['CD'][i_point] = dest_coord
    
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

def calc_transformation_matrix(coord, grid_i, set_i, coord_i, grid_d, set_d, coord_d, dimensions=''):
    # T_i and T_d are the translation matrices that do the projection to the coordinate systems of gird_i and grid_d
    # Parameters coord_i and coord_d allow to switch between the coordinate systems CD and CP (compare Nastran User Guide) with 
    # - CP = coordinate system of grid point offset
    # - CD = coordinate system of loads vector
    # Example of application: 
    # splinematrix = T_d.T.dot(T_di).dot(T_i)
    # Pmon_local = T_d.T.dot(T_i).dot(Pmon_global)
    
    if dimensions != '' and len(dimensions) == 2:
        dimensions_i = dimensions[0]
        dimensions_d = dimensions[1]
    else:
        dimensions_i = 6*len(grid_i['set'+set_i])
        dimensions_d = 6*len(grid_d['set'+set_d])
    
    # Using sparse matrices is faster and more efficient.
    T_i = sp.lil_matrix((dimensions_i,dimensions_i))
    for i_i in range(len(grid_i['ID'])):
        pos_coord_i = coord['ID'].index(grid_i[coord_i][i_i])
        T_i = spline_functions.sparse_insert( T_i, coord['dircos'][pos_coord_i], grid_i['set'+set_i][i_i,0:3], grid_i['set'+set_i][i_i,0:3] )
        T_i = spline_functions.sparse_insert( T_i, coord['dircos'][pos_coord_i], grid_i['set'+set_i][i_i,3:6], grid_i['set'+set_i][i_i,3:6] )
        
    T_d = sp.lil_matrix((dimensions_d,dimensions_d))
    for i_d in range(len(grid_d['ID'])):
        pos_coord_d = coord['ID'].index(grid_d[coord_d][i_d])
        T_d = spline_functions.sparse_insert( T_d, coord['dircos'][pos_coord_d], grid_d['set'+set_d][i_d,0:3], grid_d['set'+set_d][i_d,0:3] )
        T_d = spline_functions.sparse_insert( T_d, coord['dircos'][pos_coord_d], grid_d['set'+set_d][i_d,3:6], grid_d['set'+set_d][i_d,3:6] )
    return T_i, T_d