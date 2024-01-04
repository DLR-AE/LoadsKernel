from itertools import groupby
import numpy as np
import scipy.sparse as sp
from loadskernel.utils.sparse_matrices import insert_lil


def all_equal(iterable):
    # As suggested on stackoverflow, this is the fastest way to check if all element in an array are equal
    # https://stackoverflow.com/questions/3844801/check-if-all-elements-in-a-list-are-identical
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def grid_trafo(grid, coord, dest_coord):
    """
    This function transforms a grid into a new coordiante system.
    The coordiante transformatin can be speed up by using a matrix operation, but only in case all point are
    in the same coordinate system. To handle different coorinate systems in one grid (e.g. Nastran GRID points
    given with different CP), the coordinate transformation has to be applied gridpoint-wise.
    """
    pos_coord_dest = np.where(np.array(coord['ID']) == dest_coord)[0][0]
    if all_equal(grid['CP']):
        # get the right transformation matrices
        pos_coord_orig = np.where(np.array(coord['ID']) == grid['CP'][0])[0][0]
        # perform transformation
        offset_tmp = coord['dircos'][pos_coord_orig].dot(grid['offset'].T).T + coord['offset'][pos_coord_orig]
        offset = coord['dircos'][pos_coord_dest].T.dot(offset_tmp.T).T + coord['offset'][pos_coord_dest]
        # store new offsets in grid
        grid['offset'] = offset
        grid['CP'] = np.array([dest_coord] * grid['n'])
    else:
        for i_point in range(len(grid['ID'])):
            pos_coord_orig = np.where(np.array(coord['ID']) == grid['CP'][i_point])[0][0]
            offset_tmp = np.dot(coord['dircos'][pos_coord_orig], grid['offset'][i_point]) + coord['offset'][pos_coord_orig]
            offset = np.dot(coord['dircos'][pos_coord_dest].T, offset_tmp) + coord['offset'][pos_coord_dest]
            grid['offset'][i_point] = offset
            grid['CP'][i_point] = dest_coord


def vector_trafo(grid, coord, forcevector, dest_coord):
    """
    This function transforms a force (or displacement) vector into a new coordiante system. It is assumed
    the force and moments vector is in the coordinate system defined with CD. As above, matrix operations are
    applied if all source coord systems are identical.
    """
    pos_coord_dest = np.where(np.array(coord['ID']) == dest_coord)[0][0]
    if all_equal(grid['CD']):
        # get the right transformation matrices
        pos_coord_orig = np.where(np.array(coord['ID']) == grid['CD'][0])[0][0]
        # expand for 6 degrees of freedom
        dircos_source = np.zeros((6, 6))
        dircos_source[0:3, 0:3] = coord['dircos'][pos_coord_orig]
        dircos_source[3:6, 3:6] = coord['dircos'][pos_coord_orig]
        dircos_dest = np.zeros((6, 6))
        dircos_dest[0:3, 0:3] = coord['dircos'][pos_coord_dest]
        dircos_dest[3:6, 3:6] = coord['dircos'][pos_coord_dest]
        # perform transformation
        forcevector_trans = dircos_dest.T.dot(dircos_source.dot(forcevector[grid['set']].T)).T.reshape(1, -1).squeeze()

    else:
        forcevector_trans = np.zeros(np.shape(forcevector))
        for i_station in range(grid['n']):
            pos_coord_orig = np.where(np.array(coord['ID']) == grid['CD'][i_station])[0][0]

            dircos_source = np.zeros((6, 6))
            dircos_source[0:3, 0:3] = coord['dircos'][pos_coord_orig]
            dircos_source[3:6, 3:6] = coord['dircos'][pos_coord_orig]
            dircos_dest = np.zeros((6, 6))
            dircos_dest[0:3, 0:3] = coord['dircos'][pos_coord_dest]
            dircos_dest[3:6, 3:6] = coord['dircos'][pos_coord_dest]

            forcevector_trans[grid['set'][i_station]] = dircos_dest.T.dot(dircos_source.dot(
                forcevector[grid['set'][i_station]]))

    return forcevector_trans


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
        dimensions_i = 6 * len(grid_i['set' + set_i])
        dimensions_d = 6 * len(grid_d['set' + set_d])

    # Using sparse matrices is faster and more efficient.
    T_i = sp.lil_matrix((dimensions_i, dimensions_i))
    for i_i in range(len(grid_i['ID'])):
        pos_coord_i = coord['ID'].index(grid_i[coord_i][i_i])
        T_i = insert_lil(T_i, coord['dircos'][pos_coord_i], grid_i['set' + set_i][i_i, 0:3], grid_i['set' + set_i][i_i, 0:3])
        T_i = insert_lil(T_i, coord['dircos'][pos_coord_i], grid_i['set' + set_i][i_i, 3:6], grid_i['set' + set_i][i_i, 3:6])

    T_d = sp.lil_matrix((dimensions_d, dimensions_d))
    for i_d in range(len(grid_d['ID'])):
        pos_coord_d = coord['ID'].index(grid_d[coord_d][i_d])
        T_d = insert_lil(T_d, coord['dircos'][pos_coord_d], grid_d['set' + set_d][i_d, 0:3], grid_d['set' + set_d][i_d, 0:3])
        T_d = insert_lil(T_d, coord['dircos'][pos_coord_d], grid_d['set' + set_d][i_d, 3:6], grid_d['set' + set_d][i_d, 3:6])
    return T_i.tocsc(), T_d.tocsc()
