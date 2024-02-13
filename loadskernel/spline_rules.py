# -*- coding: utf-8 -*-

import numpy as np
import logging
import loadskernel.io_functions.read_mona as read_mona

"""
Spline rules describe how a rigid body spline is constructed. The mapping of dependent to
independent grids is stored in a dictionary with the following nomenclature:
splinerules = {'ID of independent grid': [list of dependent grid IDs]}
"""


def nearest_neighbour(grid_i, set_i, grid_d, set_d):
    logging.info('Searching nearest neighbour of {:.0f} dependent nodes in {:.0f} independent nodes...'.format(
        len(grid_d['ID']), len(grid_i['ID'])))

    splinerules = {}
    for i_d in range(grid_d['n']):
        dist = np.sum(
            (grid_i['offset' + set_i] - grid_d['offset' + set_d][i_d]) ** 2, axis=1) ** 0.5
        neighbour = grid_i['ID'][dist.argmin()]
        if neighbour not in splinerules:
            splinerules[neighbour] = [grid_d['ID'][i_d]]
        else:
            splinerules[neighbour] += [grid_d['ID'][i_d]]

    return splinerules


def rules_point(grid_i, grid_d):
    # All dependent grids are mapped to one grid point, which might be CG or MAC
    # Assumption: the relevant point is expected to be the only/first point in the independet grid
    splinerules = {}
    assert len(grid_i['ID']) == 1, "The independent grid 'grid_i' may have only one grid point for this kind of spline rules."
    splinerules[grid_i['ID'][0]] = list(grid_d['ID'])
    return splinerules


def rules_aeropanel(aerogrid):
    # Map every point k to its corresponding point j
    splinerules = {}
    for id in aerogrid['ID']:
        splinerules[id] = [id]
    return splinerules


def monstations_from_bdf(mongrid, filenames):
    if mongrid['n'] != len(filenames):
        logging.error('Number of Stations in mongrid ({:.0f}) and number of bdfs ({:.0f}) unequal!'.format(
            mongrid['n'], len(filenames)))
    splinerules = {}
    for i_station in range(mongrid['n']):
        tmp = read_mona.Modgen_GRID(filenames[i_station])
        splinerules[mongrid['ID'][i_station]] = list(tmp['ID'])
    return splinerules


def monstations_from_aecomp(mongrid, aecomp, sets):
    splinerules = {}
    for i_station in range(mongrid['n']):
        i_aecomp = aecomp['name'].index(mongrid['comp'][i_station])
        i_sets = [sets['values'][sets['ID'].index(
            x)] for x in aecomp['list_id'][i_aecomp]]
        # combine the IDs in case multiple sets are given by one AECOMP card
        splinerules[mongrid['ID'][i_station]] = np.unique(
            np.concatenate(i_sets).ravel()).tolist()

    return splinerules
