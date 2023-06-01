# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:32:03 2014

@author: voss_ar
"""

import numpy as np
import logging
import loadskernel.io_functions.read_mona as read_mona


def nearest_neighbour(grid_i,  set_i,  grid_d, set_d):
    logging.info('Searching nearest neighbour of {:.0f} dependent nodes in {:.0f} independent nodes...'.format(len(grid_d['ID']) , len(grid_i['ID'])))
    len(grid_d['ID'])
    single_ids = []    
    neighbours = []    
    for i_d in range(len(grid_d['ID'])):
        dist = np.sum((grid_i['offset'+set_i] - grid_d['offset'+set_d][i_d])**2, axis=1)**0.5
        single_ids.append([grid_d['ID'][i_d]])         
        neighbours.append([grid_i['ID'][dist.argmin()]])
        
    splinerules = {"method": 'rb',
                   "ID_i": neighbours,
                   "ID_d": single_ids,
                    }
    return splinerules

def rules_point(grid_i, grid_d):
    # all dependent grids are mapped to one grid point, which might be CG or MAC
    # Assumption: the relevant point is expected to be the only/first point in the independet grid
    splinerules = {"method": 'rb',
                   "ID_i": [grid_i['ID']],
                   "ID_d": [grid_d['ID']],
                    }
    return splinerules
    
def rules_aeropanel(aerogrid):
    # map every point k to its corresponding point j
    single_ids = []
    for id in aerogrid['ID']:
        single_ids.append([id]) 
    
    splinerules = {"method": 'rb',
                   "ID_i": single_ids,
                   "ID_d": single_ids,
                    }
    return splinerules

def monstations_from_bdf(mongrid, filenames):
    if mongrid['n'] != len(filenames):
        logging.error('Number of Stations in mongrid ({:.0f}) and number of bdfs ({:.0f}) unequal!'.format(mongrid['n'], len(filenames)))
    ID_d = []
    ID_i = []
    for i_station in range(mongrid['n']):
        tmp = read_mona.Modgen_GRID(filenames[i_station]) 
        ID_d.append(tmp['ID'])
        ID_i.append(mongrid['ID'][i_station])
                
    splinerules = {"method": 'rb',
                   "ID_i": ID_i,
                   "ID_d": ID_d,
                    }
    return splinerules
                
def monstations_from_aecomp_old(mongrid, filename):
    aecomp = read_mona.Nastran_AECOMP(filename)
    # Assumption: only SET1 is used in AECOMP, AELIST and CAERO are not yet implemented
    sets = read_mona.Nastran_SET1(filename)
    ID_d = []
    ID_i = []
    for i_station in range(mongrid['n']):
        i_aecomp = aecomp['name'].index( mongrid['comp'][i_station])
        i_set = sets['ID'].index(aecomp['list_id'][i_aecomp])
        ID_d.append(sets['values'][i_set])
        ID_i.append(mongrid['ID'][i_station])
                
    splinerules = {"method": 'rb',
                   "ID_i": ID_i,
                   "ID_d": ID_d,
                    }
    return splinerules

def monstations_from_aecomp(mongrid, aecomp, sets):
    ID_d = []
    ID_i = []
    for i_station in range(mongrid['n']):
        i_aecomp = aecomp['name'].index( mongrid['comp'][i_station])
        i_sets = [sets['ID'].index(x) for x in aecomp['list_id'][i_aecomp]]
        # combine the IDs in case multiple sets are given by one AECOMP card
        combined_ids = []
        for i_set in i_sets: 
            combined_ids += list(sets['values'][i_set])
        ID_d.append(combined_ids)
        ID_i.append(mongrid['ID'][i_station])
                
    splinerules = {"method": 'rb',
                   "ID_i": ID_i,
                   "ID_d": ID_d,
                    }
    return splinerules
