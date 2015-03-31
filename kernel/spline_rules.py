# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:32:03 2014

@author: voss_ar
"""

import numpy as np
import read_geom

def nearest_neighbour(grid_i, grid_d):
    print 'Searching nearest neighbour of {:.0f} dependent nodes in {:.0f} independent nodes...'.format(len(grid_d['ID']) , len(grid_i['ID']))
    len(grid_d['ID'])
    single_ids = []    
    neighbours = []    
    for i_d in range(len(grid_d['ID'])):
        dist = np.sum((grid_i['offset'] - grid_d['offset_k'][i_d])**2, axis=1)**0.5
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
        

def monstations_from_report(mongrid, filenames):
    if mongrid['n'] != len(filenames):
        print 'Number of Stations in mongrid ({:.0f}) and number of reports ({:.0f}) unequal!'.format(mongrid['n'], len(filenames))
    ID_d = []
    ID_i = []
    for i_station in range(mongrid['n']):
                ID_d.append(read_geom.Nastran_NodeLocationReport(filenames[i_station]))
                ID_i.append(mongrid['ID'][i_station])
                
    splinerules = {"method": 'rb',
                   "ID_i": ID_i,
                   "ID_d": ID_d,
                    }
    return splinerules
                
                
                
                
                
                