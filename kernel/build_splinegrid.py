# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:11:26 2015

@author: voss_ar
"""

import read_geom
import numpy as np


def build_splinegrid(strcgrid, filename):

    subgrid = read_geom.Modgen_GRID(filename)
    splinegrid = {'ID': [], 'CD': [], 'CP': [], 'set': [], 'offset': [],  'n': 0, }
    for i_ID in subgrid['ID']:
        pos = np.where(i_ID == strcgrid['ID'])[0][0]
        splinegrid['ID'].append(strcgrid['ID'][pos])
        splinegrid['CD'].append(strcgrid['CD'][pos])
        splinegrid['CP'].append(strcgrid['CP'][pos])
        splinegrid['n'] += 1
        splinegrid['set'].append(strcgrid['set'][pos])
        splinegrid['offset'].append(strcgrid['offset'][pos])
    
    splinegrid['ID'] = np.array(splinegrid['ID'])
    splinegrid['CD'] = np.array(splinegrid['CD'])
    splinegrid['CP'] = np.array(splinegrid['CP'])
    splinegrid['set'] = np.array(splinegrid['set'])
    splinegrid['offset'] = np.array(splinegrid['offset']) 

    return splinegrid
    
def grid_thin_out_random(grid, thin_out_factor):
    randomnumbers = np.random.rand(grid['n'])
    pos = np.where(randomnumbers < thin_out_factor)[0]
    grid_thin= {'ID': grid['ID'][pos],
                'CD': grid['CD'][pos],
                'CP': grid['CP'][pos],
                'set': grid['set'][pos],
                'offset': grid['offset'][pos],
                'n': len(pos),
               }
    return grid_thin
    
def grid_thin_out_radius(grid, radius):
    pos = range(grid['n'])
    i = 0
    while i < len(pos):
        dist = np.sum((grid['offset'][pos] - grid['offset'][pos[i]])**2, axis=1)**0.5
        dist[i] += radius * 1.1
        pos = np.delete( pos, np.where(dist <= radius)[0] )
        i += 1
        
    grid_thin= {'ID': grid['ID'][pos],
                'CD': grid['CD'][pos],
                'CP': grid['CP'][pos],
                'set': grid['set'][pos],
                'offset': grid['offset'][pos],
                'n': len(pos),
               }
    return grid_thin 
    
