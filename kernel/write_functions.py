# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:44:17 2015

@author: voss_ar
"""
import numpy as np

def write_SET1(fid, SID, entrys):
    entries_str = ''
    for entry in entrys:
        entries_str += '{:>8d}'.format(np.int(entry))
    if len(entries_str) <= 56:
        line = 'SET1    {:>8d}{:s}\n'.format(SID, entries_str)
        fid.write(line)
    else: 
        line = 'SET1    {:>8d}{:s}+\n'.format(SID, entries_str[:56])
        entries_str = entries_str[56:]
        fid.write(line)
    
        while len(entries_str) > 64:
            line = '+       {:s}+\n'.format(entries_str[:64])
            entries_str = entries_str[64:]
            fid.write(line)
        line = '+       {:s}\n'.format(entries_str)
        fid.write(line)
        

            
def write_MONPNT1(fid,mongrid, rules):
    for i_station in range(mongrid['n']):

        # MONPNT1 NAME LABEL
        # AXES COMP CID X Y Z
        # AECOMP WING AELIST 1001 1002
        # AELIST SID E1 E2 E3 E4 E5 E6 E7
    
        line = 'MONPNT1 Mon{: <5d}Label{:<51d}+\n'.format(int(mongrid['ID'][i_station]), int(mongrid['ID'][i_station]))
        fid.write(line)
        line = '+         123456 Comp{:<3d}{:>8d}{:>8.7s}{:>8.7s}{:>8.7s}{:>8d}\n'.format(int(mongrid['ID'][i_station]), int(mongrid['CP'][i_station]), str(mongrid['offset'][i_station][0]), str(mongrid['offset'][i_station][1]), str(mongrid['offset'][i_station][2]), int(mongrid['CD'][i_station]) )
        fid.write(line)
        line = 'AECOMP   Comp{:<3d}    SET1{:>8d}\n'.format(int(mongrid['ID'][i_station]), int(mongrid['ID'][i_station]) )
        fid.write(line)
        write_SET1(fid, np.int(mongrid['ID'][i_station]), rules['ID_d'][i_station])


def write_force_and_moment_cards(fid, grid, Pg, SID):
    # FORCE and MOMENT cards with all values equal to zero are ommitted to avoid problems when importing to Nastran.
    for i in range(grid['n']):
        if np.any(Pg[grid['set'][i,0:3]] != 0.0):
            line = 'FORCE   ' + '{:>8d}{:>8d}{:>8d}{:>8.7s}{:>8.7s}{:>8.7s}{:>8.7s}\n'.format(SID, np.int(grid['ID'][i]), np.int(grid['CD'][i]), str(1.0), str(Pg[grid['set'][i,0]]), str(Pg[grid['set'][i,1]]), str(Pg[grid['set'][i,2]]) )
            fid.write(line)
        if np.any(Pg[grid['set'][i,3:6]] != 0.0):
            line = 'MOMENT  ' + '{:>8d}{:>8d}{:>8d}{:>8.7s}{:>8.7s}{:>8.7s}{:>8.7s}\n'.format(SID, np.int(grid['ID'][i]), np.int(grid['CD'][i]), str(1.0), str(Pg[grid['set'][i,3]]), str(Pg[grid['set'][i,4]]), str(Pg[grid['set'][i,5]]) )
            fid.write(line)
