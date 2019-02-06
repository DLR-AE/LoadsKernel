# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:07:06 2015

@author: voss_ar
"""

#import Scientific.IO.NetCDF as netcdf
import scipy.io.netcdf as netcdf
import matplotlib.pyplot as plt
import numpy as np
import logging

import spline_rules
import spline_functions

def process_matrix(model, matrix, plot=False):

    for param in matrix.keys():
        for aero_key in matrix[param].keys():
            # if any markers are specified, extract affiliated points from cfd grid
            # otherwise, the coordinates can be extracted from the pval-file and no cfd gird is requiered!
            markers = matrix[param][aero_key]['markers']
            if markers != 'all':
                filename_grid = matrix[param][aero_key]['filename_grid']
                logging.info( 'Extracting points belonging to marker(s) {} from grid {}'.format(str(markers), filename_grid))
                # --- get points on surfaces according to marker ---
                ncfile_grid = netcdf.NetCDFFile(filename_grid, 'r')
                boundarymarker_surfaces = ncfile_grid.variables['boundarymarker_of_surfaces'][:]
                points_of_surfacetriangles = ncfile_grid.variables['points_of_surfacetriangles'][:]
                # expand triangles and merge with quadrilaterals
                if 'points_of_surfacequadrilaterals' in ncfile_grid.variables:
                    points_of_surfacetriangles = np.hstack((points_of_surfacetriangles, np.zeros((len(points_of_surfacetriangles),1), dtype=int) ))
                    points_of_surfacequadrilaterals = ncfile_grid.variables['points_of_surfacequadrilaterals'][:]
                    points_of_surface = np.vstack((points_of_surfacetriangles, points_of_surfacequadrilaterals))
                else:
                    points_of_surface = points_of_surfacetriangles
                    
                # find global id of points on surface defined by markers 
                points = np.array([], dtype=int)
                for marker in markers:
                    points = np.hstack(( points, np.where(boundarymarker_surfaces == marker)[0] ))
                points = np.unique(points_of_surface[points])
        
            matrix[param][aero_key]['Pk'] = []
            matrix[param][aero_key]['cfdgrid'] = []
            matrix[param][aero_key]['PHIcfd_k'] = []
            for i_value in range(len(matrix[param][aero_key]['values'])):
                # read pval
                filename_pval = matrix[param][aero_key]['filenames_surface_pval'][i_value]
                logging.info( '{}, {}, {}: reading {}'.format( param, aero_key, str(matrix[param][aero_key]['values'][i_value]),filename_pval))
                ncfile_pval = netcdf.NetCDFFile(filename_pval, 'r')
                # check if pval contains all required data
                for needed_key in ['global_id', 'x', 'y', 'z', 'x-force', 'y-force', 'z-force']:
                    if not ncfile_pval.variables.has_key(needed_key): 
                        logging.error( "'{}' is missing in {}".format(needed_key, matrix[param][aero_key]['filenames_surface_pval'][i_value]))
                
                global_id = ncfile_pval.variables['global_id'][:].copy()
                # if markers are specified, determine the positions of the points in the pval file
                # if markes = 'all', take all data from pval file
                if markers != 'all':
                    pos = []
                    for point in points: 
                        pos.append(np.where(global_id == point)[0][0]) 
                else:
                    pos = range(len(global_id))    
                    
                # build cfdgrid
                cfdgrid = {}
                cfdgrid['ID'] = global_id[pos]
                cfdgrid['CP'] = np.zeros(cfdgrid['ID'].shape)
                cfdgrid['CD'] = np.zeros(cfdgrid['ID'].shape)
                cfdgrid['n'] = len(cfdgrid['ID'])
                cfdgrid['offset'] = np.vstack((ncfile_pval.variables['x'][:][pos].copy(), ncfile_pval.variables['y'][:][pos].copy(),ncfile_pval.variables['z'][:][pos].copy() )).T
                cfdgrid['set'] = np.arange(6*cfdgrid['n']).reshape(-1,6)
                # store cfdgrid
                matrix[param][aero_key]['cfdgrid'].append(cfdgrid)
                
                # To save some time, check if the grids are identical.  
                # - number of points should be identical
                # - offsets should be identical
                # If the first condition returns false, the next one is not checked. This is useful to avoid errors if e.g. offsets are not comparablr because they have different lenght.
                if i_value != 0 and matrix[param][aero_key]['cfdgrid'][i_value-1]['n'] == cfdgrid['n'] and np.all(matrix[param][aero_key]['cfdgrid'][i_value-1]['offset'] == cfdgrid['offset']):
                    logging.info( ' --> assuming identical cfd grids, re-using spline matrix')
                    PHIcfd_k = matrix[param][aero_key]['PHIcfd_k'][i_value-1]                                        
                else:
                    # build spline
                    rules = spline_rules.nearest_neighbour( model.aerogrid, '_k', cfdgrid, '')    
                    PHIcfd_k = spline_functions.spline_rb(model.aerogrid, '_k', cfdgrid, '', rules, model.coord,  sparse_output=True) 
                    if plot:
                        spline_functions.plot_splinerules(model.aerogrid, '_k', cfdgrid, '', rules, model.coord) 

                # store spline
                matrix[param][aero_key]['PHIcfd_k'].append(PHIcfd_k)
                    
                # build force vector from cfd                    
                Pcfd = np.zeros(cfdgrid['n']*6)
                Pcfd[cfdgrid['set'][:,0]] = ncfile_pval.variables['x-force'][:][pos].copy()
                Pcfd[cfdgrid['set'][:,1]] = ncfile_pval.variables['y-force'][:][pos].copy()
                Pcfd[cfdgrid['set'][:,2]] = ncfile_pval.variables['z-force'][:][pos].copy()
                # translate cfd forces onto aerogrid
                Pk = PHIcfd_k.T.dot(Pcfd)
                # store Pk
                matrix[param][aero_key]['Pk'].append(Pk)
                
                # Check:
                #np.sum(Pk[model.aerogrid['set_k'][:,2]])       
                #np.sum(Pcfd[cfdgrid['set'][:,2]])
                #np.sum(ncfile_pval.variables['z-force'][:])

                if plot:
                    q_dyn = matrix[param][aero_key]['q_dyn']
                    Pmac = np.dot(model.Dkx1.T, Pk)
                    Cmac = Pmac * 0.0
                    Cmac[0:3] = Pmac[0:3] / q_dyn / model.macgrid['A_ref']
                    Cmac[3] = Pmac[3] / q_dyn / model.macgrid['A_ref'] / model.macgrid['b_ref']
                    Cmac[4] = Pmac[4] / q_dyn / model.macgrid['A_ref'] / model.macgrid['c_ref']
                    Cmac[5] = Pmac[5] / q_dyn / model.macgrid['A_ref'] / model.macgrid['b_ref']
                     
                    Cx_tau = np.sum(ncfile_pval.variables['x-force'][:][pos]) / q_dyn / model.macgrid['A_ref']
                    Cy_tau = np.sum(ncfile_pval.variables['y-force'][:][pos]) / q_dyn / model.macgrid['A_ref']
                    Cz_tau = np.sum(ncfile_pval.variables['z-force'][:][pos]) / q_dyn / model.macgrid['A_ref']
                                    
                    desc = 'param = ' + param + ', value = ' + str(matrix[param][aero_key]['values'][i_value]) + ', ' + aero_key
                    logging.info( desc)
                    logging.info( 'Cx: {:.4}, Cx_tau: {:.4}'.format(float(Cmac[0]), Cx_tau))
                    logging.info( 'Cy: {:.4}, Cy_tau: {:.4}'.format(float(Cmac[1]), Cy_tau))
                    logging.info( 'Cz: {:.4}, Cz_tau: {:.4}'.format(float(Cmac[2]), Cz_tau))
                     
                    cp = Pk[model.aerogrid['set_k'][:,0]] / q_dyn / model.aerogrid['A']
                    ax1 = plot_aerogrid(model.aerogrid, cp, 'jet', -0.5, 0.5)
                    ax1.set_title('CP_x after splining to VLM/DLM panels')
                    
                    cp = Pk[model.aerogrid['set_k'][:,1]] / q_dyn / model.aerogrid['A']
                    ax2 = plot_aerogrid(model.aerogrid, cp, 'jet', -0.5, 0.5)
                    ax2.set_title('CP_y after splining to VLM/DLM panels')
                    
                    cp = Pk[model.aerogrid['set_k'][:,2]] / q_dyn / model.aerogrid['A']
                    ax3 = plot_aerogrid(model.aerogrid, cp, 'jet', -0.5, 0.5)
                    ax3.set_title('CP_z after splining to VLM/DLM panels')

                    plt.show()
                    
                    
                ncfile_pval.close()
    return matrix
