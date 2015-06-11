# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 15:07:06 2015

@author: voss_ar
"""

#import Scientific.IO.NetCDF as netcdf
import scipy.io.netcdf as netcdf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import griddata

from trim_tools import * 
import cPickle
import scipy



def test(model, trimcase):
    
    alphas = [-1.0, 2.0, 4.0, 4.588, 10.0] #3.955 #4.588
    alphas2 = [2.0, 3.0]    
    filename_grid = '/scratch/DLR-F19_tau/tau.grid'
        
    filenames_pval = ['/scratch/DLR-F19_tau/Ma08_alpha-1/sol_steady.surface.pval.688',
                      '/scratch/DLR-F19_tau/Ma08_alpha2/sol_steady.surface.pval.507',
                      '/scratch/DLR-F19_tau/Ma08_alpha4/sol_steady.surface.pval.443',
                      '/scratch/DLR-F19_tau/Ma08_alpha4.588/sol_steady.surface.pval.462',
                      '/scratch/DLR-F19_tau/Ma08_alpha10/sol_steady.surface.pval.1500',
                     ] 
    marker_upper = [2] #[12, 15, 18, 21]
    marker_lower = [3] #[13, 16, 19, 22]

    i_atmo     = model.atmo['key'].index(trimcase['altitude'])
    rho = model.atmo['rho'][i_atmo]
    Vtas = trimcase['Ma'] * model.atmo['a'][i_atmo]
    q_dyn = rho/2.0*Vtas**2    
    
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
    points_upper = np.array([], dtype=int)
    for marker in marker_upper:
        points_upper = np.hstack(( points_upper, np.where(boundarymarker_surfaces == marker)[0] ))
    points_upper = np.unique(points_of_surface[points_upper])
    points_lower = np.array([], dtype=int)
    for marker in marker_lower:
        points_lower = np.hstack(( points_lower, np.where(boundarymarker_surfaces == marker)[0] ))
    points_lower = np.unique(points_of_surface[points_lower])
    
    # --- get dlm data and Cp ---
    x_dlm = model.aerogrid['offset_k'][:,0]
    y_dlm = model.aerogrid['offset_k'][:,1]
    z_dlm = model.aerogrid['offset_k'][:,2]
    
    i_aero     = model.aero['key'].index(trimcase['aero'])
    Qjj        = model.aero['Qjj'][i_aero]    
    wj_cam = np.sin(model.camber_twist['cam_rad'] )
    #cp_dlm =  np.dot(Qjj, wjx1 + wj_cam)
    
    downwash_corr = {'alpha': [],
                     'wj': [],
                     'wj_corrfac': [],
                    }
    
    for i_alpha in range(len(alphas)):   
        print 'working on alpha ' + str(alphas[i_alpha])
        # --- load pval and find values with global id ---
        ncfile_pval = netcdf.NetCDFFile(filenames_pval[i_alpha], 'r')
        if i_alpha == 0:
            global_id = ncfile_pval.variables['global_id'][:]
            pos_upper = []
            for point in points_upper:
                pos_upper.append(np.where(global_id == point)[0][0])        
            pos_lower = []
            for point in points_lower:
                pos_lower.append(np.where(global_id == point)[0][0])        
                
        x_pval_upper = ncfile_pval.variables['x'][:][pos_upper]
        y_pval_upper = ncfile_pval.variables['y'][:][pos_upper]
        z_pval_upper = ncfile_pval.variables['z'][:][pos_upper]
        cp_upper = ncfile_pval.variables['cp'][:][pos_upper]
        
        x_pval_lower = ncfile_pval.variables['x'][:][pos_lower]
        y_pval_lower = ncfile_pval.variables['y'][:][pos_lower]
        z_pval_lower = ncfile_pval.variables['z'][:][pos_lower]
        cp_lower = ncfile_pval.variables['cp'][:][pos_lower]
        
        # interpolate CFD results on dlm grid
        cp_interp_upper = griddata((x_pval_upper,y_pval_upper), cp_upper, (x_dlm, y_dlm), method='linear')
        cp_interp_lower = griddata((x_pval_lower,y_pval_lower), cp_lower, (x_dlm, y_dlm), method='linear')
        # sumstract upper side form lower side, this gives positive Cps
        cp_interp =  cp_interp_lower - cp_interp_upper
        
        wjx1 = np.sin(alphas[i_alpha]/180*np.pi) * np.ones((1,len(Qjj)) )[0]
        # calculate correction
        wj_soll = scipy.linalg.solve(Qjj, cp_interp)
        wj_corrfac = (wj_soll - wj_cam) / wjx1
        #bla = wjx1 * wj_corrfac  + wj_cam - wj_soll
        # save correction
        downwash_corr['alpha'].append(alphas[i_alpha])
        downwash_corr['wj'].append(np.sin(alphas[i_alpha]/180*np.pi))
        downwash_corr['wj_corrfac'].append(wj_corrfac)
        
        fl_interp = q_dyn * model.aerogrid['N'].T*model.aerogrid['A']*cp_interp
        # Bemerkung: greifen die Luftkraefte bei j,l oder k an?
        # Dies würde das Cmy beeinflussen!
        # Gewaehlt: l
        Pl_interp = np.zeros((1,840*6))[0]
        Pl_interp[model.aerogrid['set_l'][:,0]] = fl_interp[0,:]
        Pl_interp[model.aerogrid['set_l'][:,1]] = fl_interp[1,:]
        Pl_interp[model.aerogrid['set_l'][:,2]] = fl_interp[2,:]
        Pk_interp = np.dot(model.Dlk.T, Pl_interp)
        
        Pmac_interp = np.dot(model.Dkx1.T, Pk_interp)
        Cmac_interp = Pmac_interp * 0.0
        Cmac_interp[0:3] = Pmac_interp[0:3] / q_dyn / model.macgrid['A_ref']
        Cmac_interp[3] = Pmac_interp[3] / q_dyn / model.macgrid['A_ref'] / model.macgrid['b_ref']
        Cmac_interp[4] = Pmac_interp[4] / q_dyn / model.macgrid['A_ref'] / model.macgrid['c_ref']
        Cmac_interp[5] = Pmac_interp[5] / q_dyn / model.macgrid['A_ref'] / model.macgrid['b_ref']

        plt.figure()
        plt.scatter(x_dlm, y_dlm, c=cp_interp, s=50, linewidths=0.0, cmap='jet', vmin=0.0, vmax=1.0) #bwr #vmin=np.min([cp_interp, cp_dlm]), vmax=np.max([cp_interp, cp_dlm]
        plt.title('Cp CFD')
        desc = 'Ma = ' + str(trimcase['Ma']) + ', alpha = ' + str(alphas[i_alpha]) + ', ' + str(trimcase['altitude'])
        plt.text(0, -8, desc)
        plt.text(0, -9.0, 'Cz: %.4f' % float(Cmac_interp[2]) )
        plt.text(0, -9.5, 'Cmy: %.4f' % float(Cmac_interp[4]) )
        plt.text(0, -10.0, 'Cmx: %.4f' % float(Cmac_interp[3]) )
        
        plt.colorbar()

         
        
    f = open('/scratch/DLR-F19_tau/downwash_corr.pickle', 'w')
    cPickle.dump(downwash_corr, f, cPickle.HIGHEST_PROTOCOL)
    f.close()
    
    
    wj_corrfac = scipy.interpolate.interp1d(np.array(downwash_corr['alpha']), np.array(downwash_corr['wj_corrfac']).T)
    
       
    for alpha in alphas2:
        wjx1 = np.sin(alpha/180*np.pi)
        wj_cam = np.sin(model.camber_twist['cam_rad'] )
        cp_dlm =  np.dot(Qjj, wjx1 * wj_corrfac(alpha) + wj_cam)
        fl_dlm = q_dyn * model.aerogrid['N'].T*model.aerogrid['A']*cp_dlm
        # Bemerkung: greifen die Luftkraefte bei j,l oder k an?
        # Dies würde das Cmy beeinflussen!
        # Gewaehlt: l
        Pl_dlm = np.zeros((1,840*6))[0]
        Pl_dlm[model.aerogrid['set_l'][:,0]] = fl_dlm[0,:]
        Pl_dlm[model.aerogrid['set_l'][:,1]] = fl_dlm[1,:]
        Pl_dlm[model.aerogrid['set_l'][:,2]] = fl_dlm[2,:]
        Pk_dlm = np.dot(model.Dlk.T, Pl_dlm)
        
        Pmac_dlm = np.dot(model.Dkx1.T, Pk_dlm)
        Cmac_dlm = Pmac_dlm * 0.0
        Cmac_dlm[0:3] = Pmac_dlm[0:3] / q_dyn / model.macgrid['A_ref']
        Cmac_dlm[3] = Pmac_dlm[3] / q_dyn / model.macgrid['A_ref'] / model.macgrid['b_ref']
        Cmac_dlm[4] = Pmac_dlm[4] / q_dyn / model.macgrid['A_ref'] / model.macgrid['c_ref']
        Cmac_dlm[5] = Pmac_dlm[5] / q_dyn / model.macgrid['A_ref'] / model.macgrid['b_ref'] 

        plt.figure()
        plt.scatter(x_dlm, y_dlm, c=cp_dlm, s=50, linewidths=0.0, cmap='jet', vmin=0.0, vmax=1.0) #bwr #vmin=np.min([cp_interp, cp_dlm]), vmax=np.max([cp_interp, cp_dlm]
        plt.title('Cp DLM corr')
        desc = 'Ma = ' + str(trimcase['Ma']) + ', alpha = ' + str(alpha) + ', ' + str(trimcase['altitude'])
        plt.text(0, -8, desc)
        plt.text(0, -9.0, 'Cz: %.4f' % float(Cmac_dlm[2]) )
        plt.text(0, -9.5, 'Cmy: %.4f' % float(Cmac_dlm[4]) )
        plt.text(0, -10.0, 'Cmx: %.4f' % float(Cmac_dlm[3]) )
        
        plt.colorbar()
        
        
        
    
    plt.show()
    print 'Done.'
