

import scipy.io.netcdf as netcdf
import matplotlib.pyplot as plt
import numpy as np

import spline_rules
import spline_functions
from build_aero import plot_aerogrid

def controlsurface_meshdefo(model, jcl, path_output):
        
    if jcl.aero.has_key('hingeline') and jcl.aero['hingeline'] == 'y':
        hingeline = 'y'
    elif jcl.aero.has_key('hingeline') and jcl.aero['hingeline'] == 'z':
        hingeline = 'z'
    else: # default
        hingeline = 'y'
                
    for x2_key in model.x2grid['key']:        
        if jcl.meshdefo.has_key(x2_key):
            markers = jcl.meshdefo[x2_key]['markers']
            filename_grid = jcl.meshdefo[x2_key]['filename_grid']
            print 'Extracting points belonging to marker(s) {} from grid {}'.format(str(markers), filename_grid)
            # --- get points on surfaces according to marker ---
            ncfile_grid = netcdf.NetCDFFile(filename_grid, 'r')
            boundarymarker_surfaces = ncfile_grid.variables['boundarymarker_of_surfaces'][:]
            points_of_surfacetriangles = ncfile_grid.variables['points_of_surfacetriangles'][:]
            # expand triangles and merge with quadrilaterals
            if 'points_of_surfacequadrilaterals' in ncfile_grid.variables:
                points_of_surfacetriangles = np.hstack((points_of_surfacetriangles, np.zeros((len(points_of_surfacetriangles), 1), dtype=int)))
                points_of_surfacequadrilaterals = ncfile_grid.variables['points_of_surfacequadrilaterals'][:]
                points_of_surface = np.vstack((points_of_surfacetriangles, points_of_surfacequadrilaterals))
            else:
                points_of_surface = points_of_surfacetriangles
                
            # find global id of points on surface defined by markers 
            points = np.array([], dtype=int)
            for marker in markers:
                points = np.hstack((points, np.where(boundarymarker_surfaces == marker)[0]))
            points = np.unique(points_of_surface[points])
            
            # build cfdgrid
            cfdgrid = {}
            cfdgrid['ID'] = points
            cfdgrid['CP'] = np.zeros(cfdgrid['ID'].shape)
            cfdgrid['CD'] = np.zeros(cfdgrid['ID'].shape)
            cfdgrid['n'] = len(cfdgrid['ID'])   
            cfdgrid['offset'] = np.vstack((ncfile_grid.variables['points_xc'][:][points].copy(), ncfile_grid.variables['points_yc'][:][points].copy(), ncfile_grid.variables['points_zc'][:][points].copy() )).T
            cfdgrid['set'] = np.arange(6*cfdgrid['n']).reshape(-1,6)
            ncfile_grid.close()
#             from mayavi import mlab
#             p_scale = 0.1 # points
#             mlab.figure()
#             mlab.points3d(cfdgrid['offset'][:,0], cfdgrid['offset'][:,1], cfdgrid['offset'][:,2], scale_factor=p_scale)
                    
            # build spline matrix
            PHIj_cfd = spline_functions.spline_rbf(model.aerogrid, '_j', cfdgrid, '', rbf_type='tps', surface_spline=True)
            
            i_x2 = model.x2grid['key'].index(x2_key) # get position i_x2 of current control surface
            for value in jcl.meshdefo[x2_key]['values']:
                if hingeline == 'y':
                    Ujx2 = np.dot(model.Djx2[i_x2],[0,0,0,0,value/180.0*np.pi,0])
                elif hingeline == 'z':
                    Ujx2 = np.dot(model.Djx2[i_x2],[0,0,0,0,0,value/180.0*np.pi])
                        
                Ucfd = PHIj_cfd.dot(Ujx2)
                
#                 from mayavi import mlab
#                 p_scale = 0.1 # points
#                 mlab.figure()
#                 mlab.points3d(model.aerogrid['offset_j'][:,0] + Ujx2[model.aerogrid['set_j'][:,0]], model.aerogrid['offset_j'][:,1] + Ujx2[model.aerogrid['set_j'][:,1]], model.aerogrid['offset_j'][:,2] + Ujx2[model.aerogrid['set_j'][:,2]], scale_factor=p_scale, color=(1,0,0))
#                 mlab.points3d(cfdgrid['offset'][:,0] + Ucfd[cfdgrid['set'][:,0]], cfdgrid['offset'][:,1] + Ucfd[cfdgrid['set'][:,1]], cfdgrid['offset'][:,2] + Ucfd[cfdgrid['set'][:,2]], scale_factor=p_scale/5.0, color=(0,0,1))
#                 mlab.show()
                
                filename_defo = path_output + 'surface_defo_' + x2_key + '_' + str(value) + '.nc'
                print 'Writing ' + filename_defo
                f = netcdf.netcdf_file(filename_defo, 'w')
                f.history = 'Surface deformations created by Loads Kernel'
                f.createDimension('no_of_points', cfdgrid['n'])
                
                global_id = f.createVariable('global_id', 'i', ('no_of_points',))
                x = f.createVariable('x', 'd', ('no_of_points',))
                y = f.createVariable('y', 'd', ('no_of_points',))
                z = f.createVariable('z', 'd', ('no_of_points',))
                dx = f.createVariable('dx', 'd', ('no_of_points',))
                dy = f.createVariable('dy', 'd', ('no_of_points',))
                dz = f.createVariable('dz', 'd', ('no_of_points',))
                
                global_id[:] = cfdgrid['ID']
                x[:] = cfdgrid['offset'][:,0]
                y[:] = cfdgrid['offset'][:,1]
                z[:] = cfdgrid['offset'][:,2]
                dx[:] = Ucfd[cfdgrid['set'][:,0]]
                dy[:] = Ucfd[cfdgrid['set'][:,1]]
                dz[:] = Ucfd[cfdgrid['set'][:,2]]
            
                f.close()
