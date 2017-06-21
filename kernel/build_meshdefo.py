

import scipy.io.netcdf as netcdf
import matplotlib.pyplot as plt
import numpy as np
import logging, h5py, shutil

import spline_rules, spline_functions, build_splinegrid
from build_aero import plot_aerogrid

def controlsurface_meshdefo(model, jcl, job_name, path_output):
        
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
            cfdgrid = read_cfdmesh_netcdf(filename_grid, markers)
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
                
                filename_defo = path_output + 'surface_defo_' + job_name + '_' + x2_key + '_' + str(value) + '.nc'
                logging.info( 'Writing ' + filename_defo)
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
                
def Ug_f_meshdefo(model, jcl, responses, job_name, path_output):
                
    for response in responses:        
        if jcl.meshdefo.has_key('surface'):
            if jcl.meshdefo.has_key('fileformat') and jcl.meshdefo['fileformat']=='cgns':
                cfdgrid = read_cfdmesh_cgns(jcl.meshdefo['surface']['filename_grid'])
            else: # assume netcdf file format (Tau default)
                cfdgrid = read_cfdmesh_netcdf(jcl.meshdefo['surface']['filename_grid'], jcl.meshdefo['surface']['markers'])
            # build spline matrix
            #splinegrid = build_splinegrid.grid_thin_out_random(model.strcgrid, 0.5)
            splinegrid = build_splinegrid.grid_thin_out_radius(model.strcgrid, 0.4)
            #splinegrid = model.splinegrid
            PHIstrc_cfd = spline_functions.spline_rbf(splinegrid, '', cfdgrid, '', rbf_type='tps', surface_spline=False, dimensions=[model.strcgrid['n']*6, cfdgrid['n']*6])

            i_mass     = model.mass['key'].index(jcl.trimcase[response['i']]['mass'])
            PHIf_strc  = model.mass['PHIf_strc'][i_mass]
            n_modes    = model.mass['n_modes'][i_mass]
            Uf = response['X'][12:12+n_modes]
            Ug_f_body = np.dot(PHIf_strc.T, Uf.T).T #*100.0
            Ucfd = PHIstrc_cfd.dot(Ug_f_body)
                
            from mayavi import mlab
            p_scale = 0.05 # points
            mlab.figure()
            mlab.points3d(model.strcgrid['offset'][:,0], model.strcgrid['offset'][:,1], model.strcgrid['offset'][:,2] ,  scale_factor=p_scale, color=(1,1,1))
            mlab.points3d(splinegrid['offset'][:,0], splinegrid['offset'][:,1], splinegrid['offset'][:,2] ,  scale_factor=p_scale*2.0, color=(0,1,1))
            mlab.points3d(model.strcgrid['offset'][:,0] + Ug_f_body[model.strcgrid['set'][:,0]], model.strcgrid['offset'][:,1] + Ug_f_body[model.strcgrid['set'][:,1]], model.strcgrid['offset'][:,2] + Ug_f_body[model.strcgrid['set'][:,2]],  scale_factor=p_scale, color=(1,0,0))
            mlab.points3d(cfdgrid['offset'][:,0], cfdgrid['offset'][:,1], cfdgrid['offset'][:,2], scale_factor=p_scale/5.0, color=(0,0,0), mode='point')
            mlab.points3d(cfdgrid['offset'][:,0] + Ucfd[cfdgrid['set'][:,0]], cfdgrid['offset'][:,1] + Ucfd[cfdgrid['set'][:,1]], cfdgrid['offset'][:,2] + Ucfd[cfdgrid['set'][:,2]], scale_factor=p_scale/5.0, color=(0,0,1), mode='point')
            mlab.show()
                
            filename_defo = path_output + 'surface_defo_' + job_name + '_subcase_' + str(jcl.trimcase[response['i']]['subcase'])
            write_cfdmesh_netcdf(cfdgrid, Ucfd, filename_defo)
            write_cfdmesh_cgns(cfdgrid, Ucfd, filename_defo, jcl.meshdefo['surface']['filename_grid'])

def write_cfdmesh_netcdf(cfdgrid, Ucfd, filename_defo):
    logging.info( 'Writing ' + filename_defo + '.nc')
    f = netcdf.netcdf_file(filename_defo + '.nc', 'w')
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
    
def read_cfdmesh_netcdf(filename_grid, markers):
    logging.info( 'Extracting points belonging to marker(s) {} from grid {}'.format(str(markers), filename_grid))
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

    return cfdgrid

def write_cfdmesh_cgns(cfdgrid, Ucfd, filename_defo, filename_grid):
    logging.info( 'Copying ' + filename_grid)
    shutil.copyfile(filename_grid, filename_defo + '.cgns')
    f_scale = 1000.0 # convert to CGNS units: 1000.0 if mesh is desired in [mm], 1.0 if desired in [m]
    logging.info( 'Writing into ' + filename_defo + '.cgns')
    f = h5py.File(filename_defo + '.cgns', 'r+')
    keys = f['Base'].keys()
    keys.sort()
    i = 0
    for key in keys:
        # loop over domains
        if key[:4] == 'dom-':
            logging.info(' - {} updated'.format(key))
            domain = f['Base'][key]
            shape = domain['GridCoordinates']['CoordinateX'][' data'].shape
            size = domain['GridCoordinates']['CoordinateX'][' data'].size
            domain['GridCoordinates']['CoordinateX'][' data'][:] = ((cfdgrid['offset'][i:i+size,0] + Ucfd[cfdgrid['set'][i:i+size,0]])*f_scale).reshape(shape)
            domain['GridCoordinates']['CoordinateY'][' data'][:] = ((cfdgrid['offset'][i:i+size,1] + Ucfd[cfdgrid['set'][i:i+size,1]])*f_scale).reshape(shape)
            domain['GridCoordinates']['CoordinateZ'][' data'][:] = ((cfdgrid['offset'][i:i+size,2] + Ucfd[cfdgrid['set'][i:i+size,2]])*f_scale).reshape(shape)
            i += size
        else:
            logging.info(' - {} skipped'.format(key))    
    f.close()
    
def read_cfdmesh_cgns(filename_grid):
    logging.info( 'Extracting all points from grid {}'.format(filename_grid))
    f = h5py.File(filename_grid, 'r')
    x = np.array([])
    y = np.array([])
    z = np.array([])
    f_scale = 1.0/1000.0 # convert to SI units: 0.001 if mesh is given in [mm], 1.0 if given in [m]
    keys = f['Base'].keys()
    keys.sort()
    for key in keys:
        # loop over domains
        if key[:4] == 'dom-':
            logging.info(' - {} included'.format(key))
            domain = f['Base'][key]
            x = np.concatenate((x, domain['GridCoordinates']['CoordinateX'][' data'][:].reshape(-1)*f_scale))
            y = np.concatenate((y, domain['GridCoordinates']['CoordinateY'][' data'][:].reshape(-1)*f_scale))
            z = np.concatenate((z, domain['GridCoordinates']['CoordinateZ'][' data'][:].reshape(-1)*f_scale))
        else:
            logging.info(' - {} skipped'.format(key))
    f.close()
    n = len(x)
    # build cfdgrid
    cfdgrid = {}
    cfdgrid['ID'] = np.arange(n)+1
    cfdgrid['CP'] = np.zeros(n)
    cfdgrid['CD'] = np.zeros(n)
    cfdgrid['n'] = n 
    cfdgrid['offset'] = np.vstack((x,y,z)).T
    cfdgrid['set'] = np.arange(6*cfdgrid['n']).reshape(-1,6)

    return cfdgrid