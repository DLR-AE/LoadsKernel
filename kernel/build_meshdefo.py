

import scipy.io.netcdf as netcdf
import matplotlib.pyplot as plt
import numpy as np
import logging, h5py, shutil

import spline_rules, spline_functions, build_splinegrid
from build_aero import plot_aerogrid

class meshdefo:
    def  __init__(self, jcl, model, responses, plotting=False):
        self.model = model
        self.responses = responses
        self.jcl = jcl
        self.plotting=plotting
        if not jcl.meshdefo.has_key('surface'):
            logging.error('jcl.meshdefo has no key "surface"')

    def controlsurfaces(self, job_name, path_output):
         
        if self.jcl.aero.has_key('hingeline') and self.jcl.aero['hingeline'] == 'y':
            hingeline = 'y'
        elif self.jcl.aero.has_key('hingeline') and self.jcl.aero['hingeline'] == 'z':
            hingeline = 'z'
        else: # default
            hingeline = 'y'
        splinegrid = self.model.aerogrid
        for x2_key in self.model.x2grid['key']:        
            if self.jcl.meshdefo.has_key(x2_key):
                logging.info('Apply control surface deflections of {} for {} [deg] to cfdgrid'.format(x2_key, str(self.jcl.meshdefo[x2_key]['values'])))   
                i_x2 = self.model.x2grid['key'].index(x2_key) # get position i_x2 of current control surface
                for value in self.jcl.meshdefo[x2_key]['values']:
                    if hingeline == 'y':
                        Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,value/180.0*np.pi,0])
                    elif hingeline == 'z':
                        Ujx2 = np.dot(self.model.Djx2[i_x2],[0,0,0,0,0,value/180.0*np.pi])
                        
                    self.transfer_deformations(splinegrid, Ujx2, '_k', surface_spline=True)
                    self.write_defo(job_name, path_output, path_output + 'surface_defo_' + job_name + '_' + x2_key + '_' + str(value) )                    
                    
    def Ug_f(self, job_name, path_output):
        for response in self.responses:
            logging.info('Apply flexible deformations from subcase {} to cfdgrid'.format(str(self.jcl.trimcase[response['i']]['subcase'])))          
            # set-up spline grid
            #splinegrid = build_splinegrid.grid_thin_out_random(model.strcgrid, 0.5)
            splinegrid = build_splinegrid.grid_thin_out_radius(self.model.strcgrid, 0.4)
            #splinegrid = model.strcgrid
            # get structural deformation
            i_mass     = self.model.mass['key'].index(self.jcl.trimcase[response['i']]['mass'])
            PHIf_strc  = self.model.mass['PHIf_strc'][i_mass]
            n_modes    = self.model.mass['n_modes'][i_mass]
            Uf = response['X'][12:12+n_modes]
            Ug_f_body = np.dot(PHIf_strc.T, Uf.T).T #*100.0
            
            self.transfer_deformations(splinegrid, Ug_f_body)
            self.write_defo(job_name, path_output, path_output + 'surface_defo_' + job_name + '_subcase_' + str(self.jcl.trimcase[response['i']]['subcase']))
    
    def read_cfdgrids(self):
        if self.jcl.meshdefo['surface'].has_key('fileformat') and self.jcl.meshdefo['surface']['fileformat']=='cgns':
            self.read_cfdmesh_cgns()
        elif self.jcl.meshdefo['surface'].has_key('fileformat') and self.jcl.meshdefo['surface']['fileformat']=='netcdf':
            self.read_cfdmesh_netcdf()
        else:
            logging.error('jcl.meshdefo["surface"]["fileformat"] must be "netcdf" or "cgns"' )
            return
        
    def transfer_deformations(self, grid_i, U_i, set_i = '', surface_spline=False):
        if self.plotting:
            # set-up plot
            from mayavi import mlab
            p_scale = 0.05 # points
            mlab.figure()
            mlab.points3d(grid_i['offset'+set_i][:,0], grid_i['offset'+set_i][:,1], grid_i['offset'+set_i][:,2] ,  scale_factor=p_scale, color=(1,1,1))
            mlab.points3d(grid_i['offset'+set_i][:,0] + U_i[grid_i['set'+set_i][:,0]], grid_i['offset'+set_i][:,1] + U_i[grid_i['set'+set_i][:,1]], grid_i['offset'+set_i][:,2] + U_i[grid_i['set'+set_i][:,2]],  scale_factor=p_scale, color=(1,0,0))
        self.Ucfd = []
        for grid_d in self.cfdgrids:
            logging.info('Working on marker {}'.format(grid_d['desc']))
            # build spline matrix
            PHIi_d = spline_functions.spline_rbf(grid_i, set_i, grid_d, '', rbf_type='tps', surface_spline=surface_spline, dimensions=[U_i.size, grid_d['n']*6])
            # store deformation of cfdgrid
            self.Ucfd.append(PHIi_d.dot(U_i))
            if self.plotting:
                U_d = PHIi_d.dot(U_i)
                mlab.points3d(grid_d['offset'][:,0], grid_d['offset'][:,1], grid_d['offset'][:,2], color=(0,0,0), mode='point')
                mlab.points3d(grid_d['offset'][:,0] + U_d[grid_d['set'][:,0]], grid_d['offset'][:,1] + U_d[grid_d['set'][:,1]], grid_d['offset'][:,2] + U_d[grid_d['set'][:,2]], color=(0,0,1), mode='point')
            
        if self.plotting:
            mlab.show()
            
    def write_defo(self, job_name, path_output, filename_defo):
        if self.jcl.meshdefo['surface'].has_key('fileformat') and self.jcl.meshdefo['surface']['fileformat']=='cgns':
            self.write_defo_cgns(filename_defo)
        elif self.jcl.meshdefo['surface'].has_key('fileformat') and self.jcl.meshdefo['surface']['fileformat']=='netcdf':
            self.write_defo_netcdf(filename_defo)
        else:
            logging.error('jcl.meshdefo["surface"]["fileformat"] must be "netcdf" or "cgns"' )
            return

    def write_defo_netcdf(self, filename_defo):
        logging.info( 'Writing ' + filename_defo + '.nc')
        f = netcdf.netcdf_file(filename_defo + '.nc', 'w')
        f.history = 'Surface deformations created by Loads Kernel'
        # calc total number of points
        n = 0
        for cfdgrid in self.cfdgrids: n += cfdgrid['n'] 
        f.createDimension('no_of_points', n)
        # create variables
        global_id = f.createVariable('global_id', 'i', ('no_of_points',))
        x = f.createVariable('x', 'd', ('no_of_points',))
        y = f.createVariable('y', 'd', ('no_of_points',))
        z = f.createVariable('z', 'd', ('no_of_points',))
        dx = f.createVariable('dx', 'd', ('no_of_points',))
        dy = f.createVariable('dy', 'd', ('no_of_points',))
        dz = f.createVariable('dz', 'd', ('no_of_points',))
        # fill variables with data
        i = 0
        for cfdgrid, Ucfd in zip(self.cfdgrids, self.Ucfd):
            size = cfdgrid['n']
            global_id[i:i+size] = cfdgrid['ID']
            x[i:i+size] = cfdgrid['offset'][:,0]
            y[i:i+size] = cfdgrid['offset'][:,1]
            z[i:i+size] = cfdgrid['offset'][:,2]
            dx[i:i+size] = Ucfd[cfdgrid['set'][:,0]]
            dy[i:i+size] = Ucfd[cfdgrid['set'][:,1]]
            dz[i:i+size] = Ucfd[cfdgrid['set'][:,2]]
            i += size
        f.close()

    def write_defo_cgns(self, filename_defo):
        filename_grid = self.jcl.meshdefo['surface']['filename_grid']
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
                # get matching defo
                Ucfd = self.Ucfd[i]
                cfdgrid = self.cfdgrids[i]
                # write defo into file
                shape = domain['GridCoordinates']['CoordinateX'][' data'].shape
                size = domain['GridCoordinates']['CoordinateX'][' data'].size
                domain['GridCoordinates']['CoordinateX'][' data'][:] = ((cfdgrid['offset'][:,0] + Ucfd[cfdgrid['set'][:,0]])*f_scale).reshape(shape)
                domain['GridCoordinates']['CoordinateY'][' data'][:] = ((cfdgrid['offset'][:,1] + Ucfd[cfdgrid['set'][:,1]])*f_scale).reshape(shape)
                domain['GridCoordinates']['CoordinateZ'][' data'][:] = ((cfdgrid['offset'][:,2] + Ucfd[cfdgrid['set'][:,2]])*f_scale).reshape(shape)
                i += 1
            else:
                logging.info(' - {} skipped'.format(key))    
        f.close()
        
    def read_cfdmesh_cgns(self, merge_domains=False):
        filename_grid = self.jcl.meshdefo['surface']['filename_grid']
        logging.info( 'Extracting all points from grid {}'.format(filename_grid))
        f = h5py.File(filename_grid, 'r')
        f_scale = 1.0/1000.0 # convert to SI units: 0.001 if mesh is given in [mm], 1.0 if given in [m]
        keys = f['Base'].keys()
        keys.sort()
        self.cfdgrids = []
        if merge_domains:
            x = np.array([])
            y = np.array([])
            z = np.array([])
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
            cfdgrid['desc'] = 'all domains'
            self.cfdgrids.append(cfdgrid)
        else:
            for key in keys:
                # loop over domains
                if key[:4] == 'dom-':
                    logging.info(' - {} included'.format(key))
                    domain = f['Base'][key]
                    x = domain['GridCoordinates']['CoordinateX'][' data'][:].reshape(-1)*f_scale
                    y = domain['GridCoordinates']['CoordinateY'][' data'][:].reshape(-1)*f_scale
                    z = domain['GridCoordinates']['CoordinateZ'][' data'][:].reshape(-1)*f_scale
                    n = len(x)
                    # build cfdgrid
                    cfdgrid = {}
                    cfdgrid['ID'] = np.arange(n)+1
                    cfdgrid['CP'] = np.zeros(n)
                    cfdgrid['CD'] = np.zeros(n)
                    cfdgrid['n'] = n 
                    cfdgrid['offset'] = np.vstack((x,y,z)).T
                    cfdgrid['set'] = np.arange(6*cfdgrid['n']).reshape(-1,6)
                    cfdgrid['desc'] = key
                    self.cfdgrids.append(cfdgrid)
                else:
                    logging.info(' - {} skipped'.format(key))
        f.close()
        
    def read_cfdmesh_netcdf(self, merge_domains=False):
        filename_grid = self.jcl.meshdefo['surface']['filename_grid']
        markers = self.jcl.meshdefo['surface']['markers']
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
        self.cfdgrids = []
        if merge_domains:   
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
            cfdgrid['desc'] = 'all markers'
            self.cfdgrids.append(cfdgrid)
        else:
            for marker in markers:
                points = np.where(boundarymarker_surfaces == marker)[0]
                points = np.unique(points_of_surface[points])
                # build cfdgrid
                cfdgrid = {}
                cfdgrid['ID'] = points
                cfdgrid['CP'] = np.zeros(cfdgrid['ID'].shape)
                cfdgrid['CD'] = np.zeros(cfdgrid['ID'].shape)
                cfdgrid['n'] = len(cfdgrid['ID'])   
                cfdgrid['offset'] = np.vstack((ncfile_grid.variables['points_xc'][:][points].copy(), ncfile_grid.variables['points_yc'][:][points].copy(), ncfile_grid.variables['points_zc'][:][points].copy() )).T
                cfdgrid['set'] = np.arange(6*cfdgrid['n']).reshape(-1,6)
                cfdgrid['desc'] = str(marker)
                self.cfdgrids.append(cfdgrid)
        ncfile_grid.close()


