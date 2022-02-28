

import scipy.io.netcdf as netcdf
import numpy as np
import logging, h5py, shutil

import loadskernel.spline_functions as spline_functions
import loadskernel.build_splinegrid as build_splinegrid

class Meshdefo:
    def  __init__(self, jcl, model, plotting=False):
        self.jcl        = jcl
        self.model      = model
        self.cfdgrids   = model.cfdgrids
        self.plotting   = plotting
        if not 'surface' in jcl.meshdefo:
            logging.error('jcl.meshdefo has no key "surface"')
            
    def Ux2(self, Ux2):
        Ujx2 = np.zeros(self.model.aerogrid['n']*6)
        if 'hingeline' in self.jcl.aero and self.jcl.aero['hingeline'] == 'y':
            hingeline = 'y'
        elif 'hingeline' in self.jcl.aero and self.jcl.aero['hingeline'] == 'z':
            hingeline = 'z'
        else: # default
            hingeline = 'y'
        for x2_key in self.model.x2grid['key']:        
            i_x2 = self.model.x2grid['key'].index(x2_key) # get position i_x2 of current control surface
            logging.info('Apply control surface deflections of {} for {} [deg] to cfdgrid'.format(x2_key, Ux2[i_x2]/np.pi*180.0))   
            if hingeline == 'y':
                Ujx2 += np.dot(self.model.Djx2[i_x2],[0,0,0,0,Ux2[i_x2],0])
            elif hingeline == 'z':
                Ujx2 += np.dot(self.model.Djx2[i_x2],[0,0,0,0,0,Ux2[i_x2]])
        self.transfer_deformations(self.model.aerogrid, Ujx2, '_k', rbf_type='wendland2', surface_spline=False, support_radius=1.5)
                    
    def Uf(self, Uf, trimcase):
        logging.info('Apply flexible deformations to cfdgrid')
        # set-up spline grid
        if self.jcl.spline['splinegrid'] == True:
            splinegrid = self.model.splinegrid
        else:
            #splinegrid = build_splinegrid.grid_thin_out_random(model.strcgrid, 0.5)
            splinegrid = build_splinegrid.grid_thin_out_radius(self.model.strcgrid, 0.4)
        
        # get structural deformation
        i_mass     = self.model.mass['key'].index(trimcase['mass'])
        PHIf_strc  = self.model.mass['PHIf_strc'][i_mass]
        Ug_f_body = np.dot(PHIf_strc.T, Uf.T).T
        
        self.transfer_deformations(splinegrid, Ug_f_body, '', rbf_type='tps', surface_spline=False)

    def init_deformations(self):
        # create empty deformation vectors for cfdgrids
        self.Ucfd = []
        for cfdgrid in self.cfdgrids:
            self.Ucfd.append(np.zeros(cfdgrid['n']*6))
    
    def transfer_deformations(self, grid_i, U_i, set_i, rbf_type, surface_spline, support_radius=2.0):
        if self.plotting:
            # set-up plot
            from mayavi import mlab
            p_scale = 0.05 # points
            mlab.figure()
            mlab.points3d(grid_i['offset'+set_i][:,0], grid_i['offset'+set_i][:,1], grid_i['offset'+set_i][:,2] ,  scale_factor=p_scale, color=(1,1,1))
            mlab.points3d(grid_i['offset'+set_i][:,0] + U_i[grid_i['set'+set_i][:,0]], grid_i['offset'+set_i][:,1] + U_i[grid_i['set'+set_i][:,1]], grid_i['offset'+set_i][:,2] + U_i[grid_i['set'+set_i][:,2]],  scale_factor=p_scale, color=(1,0,0))
        for grid_d, Ucfd in zip(self.cfdgrids, self.Ucfd):
            logging.debug('Working on marker {}'.format(grid_d['desc']))
            # build spline matrix
            PHIi_d = spline_functions.spline_rbf(grid_i, set_i, grid_d, '', 
                                                 rbf_type=rbf_type, surface_spline=surface_spline, 
                                                 support_radius=support_radius, dimensions=[U_i.size, grid_d['n']*6])
            # store deformation of cfdgrid
            Ucfd += PHIi_d.dot(U_i)
            if self.plotting:
                U_d = PHIi_d.dot(U_i)
                mlab.points3d(grid_d['offset'][:,0], grid_d['offset'][:,1], grid_d['offset'][:,2], color=(0,0,0), mode='point')
                mlab.points3d(grid_d['offset'][:,0] + U_d[grid_d['set'][:,0]], grid_d['offset'][:,1] + U_d[grid_d['set'][:,1]], grid_d['offset'][:,2] + U_d[grid_d['set'][:,2]], color=(0,0,1), mode='point')
            del PHIi_d
        if self.plotting:
            mlab.show()
    
    def write_deformations(self, filename_defo):
        if 'fileformat' in self.jcl.meshdefo['surface'] and self.jcl.meshdefo['surface']['fileformat']=='cgns':
            self.write_defo_cgns(filename_defo)
        elif 'fileformat' in self.jcl.meshdefo['surface'] and self.jcl.meshdefo['surface']['fileformat']=='netcdf':
            self.write_defo_netcdf(filename_defo)
        else:
            logging.error('jcl.meshdefo["surface"]["fileformat"] must be "netcdf" or "cgns"' )
            return

    def write_defo_netcdf(self, filename_defo):
        logging.info( 'Writing ' + filename_defo + '.nc')
        f = netcdf.netcdf_file(filename_defo + '.nc', 'w')
        f.history = 'Surface deformations created by Loads Kernel'
        
        # Assemble temporary output. One point may belong to multiple markers.
        tmp_IDs = np.array([], dtype='int')
        tmp_x = np.array([])
        tmp_y = np.array([])
        tmp_z = np.array([])
        tmp_dx = np.array([])
        tmp_dy = np.array([])
        tmp_dz = np.array([])
        for cfdgrid, Ucfd in zip(self.cfdgrids, self.Ucfd): 
            tmp_IDs = np.concatenate((tmp_IDs, cfdgrid['ID']))
            tmp_x = np.concatenate((tmp_x, cfdgrid['offset'][:,0]))
            tmp_y = np.concatenate((tmp_y, cfdgrid['offset'][:,1]))
            tmp_z = np.concatenate((tmp_z, cfdgrid['offset'][:,2]))
            tmp_dx = np.concatenate((tmp_dx, Ucfd[cfdgrid['set'][:,0]]))
            tmp_dy = np.concatenate((tmp_dy, Ucfd[cfdgrid['set'][:,1]]))
            tmp_dz = np.concatenate((tmp_dz, Ucfd[cfdgrid['set'][:,2]]))
        IDs, pos = np.unique(tmp_IDs, return_index=True)
        
        f.createDimension('no_of_points', IDs.__len__())
        # create variables
        global_id = f.createVariable('global_id', 'i', ('no_of_points',))
        x = f.createVariable('x', 'd', ('no_of_points',))
        y = f.createVariable('y', 'd', ('no_of_points',))
        z = f.createVariable('z', 'd', ('no_of_points',))
        dx = f.createVariable('dx', 'd', ('no_of_points',))
        dy = f.createVariable('dy', 'd', ('no_of_points',))
        dz = f.createVariable('dz', 'd', ('no_of_points',))
        # fill variables with data
        global_id[:] = IDs
        x[:] = tmp_x[pos]
        y[:] = tmp_y[pos]
        z[:] = tmp_z[pos]
        dx[:] = tmp_dx[pos]
        dy[:] = tmp_dy[pos]
        dz[:] = tmp_dz[pos]

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
    