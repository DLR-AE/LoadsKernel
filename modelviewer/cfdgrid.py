

import scipy.io.netcdf as netcdf
import numpy as np
import h5py, shutil

class TauGrid:
    def  __init__(self):
        pass

    def load_file(self, filename):
        self.filename_grid = filename 
        self.get_markers()
        self.read_cfdmesh_netcdf()
        
    def get_markers(self):
        ncfile_grid = netcdf.NetCDFFile(self.filename_grid, 'r')
        self.markers = ncfile_grid.variables['marker'][:].tolist()

    def read_cfdmesh_netcdf(self, merge_domains=False):
        markers = self.markers
        #logging.info( 'Extracting points belonging to marker(s) {} from grid {}'.format(str(markers), self.filename_grid))
        # --- get all points on surfaces ---
        ncfile_grid = netcdf.NetCDFFile(self.filename_grid, 'r')
        boundarymarker_surfaces = ncfile_grid.variables['boundarymarker_of_surfaces'][:]
        points_of_surface = []
        # merge triangles with quadrilaterals
        if 'points_of_surfacetriangles' in ncfile_grid.variables:
            points_of_surface += ncfile_grid.variables['points_of_surfacetriangles'][:].tolist()
        if 'points_of_surfacequadrilaterals' in ncfile_grid.variables:
            points_of_surface += ncfile_grid.variables['points_of_surfacequadrilaterals'][:].tolist()
        
        self.cfdgrids = []
        if merge_domains:   
            # --- get points on surfaces according to marker ---
            surfaces = np.array([], dtype=int)
            for marker in markers:
                surfaces = np.hstack((surfaces, np.where(boundarymarker_surfaces == marker)[0]))
            points = np.unique([points_of_surface[s] for s in surfaces])
            # build cfdgrid
            cfdgrid = {}
            cfdgrid['ID'] = points
            cfdgrid['CP'] = np.zeros(cfdgrid['ID'].shape)
            cfdgrid['CD'] = np.zeros(cfdgrid['ID'].shape)
            cfdgrid['n'] = len(cfdgrid['ID'])   
            cfdgrid['offset'] = np.vstack((ncfile_grid.variables['points_xc'][:][points].copy(), ncfile_grid.variables['points_yc'][:][points].copy(), ncfile_grid.variables['points_zc'][:][points].copy() )).T
            cfdgrid['set'] = np.arange(6*cfdgrid['n']).reshape(-1,6)
            cfdgrid['desc'] = markers
            cfdgrid['points_of_surface'] = [points_of_surface[s] for s in surfaces]
            self.cfdgrids.append(cfdgrid)
        else:
            for marker in markers:
                # --- get points on surfaces according to marker ---
                surfaces = np.where(boundarymarker_surfaces == marker)[0]
                points = np.unique([points_of_surface[s] for s in surfaces])                
                # build cfdgrid
                cfdgrid = {}
                cfdgrid['ID'] = points
                cfdgrid['CP'] = np.zeros(cfdgrid['ID'].shape)
                cfdgrid['CD'] = np.zeros(cfdgrid['ID'].shape)
                cfdgrid['n'] = len(cfdgrid['ID'])   
                cfdgrid['offset'] = np.vstack((ncfile_grid.variables['points_xc'][:][points].copy(), ncfile_grid.variables['points_yc'][:][points].copy(), ncfile_grid.variables['points_zc'][:][points].copy() )).T
                cfdgrid['set'] = np.arange(6*cfdgrid['n']).reshape(-1,6)
                cfdgrid['desc'] = str(marker)
                cfdgrid['points_of_surface'] = [points_of_surface[s] for s in surfaces]
                self.cfdgrids.append(cfdgrid)
        ncfile_grid.close()

