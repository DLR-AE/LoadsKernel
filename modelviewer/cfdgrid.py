

import scipy.io.netcdf as netcdf
import numpy as np
import h5py, shutil, imp
read_module_from_LK = imp.load_source('read_cfdgrids', '../kernel/read_cfdgrids.py')

class TauGrid(read_module_from_LK.read_cfdgrids):
    def  __init__(self):
        pass

    def load_file(self, filename):
        self.filename_grid = filename 
        self.get_markers()
        self.read_cfdmesh_netcdf()
        
    def get_markers(self):
        ncfile_grid = netcdf.NetCDFFile(self.filename_grid, 'r')
        self.markers = ncfile_grid.variables['marker'][:].tolist()
