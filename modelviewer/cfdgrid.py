

import scipy.io.netcdf as netcdf
import loadskernel.read_cfdgrids

class TauGrid(loadskernel.read_cfdgrids.ReadCfdgrids):
    def  __init__(self):
        pass

    def load_file(self, filename):
        self.filename_grid = filename 
        self.get_markers()
        self.read_cfdmesh_netcdf()
        
    def get_markers(self):
        ncfile_grid = netcdf.NetCDFFile(self.filename_grid, 'r')
        self.markers = ncfile_grid.variables['marker'][:].tolist()
