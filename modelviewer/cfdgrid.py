from scipy.io import netcdf
import loadskernel.io_functions.read_cfdgrids


class TauGrid(loadskernel.io_functions.read_cfdgrids.ReadCfdgrids):

    def __init__(self):
        pass

    def load_file(self, filename):
        self.filename_grid = filename
        self.get_markers()
        self.read_cfdmesh_netcdf()

    def get_markers(self):
        ncfile_grid = netcdf.NetCDFFile(self.filename_grid, 'r')
        self.markers = ncfile_grid.variables['marker'][:].tolist()


class SU2Grid(loadskernel.io_functions.read_cfdgrids.ReadCfdgrids):

    def __init__(self):
        pass

    def load_file(self, filename):
        self.filename_grid = filename
        self.read_cfdmesh_su2()
