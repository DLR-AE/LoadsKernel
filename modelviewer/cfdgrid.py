from scipy.io import netcdf
import loadskernel.io_functions.read_cfdgrids


class TauGrid(loadskernel.io_functions.read_cfdgrids.ReadCfdgrids):

    def load_file(self, filename):
        self.filename_grid = filename
        self.get_markers()
        self.read_netcdf(self.filename_grid, self.markers)

    def get_markers(self):
        ncfile_grid = netcdf.netcdf_file(self.filename_grid, 'r')
        self.markers = ncfile_grid.variables['marker'][:].tolist()


class SU2Grid(loadskernel.io_functions.read_cfdgrids.ReadCfdgrids):

    def load_file(self, filename):
        self.read_su2(filename)
