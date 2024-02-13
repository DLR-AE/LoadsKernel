import os
import sys
from tvtk.api import tvtk
try:
    import pyiges
except ImportError:
    pass


class IgesMesh():

    def __init__(self):
        self.meshes = []

    def load_file(self, filename):
        self.filename = filename
        self.read_iges()

    def read_iges(self):
        """
        The Pyvista package provides a nice reader for iges files (pyiges) and has a converter to VTK.
        This VTK object can be plotted with mayavi natively so that we don't need to switch everything
        to pyvista just for the iges plotting.
        """
        if 'pyiges' in sys.modules:
            iges_object = pyiges.read(self.filename)
            pyvista_mb = iges_object.to_vtk(
                lines=False, bsplines=False, surfaces=True, points=False, merge=False)
            """
            Since version 0.32, Pyvista returns a MultiBlockDataSet, with is not hashable anymore.
            According to a discussion on github, this is a feature, not a bug (https://github.com/pyvista/pyvista/issues/1751)
            The following lines do the conversion to a "true" VTK MultiBlockDataSet, which is hashable and can be
            plotted in mayavi without any further issues.
            """
            # Create a new, empty vtk dataset
            vtk_mb = tvtk.MultiBlockDataSet()
            # Add the pyvista data block-by-block to the vtk dataset
            for i, data in enumerate(pyvista_mb):
                vtk_mb.set_block(i, data)
            # Store everything
            self.meshes.append({'desc': os.path.split(self.filename)[-1],
                                'vtk': vtk_mb})
        else:
            print('Pyiges modul not found, can not read iges file.')
