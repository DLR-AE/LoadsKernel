import copy
import numpy as np

from loadskernel.fem_interfaces.nastran_interface import NastranInterface
from loadskernel.io_functions import read_mona
from loadskernel.io_functions import read_op4


class Nastranf06Interface(NastranInterface):

    def get_stiffness_matrix(self):
        self.KGG = read_op4.load_matrix(self.jcl.geom['filename_KGG'], sparse_output=True, sparse_format=True)
        self.GM = None

    def modes_from_SOL103(self):
        # Mff, Kff and PHIstrc_f
        eigenvalues, eigenvectors, _ = read_mona.NASTRAN_f06_modal(self.jcl.mass['filename_S103'][self.i_mass])
        nodes_selection = self.strcgrid['ID']
        modes_selection = copy.deepcopy(self.jcl.mass['modes'][self.i_mass])
        if self.jcl.mass['omit_rb_modes']:
            modes_selection += 6
        eigenvalues, eigenvectors = read_mona.reduce_modes(eigenvalues, eigenvectors, nodes_selection, modes_selection)
        PHIf_strc = np.zeros((len(self.jcl.mass['modes'][self.i_mass]), len(self.strcgrid['ID']) * 6))
        for i, mode in enumerate(modes_selection):
            eigenvector = eigenvectors[str(mode)][:, 1:]
            PHIf_strc[i, :] = eigenvector.reshape((1, -1))[0]
        self.PHIstrc_f = PHIf_strc.T
        self.eigenvalues_f = np.array(eigenvalues['GeneralizedStiffness'])

    def cg_from_SOL103(self):
        massmatrix_0, inertia, offset_cg, CID = read_mona.Nastran_weightgenerator(self.jcl.mass['filename_S103'][self.i_mass])
        cggrid = {'ID': np.array([9000 + self.i_mass]),
                  'offset': np.array([offset_cg]),
                  'set': np.array([[0, 1, 2, 3, 4, 5]]),
                  'CD': np.array([CID]),
                  'CP': np.array([CID]),
                  'coord_desc': 'bodyfixed',
                  }
        cggrid_norm = {'ID': np.array([9300 + self.i_mass]),
                       'offset': np.array([[-offset_cg[0], offset_cg[1], -offset_cg[2]]]),
                       'set': np.array([[0, 1, 2, 3, 4, 5]]),
                       'CD': np.array([9300]),
                       'CP': np.array([9300]),
                       'coord_desc': 'bodyfixed_DIN9300',
                       }

        # assemble mass matrix about center of gravity, relativ to the axis of the basic coordinate system
        # DO NOT switch signs for coupling terms of I to suite EoMs, Nastran already did this!
        Mb = np.zeros((6, 6))
        Mb[0, 0] = massmatrix_0[0, 0]
        Mb[1, 1] = massmatrix_0[0, 0]
        Mb[2, 2] = massmatrix_0[0, 0]
        Mb[3:6, 3:6] = inertia  # np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]]) * inertia

        # store for later internal use
        self.cggrid = cggrid
        self.cggrid_norm = cggrid_norm

        return Mb, cggrid, cggrid_norm
