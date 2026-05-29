# Built-ins
import logging

# Libs
from scipy.sparse import csc_matrix

# Own modules
from loadskernel.fem_interfaces.nastran_interface import NastranInterface
from loadskernel.io_functions import read_op2


class Nastran95Interface(NastranInterface):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.KGG: csc_matrix = csc_matrix((0, 0))
        self.GM: csc_matrix = csc_matrix((0, 0))
        self.MGG: csc_matrix = csc_matrix((0, 0))
        self.i_mass: int = 0
        self.op2_geom = None

    def get_stiffness_matrix(self):
        # Use internal op2 reader. The matrices are numerically eqivalent compared with pyNastran's op2 reader (tested with the
        # DC3 model using np.allclose).
        if 'filename_op2' in self.jcl.geom:
            self.op2_geom = read_op2.read_op2(self.jcl.geom['filename_op2'])
            # Convert to sparse foramt.
            self.KGG = csc_matrix(self.op2_geom['KGG'])
            self.GM = csc_matrix(self.op2_geom['GM'].T)
        else:
            logging.error('Please provide filename of .op2 file containing matrices Kgg and GM.')

    def get_mass_matrix(self, i_mass):
        self.i_mass = i_mass
        if 'filename_op2' in self.jcl.mass:
            op2_mass = read_op2.read_op2(self.jcl.mass['filename_op2'][self.i_mass])
            self.MGG = csc_matrix(op2_mass['MGG'])
        else:
            logging.error('Please provide filename(s) of .op2 files containing the matrices Mgg.')

        return self.MGG

    def get_dofs(self):
        # See if a USET table was included in the OP2 file from the geom setion (along with the stiffness matrix).
        if self.op2_geom['uset'] is None:
            logging.error('No USET found in OP2-file %s !', self.jcl.geom['filename_op2'])
        self.get_sets_from_bitposes(self.op2_geom['uset'])
