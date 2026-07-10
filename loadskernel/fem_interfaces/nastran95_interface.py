# Built-ins
import logging

# Libs
import numpy as np
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

    def get_sets_from_bitposes(self, x_dec):
        """
        Reference:
        National Aeronautics and Space Administration, The Nastran Programmer's Manual, NASA SP-223(01). Washington, D.C.,
        COSMIC, 1972.
        Section 2.3.13.3 USET (TABLE), page 2.3-61
        Assumption: There is (only) the f-, s- & m-set

        Bit positons  | decimal notation |  old sets        | new set
        -------------------------------------------------------------
        31, 30 and 25 | 2, 4 and 128     | 'S', 'O' and 'A' | g-set
        22            | 1024             | 'SB'             | s-set
        32            | 1                | 'M'              | m-set
        """
        # We don't know why, but the USET values exported from Nastran 95 are not the expected binary numbers but other numbers
        # (e.g. 17, 496 or 1074, possibly depending on the operating system). The test case (DC3) didn't have any s-set.
        logging.info('Extracting bit positions from Nastran 95 USET to determine DoFs')
        # The DoFs of f-, s- and m-set are indexed with respect to g-set
        self.pos_m = [i for i, x in enumerate(x_dec) if x in [1, 17]]
        self.pos_f = [i for i, x in enumerate(x_dec) if x in [2, 4, 128, 496, 1074]]
        # Not (yet) sure if this is a bug, but in some models there are IDs labled with bit positions 22 and 24,
        # resulting in a decimal notation of 1280. So far, they were treated like the s-set.
        self.pos_s = [i for i, x in enumerate(x_dec) if x in [1024, 1280]]
        # The n-set is the sum of s-set and f-set
        self.pos_n = self.pos_s + self.pos_f
        # Sort the n-set by the DoFs
        sorting = np.argsort(self.pos_n)
        self.pos_n = [self.pos_n[i] for i in sorting]
        # Free DoFs (f-set) indexed with respect to n-set
        self.pos_fn = list(np.where(sorting >= len(self.pos_s))[0])
