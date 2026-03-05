# Built-ins
import logging
import sys

# Libs
from scipy.sparse import csc_matrix

# Own modules
from loadskernel.fem_interfaces.nastran_interface import NastranInterface

# In case a user has only the core packages and no extras installed, we want to avoid an import error when importing the
# Nastran95Interface class. The OP2 read is only performed when the stiffness or mass matrix is requested, so the error
# is not relevant until then. In that case, we catch the error and issue an error message.
try:
    from pyNastran.op2.op2 import OP2
except ImportError:
    pass


class Nastran95Interface(NastranInterface):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.KGG: csc_matrix = csc_matrix((0, 0))
        self.GM: csc_matrix = csc_matrix((0, 0))
        self.MGG: csc_matrix = csc_matrix((0, 0))
        self.i_mass: int = 0
        # Check if OP2 from pyNastran was imported successfully, see try/except statement in the import section.
        if "pyNastran" not in sys.modules:
            logging.error(
                'pyNastran was/could NOT be imported!'
                'The Nastran95Interface will not be able to read the stiffness and mass matrices from the OP2 file. '
                'Please install pyNastran or the Loads Kernel with the extras to use this feature.'
            )

    def get_stiffness_matrix(self):
        op2_model = OP2()
        try:
            op2_model.read_op2(self.jcl.geom['filename_op2'])
        except Exception as e:
            logging.warning("An error occurred during OP2 read but was ignored.\n%s", e)
        self.KGG = csc_matrix(op2_model.matrices['KGG'].data)
        self.GM = csc_matrix(op2_model.matrices['GM'].data).T

    def get_mass_matrix(self, i_mass):
        self.i_mass = i_mass

        op2_model = OP2()
        try:
            op2_model.read_op2(self.jcl.geom['filename_op2'])
        except Exception as e:
            logging.warning("An error occurred during OP2 read but was ignored.\n%s", e)
        self.MGG = csc_matrix(op2_model.matrices['MGG'].data)

        return self.MGG
