from pyNastran.op2.op2 import OP2
from scipy.sparse import csc_matrix

from loadskernel.fem_interfaces.nastran_interface import NastranInterface


class Nastran95Interface(NastranInterface):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.KGG: csc_matrix = csc_matrix((0, 0))
        self.GM: csc_matrix = csc_matrix((0, 0))
        self.MGG: csc_matrix = csc_matrix((0, 0))
        self.i_mass: int = 0

    def get_stiffness_matrix(self):
        op2_model = OP2()
        try:
            op2_model.read_op2(self.jcl.geom['filename_op2'])
        except Exception as e:
            print(f"Warning: an error occurred during OP2 read but was ignored.\n{e}")
        self.KGG = csc_matrix(op2_model.matrices['KGG'].data)
        self.GM = csc_matrix(op2_model.matrices['GM'].data).T

    def get_mass_matrix(self, i_mass):
        self.i_mass = i_mass

        op2_model = OP2()
        try:
            op2_model.read_op2(self.jcl.geom['filename_op2'])
        except Exception as e:
            print(f"Warning: an error occurred during OP2 read but was ignored.\n{e}")
        self.MGG = csc_matrix(op2_model.matrices['MGG'].data)

        return self.MGG
