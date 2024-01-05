import copy
import logging
import numpy as np

from loadskernel.fem_interfaces.nastran_interface import NastranInterface
from loadskernel.io_functions import read_b2000


class B2000Interface(NastranInterface):

    def get_stiffness_matrix(self):
        self.KGG = read_b2000.read_csv(self.jcl.geom['filename_KGG'], sparse_output=True)
        self.GM = None

    def get_mass_matrix(self, i_mass):
        self.i_mass = i_mass
        self.MGG = read_b2000.read_csv(self.jcl.mass['filename_MGG'][self.i_mass], sparse_output=True)
        return self.MGG

    def get_dofs(self):
        """
        Instead of the u-set, B2000 uses a matrix R to relate the g-set to the f-set.
        Example:
        Kff = R.T * Kgg * R
        ug = R * uf
        """
        self.Rtrans = read_b2000.read_csv(self.jcl.geom['filename_Rtrans'], sparse_output=True)

    def prepare_stiffness_matrices(self):
        self.KFF = self.Rtrans.dot(self.KGG).dot(self.Rtrans.T)

    def prepare_mass_matrices(self):
        logging.info('Prepare mass matrices for independent and free DoFs (f-set)')
        self.MFF = self.Rtrans.dot(self.MGG).dot(self.Rtrans.T)

    def modalanalysis(self):
        modes_selection = copy.deepcopy(self.jcl.mass['modes'][self.i_mass])
        if self.jcl.mass['omit_rb_modes']:
            modes_selection += 6
        eigenvalue, eigenvector = self.calc_elastic_modes(self.KFF, self.MFF, modes_selection.max())
        logging.info('From these {} modes, the following {} modes are selected: {}'.format(
            modes_selection.max(), len(modes_selection), modes_selection))
        self.eigenvalues_f = eigenvalue[modes_selection - 1]
        # reconstruct modal matrix for g-set / strcgrid
        self.PHIstrc_f = np.zeros((6 * self.strcgrid['n'], len(modes_selection)))
        i = 0  # counter selected modes
        for i_mode in modes_selection - 1:
            # deformation of f-set due to i_mode is the ith column of the eigenvector
            Uf = eigenvector[:, i_mode].real.reshape((-1, 1))
            Ug = self.Rtrans.T.dot(Uf)
            self.PHIstrc_f[:, i] = Ug.squeeze()
            i += 1
