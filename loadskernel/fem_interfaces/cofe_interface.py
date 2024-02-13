import scipy

from loadskernel.fem_interfaces.nastran_interface import NastranInterface


class CoFEInterface(NastranInterface):

    def get_stiffness_matrix(self):
        with open(self.jcl.geom['filename_CoFE']) as fid:
            CoFE_data = scipy.io.loadmat(fid)
        self.KGG = CoFE_data['KGG']
        self.GM = CoFE_data['GM'].T  # convert from CoFE to Nastran

    def get_mass_matrix(self):
        with open(self.jcl.geom['filename_CoFE']) as fid:
            CoFE_data = scipy.io.loadmat(fid)
        self.MGG = CoFE_data['MGG']
        return self.MGG

    def get_dofs(self):
        # Prepare some data required for modal analysis which is not mass case dependent.
        with open(self.jcl.geom['filename_CoFE']) as fid:
            CoFE_data = scipy.io.loadmat(fid)

        # The DoFs of f-, s- and m-set are indexed with respect to g-set
        # Convert indexing from Matlab to Python
        self.pos_f = CoFE_data['nf_g'].squeeze() - 1
        self.pos_s = CoFE_data['s'].squeeze() - 1
        self.pos_m = CoFE_data['m'].squeeze() - 1
        self.pos_n = CoFE_data['n'].squeeze() - 1

        # Free DoFs (f-set) indexed with respect to n-set
        self.pos_fn = CoFE_data['nf_n'].squeeze() - 1
