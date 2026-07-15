import logging
import numpy as np

from loadskernel import build_splinegrid


class SurfaceMeshDefo():
    """
    This is a base class for all CFD interfaces that provides some mesh deformation functionalities.
    It makes sure that the same mesh deformation methods and parameters are used for all CFD interfaces.
    The way deformations are transferred to the CFD mesh is CFD-solver specific and is therefore
    implemented in the respective CFD interface class.
    """

    def apply_Ux2(self, Ux2):
        if np.any(Ux2):
            logging.info('Apply control surface deflections to cfd surface mesh.')
            Ujx2 = np.zeros(self.aerogrid['n'] * 6)
            if 'hingeline' in self.jcl.aero and self.jcl.aero['hingeline'] == 'y':
                hingeline = 'y'
            elif 'hingeline' in self.jcl.aero and self.jcl.aero['hingeline'] == 'z':
                hingeline = 'z'
            else:  # default
                hingeline = 'y'
            for i_x2 in range(len(self.x2grid['key'])):
                logging.debug('Apply deflection of {} for {:0.4f} [deg].'.format(
                    self.x2grid['key'][i_x2], Ux2[i_x2] / np.pi * 180.0))
                if hingeline == 'y':
                    Ujx2 += np.dot(self.Djx2[i_x2], [0, 0, 0, 0, Ux2[i_x2], 0])
                elif hingeline == 'z':
                    Ujx2 += np.dot(self.Djx2[i_x2], [0, 0, 0, 0, 0, Ux2[i_x2]])
            # Hand-over the surface deformations to the CFD interface.
            self.transfer_deformations_Ux2(self.aerogrid, Ujx2, '_k', rbf_type='wendland2',
                                           surface_spline=False, support_radius=1.5)
        else:
            logging.info('Apply NO control surface deflections to cfd surface mesh.')

    def apply_Uf(self, Uf):
        if 'flex' in self.jcl.aero and self.jcl.aero['flex'] and np.any(Uf):
            logging.info('Apply flexible deformations to cfd surface mesh.')
            # set-up spline grid
            if self.jcl.spline['splinegrid']:
                # make sure that there are no double points in the spline grid as this would cause a singularity of the
                # spline matrix.
                splinegrid = build_splinegrid.grid_thin_out_radius(self.splinegrid, 0.01)
            else:
                # splinegrid = build_splinegrid.grid_thin_out_random(model.strcgrid, 0.5)
                splinegrid = build_splinegrid.grid_thin_out_radius(self.strcgrid, 0.4)

            # get structural deformation
            PHIf_strc = self.mass['PHIf_strc']
            Ug_f_body = np.dot(PHIf_strc.T, Uf.T).T
            # Hand-over the surface deformations to the CFD interface.
            self.transfer_deformations_Uf(splinegrid, Ug_f_body, '', rbf_type='tps', surface_spline=False)
        else:
            logging.info('Apply NO flexible deformations to cfd surface mesh.')
