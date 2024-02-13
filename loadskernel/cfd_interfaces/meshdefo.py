import logging
import numpy as np

from loadskernel import build_splinegrid


class Meshdefo():
    """
    This is a base class for all CFD interfaces that provides some mesh deformation functionalities.
    """

    def Ux2(self, Ux2):
        logging.info('Apply control surface deflections to cfdgrid.')
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
        self.transfer_deformations(self.aerogrid, Ujx2, '_k', rbf_type='wendland2', surface_spline=False, support_radius=1.5)

    def Uf(self, Uf):
        if 'flex' in self.jcl.aero and self.jcl.aero['flex']:
            logging.info('Apply flexible deformations to cfdgrid.')
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

            self.transfer_deformations(splinegrid, Ug_f_body, '', rbf_type='tps', surface_spline=False)
        else:
            logging.info('Apply NO flexible deformations to cfdgrid.')
