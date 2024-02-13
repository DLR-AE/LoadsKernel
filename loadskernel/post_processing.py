import copy
import logging
import numpy as np

from loadskernel.solution_tools import calc_drehmatrix
from loadskernel.grid_trafo import grid_trafo, vector_trafo
from loadskernel.io_functions.data_handling import load_hdf5_sparse_matrix, load_hdf5_dict


class PostProcessing():
    """
    In this class calculations are made that follow up on every simulation.
    The functions should be able to handle both trim calculations and time simulations.
    """

    def __init__(self, jcl, model, trimcase, response):
        self.jcl = jcl
        self.model = model
        self.trimcase = trimcase
        self.response = response

        mass = self.model['mass'][trimcase['mass']]
        self.n_modes = mass['n_modes'][()]
        self.Mgg = load_hdf5_sparse_matrix(mass['MGG'])
        self.PHIf_strc = mass['PHIf_strc'][()]
        self.PHIstrc_cg = mass['PHIstrc_cg'][()]
        self.PHIcg_norm = mass['PHIcg_norm'][()]

        self.cggrid = load_hdf5_dict(mass['cggrid'])
        self.strcgrid = load_hdf5_dict(self.model['strcgrid'])
        self.mongrid = load_hdf5_dict(self.model['mongrid'])
        self.coord = load_hdf5_dict(self.model['coord'])

        self.PHIk_strc = load_hdf5_sparse_matrix(self.model['PHIk_strc'])
        self.PHIstrc_mon = load_hdf5_sparse_matrix(self.model['PHIstrc_mon'])

        if hasattr(self.jcl, 'landinggear') or hasattr(self.jcl, 'engine'):
            self.extragrid = load_hdf5_dict(self.model['extragrid'])

        if self.jcl.aero['method'] in ['cfd_steady', 'cfd_unsteady']:
            # get cfd splining matrices
            self.PHIcfd_strc = load_hdf5_sparse_matrix(self.model['PHIcfd_strc'])

    def force_summation_method(self):
        logging.info('calculating forces & moments on structural set (force summation method)...')
        response = self.response

        response['Pg_iner'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        response['Pg_aero'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        # response['Pg_unsteady']    = np.zeros((len(response['t']), 6*self.strcgrid['n']))
        # response['Pg_gust']        = np.zeros((len(response['t']), 6*self.strcgrid['n']))
        # response['Pg_cs']          = np.zeros((len(response['t']), 6*self.strcgrid['n']))
        # response['Pg_idrag']       = np.zeros((len(response['t']), 6*self.strcgrid['n']))
        response['Pg_ext'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        response['Pg_cfd'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        response['Pg'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        response['d2Ug_dt2'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        for i_step in range(len(response['t'])):
            if hasattr(self.jcl, 'eom') and self.jcl.eom['version'] == 'waszak':
                # Fuer Bewegungsgleichungen z.B. von Waszack muessen die zusaetzlichen Terme hier ebenfalls beruecksichtigt
                # werden!
                d2Ug_dt2_r = self.PHIstrc_cg.dot(np.hstack((response['d2Ucg_dt2'][i_step, 0:3] - response['g_cg'][i_step, :]
                                                            - np.cross(response['dUcg_dt'][i_step, 0:3],
                                                                       response['dUcg_dt'][i_step, 3:6]),
                                                            response['d2Ucg_dt2'][i_step, 3:6])))
            else:
                # Nastran
                d2Ug_dt2_r = self.PHIstrc_cg.dot(np.hstack((response['d2Ucg_dt2'][i_step, 0:3] - response['g_cg'][i_step, :],
                                                            response['d2Ucg_dt2'][i_step, 3:6])))

            d2Ug_dt2_f = self.PHIf_strc.T.dot(response['d2Uf_dt2'][i_step, :])
            Pg_iner_r = -self.Mgg.dot(d2Ug_dt2_r)
            Pg_iner_f = -self.Mgg.dot(d2Ug_dt2_f)
            response['Pg_iner'][i_step, :] = Pg_iner_r + Pg_iner_f
            response['Pg_aero'][i_step, :] = self.PHIk_strc.T.dot(response['Pk_aero'][i_step, :])
            # response['Pg_gust'][i_step, :] = self.PHIk_strc.T.dot(response['Pk_gust'][i_step, :])
            # response['Pg_unsteady'][i_step, :]   = self.PHIk_strc.T.dot(response['Pk_unsteady'][i_step, :])
            # response['Pg_cs'][i_step, :]   = self.PHIk_strc.T.dot(response['Pk_cs'][i_step, :])
            # response['Pg_idrag'][i_step, :]= self.PHIk_strc.T.dot(response['Pk_idrag'][i_step, :])
            if self.jcl.aero['method'] in ['cfd_steady', 'cfd_unsteady']:
                response['Pg_cfd'][i_step, :] = self.PHIcfd_strc.T.dot(response['Pcfd'][i_step, :])
            if hasattr(self.jcl, 'landinggear') or hasattr(self.jcl, 'engine'):
                response['Pg_ext'][i_step, self.extragrid['set_strcgrid']] = response['Pextra'][i_step, self.extragrid['set']]
            response['Pg'][i_step, :] = response['Pg_aero'][i_step, :] + response['Pg_iner'][i_step, :] \
                + response['Pg_ext'][i_step, :] + response['Pg_cfd'][i_step, :]
            response['d2Ug_dt2'][i_step, :] = d2Ug_dt2_r + d2Ug_dt2_f

    def modal_displacement_method(self):
        logging.info(
            'calculating forces & moments on structural set (modal displacement method)...')
        logging.warning(
            'using the modal displacement method is not recommended, use force summation method instead.')
        response = self.response
        Kgg = load_hdf5_sparse_matrix(self.model['KGG'])

        response['Pg'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        for i_step in range(len(response['t'])):
            Uf = response['X'][i_step, :][12:12 + self.n_modes]
            Ug_f_body = self.PHIf_strc.T.dot(Uf)
            # MDM: p = K*u
            response['Pg'][i_step, :] = Kgg.dot(Ug_f_body)

    def euler_transformation(self):
        logging.info('apply euler angles...')
        response = self.response

        response['Pg_iner_global'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        response['Pg_aero_global'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        # response['Pg_gust_global'] = np.zeros((len(response['t']), 6*self.strcgrid['n']))
        # response['Pg_unsteady_global'] = np.zeros((len(response['t']), 6*self.strcgrid['n']))
        # response['Pg_cs_global']   = np.zeros((len(response['t']), 6*self.strcgrid['n']))
        # response['Pg_idrag_global']= np.zeros((len(response['t']), 6*self.strcgrid['n']))
        response['Pg_ext_global'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        response['Pg_cfd_global'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        response['Ug_r'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        response['Ug_f'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))
        response['Ug'] = np.zeros((len(response['t']), 6 * self.strcgrid['n']))

        for i_step in range(len(response['t'])):
            # Including rotation about euler angles in calculation of Ug_r and Ug_f
            # This is mainly done for plotting and time animation.

            # setting up coordinate system
            coord_tmp = copy.deepcopy(self.coord)
            coord_tmp['ID'] = np.append(coord_tmp['ID'], [1000000, 1000001])
            coord_tmp['RID'] = np.append(coord_tmp['RID'], [0, 0])
            coord_tmp['dircos'] = np.append(
                coord_tmp['dircos'],
                [self.PHIcg_norm[0:3, 0:3].dot(calc_drehmatrix(response['X'][i_step, :][3],
                                                               response['X'][i_step, :][4],
                                                               response['X'][i_step, :][5])),
                 np.eye(3)], axis=0)
            coord_tmp['offset'] = np.append(coord_tmp['offset'], [response['X'][i_step, :][0:3],
                                                                  -self.cggrid['offset'][0]], axis=0)

            # apply transformation to strcgrid
            strcgrid_tmp = copy.deepcopy(self.strcgrid)
            strcgrid_tmp['CP'] = np.repeat(1000001, self.strcgrid['n'])
            grid_trafo(strcgrid_tmp, coord_tmp, 1000000)
            response['Ug_r'][i_step, self.strcgrid['set'][:, 0]] = strcgrid_tmp['offset'][:, 0] - self.strcgrid['offset'][:, 0]
            response['Ug_r'][i_step, self.strcgrid['set'][:, 1]] = strcgrid_tmp['offset'][:, 1] - self.strcgrid['offset'][:, 1]
            response['Ug_r'][i_step, self.strcgrid['set'][:, 2]] = strcgrid_tmp['offset'][:, 2] - self.strcgrid['offset'][:, 2]
            # apply transformation to flexible deformations vector
            Uf = response['X'][i_step, :][12:12 + self.n_modes]
            Ug_f_body = np.dot(self.PHIf_strc.T, Uf.T).T
            strcgrid_tmp = copy.deepcopy(self.strcgrid)
            response['Ug_f'][i_step, :] = vector_trafo(strcgrid_tmp, coord_tmp, Ug_f_body, dest_coord=1000000)
            response['Pg_aero_global'][i_step, :] = vector_trafo(strcgrid_tmp, coord_tmp, response['Pg_aero'][i_step, :],
                                                                 dest_coord=1000000)
            response['Pg_iner_global'][i_step, :] = vector_trafo(strcgrid_tmp, coord_tmp, response['Pg_iner'][i_step, :],
                                                                 dest_coord=1000000)
            response['Pg_ext_global'][i_step, :] = vector_trafo(strcgrid_tmp, coord_tmp, response['Pg_ext'][i_step, :],
                                                                dest_coord=1000000)
            response['Pg_cfd_global'][i_step, :] = vector_trafo(strcgrid_tmp, coord_tmp, response['Pg_cfd'][i_step, :],
                                                                dest_coord=1000000)
            response['Ug'][i_step, :] = response['Ug_r'][i_step, :] + response['Ug_f'][i_step, :]

    def cuttingforces(self):
        logging.info('calculating cutting forces & moments...')
        response = self.response
        response['Pmon_local'] = np.zeros((len(response['t']), 6 * self.mongrid['n']))
        for i_step in range(len(response['t'])):
            response['Pmon_local'][i_step, :] = self.PHIstrc_mon.T.dot(response['Pg'][i_step, :])
