# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.io
import copy
import logging
import time

from panelaero import VLM, DLM

from loadskernel.fem_interfaces import nastran_interface, nastran_f06_interface, cofe_interface, b2000_interface
from loadskernel import build_aero_functions
from loadskernel import spline_rules
from loadskernel import spline_functions
from loadskernel import build_splinegrid
from loadskernel.io_functions import read_mona, read_op4, read_bdf
from loadskernel.io_functions import read_cfdgrids
from loadskernel import grid_trafo
from loadskernel.atmosphere import isa as atmo_isa
from loadskernel.engine_interfaces import propeller
from loadskernel.fem_interfaces import fem_helper


class Model():

    def __init__(self, jcl, path_output):
        # store the inputs
        self.jcl = jcl
        self.path_output = path_output
        # init the bdf reader
        self.bdf_reader = read_bdf.Reader()

    def build_model(self):
        # run all build function/stages
        self.build_coord()
        self.build_strc()
        self.build_strcshell()
        self.build_mongrid()
        self.build_extragrid()
        self.build_sensorgrid()
        self.build_atmo()
        self.build_aero()
        self.build_prop()
        self.build_splines()
        self.build_cfdgrid()
        self.build_structural_dynamics()
        # destroy the bdf reader and other stuff before saving the model
        delattr(self, 'bdf_reader')
        delattr(self, 'path_output')
        delattr(self, 'jcl')

    def build_coord(self):
        self.coord = {'ID': [0, 9300],
                      'RID': [0, 0],
                      'dircos': [np.eye(3), np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])],
                      'offset': [np.array([0, 0, 0]), np.array([0, 0, 0])],
                      }

    def build_strc(self):
        logging.info('Building structural model...')
        if self.jcl.geom['method'] == 'mona':
            # parse given bdf files
            self.bdf_reader.process_deck(self.jcl.geom['filename_grid'])
            # assemble strcgrid, sort grids to be in accordance with matricies such as Mgg from Nastran
            self.strcgrid = read_mona.add_GRIDS(self.bdf_reader.cards['GRID'].sort_values('ID'))
            # build additional coordinate systems
            read_mona.add_CORD2R(self.bdf_reader.cards['CORD2R'], self.coord)
            read_mona.add_CORD1R(self.bdf_reader.cards['CORD1R'], self.coord, self.strcgrid)

        elif self.jcl.geom['method'] == 'CoFE':
            with open(self.jcl.geom['filename_CoFE']) as fid:
                CoFE_data = scipy.io.loadmat(fid)
            n = CoFE_data['gnum'].squeeze().__len__()
            self.strcgrid = {'ID': CoFE_data['gnum'].squeeze(),
                             'CD': np.zeros(n),  # is this correct?
                             'CP': np.zeros(n),  # is this correct?
                             'n': n,
                             # convert indexing from Matlab to Python
                             'set': CoFE_data['gnum2gdof'].T - 1,
                             'offset': CoFE_data['gcoord'].T,
                             }

        # make sure the strcgrid is in one common coordinate system with ID = 0 (basic system)
        grid_trafo.grid_trafo(self.strcgrid, self.coord, 0)
        # provide a matrix PHIgg for the force and deformation transformation from the grid point coordinate system
        # specified by CD into the basic system with ID = 0
        _, self.PHIgg = grid_trafo.calc_transformation_matrix(self.coord,
                                                              self.strcgrid, '', 'CP',
                                                              self.strcgrid, '', 'CD',)
        # then set the coorinate sytems CD = 0
        self.strcgrid['CD'] = np.zeros(self.strcgrid['n'])

        logging.info('The structural model consists of {} grid points ({} DoFs) and {} coordinate systems.'.format(
            self.strcgrid['n'], self.strcgrid['n'] * 6, len(self.coord['ID'])))

    def build_strcshell(self):
        if 'filename_shell' in self.jcl.geom and not self.jcl.geom['filename_shell'] == []:
            # parse given bdf files
            self.bdf_reader.process_deck(self.jcl.geom['filename_shell'])
            # assemble strcshell from CTRIA3 and CQUAD4 elements
            self.strcshell = read_mona.add_shell_elements(pd.concat([self.bdf_reader.cards['CQUAD4'],
                                                                     self.bdf_reader.cards['CTRIA3']], ignore_index=True))

    def build_mongrid(self):
        if self.jcl.geom['method'] in ['mona', 'CoFE']:
            if 'filename_mongrid' in self.jcl.geom and not self.jcl.geom['filename_mongrid'] == '':
                logging.info('Building Monitoring Stations from GRID data...')
                self.mongrid = read_mona.Modgen_GRID(self.jcl.geom['filename_mongrid'])
                # we dont't get names for the monstations from simple grid points, so we make up a name
                self.mongrid['name'] = ['MON{:s}'.format(str(ID)) for ID in self.mongrid['ID']]
                # build additional coordinate systems
                self.bdf_reader.process_deck(self.jcl.geom['filename_moncoord'])
                read_mona.add_CORD2R(self.bdf_reader.cards['CORD2R'], self.coord)
                read_mona.add_CORD1R(self.bdf_reader.cards['CORD1R'], self.coord, self.strcgrid)
                rules = spline_rules.monstations_from_bdf(self.mongrid, self.jcl.geom['filename_monstations'])
                self.build_mongrid_matrices(rules)
            elif 'filename_monpnt' in self.jcl.geom and not self.jcl.geom['filename_monpnt'] == '':
                logging.info('Reading Monitoring Stations from MONPNTs...')
                # parse given bdf files
                self.bdf_reader.process_deck(self.jcl.geom['filename_monpnt'])
                # assemble mongrid
                self.mongrid = read_mona.add_MONPNT1(self.bdf_reader.cards['MONPNT1'])
                # build additional coordinate systems
                read_mona.add_CORD2R(self.bdf_reader.cards['CORD2R'], self.coord)
                read_mona.add_CORD1R(self.bdf_reader.cards['CORD1R'], self.coord, self.strcgrid)
                # get aecomp and sets
                aecomp = read_mona.add_AECOMP(self.bdf_reader.cards['AECOMP'])
                sets = read_mona.add_SET1(self.bdf_reader.cards['SET1'])

                rules = spline_rules.monstations_from_aecomp(self.mongrid, aecomp, sets)
                self.build_mongrid_matrices(rules)
            else:
                logging.warning('No Monitoring Stations are created!')
                """
                This is an empty dummy monitoring stations, which is necessary when no monitoring stations are defined,
                because monstations are expected to exist for example for the calculation of cutting forces, which are in
                turn expected in the post processing.  However, this procedure allows the code to run without any given
                monitoring stations, which are not available for all models.
                """
                self.mongrid = {'ID': np.array([0]),
                                'name': ['dummy'],
                                'label': ['dummy'],
                                'CP': np.array([0]),
                                'CD': np.array([0]),
                                'offset': np.array([0.0, 0.0, 0.0]),
                                'set': np.arange(6).reshape((1, 6)),
                                'n': 1,
                                }
                self.PHIstrc_mon = np.zeros((self.strcgrid['n'] * 6, 6))

    def build_mongrid_matrices(self, rules):
        PHIstrc_mon = spline_functions.spline_rb(self.mongrid, '', self.strcgrid, '', rules, self.coord, sparse_output=True)
        # The line above gives the loads coordinate system 'CP' (mostly in global coordinates).
        # Next, make sure that PHIstrc_mon maps the loads in the local coordinate system given in 'CD'.
        # Previously, step this has been accomplished in the post-processing. However, using matrix operations
        # and incorporating the coordinate transformation in the pre-processing is much more convenient.
        T_i, T_d = grid_trafo.calc_transformation_matrix(self.coord,
                                                         self.mongrid, '', 'CP',
                                                         self.mongrid, '', 'CD',)
        self.PHIstrc_mon = T_d.T.dot(T_i).dot(PHIstrc_mon.T).T
        self.mongrid_rules = rules  # save rules for optional writing of MONPNT1 cards

    def build_extragrid(self):
        if hasattr(self.jcl, 'landinggear'):
            logging.info('Building extragrid from landing gear attachment points...')
            self.extragrid = build_splinegrid.build_subgrid(self.strcgrid, self.jcl.landinggear['attachment_point'])
        elif hasattr(self.jcl, 'engine'):
            logging.info('Building extragrid from engine attachment points...')
            self.extragrid = build_splinegrid.build_subgrid(self.strcgrid, self.jcl.engine['attachment_point'])
        else:
            return
        self.extragrid['set_strcgrid'] = copy.deepcopy(self.extragrid['set'])
        self.extragrid['set'] = np.arange(0, 6 * self.extragrid['n']).reshape(-1, 6)

    def build_sensorgrid(self):
        if hasattr(self.jcl, 'sensor'):
            logging.info('Building sensorgrid from sensor attachment points...')
            self.sensorgrid = build_splinegrid.build_subgrid(self.strcgrid, self.jcl.sensor['attachment_point'])
        else:
            return
        self.sensorgrid['set_strcgrid'] = copy.deepcopy(self.sensorgrid['set'])
        self.sensorgrid['set'] = np.arange(0, 6 * self.sensorgrid['n']).reshape(-1, 6)

    def build_atmo(self):
        logging.info('Building atmo model...')
        self.atmo = {}
        if self.jcl.atmo['method'] == 'ISA':
            for i_atmo in range(len(self.jcl.atmo['key'])):
                p, rho, T, a = atmo_isa(self.jcl.atmo['h'][i_atmo])
                self.atmo[self.jcl.atmo['key'][i_atmo]] = {
                    'h': self.jcl.atmo['h'][i_atmo],
                    'p': p,
                    'rho': rho,
                    'T': T,
                    'a': a,
                }
        else:
            logging.error('Unknown atmo method: ' + str(self.jcl.aero['method']))

    def build_aero(self):
        logging.info('Building aero model...')
        if self.jcl.aero['method'] in ['mona_steady', 'mona_unsteady', 'hybrid', 'nonlin_steady',
                                       'cfd_steady', 'cfd_unsteady', 'freq_dom']:
            self.build_aerogrid()
            self.build_aero_matrices()
            self.build_W2GJ()
            self.build_macgrid()
            self.build_cs()
            self.build_AICs_steady()
            self.build_AICs_unsteady()
        else:
            logging.error('Unknown aero method: ' + str(self.jcl.aero['method']))

        logging.info('The aerodynamic model consists of {} panels and {} control surfaces.'.format(
            self.aerogrid['n'], len(self.x2grid['ID_surf'])))

    def build_aerogrid(self):
        # To avoid interference with other CQUAD4 cards parsed earlier, clear those dataframes first
        if 'method_caero' in self.jcl.aero and self.jcl.aero['method_caero'] == 'CQUAD4':
            self.bdf_reader.cards['CQUAD4'].drop(self.bdf_reader.cards['CQUAD4'].index, inplace=True)
            self.bdf_reader.cards['GRID'].drop(self.bdf_reader.cards['GRID'].index, inplace=True)
        # Then parse given bdf files
        self.bdf_reader.process_deck(self.jcl.aero['filename_caero_bdf'])
        if 'method_caero' in self.jcl.aero:
            self.aerogrid = build_aero_functions.build_aerogrid(self.bdf_reader, method_caero=self.jcl.aero['method_caero'])
        else:  # use default method defined in function
            self.aerogrid = build_aero_functions.build_aerogrid(self.bdf_reader)

    def build_aero_matrices(self):
        # cast normal vector of panels into a matrix of form (n, n*6)
        self.aerogrid['Nmat'] = sp.lil_matrix((self.aerogrid['n'], self.aerogrid['n'] * 6), dtype=float)
        for x in range(self.aerogrid['n']):
            self.aerogrid['Nmat'][x, self.aerogrid['set_k'][x, (0, 1, 2)]] = self.aerogrid['N'][x, (0, 1, 2)]
        self.aerogrid['Nmat'] = self.aerogrid['Nmat'].tocsc()
        # cast downwash due to rotations of panels into a matrix notation
        self.aerogrid['Rmat'] = sp.lil_matrix((self.aerogrid['n'] * 6, self.aerogrid['n'] * 6), dtype=float)
        for x in range(self.aerogrid['n']):
            self.aerogrid['Rmat'][x * 6 + 1, self.aerogrid['set_k'][x, 5]] = -1.0  # rotation about z-axis yields y-downwash
            self.aerogrid['Rmat'][x * 6 + 2, self.aerogrid['set_k'][x, 4]] = 1.0  # rotation about y-axis yields z-downwash
        self.aerogrid['Rmat'] = self.aerogrid['Rmat'].tocsc()
        # cast areas of panels into matrix notation
        self.aerogrid['Amat'] = sp.diags(self.aerogrid['A'], format='csc')

    def build_W2GJ(self):
        # Correctionfor camber and twist, W2GJ
        if 'filename_DMI_W2GJ' in self.jcl.aero and self.jcl.aero['filename_DMI_W2GJ']:
            for i_file in range(len(self.jcl.aero['filename_DMI_W2GJ'])):
                DMI = read_mona.Nastran_DMI(self.jcl.aero['filename_DMI_W2GJ'][i_file])
                if i_file == 0:
                    data = DMI['data'].toarray().squeeze()
                else:
                    data = np.hstack((data, DMI['data'].toarray().squeeze()))
            self.camber_twist = {'ID': self.aerogrid['ID'], 'cam_rad': data}
        else:
            logging.info('No W2GJ data (correction of camber and twist) given, setting to zero')
            self.camber_twist = {'ID': self.aerogrid['ID'], 'cam_rad': np.zeros(self.aerogrid['ID'].shape)}

    def build_macgrid(self):
        # build mac grid from geometry, except other values are given in general section of jcl
        self.macgrid = build_aero_functions.build_macgrid(self.aerogrid, self.jcl.general['b_ref'])
        if 'A_ref' in self.jcl.general:
            self.macgrid['A_ref'] = self.jcl.general['A_ref']
        if 'c_ref' in self.jcl.general:
            self.macgrid['c_ref'] = self.jcl.general['c_ref']
        if 'b_ref' in self.jcl.general:
            self.macgrid['b_ref'] = self.jcl.general['b_ref']
        if 'MAC_ref' in self.jcl.general:
            self.macgrid['offset'] = np.array([self.jcl.general['MAC_ref']])

        rules = spline_rules.rules_aeropanel(self.aerogrid)
        self.PHIjk = spline_functions.spline_rb(self.aerogrid, '_k', self.aerogrid, '_j',
                                                rules, self.coord, sparse_output=True)
        self.PHIlk = spline_functions.spline_rb(self.aerogrid, '_k', self.aerogrid, '_l',
                                                rules, self.coord, sparse_output=True)

        rules = spline_rules.rules_point(self.macgrid, self.aerogrid)
        self.Dkx1 = spline_functions.spline_rb(self.macgrid, '', self.aerogrid, '_k',
                                               rules, self.coord, sparse_output=False)
        self.Djx1 = self.PHIjk.dot(self.Dkx1)

    def build_cs(self):
        # control surfaces
        self.bdf_reader.process_deck(self.jcl.aero['filename_aesurf'] + self.jcl.aero['filename_aelist'])
        self.x2grid, self.coord = build_aero_functions.build_x2grid(self.bdf_reader, self.aerogrid, self.coord)
        self.Djx2 = []
        for i_surf in range(len(self.x2grid['ID_surf'])):

            hingegrid = {'ID': np.array([self.x2grid['ID_surf'][i_surf]]),
                         'offset': np.array([[0, 0, 0]]),
                         'CD': np.array([self.x2grid['CID'][i_surf]]),
                         'CP': np.array([self.x2grid['CID'][i_surf]]),
                         'set': np.array([[0, 1, 2, 3, 4, 5]]),
                         }
            surfgrid = {'ID': self.x2grid[i_surf]['ID'],
                        'offset_j': np.array(self.x2grid[i_surf]['offset_j']),
                        'CD': self.x2grid[i_surf]['CD'],
                        'CP': self.x2grid[i_surf]['CP'],
                        'set_j': np.array(self.x2grid[i_surf]['set_j']),
                        }
            dimensions = [6, 6 * len(self.aerogrid['ID'])]
            rules = spline_rules.rules_point(hingegrid, surfgrid)
            self.Djx2.append(spline_functions.spline_rb(hingegrid, '', surfgrid, '_j', rules, self.coord, dimensions))

    def build_AICs_steady(self):
        # set-up empty data structure
        self.aero = {}
        for key in self.jcl.aero['key']:
            self.aero[key] = {}
        if self.jcl.aero['method_AIC'] == 'nastran':
            for i_aero in range(len(self.jcl.aero['key'])):
                Ajj = read_op4.load_matrix(self.jcl.aero['filename_AIC'][i_aero], sparse_output=False, sparse_format=False)
                if 'given_AIC_is_transposed' in self.jcl.aero and self.jcl.aero['given_AIC_is_transposed']:
                    Qjj = np.linalg.inv(np.real(Ajj))
                else:
                    Qjj = np.linalg.inv(np.real(Ajj).T)
                self.aero[self.jcl.aero['key'][i_aero]]['Qjj'] = Qjj
        elif self.jcl.aero['method_AIC'] in ['vlm', 'dlm', 'ae']:
            logging.info('Calculating steady AIC matrices ({} panels, k=0.0) for {} Mach number(s)...'.format(
                self.aerogrid['n'], len(self.jcl.aero['key'])))
            t_start = time.time()
            if 'xz_symmetry' in self.jcl.aero:
                logging.info(
                    ' - XZ Symmetry: {}'.format(str(self.jcl.aero['xz_symmetry'])))
                Qjj, Bjj = VLM.calc_Qjjs(aerogrid=copy.deepcopy(self.aerogrid), Ma=self.jcl.aero['Ma'],
                                         xz_symmetry=self.jcl.aero['xz_symmetry'])  # dim: Ma,n,n
            else:
                Qjj, Bjj = VLM.calc_Qjjs(aerogrid=copy.deepcopy(
                    self.aerogrid), Ma=self.jcl.aero['Ma'])  # dim: Ma,n,n
                Gamma_jj, Q_ind_jj = VLM.calc_Gammas(aerogrid=copy.deepcopy(self.aerogrid),
                                                     Ma=self.jcl.aero['Ma'])  # dim: Ma,n,n
                for key, Gamma_jj, Q_ind_jj in zip(self.jcl.aero['key'], Gamma_jj, Q_ind_jj):
                    self.aero[key]['Gamma_jj'] = Gamma_jj
                    self.aero[key]['Q_ind_jj'] = Q_ind_jj

            for key, Qjj, Bjj in zip(self.jcl.aero['key'], Qjj, Bjj):
                self.aero[key]['Qjj'] = Qjj
                self.aero[key]['Bjj'] = Bjj

            logging.info('done in %.2f [sec].' % (time.time() - t_start))
        else:
            logging.error('Unknown AIC method: ' + str(self.jcl.aero['method_AIC']))

    def build_AICs_unsteady(self):
        if self.jcl.aero['method'] in ['mona_unsteady']:
            if self.jcl.aero['method_AIC'] == 'dlm':
                self.build_AICs_DLM()
                self.build_rfa()
            elif self.jcl.aero['method_AIC'] == 'nastran':
                self.build_AICs_Nastran()
                self.build_rfa()
            else:
                logging.error('Unknown AIC method: ' + str(self.jcl.aero['method_AIC']))
        elif self.jcl.aero['method'] in ['freq_dom']:
            if self.jcl.aero['method_AIC'] == 'dlm':
                self.build_AICs_DLM()
            elif self.jcl.aero['method_AIC'] == 'nastran':
                self.build_AICs_Nastran()
            self.aero['n_poles'] = 0

    def build_AICs_DLM(self):
        logging.info('Calculating unsteady AIC matrices ({} panels, k={} (Nastran Definition!)) \
            for {} Mach number(s)...'.format(self.aerogrid['n'], self.jcl.aero['k_red'], len(self.jcl.aero['key'])))
        if 'xz_symmetry' in self.jcl.aero:
            logging.info(
                ' - XZ Symmetry: {}'.format(str(self.jcl.aero['xz_symmetry'])))
            xz_symmetry = self.jcl.aero['xz_symmetry']
        else:
            xz_symmetry = False
        # Definitions for reduced frequencies:
        # ae_getaic: k = omega/U
        # Nastran:   k = 0.5*cref*omega/U
        t_start = time.time()
        Qjj = DLM.calc_Qjjs(aerogrid=copy.deepcopy(self.aerogrid),
                            Ma=self.jcl.aero['Ma'],
                            k=np.array(self.jcl.aero['k_red']) / (0.5 * self.jcl.general['c_ref']),
                            xz_symmetry=xz_symmetry)
        for key, Qjj in zip(self.jcl.aero['key'], Qjj):
            self.aero[key]['Qjj_unsteady'] = Qjj
            self.aero[key]['k_red'] = self.jcl.aero['k_red']
        logging.info('done in %.2f [sec].' % (time.time() - t_start))

    def build_AICs_Nastran(self):
        for i_aero in range(len(self.jcl.aero['key'])):
            self.aero[self.jcl.aero['key'][i_aero]]['Qjj_unsteady'] = np.zeros((len(self.jcl.aero['k_red']),
                                                                                self.aerogrid['n'],
                                                                                self.aerogrid['n']), dtype=complex)
            for i_k in range(len(self.jcl.aero['k_red'])):
                Ajj = read_op4.load_matrix(self.jcl.aero['filename_AIC_unsteady'][i_aero][i_k],
                                           sparse_output=False, sparse_format=False)
                if 'given_AIC_is_transposed' in self.jcl.aero and self.jcl.aero['given_AIC_is_transposed']:
                    Qjj = np.linalg.inv(Ajj.T)
                else:
                    Qjj = np.linalg.inv(Ajj)
                self.aero[self.jcl.aero['key'][i_aero]]['Qjj_unsteady'][i_k, :, :] = Qjj
            self.aero[self.jcl.aero['key'][i_aero]]['k_red'] = self.jcl.aero['k_red']

    def build_rfa(self):
        for key in self.jcl.aero['key']:
            ABCD, n_poles, betas, RMSE = build_aero_functions.rfa(Qjj=self.aero[key]['Qjj_unsteady'],
                                                                  k=self.jcl.aero['k_red'],
                                                                  n_poles=self.jcl.aero['n_poles'],
                                                                  filename=self.path_output + 'rfa_{}.png'.format(key))
            self.aero[key]['ABCD'] = ABCD
            self.aero[key]['RMSE'] = RMSE
            self.aero[key]['n_poles'] = n_poles
            self.aero[key]['betas'] = betas
            # Remove unsteady AICs to save memory. No longer critical with the HDF5 data format.
            del self.aero[key]['Qjj_unsteady']

    def build_prop(self):
        if hasattr(self.jcl, 'engine'):
            if self.jcl.engine['method'] == 'VLM4Prop':
                logging.info('Building VLM4Prop model...')
                self.prop = propeller.VLM4PropModel(self.jcl.engine['propeller_input_file'], self.coord, self.atmo)
                self.prop.build_aerogrid()
                self.prop.build_pacgrid()
                self.prop.build_AICs_steady(self.jcl.engine['Ma'])
            else:
                logging.debug('Engine method {} has / needs no propulsion properties.'.format(str(self.jcl.engine['method'])))

    def build_splines(self):
        # ----------------
        # ---- splines ---
        # ----------------
        # PHIk_strc with 'nearest_neighbour', 'rbf' or 'nastran'
        if self.jcl.spline['method'] in ['rbf', 'nearest_neighbour']:
            if 'splinegrid' in self.jcl.spline and self.jcl.spline['splinegrid'] is True:
                # this optin is only valid if spline['method'] == 'rbf' or 'rb'
                logging.info('Coupling aerogrid to strcgrid via splinegrid:')
                self.splinegrid = build_splinegrid.build_splinegrid(self.strcgrid, self.jcl.spline['filename_splinegrid'])
                # self.splinegrid = build_splinegrid.grid_thin_out_radius(self.splinegrid, 0.01)
            else:
                logging.info('Coupling aerogrid directly. Doing cleanup/thin out of strcgrid to avoid singularities \
                    (safety first!)')
                self.splinegrid = build_splinegrid.grid_thin_out_radius(self.strcgrid, 0.01)
        else:
            self.splinegrid = self.strcgrid
        logging.info('The spline model consists of {} grid points.'.format(
            self.splinegrid['n']))

        if self.jcl.spline['method'] == 'rbf':
            self.PHIk_strc = spline_functions.spline_rbf(self.splinegrid, '', self.aerogrid, '_k', 'tps',
                                                         dimensions=[len(self.strcgrid['ID']) * 6,
                                                                     len(self.aerogrid['ID']) * 6])
            # rbf-spline not (yet) stable for translation of forces and moments to structure grid,
            # so use rb-spline with nearest neighbour search instead
        elif self.jcl.spline['method'] == 'nearest_neighbour':
            self.coupling_rules = spline_rules.nearest_neighbour(self.splinegrid, '', self.aerogrid, '_k')
            self.PHIk_strc = spline_functions.spline_rb(self.splinegrid, '', self.aerogrid, '_k',
                                                        self.coupling_rules, self.coord,
                                                        dimensions=[len(self.strcgrid['ID']) * 6,
                                                                    len(self.aerogrid['ID']) * 6],
                                                        sparse_output=True)
        elif self.jcl.spline['method'] == 'nastran':
            self.PHIk_strc = spline_functions.spline_nastran(
                self.jcl.spline['filename_f06'], self.strcgrid, self.aerogrid)
        else:
            logging.error('Unknown spline method.')

    def build_cfdgrid(self):
        # -------------------
        # ---- mesh defo ---
        # -------------------
        if self.jcl.aero['method'] in ['cfd_steady', 'cfd_unsteady']:
            cfdgrids = read_cfdgrids.ReadCfdgrids(self.jcl)
            cfdgrids.read_surface(merge_domains=True)
            cfdgrids.read_surface(merge_domains=False)
            self.cfdgrid = cfdgrids.cfdgrid
            self.cfdgrids = cfdgrids.cfdgrids
            logging.info('The CFD surface grid consists of {} grid points and {} boundary markers.'.format(
                self.cfdgrid['n'], self.cfdgrids.__len__()))

            # Option A: CFD forces are transferred to the aerogrid.
            # This allows a direct integration into existing procedures and a comparison to VLM forces.
            rules = spline_rules.nearest_neighbour(self.aerogrid, '_k', self.cfdgrid, '')
            self.PHIk_cfd = spline_functions.spline_rb(self.aerogrid, '_k', self.cfdgrid, '',
                                                       rules, self.coord,
                                                       dimensions=[self.aerogrid['n'] * 6,
                                                                   self.cfdgrid['n'] * 6],
                                                       sparse_output=True)
            # Option B: CFD forces are directly transferred to the strcgrid.
            # This is more physical and for example allows the application of forces on upper and lower side.
            # The splinegrid from above is re-used.
            rules = spline_rules.nearest_neighbour(self.splinegrid, '', self.cfdgrid, '')
            self.PHIcfd_strc = spline_functions.spline_rb(self.splinegrid, '', self.cfdgrid, '',
                                                          rules, self.coord,
                                                          dimensions=[self.strcgrid['n'] * 6,
                                                                      self.cfdgrid['n'] * 6],
                                                          sparse_output=True)

    def build_structural_dynamics(self):
        logging.info('Building stiffness and mass model...')
        self.mass = {}
        if self.jcl.mass['method'] in ['mona', 'f06', 'modalanalysis', 'guyan', 'CoFE', 'B2000']:

            # select the fem interface
            if self.jcl.mass['method'] in ['modalanalysis', 'guyan']:
                fem_interface = nastran_interface.NastranInterface(self.jcl, self.strcgrid, self.coord)
            elif self.jcl.mass['method'] in ['mona', 'f06']:
                fem_interface = nastran_f06_interface.Nastranf06Interface(self.jcl, self.strcgrid, self.coord)
            elif self.jcl.mass['method'] in ['B2000']:
                fem_interface = b2000_interface.B2000Interface(self.jcl, self.strcgrid, self.coord)
            elif self.jcl.mass['method'] in ['CoFE']:
                fem_interface = cofe_interface.CoFEInterface(self.jcl, self.strcgrid, self.coord)

            # the stiffness matrix is needed for all methods / fem interfaces
            fem_interface.get_stiffness_matrix()
            # the stiffness matrix is needed for the modal discplacement method
            self.KGG = fem_interface.KGG
            # Check if matrix is symmetric
            if not fem_helper.check_matrix_symmetry(self.KGG):
                logging.warning('Stiffness matrix Kgg is NOT symmetric.')

            # do further processing of the stiffness matrix
            if self.jcl.mass['method'] in ['modalanalysis', 'guyan', 'CoFE', 'B2000']:
                fem_interface.get_dofs()
                fem_interface.prepare_stiffness_matrices()
            if self.jcl.mass['method'] in ['guyan']:
                fem_interface.get_aset(self.bdf_reader)
                fem_interface.prepare_stiffness_matrices_for_guyan()

            # loop over mass configurations
            for i_mass in range(len(self.jcl.mass['key'])):
                self.build_mass_matrices(fem_interface, i_mass)
                self.build_translation_matrices(self.jcl.mass['key'][i_mass])
        else:
            logging.error('Unknown mass method: ' + str(self.jcl.mass['method']))

    def build_mass_matrices(self, fem_interface, i_mass):
        logging.info('Mass configuration {} of {}: {} '.format(i_mass + 1, len(self.jcl.mass['key']),
                                                               self.jcl.mass['key'][i_mass]))

        # the mass matrix is needed for all methods / fem interfaces
        MGG = fem_interface.get_mass_matrix(i_mass)
        # Check if matrix is symmetric
        if not fem_helper.check_matrix_symmetry(MGG):
            logging.warning('Mass matrix Mgg is NOT symmetric.')

        # getting the eigenvalues and -vectors depends on the method / fem solver
        if self.jcl.mass['method'] in ['modalanalysis', 'guyan', 'CoFE', 'B2000']:
            Mb, cggrid, cggrid_norm = fem_interface.calc_cg()
            fem_interface.prepare_mass_matrices()
            if self.jcl.mass['method'] in ['modalanalysis', 'CoFE', 'B2000']:
                fem_interface.modalanalysis()
            elif self.jcl.mass['method'] in ['guyan']:
                fem_interface.guyanreduction()
        elif self.jcl.mass['method'] in ['mona', 'f06']:
            Mb, cggrid, cggrid_norm = fem_interface.cg_from_SOL103()
            fem_interface.modes_from_SOL103()

        # calculate all generalized matrices
        Mff, Kff, Dff, PHIf_strc, Mhh, Khh, Dhh, PHIh_strc = fem_interface.calc_modal_matrices()
        # apply coodinate transformation so that all deformations are given in the basic coordinate system
        PHIf_strc = self.PHIgg.dot(PHIf_strc.T).T
        PHIh_strc = self.PHIgg.dot(PHIh_strc.T).T
        # store everything
        self.mass[self.jcl.mass['key'][i_mass]] = {
            'Mb': Mb,
            'MGG': MGG,
            'cggrid': cggrid,
            'cggrid_norm': cggrid_norm,
            'PHIf_strc': PHIf_strc,
            'Mff': Mff,
            'Kff': Kff,
            'Dff': Dff,
            'PHIh_strc': PHIh_strc,
            'Mhh': Mhh,
            'Khh': Khh,
            'Dhh': Dhh,
            'n_modes': len(self.jcl.mass['modes'][i_mass]),
        }

    def build_translation_matrices(self, key):
        """
        In this function, we do a lot of splining. Depending on the intended solution sequence,
        different matrices are required.
        """

        cggrid = self.mass[key]['cggrid']
        cggrid_norm = self.mass[key]['cggrid_norm']
        MGG = self.mass[key]['MGG']
        PHIf_strc = self.mass[key]['PHIf_strc']
        PHIh_strc = self.mass[key]['PHIh_strc']

        rules = spline_rules.rules_point(cggrid, self.strcgrid)
        PHIstrc_cg = spline_functions.spline_rb(cggrid, '', self.strcgrid, '', rules, self.coord)

        if self.jcl.aero['method'] in ['cfd_steady', 'cfd_unsteady']:
            rules = spline_rules.rules_point(cggrid, self.cfdgrid)
            PHIcfd_cg = spline_functions.spline_rb(cggrid, '', self.cfdgrid, '', rules, self.coord)
            # some pre-multiplications to speed-up main processing
            PHIcfd_f = self.PHIcfd_strc.dot(PHIf_strc.T)
            self.mass[key]['PHIcfd_cg'] = PHIcfd_cg
            self.mass[key]['PHIcfd_f'] = PHIcfd_f

        if hasattr(self, 'extragrid'):
            rules = spline_rules.rules_point(cggrid, self.extragrid)
            PHIextra_cg = spline_functions.spline_rb(cggrid, '', self.extragrid, '', rules, self.coord)
            PHIf_extra = PHIf_strc[:, self.extragrid['set_strcgrid'].reshape(1, -1)[0]]
            self.mass[key]['PHIextra_cg'] = PHIextra_cg
            self.mass[key]['PHIf_extra'] = PHIf_extra

        if hasattr(self, 'sensorgrid'):
            rules = spline_rules.rules_point(cggrid, self.sensorgrid)
            PHIsensor_cg = spline_functions.spline_rb(cggrid, '', self.sensorgrid, '', rules, self.coord)
            PHIf_sensor = PHIf_strc[:, self.sensorgrid['set_strcgrid'].reshape(1, -1)[0]]
            self.mass[key]['PHIsensor_cg'] = PHIsensor_cg
            self.mass[key]['PHIf_sensor'] = PHIf_sensor

        rules = spline_rules.rules_point(cggrid, self.macgrid)
        PHImac_cg = spline_functions.spline_rb(cggrid, '', self.macgrid, '', rules, self.coord)
        rules = spline_rules.rules_point(self.macgrid, cggrid)
        PHIcg_mac = spline_functions.spline_rb(self.macgrid, '', cggrid, '', rules, self.coord)

        rules = spline_rules.rules_point(cggrid, cggrid_norm)
        PHInorm_cg = spline_functions.spline_rb(cggrid, '', cggrid_norm, '', rules, self.coord)
        rules = spline_rules.rules_point(cggrid_norm, cggrid)
        PHIcg_norm = spline_functions.spline_rb(cggrid_norm, '', cggrid, '', rules, self.coord)

        # some pre-multiplications to speed-up main processing
        PHIjf = self.PHIjk.dot(self.PHIk_strc.dot(PHIf_strc.T))
        PHIlf = self.PHIlk.dot(self.PHIk_strc.dot(PHIf_strc.T))
        PHIkf = self.PHIk_strc.dot(PHIf_strc.T)
        PHIjh = self.PHIjk.dot(self.PHIk_strc.dot(PHIh_strc.T))
        PHIlh = self.PHIlk.dot(self.PHIk_strc.dot(PHIh_strc.T))
        PHIkh = self.PHIk_strc.dot(PHIh_strc.T)

        Mfcg = PHIf_strc.dot(-MGG.dot(PHIstrc_cg))

        # save all matrices to data structure
        self.mass[key]['Mfcg'] = Mfcg
        self.mass[key]['PHImac_cg'] = PHImac_cg
        self.mass[key]['PHIcg_mac'] = PHIcg_mac
        self.mass[key]['PHIcg_norm'] = PHIcg_norm
        self.mass[key]['PHInorm_cg'] = PHInorm_cg
        self.mass[key]['PHIstrc_cg'] = PHIstrc_cg
        self.mass[key]['PHIjf'] = PHIjf
        self.mass[key]['PHIlf'] = PHIlf
        self.mass[key]['PHIkf'] = PHIkf
        self.mass[key]['PHIjh'] = PHIjh
        self.mass[key]['PHIlh'] = PHIlh
        self.mass[key]['PHIkh'] = PHIkh
