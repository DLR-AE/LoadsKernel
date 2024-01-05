# -*- coding: utf-8 -*-
import logging
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from loadskernel.io_functions import read_mona
import loadskernel.engine_interfaces.propeller


def build_x2grid(bdf_reader, aerogrid, coord):
    aesurf = read_mona.add_AESURF(bdf_reader.cards['AESURF'])
    aelist = read_mona.add_SET1(bdf_reader.cards['AELIST'])
    # build additional coordinate systems
    read_mona.add_CORD2R(bdf_reader.cards['CORD2R'], coord)

    x2grid = {'ID_surf': aesurf['ID'],
              'CID': aesurf['CID'],
              'key': aesurf['key'],
              'eff': aesurf['eff'],
              }
    for i_surf in range(len(aesurf['ID'])):
        x2grid[i_surf] = {'CD': [], 'CP': [], 'ID': [], 'offset_j': [], 'set_j': []}
        for i_panel in aelist['values'][aelist['ID'].index(aesurf['AELIST'][i_surf])]:
            pos_panel = np.where(aerogrid['ID'] == i_panel)[0][0]
            x2grid[i_surf]['ID'].append(aerogrid['ID'][pos_panel])
            x2grid[i_surf]['CD'].append(aerogrid['CD'][pos_panel])
            x2grid[i_surf]['CP'].append(aerogrid['CP'][pos_panel])
            x2grid[i_surf]['offset_j'].append(aerogrid['offset_j'][pos_panel])
            x2grid[i_surf]['set_j'].append(aerogrid['set_j'][pos_panel])

    return x2grid, coord


def build_aerogrid(bdf_reader, filename='', method_caero='CAERO1'):
    if method_caero == 'CQUAD4':
        # all corner points are defined as grid points by ModGen
        caero_grid = read_mona.add_GRIDS(bdf_reader.cards['GRID'].sort_values('ID'))
        # four grid points are assembled to one panel, this is expressed as CQUAD4s
        caero_panels = read_mona.add_shell_elements(bdf_reader.cards['CQUAD4'].sort_values('ID'))
    elif method_caero in ['CAERO1', 'CAERO7']:
        # Adjust the counting of CAERO7 (ZAERO) panels to CAERO1 (Nastran)
        bdf_reader.cards['CAERO7']['NSPAN'] -= 1
        bdf_reader.cards['CAERO7']['NCHORD'] -= 1
        # combine CAERO7 and CAERO1 and use the same function to assemble aerodynamic panels
        combined_caero = pd.concat([bdf_reader.cards['CAERO1'], bdf_reader.cards['CAERO7']], ignore_index=True)
        caero_grid, caero_panels = read_mona.add_panels_from_CAERO(combined_caero.sort_values('ID'),
                                                                   bdf_reader.cards['AEFACT'])
    elif method_caero in ['VLM4Prop']:
        caero_grid, caero_panels, cam_rad = loadskernel.engine_interfaces.propeller.read_propeller_input(filename)
    else:
        logging.error("Error: Method %s not implemented. Available options are 'CQUAD4', 'CAERO1' and 'CAERO7'", method_caero)
    logging.info(' - from corner points and aero panels, constructing aerogrid')
    ID = []
    l = []  # length of panel
    A = []  # area of one panel
    N = []  # unit normal vector
    offset_l = []  # 25% point l
    offset_k = []  # 50% point k
    offset_j = []  # 75% downwash control point j
    offset_P1 = []  # Vortex point at 25% chord, 0% span
    offset_P3 = []  # Vortex point at 25% chord, 100% span
    r = []  # vector P1 to P3, span of panel

    for i_panel in range(len(caero_panels['ID'])):

        #
        #                   l_2
        #             4 o---------o 3
        #               |         |
        #  u -->    b_1 | l  k  j | b_2
        #               |         |
        #             1 o---------o 2
        #         y         l_1
        #         |
        #        z.--- x

        index_1 = np.where(caero_panels['cornerpoints'][i_panel][0] == caero_grid['ID'])[0][0]
        index_2 = np.where(caero_panels['cornerpoints'][i_panel][1] == caero_grid['ID'])[0][0]
        index_3 = np.where(caero_panels['cornerpoints'][i_panel][2] == caero_grid['ID'])[0][0]
        index_4 = np.where(caero_panels['cornerpoints'][i_panel][3] == caero_grid['ID'])[0][0]

        l_1 = caero_grid['offset'][index_2] - caero_grid['offset'][index_1]
        l_2 = caero_grid['offset'][index_3] - caero_grid['offset'][index_4]
        b_1 = caero_grid['offset'][index_4] - caero_grid['offset'][index_1]
        b_2 = caero_grid['offset'][index_3] - caero_grid['offset'][index_2]
        l_m = (l_1 + l_2) / 2.0
        b_m = (b_1 + b_2) / 2.0

        ID.append(caero_panels['ID'][i_panel])
        l.append(l_m[0])
        # A.append(l_m[0]*b_m[1])
        A.append(np.linalg.norm(np.cross(l_m, b_m)))
        N.append(np.cross(l_1, b_1) / np.linalg.norm(np.cross(l_1, b_1)))
        offset_l.append(caero_grid['offset'][index_1] + 0.25 * l_m + 0.50 * b_1)
        offset_k.append(caero_grid['offset'][index_1] + 0.50 * l_m + 0.50 * b_1)
        offset_j.append(caero_grid['offset'][index_1] + 0.75 * l_m + 0.50 * b_1)
        offset_P1.append(caero_grid['offset'][index_1] + 0.25 * l_1)
        offset_P3.append(caero_grid['offset'][index_4] + 0.25 * l_2)
        r.append((caero_grid['offset'][index_4] + 0.25 * l_2) - (caero_grid['offset'][index_1] + 0.25 * l_1))

    n = len(ID)
    set_l = np.arange(n * 6).reshape((n, 6))
    set_k = np.arange(n * 6).reshape((n, 6))
    set_j = np.arange(n * 6).reshape((n, 6))
    aerogrid = {'ID': np.array(ID),
                'l': np.array(l),
                'A': np.array(A),
                'N': np.array(N),
                'offset_l': np.array(offset_l),
                'offset_k': np.array(offset_k),
                'offset_j': np.array(offset_j),
                'offset_P1': np.array(offset_P1),
                'offset_P3': np.array(offset_P3),
                'r': np.array(r),
                'set_l': set_l,
                'set_k': set_k,
                'set_j': set_j,
                'CD': caero_panels['CD'],
                'CP': caero_panels['CP'],
                'n': n,
                'coord_desc': 'bodyfixed',
                'cornerpoint_panels': caero_panels['cornerpoints'],
                'cornerpoint_grids': np.hstack((caero_grid['ID'][:, None], caero_grid['offset']))
                }
    if method_caero in ['VLM4Prop']:
        aerogrid['cam_rad'] = cam_rad
    return aerogrid


def build_macgrid(aerogrid, b_ref):
    # Assumptions:
    # - the meam aerodynamic center is the 25% point on the mean aerodynamic choord
    # - the mean aerodynamic choord runs through the (geometrical) center of all panels
    A = np.sum(aerogrid['A'])
    mean_aero_choord = A / b_ref
    geo_center_x = np.dot(aerogrid['offset_k'][:, 0], aerogrid['A']) / A
    geo_center_y = np.dot(aerogrid['offset_k'][:, 1], aerogrid['A']) / A
    geo_center_z = np.dot(aerogrid['offset_k'][:, 2], aerogrid['A']) / A
    macgrid = {'ID': np.array([0]),
               'offset': np.array([[geo_center_x - 0.25 * mean_aero_choord, geo_center_y, geo_center_z]]),
               "set": np.array([[0, 1, 2, 3, 4, 5]]),
               'CD': np.array([0]),
               'CP': np.array([0]),
               'coord_desc': 'bodyfixed',
               'A_ref': A,
               'b_ref': b_ref,
               'c_ref': mean_aero_choord,
               }
    return macgrid


def rfa(Qjj, k, n_poles=2, filename='rfa.png'):
    # B = A*x
    # B ist die gegebene AIC, A die Roger-Approximation, x sind die zu findenden Koeffizienten B0,B1,...B7
    logging.info('Performing rational function approximation (RFA) on AIC matrices with {} poles...'.format(n_poles))
    k = np.array(k)
    n_k = len(k)
    betas = np.max(k) / np.arange(1, n_poles + 1)  # Roger
    # Alternative proposed by Karpel / ZAERO would be: betas = 1.7*np.max(k)*(np.arange(1,n_poles+1)/(n_poles+1.0))**2.0
    option = 2
    if option == 1:  # voll
        Ajj_real = np.array([np.ones(n_k), np.zeros(n_k), -k ** 2])
        Ajj_imag = np.array([np.zeros(n_k), k, np.zeros(n_k)])
    elif option == 2:  # ohne Beschleunigungsterm
        Ajj_real = np.array([np.ones(n_k), np.zeros(n_k), np.zeros(n_k)])
        Ajj_imag = np.array([np.zeros(n_k), k, np.zeros(n_k)])
    elif option == 3:  # ohne Beschleunigungsterm und Daempfungsterm
        Ajj_real = np.array([np.ones(n_k), np.zeros(n_k), np.zeros(n_k)])
        Ajj_imag = np.array([np.zeros(n_k), np.zeros(n_k), np.zeros(n_k)])

    for beta in betas:
        Ajj_real = np.vstack((Ajj_real, k ** 2 / (k ** 2 + beta ** 2)))
        Ajj_imag = np.vstack((Ajj_imag, k * beta / (k ** 2 + beta ** 2)))
    Ajj = np.vstack((Ajj_real.T, Ajj_imag.T))

    # komplette AIC-Matrix: hierzu wird die AIC mit nj*nj umgeformt in einen Vektor nj**2
    Qjj_reshaped = np.vstack((np.real(Qjj.reshape(n_k, -1)), np.imag(Qjj.reshape(n_k, -1))))
    n_j = Qjj.shape[1]

    logging.info('- solving B = A*x with least-squares method')
    solution, _, _, _ = np.linalg.lstsq(Ajj, Qjj_reshaped, rcond=-1)
    ABCD = solution.reshape(3 + n_poles, n_j, n_j)

    # Kontrolle
    Qjj_aprox = np.dot(Ajj, solution)
    RMSE = []
    logging.info('- root-mean-square error(s): ')
    for k_i in range(n_k):
        RMSE_real = np.sqrt(((Qjj_aprox[k_i, :].reshape(n_j, n_j)
                              - np.real(Qjj[k_i, :, :])) ** 2).sum(axis=None) / n_j ** 2)
        RMSE_imag = np.sqrt(((Qjj_aprox[k_i + n_k, :].reshape(n_j, n_j)
                              - np.imag(Qjj[k_i, :, :])) ** 2).sum(axis=None) / n_j ** 2)
        RMSE.append([RMSE_real, RMSE_imag])
        logging.info('  k = {:<6}, RMSE_real = {:<20}, RMSE_imag = {:<20}'.format(k[k_i], RMSE_real, RMSE_imag))
    # Vergroesserung des Frequenzbereichs
    k = np.arange(0.0, (k.max()), 0.001)
    n_k = len(k)

    if option == 1:  # voll
        Ajj_real = np.array([np.ones(n_k), np.zeros(n_k), -k ** 2])
        Ajj_imag = np.array([np.zeros(n_k), k, np.zeros(n_k)])
    elif option == 2:  # ohne Beschleunigungsterm
        Ajj_real = np.array([np.ones(n_k), np.zeros(n_k), np.zeros(n_k)])
        Ajj_imag = np.array([np.zeros(n_k), k, np.zeros(n_k)])
    elif option == 3:  # ohne Beschleunigungsterm und Dämpfungsterm
        Ajj_real = np.array([np.ones(n_k), np.zeros(n_k), np.zeros(n_k)])
        Ajj_imag = np.array([np.zeros(n_k), np.zeros(n_k), np.zeros(n_k)])

    for beta in betas:
        Ajj_real = np.vstack((Ajj_real, k ** 2 / (k ** 2 + beta ** 2)))
    for beta in betas:
        Ajj_imag = np.vstack((Ajj_imag, k * beta / (k ** 2 + beta ** 2)))
    # Plots vom Real- und Imaginaerteil der ersten m_n*n_n Panels
    first_panel = 0
    m_n = 3
    n_n = 3
    plt.figure()
    for m_i in range(m_n):
        for n_i in range(n_n):
            qjj = Qjj[:, first_panel + n_i, first_panel + m_i]
            qjj_aprox = np.dot(Ajj_real.T, ABCD[:, first_panel + n_i, first_panel + m_i]) \
                + np.dot(Ajj_imag.T, ABCD[:, first_panel + n_i, first_panel + m_i]) * 1j
            plt.subplot(m_n, n_n, m_n * m_i + n_i + 1)
            plt.plot(np.real(qjj), np.imag(qjj), 'b.-')
            plt.plot(np.real(qjj_aprox), np.imag(qjj_aprox), 'r-')
            plt.xlabel('real')
            plt.ylabel('imag')

    plt.savefig(filename)
    plt.close()
    return ABCD, n_poles, betas, np.array(RMSE)
