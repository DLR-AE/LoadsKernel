# -*- coding: utf-8 -*-
import copy
import logging
import os
import math
from itertools import compress

import numpy as np
import pandas as pd
import scipy.sparse as sp


def NASTRAN_f06_modal(filename, modes_selected='all', omitt_rigid_body_modes=False):
    '''
    This methode parses a NASTRAN .f06 file and searches for eigenvalues and
    eigenvectors. (Basierend auf Script von Markus.)
    '''
    filesize = float(os.stat(filename).st_size)
    logging.info('Read modal data from f06 file: %s with %.2f MB' % (filename, filesize / 1024 ** 2))

    eigenvalues = {"ModeNo": [],
                   "ExtractionOrder": [],
                   "Eigenvalue": [],
                   "Radians": [],
                   "Cycles": [],
                   "GeneralizedMass": [],
                   "GeneralizedStiffness": []}
    node_ids = []
    eigenvectors = {}

    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if str.find(str.replace(read_string, ' ', ''), 'REALEIGENVALUES') != -1:
                # print ' -> reading eigenvalues...' # Eigenwerte
                fid.readline()
                fid.readline()
                while True:
                    line = str.split(fid.readline())
                    if len(line) == 7:
                        eigenvalues["ModeNo"].append(int(line[0]))
                        eigenvalues["ExtractionOrder"].append(int(line[1]))
                        eigenvalues["Eigenvalue"].append(float(line[2]))
                        eigenvalues["Radians"].append(float(line[3]))
                        eigenvalues["Cycles"].append(float(line[4]))
                        eigenvalues["GeneralizedMass"].append(float(line[5]))
                        eigenvalues["GeneralizedStiffness"].append(float(line[6]))
                    else:
                        break

            elif str.find(str.replace(read_string, ' ', ''), 'REALEIGENVECTORNO') != -1 and read_string != '':
                eigenvector_no = int(str.split(read_string)[-1])
                if str(eigenvector_no) not in eigenvectors:
                    eigenvectors[str(eigenvector_no)] = []
                fid.readline()
                fid.readline()
                while True:
                    line = str.split(fid.readline())
                    if len(line) == 8 and line[1] == 'G':
                        node_ids.append(int(line[0]))
                        eigenvectors[str(eigenvector_no)].append([int(line[0]), float(line[2]), float(line[3]), float(line[4]),
                                                                  float(line[5]), float(line[6]), float(line[7])])
                    else:
                        break

            elif read_string == '':
                break

    logging.info('Found %i eigenvalues and %i eigenvectors for %i nodes.', len(eigenvalues["ModeNo"]),
                 len(eigenvectors.keys()), len(node_ids) / len(eigenvalues["ModeNo"]))
    return eigenvalues, eigenvectors, node_ids


def reduce_modes(eigenvalues, eigenvectors, nodes_selection, modes_selection):
    logging.info('Reduction of data to %i selected modes and %i nodes.', len(modes_selection), len(nodes_selection))
    eigenvalues_new = {"ModeNo": [],
                       "ExtractionOrder": [],
                       "Eigenvalue": [],
                       "Radians": [],
                       "Cycles": [],
                       "GeneralizedMass": [],
                       "GeneralizedStiffness": []}
    eigenvectors_new = {}
    # Searching for the indices of the selected nodes take time
    # Assumption: nodes have the same sequence in all modes
    pos_eigenvector = []
    nodes_eigenvector = np.array(eigenvectors[str(modes_selection[0])])[:, 0]
    logging.info(' - working on nodes...')
    for i_nodes in range(len(nodes_selection)):
        pos_eigenvector.append(np.where(nodes_selection[i_nodes] == nodes_eigenvector)[0][0])

    logging.info(' - working on modes...')
    for i_mode in range(len(modes_selection)):
        pos_mode = np.where(modes_selection[i_mode] == np.array(eigenvalues['ModeNo']))[0][0]
        eigenvalues_new['ModeNo'].append(eigenvalues['ModeNo'][pos_mode])
        eigenvalues_new['ExtractionOrder'].append(eigenvalues['ExtractionOrder'][pos_mode])
        eigenvalues_new['Eigenvalue'].append(eigenvalues['Eigenvalue'][pos_mode])
        eigenvalues_new['Radians'].append(eigenvalues['Radians'][pos_mode])
        eigenvalues_new['Cycles'].append(eigenvalues['Cycles'][pos_mode])
        eigenvalues_new['GeneralizedMass'].append(eigenvalues['GeneralizedMass'][pos_mode])
        eigenvalues_new['GeneralizedStiffness'].append(eigenvalues['GeneralizedStiffness'][pos_mode])

        eigenvectors_new[str(modes_selection[i_mode])] = np.array(eigenvectors[str(modes_selection[i_mode])])[pos_eigenvector]

    return eigenvalues_new, eigenvectors_new


def Nastran_weightgenerator(filename):
    logging.info('Read Weight data from f06 file: %s', filename)

    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if str.find(read_string, 'O U T P U T   F R O M   G R I D   P O I N T   W E I G H T   G E N E R A T O R') != -1:
                read_string = fid.readline()
                CID = nastran_number_converter(read_string.split()[-1], 'int')
                read_string = fid.readline()
                massmatrix_0 = []
                for i in range(6):
                    read_string = fid.readline()
                    massmatrix_0.append([nastran_number_converter(read_string.split()[1], 'float'),
                                         nastran_number_converter(read_string.split()[2], 'float'),
                                         nastran_number_converter(read_string.split()[3], 'float'),
                                         nastran_number_converter(read_string.split()[4], 'float'),
                                         nastran_number_converter(read_string.split()[5], 'float'),
                                         nastran_number_converter(read_string.split()[6], 'float'),
                                         ])
            elif str.find(read_string, 'MASS AXIS SYSTEM (S)') != -1:
                read_string = fid.readline()
                cg_y = nastran_number_converter(read_string.split()[3], 'float')
                cg_z = nastran_number_converter(read_string.split()[4], 'float')
                read_string = fid.readline()
                cg_x = nastran_number_converter(read_string.split()[2], 'float')
                offset_cg = np.array([cg_x, cg_y, cg_z])
                read_string = fid.readline()
                read_string = fid.readline()

                inertia = []
                for i in range(3):
                    read_string = fid.readline()
                    inertia.append([nastran_number_converter(read_string.split()[1], 'float'),
                                    nastran_number_converter(read_string.split()[2], 'float'),
                                    nastran_number_converter(read_string.split()[3], 'float'),
                                    ])
                break
            elif read_string == '':
                break

        return np.array(massmatrix_0), np.array(inertia), offset_cg, CID


def Modgen_GRID(filename):
    logging.info('Read GRID data from ModGen file: %s' % filename)
    grids = []
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    for line in lines:
        if line[0] != '$' and str.find(line[:8], 'GRID') != -1:
            if len(line) <= 48:  # if CD is missing, fix with CP
                line = line + '        '
            grids.append([nastran_number_converter(line[8:16], 'ID'), nastran_number_converter(line[16:24], 'CP'),
                          nastran_number_converter(line[24:32], 'float'), nastran_number_converter(line[32:40], 'float'),
                          nastran_number_converter(line[40:48], 'float'), nastran_number_converter(line[48:56], 'CD')])

    n = len(grids)
    grid = {"ID": np.array([grid[0] for grid in grids]),
            "offset": np.array([grid[2:5] for grid in grids]),
            "n": n,
            "CP": np.array([grid[1] for grid in grids]),
            "CD": np.array([grid[5] for grid in grids]),
            "set": np.arange(n * 6).reshape((n, 6)),
            }
    return grid


def add_GRIDS(pandas_grids):
    # This functions relies on the Pandas data frames from the bdf reader.
    n = pandas_grids.shape[0]
    strcgrid = {}
    strcgrid['ID'] = pandas_grids['ID'].to_numpy(dtype='int')
    strcgrid['CD'] = pandas_grids['CD'].to_numpy(dtype='int')
    strcgrid['CP'] = pandas_grids['CP'].to_numpy(dtype='int')
    strcgrid['n'] = n
    strcgrid['set'] = np.arange(n * 6).reshape((n, 6))
    strcgrid['offset'] = pandas_grids[['X1', 'X2', 'X3']].to_numpy(dtype='float')
    return strcgrid


def add_shell_elements(pandas_panels):
    # This functions relies on the Pandas data frames from the bdf reader.
    strcshell = {}
    n = pandas_panels.shape[0]
    strcshell['ID'] = pandas_panels['ID'].to_numpy(dtype='int')
    strcshell['cornerpoints'] = np.array(pandas_panels[['G1', 'G2', 'G3', 'G4']])
    strcshell['CD'] = np.zeros(n)  # Assumption: panels are given in global coord system
    strcshell['CP'] = np.zeros(n)
    strcshell['n'] = n
    return strcshell


def add_panels_from_CAERO(pandas_caero, pandas_aefact):
    logging.info('Constructing aero panels from CAERO cards')
    # from CAERO cards, construct corner points... '
    # then, combine four corner points to one panel
    grid_ID = 1  # the file number is used to set a range of grid IDs
    grids = {'ID': [], 'offset': []}
    panels = {"ID": [], 'CP': [], 'CD': [], "cornerpoints": []}
    for index, caerocard in pandas_caero.iterrows():
        # get the four corner points of the CAERO card
        X1 = caerocard[['X1', 'Y1', 'Z1']].to_numpy(dtype='float')
        X4 = caerocard[['X4', 'Y4', 'Z4']].to_numpy(dtype='float')
        X2 = X1 + np.array([caerocard['X12'], 0.0, 0.0])
        X3 = X4 + np.array([caerocard['X43'], 0.0, 0.0])
        # calculate LE, Root and Tip vectors [x,y,z]^T
        LE = X4 - X1
        Root = X2 - X1
        Tip = X3 - X4
        n_span = int(caerocard['NSPAN'])
        n_chord = int(caerocard['NCHORD'])
        if caerocard['NCHORD'] == 0:
            # look in AEFACT cards for the appropriate card and get spacing
            if pd.notna(caerocard['LCHORD']):
                d_chord = [v for v in pandas_aefact.loc[pandas_aefact['ID'] == caerocard['LCHORD'],
                                                        'values'].values[0] if v is not None]
                n_chord = len(d_chord) - 1  # n_boxes = n_division-1
            else:
                logging.error('Assumption of equal spaced CAERO7 panels is violated!')
        else:
            # assume equidistant spacing
            d_chord = np.linspace(0.0, 1.0, n_chord + 1)

        if caerocard['NSPAN'] == 0:
            # look in AEFACT cards for the appropriate card and get spacing
            if pd.notna(caerocard['LSPAN']):
                d_span = [v for v in pandas_aefact.loc[pandas_aefact['ID'] == caerocard['LSPAN'],
                                                       'values'].values[0] if v is not None]
                n_span = len(d_span) - 1  # n_boxes = n_division-1
            else:
                logging.error('Assumption of equal spaced CAERO7 panels is violated!')
        else:
            # assume equidistant spacing
            d_span = np.linspace(0.0, 1.0, n_span + 1)

        # build matrix of corner points
        # index based on n_divisions
        grids_map = np.zeros((n_chord + 1, n_span + 1), dtype='int')
        for i_strip in range(n_span + 1):
            for i_row in range(n_chord + 1):
                offset = X1 + LE * d_span[i_strip] + (Root * (1.0 - d_span[i_strip]) + Tip * d_span[i_strip]) * d_chord[i_row]
                grids['ID'].append(grid_ID)
                grids['offset'].append(offset)
                grids_map[i_row, i_strip] = grid_ID
                grid_ID += 1
        # build panels from cornerpoints
        # index based on n_boxes
        panel_ID = int(caerocard['ID'])
        for i_strip in range(n_span):
            for i_row in range(n_chord):
                panels['ID'].append(panel_ID)
                panels['CP'].append(caerocard['CP'])  # applying CP of CAERO card to all grids
                panels['CD'].append(caerocard['CP'])
                panels['cornerpoints'].append([grids_map[i_row, i_strip], grids_map[i_row + 1, i_strip],
                                               grids_map[i_row + 1, i_strip + 1], grids_map[i_row, i_strip + 1]])
                panel_ID += 1
    panels['ID'] = np.array(panels['ID'])
    panels['CP'] = np.array(panels['CP'])
    panels['CD'] = np.array(panels['CD'])
    panels['cornerpoints'] = np.array(panels['cornerpoints'])
    grids['ID'] = np.array(grids['ID'])
    grids['offset'] = np.array(grids['offset'])
    return grids, panels


def nastran_number_converter(string_in, type, default=0):
    if type in ['float', 'f']:
        try:
            out = float(string_in)
        except ValueError:
            # remove all spaces, which might also occur in between the sign and the number (e.g. in ModGen)
            string_in = string_in.replace(' ', '')
            # remove end of line
            for c in ['\n', '\r']:
                string_in = string_in.strip(c)
            if '-' in string_in[1:]:
                if string_in[0] in ['-', '+']:
                    sign = string_in[0]
                    out = float(sign + string_in[1:].replace('-', 'E-'))
                else:
                    out = float(string_in.replace('-', 'E-'))
            elif '+' in string_in[1:]:
                if string_in[0] in ['-', '+']:
                    sign = string_in[0]
                    out = float(sign + string_in[1:].replace('+', 'E+'))
                else:
                    out = float(string_in.replace('+', 'E+'))
            elif string_in == '':
                logging.debug("Could not interpret the following number: '" + string_in
                              + "' -> setting value to " + str(default))
                out = default
            else:
                logging.error("Could not interpret the following number: " + string_in)
                return
    elif type in ['int', 'i', 'ID', 'CD', 'CP']:
        try:
            out = int(string_in)
        except ValueError:
            out = default
    elif type in ['str']:
        # An dieser Stelle reicht es nicht mehr aus, nur die Leerzeichen zu entfernen...
        # whitelist = string.ascii_letters + string.digits
        # out = ''.join(filter(whitelist.__contains__, string_in))
        out = string_in.strip('*, ')
        if out == '':
            out = default

    return out


def Nastran_DMI(filename):
    DMI = {}
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if read_string[:3] == 'DMI' and nastran_number_converter(read_string[16:24], 'int') == 0:
                # this is the header
                DMI['NAME'] = str.replace(read_string[8:16], ' ', '')  # matrix name
                DMI['FORM'] = nastran_number_converter(read_string[24:32], 'int')
                DMI['TIN'] = nastran_number_converter(read_string[32:40], 'int')
                DMI['n_row'] = nastran_number_converter(read_string[56:64], 'int')
                DMI['n_col'] = nastran_number_converter(read_string[64:72], 'int')
                if DMI['TIN'] in [1, 2]:
                    DMI['data'] = sp.lil_matrix((DMI['n_col'], DMI['n_row']), dtype=float)
                elif DMI['TIN'] in [3, 4]:
                    logging.error('Complex DMI matrix input NOT supported.')
                logging.info('Read {} data from file: {}'.format(DMI['NAME'], filename))
                while True:
                    # now we read column by column
                    read_string = fid.readline()
                    if read_string[:3] == 'DMI' and nastran_number_converter(read_string[16:24], 'int') != 0:
                        # new column
                        i_col = nastran_number_converter(read_string[16:24], 'int')
                        col = read_string[24:-1]
                        # figure out how many lines the column will have
                        n_items = DMI['n_row']  # items that go into the column
                        n_lines = int(math.ceil((n_items - 3) / 4.0))
                        for _ in range(n_lines):
                            col += fid.readline()[8:-1]
                        for _ in range(n_items):
                            DMI['data'][i_col - 1, nastran_number_converter(col[:8], 'int') - 1] = \
                                nastran_number_converter(col[8:16], 'float')
                            col = col[16:]
                    else:
                        break
            elif read_string == '':
                # end of file
                break
    return DMI


def add_CORD2R(pandas_cord2r, coord):
    # This functions relies on the Pandas data frames from the bdf reader.
    for _, row in pandas_cord2r.iterrows():
        ID = int(row['ID'])
        RID = int(row['RID'])
        A = row[['A1', 'A2', 'A3']].to_numpy(dtype='float').squeeze()
        B = row[['B1', 'B2', 'B3']].to_numpy(dtype='float').squeeze()
        C = row[['C1', 'C2', 'C3']].to_numpy(dtype='float').squeeze()
        # build coord
        z = B - A
        y = np.cross(B - A, C - A)
        x = np.cross(y, z)
        dircos = np.vstack((x / np.linalg.norm(x), y / np.linalg.norm(y), z / np.linalg.norm(z))).T
        # save
        if ID not in coord['ID']:
            coord['ID'].append(ID)
            coord['RID'].append(RID)
            coord['offset'].append(A)
            coord['dircos'].append(dircos)


def add_CORD1R(pandas_cord1r, coord, strcgrid):
    # This functions relies on the Pandas data frames from the bdf reader.
    for _, row in pandas_cord1r.iterrows():
        ID = int(row['ID'])
        RID = 0
        A = strcgrid['offset'][np.where(strcgrid['ID'] == row['A'])[0][0]]
        B = strcgrid['offset'][np.where(strcgrid['ID'] == row['B'])[0][0]]
        C = strcgrid['offset'][np.where(strcgrid['ID'] == row['C'])[0][0]]
        # build coord - wie bei CORD2R
        z = B - A
        y = np.cross(B - A, C - A)
        x = np.cross(y, z)
        dircos = np.vstack((x / np.linalg.norm(x), y / np.linalg.norm(y), z / np.linalg.norm(z))).T
        # save
        if ID not in coord['ID']:
            coord['ID'].append(ID)
            coord['RID'].append(RID)
            coord['offset'].append(A)
            coord['dircos'].append(dircos)


def add_AESURF(pandas_aesurfs):
    aesurf = {}
    aesurf['ID'] = pandas_aesurfs['ID'].to_list()
    aesurf['key'] = pandas_aesurfs['LABEL'].to_list()
    aesurf['CID'] = pandas_aesurfs['CID'].to_list()
    aesurf['AELIST'] = pandas_aesurfs['AELIST'].to_list()
    aesurf['eff'] = pandas_aesurfs['EFF'].to_list()
    return aesurf


def add_SET1(pandas_sets):
    # This functions relies on the Pandas data frames from the bdf reader.
    # Due to the mixture of integers and strings ('THRU') in a SET1 card, all list items were parsed as strings.
    # This function parses the strings to integers and handles the 'THRU' option.
    set_values = []
    for _, row in pandas_sets[['values']].iterrows():
        # create a copy of the current row to work with
        my_row = copy.deepcopy(row.iloc[0])
        # remove all None values
        my_row = [item for item in my_row if item is not None]
        values = []
        while my_row:
            if my_row[0] == 'THRU':
                # Replace 'THRU' with the intermediate values
                startvalue = values[-1] + 1
                stoppvalue = nastran_number_converter(my_row[1], 'int', default=None)
                values += list(range(startvalue, stoppvalue + 1))
                # remove consumed values from row
                my_row.pop(0)
                my_row.pop(0)
            else:
                # Parse list item as interger
                values.append(nastran_number_converter(my_row[0], 'int', default=None))
                # remove consumed values from row
                my_row.pop(0)
        set_values.append(np.array(values))

    sets = {}
    sets['ID'] = pandas_sets['ID'].to_list()
    sets['values'] = set_values
    return sets


def add_AECOMP(pandas_aecomps):
    # This functions relies on the Pandas data frames from the bdf reader.
    list_id = []
    # Loop over the rows to check for NaNs and None, which occur in case an empty field was in the list.
    # Then, select only the valid list items.
    for _, row in pandas_aecomps[['LISTID']].iterrows():
        is_id = [pd.notna(x) for x in row.iloc[0]]
        list_id.append(list(compress(row.iloc[0], is_id)))

    aecomp = {}
    aecomp['name'] = pandas_aecomps['NAME'].to_list()
    aecomp['list_type'] = pandas_aecomps['LISTTYPE'].to_list()
    aecomp['list_id'] = list_id
    return aecomp


def add_MONPNT1(pandas_monpnts):
    # This functions relies on the Pandas data frames from the bdf reader.
    n = pandas_monpnts.shape[0]
    mongrid = {}
    # Eigentlich haben MONPNTs keine IDs sondern nur Namen...
    mongrid['ID'] = np.arange(1, n + 1)
    mongrid['name'] = pandas_monpnts['NAME'].to_list()
    mongrid['comp'] = pandas_monpnts['COMP'].to_list()
    mongrid['CD'] = pandas_monpnts['CD'].to_numpy(dtype='int')
    mongrid['CP'] = pandas_monpnts['CP'].to_numpy(dtype='int')
    mongrid['n'] = n
    mongrid['set'] = np.arange(n * 6).reshape((n, 6))
    mongrid['offset'] = pandas_monpnts[['X', 'Y', 'Z']].to_numpy(dtype='float')
    return mongrid
