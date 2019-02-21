# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:29:57 2014

@author: voss_ar
"""
import string
import numpy as np
import scipy.sparse as sp
import math as math
import os, logging

def NASTRAN_f06_modal(filename, modes_selected='all', omitt_rigid_body_modes=False):
    '''
    This methode parses a NASTRAN .f06 file and searches for eigenvalues and
    eigenvectors. (Basierend auf Script von Markus.)
    '''
    filesize = float(os.stat(filename).st_size)
    logging.info('Read modal data from f06 file: %s with %.2f MB' %(filename, filesize/1024**2))
    percent = 0.
    print 'Progress [%]: ',

    eigenvalues = {"ModeNo":[],
                   "ExtractionOrder":[],
                   "Eigenvalue":[],
                   "Radians":[],
                   "Cycles":[],
                   "GeneralizedMass":[],
                   "GeneralizedStiffness":[]}
    grids = []
    CaseControlEcho = []  
    node_ids = []
    eigenvectors = {}
    
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            
            # calculate progress and print to command line
            if fid.tell()/filesize*100. > percent:
                percent += 1.
                print int(percent),

            
            if string.find(string.replace(read_string,' ',''),'REALEIGENVALUES') != -1:
                #print ' -> reading eigenvalues...' # Eigenwerte
                fid.readline()
                fid.readline()
                while True:
                    line = string.split(fid.readline())
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
                    
            elif string.find(string.replace(read_string,' ',''),'REALEIGENVECTORNO') != -1 and read_string != '':
                #print ' -> reading eigenvectors...' # Eigenvektoren
                eigenvector_no = int(string.split(read_string)[-1])
                if not eigenvectors.has_key(str(eigenvector_no)):
                    eigenvectors[str(eigenvector_no)] = []
                fid.readline()
                fid.readline()
                while True:
                    line = string.split(fid.readline())
                    if len(line) == 8 and line[1] == 'G':
                        node_ids.append(int(line[0]))
                        eigenvectors[str(eigenvector_no)].append([int(line[0]), float(line[2]),float(line[3]),float(line[4]),float(line[5]),float(line[6]),float(line[7])])
                    else:
                        break
       
            elif read_string == '':
                break    

    print ' Done.'
    logging.info('Found %i eigenvalues and %i eigenvectors for %i nodes.' %(len(eigenvalues["ModeNo"]), len(eigenvectors.keys()), len(node_ids)/len(eigenvalues["ModeNo"])))

    return  eigenvalues, eigenvectors, node_ids
    
def reduce_modes(eigenvalues, eigenvectors, nodes_selection, modes_selection):
    logging.info('Reduction of data to %i selected modes and %i nodes.' %(len(modes_selection), len(nodes_selection)))
    eigenvalues_new = {"ModeNo":[],
                   "ExtractionOrder":[],
                   "Eigenvalue":[],
                   "Radians":[],
                   "Cycles":[],
                   "GeneralizedMass":[],
                   "GeneralizedStiffness":[]}
    eigenvectors_new = {}
    # Searching for the indices of the selected nodes take time
    # Assumption: nodes have the same sequence in all modes
    pos_eigenvector = []
    nodes_eigenvector = np.array(eigenvectors[str(modes_selection[0])])[:,0]
    logging.info(' - working on nodes...')
    for i_nodes in range(len(nodes_selection)):
        pos_eigenvector.append( np.where(nodes_selection[i_nodes] ==  nodes_eigenvector)[0][0] )
    
    logging.info(' - working on modes...')
    for i_mode in range(len(modes_selection)):
        pos_mode = np.where(modes_selection[i_mode]==np.array(eigenvalues['ModeNo']))[0][0]
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
    logging.info('Read Weight data from f06 file: %s' %filename)
    
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if string.find(read_string, 'O U T P U T   F R O M   G R I D   P O I N T   W E I G H T   G E N E R A T O R') !=-1:
                read_string = fid.readline()
                CID = nastran_number_converter(read_string.split()[-1], 'int')
                read_string = fid.readline()
                massmatrix_0 = []
                for i in range(6):                
                    read_string = fid.readline()
                    massmatrix_0.append([nastran_number_converter(read_string.split()[1], 'float'), \
                                         nastran_number_converter(read_string.split()[2], 'float'), \
                                         nastran_number_converter(read_string.split()[3], 'float'), \
                                         nastran_number_converter(read_string.split()[4], 'float'), \
                                         nastran_number_converter(read_string.split()[5], 'float'), \
                                         nastran_number_converter(read_string.split()[6], 'float'), \
                                       ])
            elif string.find(read_string, 'MASS AXIS SYSTEM (S)') !=-1:
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
                    inertia.append([nastran_number_converter(read_string.split()[1], 'float'), \
                                    nastran_number_converter(read_string.split()[2], 'float'), \
                                    nastran_number_converter(read_string.split()[3], 'float'), \
                                  ])
                break
            elif read_string == '':
                break 
                
        return np.array(massmatrix_0), np.array(inertia), offset_cg, CID
           
def Modgen_GRID(filename):
    logging.info('Read GRID data from ModGen file: %s' %filename)
    grids = []
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    for line in lines:
        if line[0] != '$' and string.find(line[:8], 'GRID') !=-1 :
            if len(line) <= 48: # if CD is missing, fix with CP
                line = line + '        '
            grids.append([nastran_number_converter(line[8:16], 'ID'), nastran_number_converter(line[16:24], 'CP'), nastran_number_converter(line[24:32], 'float'), nastran_number_converter(line[32:40], 'float'), nastran_number_converter(line[40:48], 'float'), nastran_number_converter(line[48:56], 'CD')])
            
    n = len(grids)
    grid = {"ID": np.array([grid[0] for grid in grids]),
            "offset":np.array([grid[2:5] for grid in grids]),
            "n": n,
            "CP": np.array([grid[1] for grid in grids]),
            "CD": np.array([grid[5] for grid in grids]),
            "set": np.arange(n*6).reshape((n,6)),
           }
    return grid

def Modgen_CQUAD4(filename):
    logging.info('Read CQUAD4/CTRIA3 data from ModGen file: %s' %filename)
    ids = []
    cornerpoints_points = []
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if string.find(read_string, 'CQUAD4') !=-1 and read_string[0] != '$':
                ids.append(nastran_number_converter(read_string[8:16], 'ID'),)
                cornerpoints_points.append([nastran_number_converter(read_string[24:32], 'ID'), nastran_number_converter(read_string[32:40], 'ID'), nastran_number_converter(read_string[40:48], 'ID'), nastran_number_converter(read_string[48:56], 'ID')])
            elif string.find(read_string, 'CTRIA3') !=-1 and read_string[0] != '$':
                ids.append(nastran_number_converter(read_string[8:16], 'ID'),)
                cornerpoints_points.append([nastran_number_converter(read_string[24:32], 'ID'), nastran_number_converter(read_string[32:40], 'ID'), nastran_number_converter(read_string[40:48], 'ID')])
            elif read_string == '':
                break
    ids = np.array(ids)
    panels = {"ID": ids,
              "cornerpoints": cornerpoints_points, # cornerpoints is a list to allow for both quad and tria elements
              "CP": np.zeros(ids.shape), # Assumption: panels are given in global coord system
              "CD": np.zeros(ids.shape),
              "n": len(ids)
             }
    return panels
    
def CAERO(filename, i_file):
    logging.info('Read CAERO1 and/or CAERO7 cards from Nastran/ZAERO bdf: %s' %filename)
    caerocards = []
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if string.find(read_string, 'CAERO1') !=-1 and read_string[0] != '$':
                # read first line of CAERO card
                caerocard = {'EID': nastran_number_converter(read_string[8:16], 'ID'),
                             'CP': nastran_number_converter(read_string[24:32], 'ID'),
                             'n_span': nastran_number_converter(read_string[32:40], 'ID'), # n_boxes
                             'n_chord': nastran_number_converter(read_string[40:48], 'ID'), # n_boxes
                             'l_span': nastran_number_converter(read_string[48:56], 'ID'),
                             'l_chord': nastran_number_converter(read_string[56:64], 'ID'),
                            }
                # read second line of CAERO card
                read_string = fid.readline()  
                caerocard['X1'] = np.array([nastran_number_converter(read_string[ 8:16], 'float'), nastran_number_converter(read_string[16:24], 'float'), nastran_number_converter(read_string[24:32], 'float')])
                caerocard['length12'] = nastran_number_converter(read_string[32:40], 'float')
                caerocard['X2'] = caerocard['X1'] + np.array([caerocard['length12'], 0.0, 0.0])
                caerocard['X4'] =np.array([nastran_number_converter(read_string[40:48], 'float'), nastran_number_converter(read_string[48:56], 'float'), nastran_number_converter(read_string[56:64], 'float')])
                caerocard['length43'] = nastran_number_converter(read_string[64:72], 'float')
                caerocard['X3'] = caerocard['X4'] + np.array([caerocard['length43'], 0.0, 0.0])
                caerocards.append(caerocard)
            if string.find(read_string, 'CAERO7') !=-1 and read_string[0] != '$':
                # The CAERO7 cards of ZAERO is nearly identical to Nastran'S CAERO1 card. 
                # However, it uses 3 lines, which makes the card more readable to the human eye.
                # Also, not the number of boxes but the number of divisions is given (n_boxes = n_division-1)
                # read first line of CAERO card
                caerocard = {'EID': nastran_number_converter(read_string[8:16], 'ID'),
                             'CP': nastran_number_converter(read_string[24:32], 'ID'),
                             'n_span': nastran_number_converter(read_string[32:40], 'ID') - 1,
                             'n_chord': nastran_number_converter(read_string[40:48], 'ID') - 1,
                            }
                if np.any([caerocard['n_span'] == 0, caerocard['n_chord'] == 0]):
                    logging.error('Assumption of equal spaced CAERO7 panels is violated!')
                # read second line of CAERO card
                read_string = fid.readline()  
                caerocard['X1'] = np.array([nastran_number_converter(read_string[ 8:16], 'float'), nastran_number_converter(read_string[16:24], 'float'), nastran_number_converter(read_string[24:32], 'float')])
                caerocard['length12'] = nastran_number_converter(read_string[32:40], 'float')
                caerocard['X2'] = caerocard['X1'] + np.array([caerocard['length12'], 0.0, 0.0])
                # read third line of CAERO card
                read_string = fid.readline()  
                caerocard['X4'] =np.array([nastran_number_converter(read_string[ 8:16], 'float'), nastran_number_converter(read_string[16:24], 'float'), nastran_number_converter(read_string[24:32], 'float')])
                caerocard['length43'] = nastran_number_converter(read_string[32:40], 'float')
                caerocard['X3'] = caerocard['X4'] + np.array([caerocard['length43'], 0.0, 0.0])
                caerocards.append(caerocard)
            elif read_string == '':
                break
    logging.info('Read AEFACT cards from Nastran bdf: %s' %filename)
    aefacts = Nastran_AEFACT(filename)        
    logging.info(' - from CAERO cards, constructing corner points and aero panels')
    # from CAERO cards, construct corner points... '
    # then, combine four corner points to one panel
    grid_ID = i_file * 100000 # the file number is used to set a range of grid IDs 
    grids = {'ID':[], 'offset':[]}
    panels = {"ID": [], 'CP':[], 'CD':[], "cornerpoints": []}
    for caerocard in caerocards:
         # calculate LE, Root and Tip vectors [x,y,z]^T
         LE   = caerocard['X4'] - caerocard['X1']
         Root = caerocard['X2'] - caerocard['X1']
         Tip  = caerocard['X3'] - caerocard['X4']
         
         if caerocard['n_chord'] == 0:
             # look in AEFACT cards for the appropriate card and get spacing
             d_chord = aefacts['values'][aefacts['ID'].index(caerocard['l_chord'])]
             caerocard['n_chord'] = len(d_chord)-1 # n_boxes = n_division-1
         else:
             # assume equidistant spacing
             d_chord = np.linspace(0.0, 1.0, caerocard['n_chord']+1 ) 
             
         if caerocard['n_span'] == 0:
             # look in AEFACT cards for the appropriate card and get spacing
             d_span = aefacts['values'][aefacts['ID'].index(caerocard['l_span'])]
             caerocard['n_span'] = len(d_span)-1 # n_boxes = n_division-1
         else:
              # assume equidistant spacing
             d_span = np.linspace(0.0, 1.0, caerocard['n_span']+1 ) 
         
         # build matrix of corner points
         # index based on n_divisions
         grids_map = np.zeros((caerocard['n_chord']+1,caerocard['n_span']+1), dtype='int')
         for i_strip in range(caerocard['n_span']+1):
             for i_row in range(caerocard['n_chord']+1):
                 offset = caerocard['X1'] \
                        + LE * d_span[i_strip] \
                        + (Root*(1.0-d_span[i_strip]) + Tip*d_span[i_strip]) * d_chord[i_row]
                 grids['ID'].append(grid_ID)
                 grids['offset'].append(offset)
                 grids_map[i_row,i_strip ] = grid_ID
                 grid_ID += 1
         # build panels from cornerpoints
         # index based on n_boxes
         panel_ID =  caerocard['EID']                  
         for i_strip in range(caerocard['n_span']):
             for i_row in range(caerocard['n_chord']):
                panels['ID'].append(panel_ID)
                panels['CP'].append(caerocard['CP']) # applying CP of CAERO card to all grids
                panels['CD'].append(caerocard['CP'])
                panels['cornerpoints'].append([ grids_map[i_row, i_strip], grids_map[i_row+1, i_strip], grids_map[i_row+1, i_strip+1], grids_map[i_row, i_strip+1] ])
                panel_ID += 1 
    panels['ID'] = np.array(panels['ID'])
    panels['CP'] = np.array(panels['CP'])
    panels['CD'] = np.array(panels['CD'])
    panels['cornerpoints'] = np.array(panels['cornerpoints'])
    grids['ID'] = np.array(grids['ID'])
    grids['offset'] = np.array(grids['offset'])
    return grids, panels      
    
def nastran_number_converter(string_in, type, default=0):
    if type in ['float']:
        try:
            out = float(string_in)
        except:
            
            string_in = string_in.replace(' ', '') # remove leading spaces
            for c in ['\n', '\r']: string_in = string_in.strip(c) # remove end of line
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
                logging.warning("Could not interpret the following number: '" + string_in + "' -> setting value to zero.")
                out = float(default)
            else: 
                logging.error("Could not interpret the following number: " + string_in)
                return
    elif type in ['int', 'ID', 'CD', 'CP']:
        try:
            out = int(string_in)
        except:
            out = int(default)  
    return out

def Nastran_DMI(filename):
    DMI = {}
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if read_string[:3] == 'DMI' and nastran_number_converter(read_string[16:24], 'int') == 0:
                # this is the header
                DMI['NAME'] = string.replace(read_string[8:16],' ','') # matrix name
                DMI['FORM'] = nastran_number_converter(read_string[24:32], 'int')
                DMI['TIN']  = nastran_number_converter(read_string[32:40], 'int')
                DMI['n_row']  = nastran_number_converter(read_string[56:64], 'int')
                DMI['n_col']  = nastran_number_converter(read_string[64:72], 'int')
                if DMI['TIN'] in [1,2]:
                    DMI['data'] = sp.lil_matrix((DMI['n_col'], DMI['n_row']), dtype=float)
                elif DMI['TIN'] in [3,4]:
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
                        n_items = DMI['n_row'] # items that go into the column
                        n_lines = int(math.ceil((n_items-3)/4.0))
                        for i_line in range(n_lines):
                            col += fid.readline()[8:-1]
                        for i_item in range(n_items):
                            DMI['data'][i_col-1, nastran_number_converter(col[:8], 'int')-1] = nastran_number_converter(col[8:16], 'float')
                            col = col[16:]
                    else:
                        break
            elif read_string == '':
                # end of file
                break
    return DMI                
     
def Nastran_OP4(filename, sparse_output=False, sparse_format=False ):
    # Assumptions:
    # - only one matrix is give per file
    # - often, matrices have sparse characteristics, but might be given in Non-Sparse format.
    #   -> the reader always assembles the matrix as sparse internally
    # Option:           Value/Type:      Desciption:
    # filename          string          path & filename of OP4 file, e.g. './mg02_DLR-F19-S/nastran/MAA.dat'
    # sparse_output     True/False      decides about output conversion to full matrix for output
    # sparse_format     True/False      True: Sparse and ASCII Format of input file
    #                                   False: Non-Sparse and ASCII Format of input file

    filesize = float(os.stat(filename).st_size)
    logging.info('Read data from OP4 file: %s with %.2f MB' %(filename, filesize/1024**2))
    percent = 0.
    print 'Progress [%]: ',
    with open(filename, 'r') as fid:
        # get information from header line
        read_string = fid.readline()
        n_col = nastran_number_converter(read_string[0:8], 'int')
        n_row = nastran_number_converter(read_string[8:16], 'int')
        # Assumptions:
        # - real values: 16 characters for one value --> five values per line 
        # - complex values: 16 characters for real and complex part each --> five values in two line
        string_length = 16 # double precision
        # According to the manual, NTYPE 1 and 3 are used for single precision while 2 and 4 are used for double precision. 
        # However, no change in string length has been observed.
        # In addition, the format '1P,5E16.9' has 9 digits after the decimal separator, which is double precision, disregarding the information given by NTYPE.
        if nastran_number_converter(read_string[24:32], 'int') in [1, 2]:
            type_real = True
            type_complex = False
            data = sp.lil_matrix((n_col, n_row), dtype=float)
        elif nastran_number_converter(read_string[24:32], 'int') in [3, 4]:
            type_real = False
            type_complex = True
            data = sp.lil_matrix((n_col, n_row), dtype=complex)
            
        else:
            logging.error('Unknown format: ' + read_string[24:32] )
        while True:    
            # read header of data block
            read_string = fid.readline()
            
            # calculate progress and print to command line
            if fid.tell()/filesize*100. > percent:
                percent += 1.
                print int(percent),
            
            if read_string == '' or nastran_number_converter(read_string[0:8], 'int') > n_col:
                # end of file reached or the last "dummy column" is reached
                break
            elif sparse_format:
                if n_row > 65535:
                    bigmat = True
                    # For BIGMAT matrices, the length L and row position i_row are given directly in record 3.
                    # L, IROW
                else:
                    bigmat = False
                    # Non-BIGMAT matrices have a string header IS, located in record 3, from which the length L and row position i_row are derived by
                    # L = INT(IS/65536) - 1
                    # IROW = IS - 65536(L + 1)
                i_col = nastran_number_converter(read_string[0:8], 'int') - 1
                n_words = nastran_number_converter(read_string[16:24], 'int')
                while n_words > 0:
                    
                    read_string = fid.readline()
                    if not bigmat:
                        IS = nastran_number_converter(read_string[0:16], 'int')
                        L = np.int(IS/65536 - 1)
                        n_words -= L+1
                        i_row = IS - 65536*(L+1) -1
                    else:
                        L = np.int(nastran_number_converter(read_string[0:8], 'int') - 1)
                        n_words -= L+2
                        i_row = nastran_number_converter(read_string[8:16], 'int') -1
                    # figure out how many lines the datablock will have
                    if type_real:
                        n_items = L/2 # items that go into the column
                        n_lines = int(math.ceil(n_items/5.0))
                    elif type_complex:
                        n_items = L/2 / 2 # items that go into the column
                        n_lines = int(math.ceil(n_items/2.5))
                    # read lines of datablock
                    row = ''
                    for i_line in range(n_lines):
                        row += fid.readline()[:-1]
                        
                    for i_item in range(n_items):
                        if type_real:
                            data[i_col, i_row + i_item] = nastran_number_converter(row[:16], 'float')
                            row = row[16:]
                        elif type_complex:
                            data[i_col, i_row + i_item] = np.complex(nastran_number_converter(row[:16], 'float'), nastran_number_converter(row[16:32], 'float'))
                            row = row[32:]
                    
            elif not sparse_format:
                i_col = nastran_number_converter(read_string[0:8], 'int') - 1
                i_row = nastran_number_converter(read_string[8:16], 'int') - 1
                
                # figure out how many lines the datablock will have
                if type_real:
                    n_items = nastran_number_converter(read_string[16:24], 'int') # items that go into the column
                    n_lines = int(math.ceil(n_items/5.0))
                elif type_complex:
                    n_items = nastran_number_converter(read_string[16:24], 'int') / 2 # items that go into the column
                    n_lines = int(math.ceil(n_items/2.5))
                # read lines of datablock
                row = ''
                for i_line in range(n_lines):
                    row += fid.readline()[:-1]
                    
                for i_item in range(n_items):
                    if type_real:
                        data[i_col, i_row + i_item] = nastran_number_converter(row[:16], 'float')
                        row = row[16:]
                    elif type_complex:
                        data[i_col, i_row + i_item] = np.complex(nastran_number_converter(row[:16], 'float'), nastran_number_converter(row[16:32], 'float'))
                        row = row[32:]
        if sparse_output:
            data = data.tocsc() # better sparse format than lil_matrix
        if not sparse_output:
            data = data.toarray()
    print 'Done.'       
    return data
  
def Modgen_CORD2R(filename, coord, grid=''):
#    coord = {'ID':[],
#             'RID':[],
#             'offset':[],
#             'dircos':[],
#            }
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if string.find(read_string, 'CORD2R') !=-1 and read_string[0] != '$':
                # extract information from CORD2R card
                line1 = read_string
                line2 = fid.readline()
                ID = nastran_number_converter(line1[8:16], 'int')
                RID = nastran_number_converter(line1[16:24], 'int')
                A = np.array([nastran_number_converter(line1[24:32], 'float'), nastran_number_converter(line1[32:40], 'float'), nastran_number_converter(line1[40:48], 'float')])
                B = np.array([nastran_number_converter(line1[48:56], 'float'), nastran_number_converter(line1[56:64], 'float'), nastran_number_converter(line1[64:72], 'float')])
                C = np.array([nastran_number_converter(line2[8:16], 'float'), nastran_number_converter(line2[16:24], 'float'), nastran_number_converter(line2[24:32], 'float')])
                # build coord                
                z = B - A
                y = np.cross(B-A, C-A)
                x = np.cross(y,z)
                dircos = np.vstack((x/np.linalg.norm(x),y/np.linalg.norm(y),z/np.linalg.norm(z))).T
                # save
                coord['ID'].append(ID)
                coord['RID'].append(RID)
                coord['offset'].append(A)
                coord['dircos'].append(dircos)
                
            elif string.find(read_string, 'CORD1R') !=-1 and read_string[0] != '$':
                # CHORD1R ist aehnlich zu CORD2R, anstelle von offsets werden als grid points angegeben 
                if grid == '':
                    logging.warning(read_string)
                    logging.warning('Found CORD1R card, but no grid is given. Coord is ignored.')
                else:
                    line1 = read_string
                    ID = nastran_number_converter(line1[8:16], 'int')
                    RID = 0
                    ID_A = nastran_number_converter(line1[16:24], 'int')
                    A =  grid['offset'][np.where(grid['ID'] == ID_A)[0][0]]
                    ID_B = nastran_number_converter(line1[24:32], 'int')
                    B = grid['offset'][np.where(grid['ID'] == ID_B)[0][0]]   
                    ID_C = nastran_number_converter(line1[32:40], 'int')
                    C = grid['offset'][np.where(grid['ID'] == ID_C)[0][0]]   
                    # build coord - wie bei CORD2R              
                    z = B - A
                    y = np.cross(B-A, C-A)
                    x = np.cross(y,z)
                    dircos = np.vstack((x/np.linalg.norm(x),y/np.linalg.norm(y),z/np.linalg.norm(z))).T
                    # save
                    coord['ID'].append(ID)
                    coord['RID'].append(RID)
                    coord['offset'].append(A)
                    coord['dircos'].append(dircos)
            elif read_string == '':
                break
        return coord
                
def Modgen_AESURF(filename):    
    aesurf = {'ID':[],
              'key':[],
              'CID':[],
              'AELIST':[],
              'eff':[],
             }
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    for line in lines:
        if string.find(line, 'AESURF') !=-1 and line[0] != '$':
            # extract information from AESURF card
            aesurf['ID'].append(nastran_number_converter(line[8:16], 'int'))
            aesurf['key'].append(string.replace(line[16:24], ' ', ''))
            aesurf['CID'].append(nastran_number_converter(line[24:32], 'int'))
            aesurf['AELIST'].append(nastran_number_converter(line[32:40], 'int'))
            aesurf['eff'].append(nastran_number_converter(line[56:64], 'float'))
    return aesurf

def Nastran_AEFACT(filename):
    # AEFACTs have the same nomenklatur as SET1s
    # Thus, reuse the Nastran_SET1() function with a different keyword
    # However, AEFACTs are mostly used with float numbers.
    return Nastran_SET1(filename, keyword='AEFACT', type='float')

def Modgen_AELIST(filename):
    # AELISTs have the same nomenklatur as SET1s
    # Thus, reuse the Nastran_SET1() function with a different keyword
    return Nastran_SET1(filename, keyword='AELIST')

def Nastran_SET1(filename, keyword='SET1', type='int'):
    
    sets = {'ID':[], 'values':[]}
    next_line = False
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
#             if string.find(read_string[:8], keyword) !=-1 and string.replace(read_string[24:32], ' ', '') == 'THRU' and read_string[:1] != '$':
#                  # Assumption: the list is defined with the help of THRU and there is only one THRU
#                 startvalue = nastran_number_converter(read_string[16:24], 'int')
#                 stoppvalue = nastran_number_converter(read_string[32:40], 'int')
#                 values = np.arange(startvalue, stoppvalue+1)
#                 sets['ID'].append(nastran_number_converter(read_string[8:16], 'int'))
#                 sets['values'].append(values)
            if string.find(read_string[:8], keyword) !=-1 and read_string[-2:-1] == '+' and read_string[:1] != '$':
                # this is the first line
                row = read_string[8:-2]
                next_line = True
            elif next_line and read_string[:1] == '+' and read_string[-2:-1] == '+':
                # these are the middle lines
                row += read_string[8:-2]
            elif np.all(next_line and read_string[:1] == '+') or np.all(string.find(read_string[:8], keyword) !=-1 and read_string[:1] != '$'):
                if np.all(string.find(read_string[:8], keyword) !=-1 and read_string[:1] != '$'):
                    # this is the first AND the last line, no more IDs to come
                    row = read_string[8:]
                else:
                    # this is the last line, no more IDs to come
                    row += read_string[8:]
                for c in ['\n', '\r']: row = row.strip(c)
                next_line = False
                # start conversion from string to list containing ID values
                sets['ID'].append(nastran_number_converter(row[:8], 'int'))
                row = row[8:]
                
                values = []
                while len(row)>0:
                    if string.replace(row[:8], ' ', '') == 'THRU':
                        startvalue = values[-1]+1
                        stoppvalue = nastran_number_converter(row[8:16], type)
                        values += range(startvalue, stoppvalue+1) 
                        row = row[16:]
                    else:
                        values.append(nastran_number_converter(row[:8], type))
                        row = row[8:]
                sets['values'].append( np.array([x for x in values if x != 0]) )
            if read_string == '':
                break
        return sets
        
def Nastran_AECOMP(filename):
    aecomp = {'name': [], 'list_type':[], 'list_id':[]}
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    for line in lines:
        if string.find(line, 'AECOMP') !=-1 and line[0] != '$':
            # Assumption: only one list is given per AECOMP
            aecomp['name'].append(string.replace(line[8:16], ' ', ''))
            aecomp['list_type'].append(string.replace(line[16:24], ' ', ''))
            aecomp['list_id'].append(nastran_number_converter(line[24:32], 'int'))
    
    return aecomp

def Nastran_MONPNT1(filename):    
    ID = []
    name = []
    label = []
    comp = []
    CP = []
    CD = []
    offset = []
    i_ID = 1
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if string.find(read_string, 'MONPNT1') !=-1 and read_string[0] != '$':
                # extract information from MONPNT1 card
                # erste Zeile
                ID.append(i_ID) # Eigentlich haben MONPNTs keine IDs sondern nur Namen...
                i_ID += 1
                # An dieser Stelle reicht es nicht mehr aus, nur die Leerzeichen zu entfernen...
                name.append(string.join((ch for ch in read_string[8:16] if ch in string.ascii_letters + string.digits + '_'), ''))
                label.append(string.join((ch for ch in read_string[16:72] if ch in string.ascii_letters + string.digits + '_'), ''))
                # zweite Zeile
                read_string = fid.readline()
                comp.append(string.replace(read_string[16:24], ' ', ''))
                CP.append(nastran_number_converter(read_string[24:32], 'int'))
                CD.append(nastran_number_converter(read_string[56:64], 'int'))
                offset.append([nastran_number_converter(read_string[32:40], 'float'), nastran_number_converter(read_string[40:48], 'float'), nastran_number_converter(read_string[48:56], 'float')])
            elif read_string == '':
                break
    
    n = len(ID)
    mongrid = {'ID':np.array(ID),
               'name':name,
               'label':label,
               'comp':comp,
               'CP':np.array(CP),
               'CD':np.array(CD),
               'offset':np.array(offset),
               'set': np.arange(n*6).reshape((n,6)),
               'n':n,
              }
    return mongrid

def Modgen_W2GJ(filename):
    logging.info('Read W2GJ data (correction of camber and twist) from ModGen file: %s' %filename)
    ID = []
    cam_rad = []
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    for line in lines:
        if string.find(line, 'CAM_RAD') !=-1 and line[0] != '$':
            pos_ID = line.split().index('ID-CAE1')
            pos_BOX = line.split().index('ID-BOX')
            pos_CAM_RAD = line.split().index('CAM_RAD')
        elif line[0] != '$':
            # ID of every single aero panel
            ID.append(nastran_number_converter(line.split()[pos_ID], 'int') + nastran_number_converter(line.split()[pos_BOX], 'int') - 1)
            cam_rad.append(nastran_number_converter(line.split()[pos_CAM_RAD], 'float'))

    camber_twist = {'ID': np.array(ID),
                    'cam_rad':np.array(cam_rad),
                   }
    return camber_twist

def Nastran_NodeLocationReport(filename):
    IDs = set()
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if string.find(read_string, 'Node ID') !=-1 and read_string[0] != '$':
                while True:
                    read_string = fid.readline()
                    if read_string.split() != [] and nastran_number_converter(read_string.split()[0], 'ID') != 0:
                        IDs.add(nastran_number_converter(read_string.split()[0], 'ID'))
                    else:
                        break
            elif read_string == '':
                break

        return list(IDs)     
