# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:29:57 2014

@author: voss_ar
"""
import string
import numpy as np
import scipy.sparse as sp
import math as math
import os

def NASTRAN_f06_modal(filename, modes_selected='all', omitt_rigid_body_modes=False):
    '''
    This methode parses a NASTRAN .f06 file and searches for eigenvalues and
    eigenvectors. (Basierend auf Script von Markus.)
    '''
    
    print 'Read modal data from f06 file: %s' %filename
    filesize = float(os.stat(filename).st_size)
    percent = 0.
    
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
    print 'Found %i eigenvalues and %i eigenvectors for %i nodes.' %(len(eigenvalues["ModeNo"]), len(eigenvectors.keys()), len(node_ids)/len(eigenvalues["ModeNo"]))

    return  eigenvalues, eigenvectors, node_ids
    
def reduce_modes(eigenvalues, eigenvectors, nodes_selection, modes_selection):
    print 'Reduction of data to %i selected modes and %i nodes.' %(len(modes_selection), len(nodes_selection))
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
    print ' - working on nodes...'
    for i_nodes in range(len(nodes_selection)):
        pos_eigenvector.append( np.where(nodes_selection[i_nodes] ==  nodes_eigenvector)[0][0] )
    
    print ' - working on modes...'
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
    print 'Read Weight data from f06 file: %s' %filename
    
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
    print 'Read GRID data from ModGen file: %s' %filename
    grids = []
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if string.find(read_string, 'GRID') !=-1 and read_string[0] != '$':
                if string.find(read_string[48:56], '\n') != -1: # if CD is missing, fix with CP
                    read_string = read_string[:48] + read_string[16:24]
                grids.append([nastran_number_converter(read_string[8:16], 'ID'), nastran_number_converter(read_string[16:24], 'CP'), nastran_number_converter(read_string[24:32], 'float'), nastran_number_converter(read_string[32:40], 'float'), nastran_number_converter(read_string[40:48], 'float'), nastran_number_converter(read_string[48:56], 'CD')])
            elif read_string == '':
                break
    grids = np.array(grids)
    n = len(grids[:,0])
    grid = {"ID": grids[:,0],
            "offset":grids[:,2:5],
            "n": n,
            "CP": grids[:,1],
            "CD": grids[:,5],
            "set": np.arange(n*6).reshape((n,6)),
           }
    return grid

def Modgen_CQUAD4(filename):
    print 'Read CQUAD4 data from ModGen file: %s' %filename
    data = []
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if string.find(read_string, 'CQUAD4') !=-1 and read_string[0] != '$':
                data.append([nastran_number_converter(read_string[8:16], 'ID'), nastran_number_converter(read_string[24:32], 'ID'), nastran_number_converter(read_string[32:40], 'ID'), nastran_number_converter(read_string[40:48], 'ID'), nastran_number_converter(read_string[48:56], 'ID')])
            elif read_string == '':
                break
    data = np.array(data)
    panels = {"ID": data[:,0],
              "cornerpoints": data[:,1:5],
             }
    return panels
    
    
def nastran_number_converter(string_in, type, default=0):
    if type in ['float']:
        try:
            out = float(string_in)
        except:
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
            else: 
                print "ERROR: could not interprete the following number: " + string_in
                return
    elif type in ['int', 'ID', 'CD', 'CP']:
        try:
            out = int(string_in)
        except:
            out = int(default)  
    return out
    
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
    print 'Read data from OP4 file: %s with %.2f MB' %(filename, filesize/1024**2)
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
        if nastran_number_converter(read_string[24:32], 'int') == 2:
            type_real = True
            type_complex = False
            data = sp.lil_matrix((n_col, n_row), dtype=float)
        elif nastran_number_converter(read_string[24:32], 'int') == 4:
            type_real = False
            type_complex = True
            data = sp.lil_matrix((n_col, n_row), dtype=complex)
            
        else:
            print 'Unknown format: ' + read_string[24:32] 
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
                    i_col = nastran_number_converter(read_string[0:8], 'int') - 1
                    n_words = nastran_number_converter(read_string[16:24], 'int')
                    while n_words > 0:
                        
                        read_string = fid.readline()
                        L = np.int(nastran_number_converter(read_string[0:16], 'int')/65536 - 1)
                        n_words -= L+1
                        i_row = nastran_number_converter(read_string[0:16], 'int') - 65536*(L+1) -1
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
                    print read_string
                    print 'Found CORD1R card, but no grid is given. Coord is ignored.'
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
        while True:
            read_string = fid.readline()
            if string.find(read_string, 'AESURF') !=-1 and read_string[0] != '$':
                # extract information from AESURF card
                aesurf['ID'].append(nastran_number_converter(read_string[8:16], 'int'))
                aesurf['key'].append(string.replace(read_string[16:24], ' ', ''))
                aesurf['CID'].append(nastran_number_converter(read_string[24:32], 'int'))
                aesurf['AELIST'].append(nastran_number_converter(read_string[32:40], 'int'))
                aesurf['eff'].append(nastran_number_converter(read_string[56:64], 'float'))
            elif read_string == '':
                break
        return aesurf

def Modgen_AELIST(filename):
    aelist = {'ID': [], 'values':[]}
    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if string.find(read_string, 'AELIST') !=-1 and read_string[0] != '$':
                if string.replace(read_string[24:32], ' ', '') == 'THRU' and read_string[-2] != '+':
                    # Assumption: the list is defined with the help of THRU and there is only one THRU
                    startvalue = nastran_number_converter(read_string[16:24], 'int')
                    stoppvalue = nastran_number_converter(read_string[32:40], 'int')
                    values = np.arange(startvalue, stoppvalue+1)
                    aelist['ID'].append(nastran_number_converter(read_string[8:16], 'int'))
                    aelist['values'].append(values)
                else:
                    print 'Notation of AELIST with single values not yet implemented!'
                    return
            elif read_string == '':
                break
        return aelist

def Modgen_SET1(filename):
    # Assumption: only one SET1 card per file

    with open(filename, 'r') as fid:
        while True:
            read_string = fid.readline()
            if string.find(read_string, 'SET1') !=-1 and read_string[0] != '$':
                row = read_string[16:-2]
            elif read_string[0] == '+' and read_string[-2:-1] == '+':
                row += read_string[8:-2]
            elif read_string[0] == '+':
                row += read_string[8:-1]
                break
        IDs = []
        while len(row)>0:
            IDs.append(nastran_number_converter(row[:8], 'int'))
            row = row[8:]
        return np.array(IDs)


def Modgen_W2GJ(filename):
    print 'Read W2GJ data (correction of camber and twist) from ModGen file: %s' %filename
    ID = []
    cam_rad = []
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    for line in lines:
        if string.find(line, 'CAM_RAD') !=-1 and line[0] != '$':
            pos_ID = line.split().index('ID-CAE1')
            pos_CAM_RAD = line.split().index('CAM_RAD')
        elif line[0] != '$':
            ID.append(nastran_number_converter(line.split()[pos_ID], 'int'))
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
