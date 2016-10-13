# JCL - Job Control File documentation

import numpy as np
from units import *

class jcl:
    def __init__(self):
        self.general = {'aircraft':'MULDICON',
                        'b_ref': 7.69,            # reference span width (from tip to tip)
                        'c_ref': 6.0,              # reference chord length
                        'A_ref': 77.8,                # reference area
                        'MAC_ref': [6.0, 0.0, 0.0], # mean aerodynamic center, also used as moments reference point
                       }
        self.efcs = {'version': 'mephisto'} # name of the corresponding class in efcs.py
        self.geom = {'method': 'mona',
                     'filename_grid':['/work/voss_ar/MULDICON_Model/mg01_Muldicon_rechts/output/Muldicon-baseline.bdf', 
                                      '/work/voss_ar/MULDICON_Model/mg02_Muldicon_links/output/Muldicon-baseline.bdf', 
                                      '/work/voss_ar/MULDICON_Model/mg03_Muldicon_Klappen/output/Klappe_innen-baseline.bdf', 
                                      '/work/voss_ar/MULDICON_Model/mg03_Muldicon_Klappen/output/Klappe_aussen-baseline.bdf',
                                      '/work/voss_ar/MULDICON_Model/mg04_Muldicon_Klappen_links/output/Klappe_innen-baseline.bdf',
                                      '/work/voss_ar/MULDICON_Model/mg04_Muldicon_Klappen_links/output/Klappe_aussen-baseline.bdf',
                                      '/work/voss_ar/MULDICON_Model/na21_Assembly/Assembly_Klappen.RBE2',
                                      '/work/voss_ar/MULDICON_Model/mg01_Muldicon_rechts/output/Muldicon-baseline.RBE3_SUBSEG2',
                                      '/work/voss_ar/MULDICON_Model/mg02_Muldicon_links/output/Muldicon-baseline.RBE3_SUBSEG2',
                                    ],     # bdf file(s) with GRIDs and CORDs (CORD1R and CORD2R)
                     'filename_monpnt': '',   # bdf file(s) with MONPNT-cards
                     # Alternative way to define monitoring stations:
                     'filename_mongrid': '/work/voss_ar/MULDICON_Model/monstations/monstations_grids.bdf',   # bdf file with GRID-cards, one monitoring station is created at each GRID point, 1st GRID point -> 1st monstation
                     'filename_moncoord':'/work/voss_ar/MULDICON_Model/monstations/monstations_grids.bdf',  # additional CORDs for monitoring stations
                     'filename_monstations': ['/work/voss_ar/MULDICON_Model/monstations/monstations_MON09.bdf'], # bdf file with GRID-cards, 1st file -> 1st monstation
                     # The following matrices are required for some mass methods. However, they are geometry dependent...
                     'filename_KGG':'',      # unused
                     'filename_KFF':'/work/voss_ar/MULDICON_Model/na22_SOL103/nastran/KAA.dat',      # stiffness matrix KFF via DMAP Alter und OP4 - required for mass method = 'modalanalysis' or 'guyan'
                     'filename_uset': '/work/voss_ar/MULDICON_Model/na22_SOL103/nastran/uset.op2',   # USET via DMAP Alter und OP4                 - required for mass method = 'modalanalysis' or 'guyan'
                     'filename_GM': '/work/voss_ar/MULDICON_Model/na22_SOL103/nastran/GM.dat',       # matrix GM via DMAP Alter und OP4            - required for mass method = 'modalanalysis' or 'guyan'
                     'filename_aset': '',   # bdf file(s) with ASET1-card                 - required for mass method = 'guyan'
                    }
        self.aero = {'method': 'mona_steady', # 'mona_steady' or 'hybrid'
                     'flex': False, # True or False, aerodynamic feedback of elastic structure
                     'method_caero': 'CQUAD4',                              # aerogrid is given by CAERO1 or by CQUAD4 cards
                     'filename_caero_bdf': ['/work/voss_ar/MULDICON_Model/mg01_Muldicon_rechts/output/Muldicon-baseline.CAERO1_bdf',
                                            '/work/voss_ar/MULDICON_Model/mg02_Muldicon_links/output/Muldicon-baseline.CAERO1_bdf'],                  # bdf file(s) with CAERO1 or CQUAD4-cards for aerogrid. IDs in ascending order.
                     'filename_deriv_4_W2GJ': ['/work/voss_ar/MULDICON_Model/mg01_Muldicon_rechts/output/Muldicon-baseline.deriv_4_W2GJ_mod',
                                               '/work/voss_ar/MULDICON_Model/mg02_Muldicon_links/output/Muldicon-baseline.deriv_4_W2GJ_mod'],    # ModGen output for camber and twist correction. Same order as the aerogrid.
                     'filename_aesurf': ['/work/voss_ar/MULDICON_Model/mg01_Muldicon_rechts/output/Muldicon-baseline.AESURF',
                                         '/work/voss_ar/MULDICON_Model/mg02_Muldicon_links/output/Muldicon-baseline.AESURF'],                # bdf file(s) with AESURF-cards
                     'filename_aelist': ['/work/voss_ar/MULDICON_Model/mg01_Muldicon_rechts/output/Muldicon-baseline.AELIST',
                                         '/work/voss_ar/MULDICON_Model/mg02_Muldicon_links/output/Muldicon-baseline.AELIST'],                # bdf file(s) with AELIST-cards
                     'hingeline': 'y',                                      # The hingeline of a CS is given by a CORD. Either the y- or the z-axis is taken as hingeline. 'y', 'z' 
                     'method_AIC': 'vlm', # 'vlm' or 'nastran'
                     'key':['VC', 'MC', 'VD', 'MD'],
                     'Ma': [0.8, 0.9, 0.89, 0.97],
                     'filename_AIC': [],  # provide OP4 files with AICs if method_AIC = 'nastran'
                    }
        self.matrix_aerodb = {} # if aero method = 'hybrid'
        
        self.spline = {'method': 'nearest_neighbour',           # 'nearest_neighbour', 'rbf' or 'nastran'
                       'filename_f06': '',          # spline matrix is written to .f06-file with PARAM    OPGTKG   1
                       # Possibility to use only a subset of the structural grid for splining. Not valid and ignored when spline method = 'nastran'
                       'splinegrid': False,                      # True or False
                       'filename_splinegrid': ''  # bdf file(s) with GRIDs
                      }
        self.mass = {'method': 'modalanalysis', # 'mona', 'modalanalysis' or 'guyan'
                       'key': ['baseline'],
                       'filename_MGG':['/work/voss_ar/MULDICON_Model/na22_SOL103/nastran/MGG.dat'],         # MGG via DMAP Alter und OP4 - always required
                       'filename_MFF':['/work/voss_ar/MULDICON_Model/na22_SOL103/nastran/MAA.dat'],         # MFF via DMAP Alter und OP4 - required for 'modalanalysis' and 'guyan'
                       'filename_S103':[],  # eigenvalues and eigenvectors from .f06-file - required for 'mona'
                       'omit_rb_modes': True, # True or False, omits first six modes
                       'modes':[np.arange(1,11)], # list(s) of modes to use 
                      }
        self.atmo = {'method':'ISA', 
                     'key':['FL000','FL055', 'FL075', 'FL200', 'FL300', 'FL450'],
                     'h': ft2m([0, 5500, 7500, 20000, 30000, 45000]), # altitude in meters
                    }
#         self.trimcase = [{'desc': 'CC.baseline.OVCFL000.testcase', # description of maneuver case, e.g. according to G. Pinho Chiozzotto, "Kriterien fuer die Erstellung eines Lastenkatalogs," Institute of Aeroelasticity, iLOADs MS1.2, Feb. 2014.
#                           'manoeuver': 'pitch&roll',      # unused
#                           'subcase': 1,        # ID number
#                           'Ma': 0.8,            # Mach number
#                           'aero': 'VC',         # aero key
#                           'altitude': 'FL000',  # atmo key
#                           'mass': 'baseline',       # mass key
#                           'Nz': 1.0,            # load factor Nz
#                           # velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
#                           'p': 0.0/180.0*np.pi,        # roll rate in rad/s
#                           'q': 0.0/180.0*np.pi,        # pitch rate in rad/s
#                           'r': 0.0,                     # yaw rate in rad/s
#                           'pdot': 0.0/180.0*np.pi,   # roll acceleration in rad/s^2
#                           'qdot': 0.0,                  # pitch acceleration in rad/s^2
#                           'rdot': 0.0,                  # yaw acceleration in rad/s^2
#                          },
#                         ]
#         self.simcase = [{}] # under development
        
        from numpy import array
        with open('/work/voss_ar/MULDICON_Model/loadcases/trim.trimcase_dict', 'r') as fid:
            trimcase_str = fid.read()
        self.trimcase = eval(trimcase_str)
        # generate empty simcases
        self.simcase = []
        for i in range(len(self.trimcase)):  self.simcase.append({})
        # End