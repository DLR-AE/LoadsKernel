# JCL - Job Control File documentation

import numpy as np
from units import *

class jcl:
    def __init__(self):
        self.general = {'aircraft':'Discus2c',
                        'b_ref': 18.0,            # reference span width (from tip to tip)
                        'c_ref': 0.685,              # reference chord length
                        'A_ref': 11.39,                # reference area
                        'MAC_ref': [0.26211, 0.0, 0.0], # mean aerodynamic center, also used as moments reference point
                       }
        self.efcs = {'version': 'discus2c'} # name of the corresponding class in efcs.py
        self.geom = {'method': 'mona',
                     'filename_grid':['/scratch/Discus2c/Discus_iLOADs_m4.bdf', 
                                      '/scratch/Discus2c/Discus_iLOADs_acc.bdf'],     # bdf file(s) with GRIDs and CORDs (CORD1R and CORD2R)
                     'filename_monpnt': '',   # bdf file(s) with MONPNT-cards
                     # Alternative way to define monitoring stations:
                     'filename_mongrid': '/scratch/Discus2c/Discus_iLOADs_mongrid.bdf',   # bdf file with GRID-cards, one monitoring station is created at each GRID point, 1st GRID point -> 1st monstation
                     'filename_moncoord':'/scratch/Discus2c/Discus_iLOADs_loadmod.bdf',  # additional CORDs for monitoring stations
                     'filename_monstations': ['/scratch/Discus2c/WR6.bdf',
                                              '/scratch/Discus2c/WR4.bdf',
                                              '/scratch/Discus2c/WR1.bdf',
                                              '/scratch/Discus2c/WL6.bdf',
                                              '/scratch/Discus2c/WL4.bdf',
                                              '/scratch/Discus2c/WL1.bdf',
                                              '/scratch/Discus2c/HTR8.bdf',
                                              '/scratch/Discus2c/HTR693.bdf',
                                              '/scratch/Discus2c/HTL1.bdf',
                                              '/scratch/Discus2c/FUS2.bdf',], # bdf file with GRID-cards, 1st file -> 1st monstation
                     # The following matrices are required for some mass methods. However, they are geometry dependent...
                     'filename_KGG':'/scratch/Discus2c/nastran/KGG_pilot1.dat',      # unused
                     'filename_KFF':'/scratch/Discus2c/nastran/KAA_pilot1.dat',      # stiffness matrix KFF via DMAP Alter und OP4 - required for mass method = 'modalanalysis' or 'guyan'
                     'filename_uset': '/scratch/Discus2c/nastran/uset_pilot1.op2',   # USET via DMAP Alter und OP4                 - required for mass method = 'modalanalysis' or 'guyan'
                     'filename_GM': '/scratch/Discus2c/nastran/GM_pilot1.dat',       # matrix GM via DMAP Alter und OP4            - required for mass method = 'modalanalysis' or 'guyan'
                     'filename_aset': '',   # bdf file(s) with ASET1-card                 - required for mass method = 'guyan'
                    }
        self.aero = {'method': 'mona_steady', # 'mona_steady' or 'hybrid'
                     'flex': True, # True or False, aerodynamic feedback of elastic structure
                     'method_caero': 'CAERO7',                              # aerogrid is given by CAERO1 or by CQUAD4 cards
                     'filename_caero_bdf': ['/scratch/Discus2c/Discus_iLOADs_ZAERO_m4_no_fuselage2.bdf'],                  # bdf file(s) with CAERO1 or CQUAD4-cards for aerogrid. IDs in ascending order.
                     'filename_deriv_4_W2GJ': [],    # ModGen output for camber and twist correction. Same order as the aerogrid.
                     'filename_aesurf': ['/scratch/Discus2c/Discus_iLOADs_cs2.bdf'],                # bdf file(s) with AESURF-cards
                     'filename_aelist': ['/scratch/Discus2c/Discus_iLOADs_cs2.bdf'],                # bdf file(s) with AELIST-cards
                     'hingeline': 'z', # 'y', 'z' 
                     'method_AIC': 'vlm', # 'vlm' or 'nastran'
                     'key':['VC'],
                     'Ma': [0.15],
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
                       'key': ['Pilot1', 'Pilot2'],
                       'filename_MGG':['/scratch/Discus2c/nastran/MGG_pilot1.dat', '/scratch/Discus2c/nastran/MGG_pilot2.dat'],         # MGG via DMAP Alter und OP4 - always required
                       'filename_MFF':['/scratch/Discus2c/nastran/MAA_pilot1.dat', '/scratch/Discus2c/nastran/MAA_pilot2.dat'],         # MFF via DMAP Alter und OP4 - required for 'modalanalysis' and 'guyan'
                       'filename_S103':['/scratch/Discus2c/nastran/SOL103_Pilot1.f06', '/scratch/Discus2c/nastran/SOL103_Pilot2.f06'],  # eigenvalues and eigenvectors from .f06-file - required for 'mona'
                       'omit_rb_modes': True, # True or False, omits first six modes
                       'modes':[np.arange(1,31), np.arange(1,31)], # list(s) of modes to use 
                      }
        self.atmo = {'method':'ISA', 
                     'key':['FL000', 'FT-P1', 'FT-P2', 'FT-P3', 'FT-P4', 'FT-P5', 'FT-P6', 'FT-P7', 'FT-P8', 'Elev3211_A', 'Elev3211_B', 'B2B_1-10A', 'B2B_1-10B'],
                     'h':  [0,        2901.4,  2657.31, 2679.31, 925.812, 2310.98, 1315.54, 1111.68, 2045.62,   1782.,      1765.,        1151.,         1120.], # altitude in meters
                    }
        self.trimcase = [
                         {'desc': 'Elev3211_A', # description of maneuver case, e.g. according to G. Pinho Chiozzotto, "Kriterien fuer die Erstellung eines Lastenkatalogs," Institute of Aeroelasticity, iLOADs MS1.2, Feb. 2014.
                         'manoeuver': 'segelflug',      # unused
                         'subcase': 1,        # ID number
                         'Ma': 0.1174,            # Mach number
                         'aero': 'VC',         # aero key
                         'altitude': 'Elev3211_A',  # atmo key
                         'mass': 'Pilot1',       # mass key
                         'Nz': 1.0,            # load factor Nz
                         # velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                         'p': 0.0,        # roll rate in rad/s
                         'q': 0.0,        # pitch rate in rad/s
                         'r': 0.0,
                         'pdot': 0.0,     # roll acceleration in rad/s^2
                         'qdot': 0.0,     # pitch acceleration in rad/s^2
                         'rdot': 0.0,
                        },
                        {'desc': 'Elev3211_B', # description of maneuver case, e.g. according to G. Pinho Chiozzotto, "Kriterien fuer die Erstellung eines Lastenkatalogs," Institute of Aeroelasticity, iLOADs MS1.2, Feb. 2014.
                         'manoeuver': 'segelflug',      # unused
                         'subcase': 2,        # ID number
                         'Ma': 0.1138,            # Mach number
                         'aero': 'VC',         # aero key
                         'altitude': 'Elev3211_B',  # atmo key
                         'mass': 'Pilot1',       # mass key
                         'Nz': 1.0,            # load factor Nz
                         # velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                         'p': 0.0,        # roll rate in rad/s
                         'q': 0.0,        # pitch rate in rad/s
                         'r': 0.0,
                         'pdot': 0.0,     # roll acceleration in rad/s^2
                         'qdot': 0.0,     # pitch acceleration in rad/s^2
                         'rdot': 0.0,
                        },
                        {'desc': 'B2B_1-10A', # description of maneuver case, e.g. according to G. Pinho Chiozzotto, "Kriterien fuer die Erstellung eines Lastenkatalogs," Institute of Aeroelasticity, iLOADs MS1.2, Feb. 2014.
                         'manoeuver': 'segelflug',      # unused
                         'subcase': 3,        # ID number
                         'Ma': 0.1135,            # Mach number
                         'aero': 'VC',         # aero key
                         'altitude': 'B2B_1-10A',  # atmo key
                         'mass': 'Pilot2',       # mass key
                         'Nz': 1.0,            # load factor Nz
                         # velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                         'p': 0.0,        # roll rate in rad/s
                         'q': 0.0,        # pitch rate in rad/s
                         'r': 0.0,
                         'pdot': 0.0,     # roll acceleration in rad/s^2
                         'qdot': 0.0,     # pitch acceleration in rad/s^2
                         'rdot': 0.0,
                        },
                        {'desc': 'B2B_1-10B', # description of maneuver case, e.g. according to G. Pinho Chiozzotto, "Kriterien fuer die Erstellung eines Lastenkatalogs," Institute of Aeroelasticity, iLOADs MS1.2, Feb. 2014.
                         'manoeuver': 'segelflug',      # unused
                         'subcase': 4,        # ID number
                         'Ma': 0.1108,            # Mach number
                         'aero': 'VC',         # aero key
                         'altitude': 'B2B_1-10B',  # atmo key
                         'mass': 'Pilot2',       # mass key
                         'Nz': 1.0,            # load factor Nz
                         # velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                         'p': 0.0,        # roll rate in rad/s
                         'q': 0.0,        # pitch rate in rad/s
                         'r': 0.0,
                         'pdot': 0.0,     # roll acceleration in rad/s^2
                         'qdot': 0.0,     # pitch acceleration in rad/s^2
                         'rdot': 0.0,
                        },
                        ]
        self.simcase = [
                        {'dt': 0.05, 
                         't_final': 6.99,
                         'gust': False,
                         'cs_signal': True,
                        },
                        {'dt': 0.05, 
                         't_final': 7.99,
                         'gust': False,
                         'cs_signal': True,
                        },
                        {'dt': 0.05, 
                         't_final': 6.49,
                         'gust': False,
                         'cs_signal': True,
                        },
                        {'dt': 0.05, 
                         't_final': 6.99,
                         'gust': False,
                         'cs_signal': True,
                        },
                        ]
#         self.simcase = [{}, {}]
        # End