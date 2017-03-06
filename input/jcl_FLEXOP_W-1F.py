# JCL - Job Control File documentation

import numpy as np
from units import *

class jcl:
    def __init__(self):
        self.general = {'aircraft':'FLEXOP',
                        'b_ref': 1.0,            # reference span width (from tip to tip)
                        'c_ref': 0.45,              # reference chord length
                        'A_ref': 1.0,                # reference area
                        'MAC_ref': [0.0, 0.0, 0.0], # mean aerodynamic center, also used as moments reference point
                       }
        self.efcs = {'version': 'flexop'} # name of the corresponding class in efcs.py
        self.geom = {'method': 'mona',
                     'filename_grid':['/scratch/FLEXOP/na24_FLEXOP_W-1F_aT_aio/na24_FLEXOP_W-1F_aT_aio.nas'],     # bdf file(s) with GRIDs and CORDs (CORD1R and CORD2R)
                     'filename_monpnt': '',   # bdf file(s) with MONPNT-cards
                     # Alternative way to define monitoring stations:
                     'filename_mongrid': '/scratch/FLEXOP/na24_FLEXOP_W-1F_aT_aio/monstations_grids.bdf',   # bdf file with GRID-cards, one monitoring station is created at each GRID point, 1st GRID point -> 1st monstation
                     'filename_moncoord':'/scratch/FLEXOP/na24_FLEXOP_W-1F_aT_aio/monstations_grids.bdf',  # additional CORDs for monitoring stations
                     'filename_monstations': ['/scratch/FLEXOP/na24_FLEXOP_W-1F_aT_aio/MON1.bdf'], # bdf file with GRID-cards, 1st file -> 1st monstation
                     # The following matrices are required for some mass methods. However, they are geometry dependent...
                     'filename_KGG':'',      # unused
                     'filename_KFF':'/scratch/FLEXOP/na24_FLEXOP_W-1F_aT_aio/KAA.dat',      # stiffness matrix KFF via DMAP Alter und OP4 - required for mass method = 'modalanalysis' or 'guyan'
                     'filename_uset': '/scratch/FLEXOP/na24_FLEXOP_W-1F_aT_aio/uset.op2',   # USET via DMAP Alter und OP4                 - required for mass method = 'modalanalysis' or 'guyan'
                     'filename_GM': '/scratch/FLEXOP/na24_FLEXOP_W-1F_aT_aio/GM.dat',       # matrix GM via DMAP Alter und OP4            - required for mass method = 'modalanalysis' or 'guyan'
                     'filename_aset': '',   # bdf file(s) with ASET1-card                 - required for mass method = 'guyan'
                    }
        self.aero = {'method': 'mona_steady', # 'mona_steady' or 'hybrid'
                     'flex': True, # True or False, aerodynamic feedback of elastic structure
                     'method_caero': 'CAERO1',                              # aerogrid is given by CAERO1 or by CQUAD4 cards
                     'filename_caero_bdf': ['/scratch/FLEXOP/na25_FLEXOP_W-1F_aU_aio/na25_FLEXOP_W-1F_aU_aio.nas'],                  # bdf file(s) with CAERO1 or CQUAD4-cards for aerogrid. IDs in ascending order.
                     'filename_deriv_4_W2GJ': [],    # ModGen output for camber and twist correction. Same order as the aerogrid.
                     'filename_aesurf': ['/scratch/FLEXOP/Yasser/V-tail_AESURF_AELIST'],                # bdf file(s) with AESURF-cards
                     'filename_aelist': ['/scratch/FLEXOP/Yasser/V-tail_AESURF_AELIST'],                # bdf file(s) with AELIST-cards
                     'hingeline': 'y',                                      # The hingeline of a CS is given by a CORD. Either the y- or the z-axis is taken as hingeline. 'y', 'z' 
                     'method_AIC': 'vlm', # 'vlm' or 'nastran'
                     'key':['M015'],
                     'Ma': [0.15],
                     'filename_AIC': [],  # provide OP4 files with AICs if method_AIC = 'nastran'
                    }
        self.matrix_aerodb = {} # if aero method = 'hybrid'
        
        self.spline = {'method': 'nearest_neighbour',           # 'nearest_neighbour', 'rbf' or 'nastran'
                       'filename_f06': 'filename.f06',          # spline matrix is written to .f06-file with PARAM    OPGTKG   1
                       # Possibility to use only a subset of the structural grid for splining. Not valid and ignored when spline method = 'nastran'
                       'splinegrid': False,                      # True or False
                       'filename_splinegrid': 'splinegrid.bdf'  # bdf file(s) with GRIDs
                      }
        self.mass = {'method': 'mona', # 'mona', 'modalanalysis' or 'guyan'
                       'key': ['M1'],
                       'filename_MGG':['/scratch/FLEXOP/na24_FLEXOP_W-1F_aT_aio/MGG.dat'],         # MGG via DMAP Alter und OP4 - always required
                       'filename_MFF':['/scratch/FLEXOP/na24_FLEXOP_W-1F_aT_aio/MAA.dat'],         # MFF via DMAP Alter und OP4 - required for 'modalanalysis' and 'guyan'
                       'filename_S103':['/scratch/FLEXOP/na24_FLEXOP_W-1F_aT_aio/na24_FLEXOP_W-1F_aT_aio.f06'],  # eigenvalues and eigenvectors from .f06-file - required for 'mona'
                       'omit_rb_modes': True, # True or False, omits first six modes
                       'modes':[np.arange(1,25)], # list(s) of modes to use 
                      }
        self.atmo = {'method':'ISA', 
                     'key':['FL000','FL055', 'FL075', 'FL200', 'FL300', 'FL450'],
                     'h': ft2m([0, 5500, 7500, 20000, 30000, 45000]), # altitude in meters
                    }
        self.trimcase = [{'desc': 'CC.M1.OM015FL000.test', # description of maneuver case, e.g. according to G. Pinho Chiozzotto, "Kriterien fuer die Erstellung eines Lastenkatalogs," Institute of Aeroelasticity, iLOADs MS1.2, Feb. 2014.
                          'manoeuver': 'pitch',      # unused
                          'subcase': 1,        # ID number
                          'Ma': 0.15,            # Mach number
                          'aero': 'M015',         # aero key
                          'altitude': 'FL000',  # atmo key
                          'mass': 'M1',       # mass key
                          'Nz': 1.0,            # load factor Nz
                          # velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          'p': 0.0/180.0*np.pi,        # roll rate in rad/s
                          'q': 0.0/180.0*np.pi,        # pitch rate in rad/s
                          'r': 0.0,                     # yaw rate in rad/s
                          'pdot': 0.0/180.0*np.pi,   # roll acceleration in rad/s^2
                          'qdot': 0.0,                  # pitch acceleration in rad/s^2
                          'rdot': 0.0,                  # yaw acceleration in rad/s^2
                         },
                        ]
        self.simcase = [{}] # For every trimcase, a corresponding simcase is required. For maneuvers, it may be empty.

        # End