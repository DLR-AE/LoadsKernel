# JCL - Job Control File documentation

import numpy as np
from kernel.units import *

class jcl:
    def __init__(self):
        self.general = {'aircraft':'DLR F-19-S',
                        'b_ref': 15.375,            # reference span width (from tip to tip)
                        'c_ref': 4.79,              # reference chord length
                        'A_ref': 77,                # reference area
                        'MAC_ref': [6.0, 0.0, 0.0], # mean aerodynamic center, also used as moments reference point
                       }
        self.efcs = {'version': 'mephisto'} # name of the corresponding class in efcs.py
        self.geom = {'method': 'mona',
                     'filename_grid':['grids.bdf'],     # bdf file(s) with GRIDs and CORDs (CORD1R and CORD2R)
                     'filename_shell':['shells.bdf'],   # bdf file(s) with CQUADs and CTRIAs, for visualization only, e.g. outer skin on the aircraft
                     'filename_monpnt': 'monpnt.bdf',   # bdf file(s) with MONPNT-cards
                     # Alternative way to define monitoring stations:
                     'filename_mongrid': 'monstations_grids.bdf',   # bdf file with GRID-cards, one monitoring station is created at each GRID point, 1st GRID point -> 1st monstation
                     'filename_moncoord':'monstations_coords.bdf',  # additional CORDs for monitoring stations
                     'filename_monstations': ['monstation_MON1.bdf', 'monstation_MON2.bdf'], # bdf file with GRID-cards, 1st file -> 1st monstation
                     # The following matrices are required for some mass methods. However, they are geometry dependent...
                     'filename_KGG':'KGG.dat',      # KGG via DMAP Alter und OP4                  - required for mass method = 'modalanalysis' or 'guyan'
                     'filename_uset': 'uset.op2',   # USET via DMAP Alter und OP4                 - required for mass method = 'modalanalysis' or 'guyan'
                     'filename_GM': 'GM.dat',       # matrix GM via DMAP Alter und OP4            - required for mass method = 'modalanalysis' or 'guyan'
                     'filename_aset': 'aset.bdf',   # bdf file(s) with ASET1-card                 - required for mass method = 'guyan'
                    }
        self.aero = {'method': 'mona_steady', # 'mona_steady', 'mona_unsteady', 'steady_nonlin' or 'cfd_steady'
                     'flex': True, # True or False, aerodynamic feedback of elastic structure
                     'method_caero': 'CAERO1',                               # aerogrid is given by CAERO1, CAERO7 or by CQUAD4 cards
                     'filename_caero_bdf': ['CAERO1_bdf'],                   # bdf file(s) with CAERO1 or CQUAD4-cards for aerogrid. IDs in ascending order.
                     'filename_deriv_4_W2GJ': ['filename.deriv_4_W2GJ'],     # ModGen output for camber and twist correction. Same order as the aerogrid.
                     # Alternative definition of camber and twist correction :
                     'filename_DMI_W2GJ': [],                                # DMI Matrix for camber and twist correction. Same order as the aerogrid.
                     'filename_aesurf': ['filename.AESURF'],                 # bdf file(s) with AESURF-cards
                     'filename_aelist': ['filename.AELIST'],                 # bdf file(s) with AELIST-cards
                     'hingeline': 'z',                                       # The hingeline of a CS is given by a CORD. Either the y- or the z-axis is taken as hingeline. 'y', 'z' 
                     'method_AIC': 'vlm',                                    # 'vlm' (internal), 'dlm' (octave) or 'nastran' (external form matrices)
                     'key':['VC', 'MC'],
                     'Ma': [0.8, 0.9],
                     'filename_AIC': ['AIC_VC.dat', 'AIC_MC.dat'],           # provide OP4 files with AICs if method_AIC = 'nastran'
                     'k_red': [0.001, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 1.5 ], # reduced frequencies for DLM, Nastran Definition!
                     'n_poles': 4,                                           # number of poles for rational function approximation (RFA)
                     'Cn_beta_corr': [ -0.012],                              # Correction coefficient at CG, negativ = destabilizing
                     'Cm_alpha_corr':[ 0.22],                                # Correction coefficient at CG, positiv = destabilizing
                     'viscous_drag': 'coefficients',                         # Correction coefficient at MAC, Cd = Cd0 + dCd/dalpha^2 * alpha^2
                     'Cd_0': [0.005],
                     'Cd_alpha^2': [0.018*6.28**2.0], 
                     'induced_drag':True,                                    # True or False, calculates local induced drag e.g. for roll-yaw-coupling
                     # Additional parameters for CFD
                     'para_path':'/scratch/tau/',
                     'para_file':'para',
                     'tau_solver': 'el',
                     'tau_cores': 16,
                    }
        self.meshdefo = {'surface':                              # general surface mesh information
                                    {'fileformat': 'netcdf',     # 'cgns', 'netcdf'
                                     'markers': [1,3],           # list of markers [1, 2, ...] of surfaces to be included in deformation
                                     'filename_grid':'tau.grid', # Tau volume or surface mesh or CGNSS surface mesh
                                    },
                         'volume':{},                            # general volume mesh information, unused
                        } 
        self.spline = {'method': 'nearest_neighbour',           # 'nearest_neighbour', 'rbf' or 'nastran'
                       'filename_f06': 'filename.f06',          # spline matrix is written to .f06-file with PARAM    OPGTKG   1
                       # Possibility to use only a subset of the structural grid for splining. Not valid and ignored when spline method = 'nastran'
                       'splinegrid': True,                      # True or False
                       'filename_splinegrid': ['splinegrid.bdf']  # bdf file(s) with GRIDs
                      }
        self.mass =    {'method': 'mona', # 'mona', 'CoFE', 'modalanalysis' or 'guyan'
                        'key': ['M1', 'M2'],
                        'filename_MGG':['MGG_M1.dat', 'MGG_M2.dat'],         # MGG via DMAP Alter und OP4 - always required
                        'filename_S103':['SOL103_M1.f06', 'SOL103_M1.f06'],  # eigenvalues and eigenvectors from .f06-file - required for 'mona'
                        'filename_CoFE':['M1.mat', 'M2.mat'],  # eigenvalues and eigenvectors from .f06-file - required for 'mona'
                        'omit_rb_modes': False, # True or False, omits first six modes
                        'modes':[np.arange(1,13), np.arange(1,16)], # list(s) of modes to use 
                       }
        self.damping = {'method': 'modal',
                        'damping': 0.02,
                       }
        self.atmo = {'method':'ISA', 
                     'key':['FL000','FL055', 'FL075', 'FL200', 'FL300', 'FL450'],
                     'h': ft2m([0, 5500, 7500, 20000, 30000, 45000]), # altitude in meters
                    }
        self.eom = {'version': 'linear'} # 'linear, 'waszak'
        # parameters for generic landing gear, see PhD Thesis of Wolf Krueger and Sunpeth Cumnuantip
        para_LG = {'stroke_length':  0.3,   # m
                    'fitting_length': 0.72, # m
                    'n':  1.4,
                    'ck': 1.0,
                    'd2':  85000.0,         # N/(m/s)^2
                    'F_static': 53326.0,    # N
                    'r_tire': 0.28,         # m
                    'c1_tire': 832000.0,    # N/m
                    'd1_tire': 4500.0,      # N/(m/s)^2
                    'm_tire': 58.0,         # kg
                  }
        self.landinggear= {'method': 'generic',                         # activates a generic landing gear model during time simulation
                           'key': ['MLG1', 'MLG2', 'NLG' ],
                           'attachment_point':[800002, 800003, 800001], # IDs of FE attachment nodes
                           'para': [para_LG, para_LG, para_LG],         # parameters for generic landing gear module, see above
                          }
        self.trimcase = [{'desc': 'CC.BFDM.OVCFL000.Vergleichsfall53', # description of maneuver case, e.g. according to G. Pinho Chiozzotto, "Kriterien fuer die Erstellung eines Lastenkatalogs," Institute of Aeroelasticity, iLOADs MS1.2, Feb. 2014.
                          'manoeuver': '',      # unused
                          'subcase': 53,        # ID number
                          'Ma': 0.8,            # Mach number
                          'aero': 'VC',         # aero key
                          'altitude': 'FL000',  # atmo key
                          'mass': 'BFDM',       # mass key
                          'Nz': 5.0,            # load factor Nz
                          # velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          'p': 34.3/180.0*np.pi,        # roll rate in rad/s
                          'q': 28.6/180.0*np.pi,        # pitch rate in rad/s
                          'r': 0.0,                     # yaw rate in rad/s
                          'pdot': -286.5/180.0*np.pi,   # roll acceleration in rad/s^2
                          'qdot': 0.0,                  # pitch acceleration in rad/s^2
                          'rdot': 0.0,                  # yaw acceleration in rad/s^2
                          'support': [0,1,2,3,4,5]      # list of DoF to be constrained
                         },
                        ]
        self.simcase = [{}] # For every trimcase, a corresponding simcase is required. For maneuvers, it may be empty.
        # a time simulation is triggered if the simcase contains at least 'dt' and 't_final'
        self.simcase = [{'dt': 0.01,            # time step size in [s]
                         't_final': 2.0,        # final simulation time  in [s]
                         'gust': True,          # True or False, enables 1-cosine gust according to CS-25
                         'gust_gradient': 9.0,  # gust gradient H in [m]
                         'gust_orientation': 0, # orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top
                         'gust_para':{'Z_mo': 12500.0, 'MLW': 65949.0, 'MTOW': 73365.0, 'MZFW': 62962.0, 'MD': 0.87, 'T1': 0.00}, # gust parameters according to CS-25
                         'cs_signal': False,    # True or False, allows playback of control surface signals via efcs
                         'controller': False,   # True or False, enables a generic controller e.g. to maintain p, q and r
                         'landinggear':False,   # True or False, enables a generic landing gear
                         'support': [0,1,2,3,4,5]      # list of DoF to be constrained
                        },
                       ] 
        # End