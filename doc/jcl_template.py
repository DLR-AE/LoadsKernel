# JCL - Job Control File documentation

import numpy as np
from loadskernel.units import *

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
                     'filename_KGG':'KGG.dat',          # KGG via DMAP Alter und OP4                  - required for mass method = 'modalanalysis', 'guyan' or 'B2000'
                     'filename_uset': 'uset.op2',       # USET via DMAP Alter und OP4                 - required for mass method = 'modalanalysis', 'guyan' 
                     'filename_GM': 'GM.dat',           # matrix GM via DMAP Alter und OP4            - required for mass method = 'modalanalysis', 'guyan'
                     'filename_aset': 'aset.bdf',       # bdf file(s) with ASET1-card                 - required for mass method = 'guyan'
                     'filename_Rtrans': 'Rtrans.csv',   # matrix R_trans frum B2000                   - required for mass method = 'B2000'
                    }
        self.aero = {'method': 'mona_steady', # 'mona_steady', 'mona_unsteady', 'nonlin_steady', 'cfd_steady' or 'freq_dom'
                     'flex': True, # True or False, aerodynamic feedback of elastic structure
                     'method_caero': 'CAERO1',                               # aerogrid is given by CAERO1, CAERO7 or by CQUAD4 cards
                     'filename_caero_bdf': ['CAERO1_bdf'],                   # bdf file(s) with CAERO1 or CQUAD4-cards for aerogrid. IDs in ascending order.
                     'filename_deriv_4_W2GJ': ['filename.deriv_4_W2GJ'],     # ModGen output for camber and twist correction. Same order as the aerogrid.
                     # Alternative definition of camber and twist correction :
                     'filename_DMI_W2GJ': [],                                # DMI Matrix for camber and twist correction. Same order as the aerogrid.
                     'filename_aesurf': ['filename.AESURF'],                 # bdf file(s) with AESURF-cards
                     'filename_aelist': ['filename.AELIST'],                 # bdf file(s) with AELIST-cards
                     'hingeline': 'z',                                       # The hingeline of a CS is given by a CORD. Either the y- or the z-axis is taken as hingeline. 'y', 'z' 
                     'method_AIC': 'vlm',                                    # 'vlm' (panel-aero), 'dlm' (panel-aero) or 'nastran' (external form matrices)
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
                     'induced_drag': False,                                    # True or False, calculates local induced drag e.g. for roll-yaw-coupling
                     # Additional parameters for CFD
                     'para_path':'/scratch/tau/',
                     'para_file':'para',
                     'cfd_solver': 'tau',                                   # 'tau' or 'su2'
                     'tau_solver': 'el',
                     'tau_cores': 16,
                    }
        self.meshdefo = {'surface':                              # general surface mesh information
                                    {'fileformat': 'netcdf',     # 'cgns', 'netcdf', 'su2'
                                     'markers': [1,3],           # list of markers [1, 2, ...] or ['upper', 'lower', ...] of surfaces to be included in deformation
                                     'filename_grid':'tau.grid', # CFD mesh
                                    },
                         'volume':{},                            # general volume mesh information, unused
                        } 
        self.spline = {'method': 'nearest_neighbour',           # 'nearest_neighbour', 'rbf' or 'nastran'
                       'filename_f06': 'filename.f06',          # spline matrix is written to .f06-file with PARAM    OPGTKG   1
                       # Possibility to use only a subset of the structural grid for splining. Not valid and ignored when spline method = 'nastran'
                       'splinegrid': True,                      # True or False
                       'filename_splinegrid': ['splinegrid.bdf']  # bdf file(s) with GRIDs
                      }
        self.mass =    {'method': 'modalanalysis', # 'f06', 'modalanalysis', 'guyan', 'CoFE', 'B2000'
                        'key': ['M1', 'M2'],
                        'filename_MGG':['MGG_M1.dat', 'MGG_M2.dat'],         # MGG via DMAP Alter and OP4 - always required
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
        self.engine= {'method': 'thrust_only', # activates an engine model: 'thrust_only', 'propellerdisk', 'PyPropMat' or 'VLM4Prop'
                      # Note: 'PyPropMAt' and 'VLM4Prop' require sensors with the same key to measure the local onflow.
                      'key': ['E-P', 'E-S' ],
                      'attachment_point':[54100003, 64100003], # IDs of FE attachment nodes
                      #'design_thrust': [47.0, 47.0],    # N
                      'thrust_vector':   [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], # body coordinate system
                      'rotation_vector': [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], # body coordinate system
                      'rotation_inertia': [0.18, 0.18], # Nms^2 
                      'diameter': [1.7, 1.7],           # propeller area in m^2
                      'n_blades': [2, 2],               # number of blades
                      'Ma': [0.25],                     # Mach number for VLM4Prop
                      'propeller_input_file':'HAP_O6_PROP_pitch.yaml', # Input-file ('.yaml') for PyPropMAt and VLM4Prop
                     }
        self.sensor= {'key': ['wind', 'E-P', 'E-S' ], # In case a wind sensor is specified here, this sensor is used to measure alpha and beta.
                      'attachment_point':[200013, 54100003, 64100003], # IDs of FE attachment nodes
                      }
        # This section controls the automatic plotting and selection of dimensioning load cases. 
        # Simply put a list of names of the monitoring stations (e.g. ['MON1', 'MON2',...]) into the dictionary 
        # of possible load plots listed below. This will generate a pdf document and nastran force and moment 
        # cards for the dimensioning load cases. 
        self.loadplots = {
                          'potatos_fz_mx': ['MON5'],
                          'potatos_mx_my': ['MON1', 'MON2', 'MON3', 'MON4', 'MON334'],
                          'potatos_fz_my': [],
                          'potatos_fy_mx': [],
                          'potatos_mx_mz': ['MON324'],
                          'potatos_my_mz': [],
                          'cuttingforces_wing': ['MON1', 'MON2', 'MON3', 'MON4'],
                          }
                      
        self.trimcase = [{'desc': 'CC.BFDM.OVCFL000.Vergleichsfall53', # description of maneuver case, e.g. according to G. Pinho Chiozzotto, "Kriterien fuer die Erstellung eines Lastenkatalogs," Institute of Aeroelasticity, iLOADs MS1.2, Feb. 2014.
                          'maneuver': '',       # blank for trim about all three axes, for more trim conditions see trim_conditions.py
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
                          'support': [0,1,2,3,4,5],     # list of DoF to be constrained
                          'thrust':'balanced',          # thrust per engine in N or 'balanced'
                         },
                        ]
        self.simcase = [{}] # For every trimcase, a corresponding simcase is required. For maneuvers, it may be empty.
        # a time simulation is triggered if the simcase contains at least 'dt' and 't_final'
        self.simcase = [{'dt': 0.01,            # time step size in [s]
                         't_final': 2.0,        # final simulation time  in [s]
                         'gust': False,         # True or False, enables 1-cosine gust according to CS-25
                         'gust_gradient': 9.0,  # gust gradient H in [m]
                         'gust_orientation': 0, # orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top, 
                         # 90 = gust from the right, 270 = gust from the left, arbitrary values possible (rotation of gust direction vector about Nastran's x-axis pointing backwards)
                         'gust_para':{'Z_mo': 12500.0, 'MLW': 65949.0, 'MTOW': 73365.0, 'MZFW': 62962.0, 'MD': 0.87, 'T1': 0.00}, # gust parameters according to CS-25
                         'WG_TAS': 0.1,         # alternatively, give gust velocity / Vtas directly
                         'turbulence': False,   # True or False, enables continuous turbulence excitation
                         'limit_turbulence': False, # True or False, calculates limit turbulence according to CS-25
                         'cs_signal': False,    # True or False, enables playback of control surface signals via efcs
                         'controller': False,   # True or False, enables a generic controller e.g. to maintain p, q and r
                         'landinggear':False,   # True or False, enables a generic landing gear
                         'support': [0,1,2,3,4,5],      # list of DoF to be constrained
                         'flutter': False,      # True or False, enables flutter check with k, ke or pk method
                         'flutter_para':{'method': 'k', 'k_red':np.linspace(2.0, 0.001, 1000)},  # flutter parameters for k and ke method
                         'flutter_para':{'method': 'pk', 'Vtas':np.linspace(100.0, 500.0, 100)}, # flutter parameters for pk method
                        },
                       ] 
        # End