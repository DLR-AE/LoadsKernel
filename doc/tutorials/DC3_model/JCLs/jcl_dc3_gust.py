"""
Job Control File documentation
The Job Control (jcl) is a python class which defines the model and and the simulation and is imported at
the beginning of every simulation. Unlike a conventional parameter file, this allows scripting/programming
of the input, e.g. to convert units, generate mutiple load cases, etc.
Note that this documentation of parameters is comprehensive, but a) not all parameters are necessary for
every kind of simulation and b) some parameters are for experts only --> your JCL might be much smaller.
"""
import numpy as np
import platform
import os
from loadskernel.units import ft2m, tas2Ma
from loadskernel import jcl_helper
import pathlib


class jcl:

    def __init__(self):
        model_root = pathlib.Path(__file__).parent.parent.resolve()

        # Give your aircraft a name and set some general parameters
        self.general = {'aircraft': 'DC3',
                        # Reference span width (from tip to tip)
                        'b_ref': 29.0,
                        # Reference chord length
                        'c_ref': 3.508,
                        # Reference area
                        'A_ref': 91.7,
                        # Mean aerodynamic center, also used as moments reference point
                        'MAC_ref': [8.566, 0.0, 0.0],
                        }
        """
        The electronic flight control system (EFCS) provides the "wireing" of the pilot commands
        xi, eta and zeta with the control surface deflections. This is aircraft-specific and needs
        to be implemented as a python module.
        """
        # Electronic flight control system
        self.efcs = {'version': 'efcs_dc3',  # Name of the corresponding python module
                     # Path where to find the EFCS module
                     'path': os.path.join(model_root, 'efcs'),
                     }
        # Read the structural geometry
        self.geom = {'method': 'mona',  # ModGen and/or Nastran (mona) BDFs
                     # bdf file(s) with GRIDs and CORDs (CORD1R and CORD2R)
                     'filename_grid': [os.path.join(model_root, 'fem', 'structure_only.bdf')],
                     # bdf file(s) with CQUADs and CTRIAs, for visualization only, e.g. outer skin on the aircraft
                     # 'filename_shell': [],
                     # bdf file(s) with MONPNT-cards
                     'filename_monpnt': os.path.join(model_root, 'fem', 'export_monitoring-stations.csv'),
                     # The following matrices are required for some mass methods. However, the stiffness is geometry
                     # and not mass dependent. Overview:
                     # KGG via DMAP Alter und OP4            - required for mass method = 'modalanalysis', 'guyan' or 'B2000'
                     # USET via DMAP Alter und OP4           - required for mass method = 'modalanalysis', 'guyan'
                     # matrix GM via DMAP Alter und OP4      - required for mass method = 'modalanalysis', 'guyan'
                     # bdf file(s) with ASET1-card           - required for mass method = 'guyan'
                     # matrix R_trans frum B2000             - required for mass method = 'B2000'
                     'filename_h5': os.path.join(model_root, 'fem', 'SOL103_structure_only.mtx.h5'),
                     'filename_uset': os.path.join(model_root, 'fem', 'uset.op2'),
                     }
        # Settings for the aerodynamic model
        self.aero = {'method': 'mona_unsteady',
                     # 'mona_steady'      - steady trim and quasi-steady time domain simulations
                     # 'mona_unsteady'    - unsteady time domain simulation based on the RFA, e.g. for gust
                     # 'freq_dom'         - frequency domain simulations, e.g. gust, continuous turbulence, flutter, etc
                     # 'nonlin_steady'    - steady trim and quasi-steady time domain simulations with some non-linearities
                     # 'cfd_steady'       - steady trim
                     # 'cfd_unsteady'     - unsteady time domain simulation, e.g. for gust
                     #
                     # True or False, aerodynamic feedback of elastic structure on aerodynamics can be deactivated.
                     # You will still see deformations, but there is no coupling.
                     'flex': True,
                     # aerogrid is given by CAERO1, CAERO7 or by CQUAD4 cards
                     'method_caero': 'CAERO1',
                     # bdf file(s) with CAERO1 or CQUAD4-cards for aerogrid. IDs in ascending order.
                     'filename_caero_bdf': [os.path.join(model_root, 'aero', 'vt', 'vt.CAERO1'),
                                            os.path.join(model_root, 'aero', 'left-ht', 'left-ht.CAERO1'),
                                            os.path.join(model_root, 'aero', 'right-ht', 'right-ht.CAERO1'),
                                            os.path.join(model_root, 'aero', 'left-wing', 'left-wing.CAERO1'),
                                            os.path.join(model_root, 'aero', 'right-wing', 'right-wing.CAERO1'),
                                            ],
                     # DMI Matrix for camber and twist correction. Same order as the aerogrid.
                     'filename_DMI_W2GJ': [os.path.join(model_root, 'fem', 'w2gj_list.DMI_merge')],
                     # bdf file(s) with AESURF-cards
                     'filename_aesurf': [os.path.join(model_root, 'aero', 'vt', 'vt.AESURF'),
                                         os.path.join(model_root, 'aero', 'left-ht', 'left-ht.AESURF'),
                                         os.path.join(model_root, 'aero', 'right-ht', 'right-ht.AESURF'),
                                         os.path.join(model_root, 'aero', 'left-wing', 'left-wing.AESURF'),
                                         os.path.join(model_root, 'aero', 'right-wing', 'right-wing.AESURF'),
                                         ],
                     # bdf file(s) with AELIST-cards
                     'filename_aelist': [os.path.join(model_root, 'aero', 'vt', 'vt.AELIST'),
                                         os.path.join(model_root, 'aero', 'left-ht', 'left-ht.AELIST'),
                                         os.path.join(model_root, 'aero', 'right-ht', 'right-ht.AELIST'),
                                         os.path.join(model_root, 'aero', 'left-wing', 'left-wing.AELIST'),
                                         os.path.join(model_root, 'aero', 'right-wing', 'right-wing.AELIST'),
                                         ],
                     # The hingeline of a CS is given by a CORD. Either the y- or the z-axis is taken as hingeline. 'y', 'z'
                     'hingeline': 'y',
                     # 'vlm' (panel-aero), 'dlm' (panel-aero) or 'nastran' (external form matrices)
                     'method_AIC': 'dlm',
                     'key': ['VC'],
                     'Ma': [0.27],
                     # reduced frequencies for DLM, Nastran Definition!
                     'k_red': [0.001, 0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0],
                     # number of poles for rational function approximation (RFA)
                     'n_poles': 4,
                     }
        # Set the was in which the aerodynamic forces are applied to the structure.
        self.spline = {'method': 'nearest_neighbour',  # Options: 'nearest_neighbour', 'rbf' or 'nastran'
                       # Possibility to use only a subset of the structural grid for splining. True or False
                       'splinegrid': False,
                       # bdf file(s) with GRIDs to ne used
                       'filename_splinegrid': ['splinegrid.bdf']
                       }
        # Settings for the structural dynamics.
        self.mass = {'method': 'modalanalysis',
                     # Implemented interfaces: 'f06', 'modalanalysis', 'guyan', 'CoFE', 'B2000'
                     'key': ['M3'],
                     # MGG via DMAP Alter and OP4 - always required
                     'filename_h5': [
                         os.path.join(model_root, 'fem', 'SOL103_M3.mtx.h5'),
                     ],
                     # True or False, omits first six modes
                     'omit_rb_modes': True,
                     # list(s) of modes to use
                     'modes': [np.arange(1, 21)],
                     }
        # Modal damping can be applied as a factor of the stiffness matrix.
        self.damping = {'method': 'modal',
                        'damping': 0.02,
                        }
        # The international standard atmosphere (ISA)
        self.atmo = {'method': 'ISA',
                     'key': ['FL000', 'FL055', 'FL075', 'FL210'],
                     # Altitude in meters
                     'h': ft2m([0, 5500, 7500, 21000, ]),
                     }
        # Setting of the rigid body equations of motion
        self.eom = {'version': 'waszak'}  # 'linear' or 'waszak'

        """
        This section controls the automatic plotting and selection of dimensioning load cases.
        Simply put a list of names of the monitoring stations (e.g. ['MON1', 'MON2',...]) into the dictionary
        of possible load plots listed below. This will generate a pdf document and nastran force and moment
        cards for the dimensioning load cases.
        """
        self.loadplots = {'potatos_fz_mx': [],
                          'potatos_mx_my': ['WL01', 'WL03', 'WL05', 'WL07', 'WL09', 'WL11', 'WL13', 'WL15', 'WL17',
                                            'WL19', 'WL21', 'WL23', 'WL25', 'WL27', 'WL29', 'WL31', 'WR31', 'WR29',
                                            'WR27', 'WR25', 'WR23', 'WR21', 'WR19', 'WR17', 'WR15', 'WR13', 'WR11',
                                            'WR09', 'WR07', 'WR05', 'WR03', 'WR01'],
                          'potatos_fz_my': [],
                          'potatos_fy_mx': [],
                          'potatos_mx_mz': [],
                          'potatos_my_mz': [],
                          'cuttingforces_wing': ['WL01', 'WL03', 'WL05', 'WL07', 'WL09', 'WL11', 'WL13', 'WL15', 'WL17',
                                                 'WL19', 'WL21', 'WL23', 'WL25', 'WL27', 'WL29', 'WL31', 'WR31', 'WR29',
                                                 'WR27', 'WR25', 'WR23', 'WR21', 'WR19', 'WR17', 'WR15', 'WR13', 'WR11',
                                                 'WR09', 'WR07', 'WR05', 'WR03', 'WR01'],
                          }
        """
        The trimcase defines the maneuver load case, one dictionary per load case.
        There may be hundreds or thousands of load cases, so at some point it might be beneficial to script this section or
        import an excel sheet.
        """
        self.trimcase = [{'desc': 'CC.M3.OVCFL000.level', # Descriptive string of the maneuver case
                          # Kind of trim condition, blank for trim about all three axes, for more trim conditions see
                          # trim_conditions.py
                          'maneuver': '',
                          # Subcase ID number, for Nastran in acending order
                          'subcase': 1,
                          # Setting of the operational point
                          # The flight speed is given by the Mach number
                          'Ma': tas2Ma(70.0, 0.0),
                          # Aero key
                          'aero': 'VC',
                          # Atmo key
                          'altitude': 'FL000',
                          # Mass key
                          'mass': 'M3',
                          # Load factor Nz
                          'Nz': 1.0,
                          # Velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          # Roll rate in rad/s
                          'p': 0.0 / 180.0 * np.pi,
                          # Pitch rate in rad/s
                          'q': 0.0 / 180.0 * np.pi,
                          # Yaw rate in rad/s
                          'r': 0.0,
                          # Roll acceleration in rad/s^2
                          'pdot': 0.0 ,
                          # Pitch acceleration in rad/s^2
                          'qdot': 0.0,
                          # Yaw acceleration in rad/s^2
                          'rdot': 0.0,
                          },
                         {'desc': 'CC.M3.OVCFL000.level',  # Descriptive string of the maneuver case
                          # Kind of trim condition, blank for trim about all three axes, for more trim conditions see
                          # trim_conditions.py
                          'maneuver': '',
                          # Subcase ID number, for Nastran in acending order
                          'subcase': 2,
                          # Setting of the operational point
                          # The flight speed is given by the Mach number
                          'Ma': tas2Ma(70.0, 0.0),
                          # Aero key
                          'aero': 'VC',
                          # Atmo key
                          'altitude': 'FL000',
                          # Mass key
                          'mass': 'M3',
                          # Load factor Nz
                          'Nz': 1.0,
                          # Velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          # Roll rate in rad/s
                          'p': 0.0 / 180.0 * np.pi,
                          # Pitch rate in rad/s
                          'q': 0.0 / 180.0 * np.pi,
                          # Yaw rate in rad/s
                          'r': 0.0,
                          # Roll acceleration in rad/s^2
                          'pdot': 0.0,
                          # Pitch acceleration in rad/s^2
                          'qdot': 0.0,
                          # Yaw acceleration in rad/s^2
                          'rdot': 0.0,
                          },
                         {'desc': 'CC.M3.OVCFL000.level',  # Descriptive string of the maneuver case
                          # Kind of trim condition, blank for trim about all three axes, for more trim conditions see
                          # trim_conditions.py
                          'maneuver': '',
                          # Subcase ID number, for Nastran in acending order
                          'subcase': 3,
                          # Setting of the operational point
                          # The flight speed is given by the Mach number
                          'Ma': tas2Ma(70.0, 0.0),
                          # Aero key
                          'aero': 'VC',
                          # Atmo key
                          'altitude': 'FL000',
                          # Mass key
                          'mass': 'M3',
                          # Load factor Nz
                          'Nz': 1.0,
                          # Velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          # Roll rate in rad/s
                          'p': 0.0 / 180.0 * np.pi,
                          # Pitch rate in rad/s
                          'q': 0.0 / 180.0 * np.pi,
                          # Yaw rate in rad/s
                          'r': 0.0,
                          # Roll acceleration in rad/s^2
                          'pdot': 0.0,
                          # Pitch acceleration in rad/s^2
                          'qdot': 0.0,
                          # Yaw acceleration in rad/s^2
                          'rdot': 0.0,
                          },
                         {'desc': 'CC.M3.OVCFL000.level',  # Descriptive string of the maneuver case
                          # Kind of trim condition, blank for trim about all three axes, for more trim conditions see
                          # trim_conditions.py
                          'maneuver': '',
                          # Subcase ID number, for Nastran in acending order
                          'subcase': 4,
                          # Setting of the operational point
                          # The flight speed is given by the Mach number
                          'Ma': tas2Ma(70.0, 0.0),
                          # Aero key
                          'aero': 'VC',
                          # Atmo key
                          'altitude': 'FL000',
                          # Mass key
                          'mass': 'M3',
                          # Load factor Nz
                          'Nz': 1.0,
                          # Velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          # Roll rate in rad/s
                          'p': 0.0 / 180.0 * np.pi,
                          # Pitch rate in rad/s
                          'q': 0.0 / 180.0 * np.pi,
                          # Yaw rate in rad/s
                          'r': 0.0,
                          # Roll acceleration in rad/s^2
                          'pdot': 0.0,
                          # Pitch acceleration in rad/s^2
                          'qdot': 0.0,
                          # Yaw acceleration in rad/s^2
                          'rdot': 0.0,
                          },
                         {'desc': 'CC.M3.OVCFL000.level', # Descriptive string of the maneuver case
                          # Kind of trim condition, blank for trim about all three axes, for more trim conditions see
                          # trim_conditions.py
                          'maneuver': '',
                          # Subcase ID number, for Nastran in acending order
                          'subcase': 5,
                          # Setting of the operational point
                          # The flight speed is given by the Mach number
                          'Ma': tas2Ma(70.0, 0.0),
                          # Aero key
                          'aero': 'VC',
                          # Atmo key
                          'altitude': 'FL000',
                          # Mass key
                          'mass': 'M3',
                          # Load factor Nz
                          'Nz': 1.0,
                          # Velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          # Roll rate in rad/s
                          'p': 0.0 / 180.0 * np.pi,
                          # Pitch rate in rad/s
                          'q': 0.0 / 180.0 * np.pi,
                          # Yaw rate in rad/s
                          'r': 0.0,
                          # Roll acceleration in rad/s^2
                          'pdot': 0.0 ,
                          # Pitch acceleration in rad/s^2
                          'qdot': 0.0,
                          # Yaw acceleration in rad/s^2
                          'rdot': 0.0,
                          },
                         {'desc': 'CC.M3.OVCFL000.level', # Descriptive string of the maneuver case
                          # Kind of trim condition, blank for trim about all three axes, for more trim conditions see
                          # trim_conditions.py
                          'maneuver': '',
                          # Subcase ID number, for Nastran in acending order
                          'subcase': 6,
                          # Setting of the operational point
                          # The flight speed is given by the Mach number
                          'Ma': tas2Ma(70.0, 0.0),
                          # Aero key
                          'aero': 'VC',
                          # Atmo key
                          'altitude': 'FL000',
                          # Mass key
                          'mass': 'M3',
                          # Load factor Nz
                          'Nz': 1.0,
                          # Velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          # Roll rate in rad/s
                          'p': 0.0 / 180.0 * np.pi,
                          # Pitch rate in rad/s
                          'q': 0.0 / 180.0 * np.pi,
                          # Yaw rate in rad/s
                          'r': 0.0,
                          # Roll acceleration in rad/s^2
                          'pdot': 0.0 ,
                          # Pitch acceleration in rad/s^2
                          'qdot': 0.0,
                          # Yaw acceleration in rad/s^2
                          'rdot': 0.0,
                          },
                         {'desc': 'CC.M3.OVCFL000.level', # Descriptive string of the maneuver case
                          # Kind of trim condition, blank for trim about all three axes, for more trim conditions see
                          # trim_conditions.py
                          'maneuver': '',
                          # Subcase ID number, for Nastran in acending order
                          'subcase': 7,
                          # Setting of the operational point
                          # The flight speed is given by the Mach number
                          'Ma': tas2Ma(70.0, 0.0),
                          # Aero key
                          'aero': 'VC',
                          # Atmo key
                          'altitude': 'FL000',
                          # Mass key
                          'mass': 'M3',
                          # Load factor Nz
                          'Nz': 1.0,
                          # Velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          # Roll rate in rad/s
                          'p': 0.0 / 180.0 * np.pi,
                          # Pitch rate in rad/s
                          'q': 0.0 / 180.0 * np.pi,
                          # Yaw rate in rad/s
                          'r': 0.0,
                          # Roll acceleration in rad/s^2
                          'pdot': 0.0 ,
                          # Pitch acceleration in rad/s^2
                          'qdot': 0.0,
                          # Yaw acceleration in rad/s^2
                          'rdot': 0.0,
                          },
                         {'desc': 'CC.M3.OVCFL000.level',  # Descriptive string of the maneuver case
                          # Kind of trim condition, blank for trim about all three axes, for more trim conditions see
                          # trim_conditions.py
                          'maneuver': '',
                          # Subcase ID number, for Nastran in acending order
                          'subcase': 8,
                          # Setting of the operational point
                          # The flight speed is given by the Mach number
                          'Ma': tas2Ma(70.0, 0.0),
                          # Aero key
                          'aero': 'VC',
                          # Atmo key
                          'altitude': 'FL000',
                          # Mass key
                          'mass': 'M3',
                          # Load factor Nz
                          'Nz': 1.0,
                          # Velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          # Roll rate in rad/s
                          'p': 0.0 / 180.0 * np.pi,
                          # Pitch rate in rad/s
                          'q': 0.0 / 180.0 * np.pi,
                          # Yaw rate in rad/s
                          'r': 0.0,
                          # Roll acceleration in rad/s^2
                          'pdot': 0.0,
                          # Pitch acceleration in rad/s^2
                          'qdot': 0.0,
                          # Yaw acceleration in rad/s^2
                          'rdot': 0.0,
                          },
                         {'desc': 'CC.M3.OVCFL000.level',  # Descriptive string of the maneuver case
                          # Kind of trim condition, blank for trim about all three axes, for more trim conditions see
                          # trim_conditions.py
                          'maneuver': '',
                          # Subcase ID number, for Nastran in acending order
                          'subcase': 9,
                          # Setting of the operational point
                          # The flight speed is given by the Mach number
                          'Ma': tas2Ma(70.0, 0.0),
                          # Aero key
                          'aero': 'VC',
                          # Atmo key
                          'altitude': 'FL000',
                          # Mass key
                          'mass': 'M3',
                          # Load factor Nz
                          'Nz': 1.0,
                          # Velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          # Roll rate in rad/s
                          'p': 0.0 / 180.0 * np.pi,
                          # Pitch rate in rad/s
                          'q': 0.0 / 180.0 * np.pi,
                          # Yaw rate in rad/s
                          'r': 0.0,
                          # Roll acceleration in rad/s^2
                          'pdot': 0.0,
                          # Pitch acceleration in rad/s^2
                          'qdot': 0.0,
                          # Yaw acceleration in rad/s^2
                          'rdot': 0.0,
                          },
                         {'desc': 'CC.M3.OVCFL000.level',  # Descriptive string of the maneuver case
                          # Kind of trim condition, blank for trim about all three axes, for more trim conditions see
                          # trim_conditions.py
                          'maneuver': '',
                          # Subcase ID number, for Nastran in acending order
                          'subcase': 10,
                          # Setting of the operational point
                          # The flight speed is given by the Mach number
                          'Ma': tas2Ma(70.0, 0.0),
                          # Aero key
                          'aero': 'VC',
                          # Atmo key
                          'altitude': 'FL000',
                          # Mass key
                          'mass': 'M3',
                          # Load factor Nz
                          'Nz': 1.0,
                          # Velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          # Roll rate in rad/s
                          'p': 0.0 / 180.0 * np.pi,
                          # Pitch rate in rad/s
                          'q': 0.0 / 180.0 * np.pi,
                          # Yaw rate in rad/s
                          'r': 0.0,
                          # Roll acceleration in rad/s^2
                          'pdot': 0.0,
                          # Pitch acceleration in rad/s^2
                          'qdot': 0.0,
                          # Yaw acceleration in rad/s^2
                          'rdot': 0.0,
                          }
                         ]
        """
        For every trimcase, a corresponding simcase is required. For maneuvers, it may be empty self.simcase = [{}].
        A time simulation is triggered if the simcase contains at least 'dt' and 't_final'
        """
        self.simcase = [{'dt': 0.01,  # Time step size of the output in [s]
                         # Final simulation time  in [s]
                         't_final': 3.0,
                         # True or False, enables 1-cosine gust according to CS-25
                         'gust': True,
                         # Gust gradient H (half gust length) in [m]
                         'gust_gradient': 9.0,
                         # Orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top,
                         # 90 = gust from the right, 270 = gust from the left, arbitrary values possible
                         # (rotation of gust direction vector about Nastran's x-axis pointing backwards)
                         'gust_orientation': 0,
                         # Gust parameters according to CS-25 to calculate the gust velocity
                         # MD is the dive speed in Mach
                         'gust_para': {'Z_mo': 8046.72, 'MLW': 11793.40, 'MTOW': 11883.98, 'MZFW': 10594.47, 'MD': 0.334,
                                       'T1': 0.0},
                         # Alternatively, give gust velocity / Vtas directly
                         # 'WG_TAS': 0.1,
                         # True or False, enables continuous turbulence excitation
                         'turbulence': False,
                         # True or False, calculates limit turbulence according to CS-25
                         'limit_turbulence': False,
                         # True or False, enables playback of control surface signals via efcs
                         'cs_signal': False,
                         # True or False, enables a generic controller e.g. to maintain p, q and r
                         'controller': False,
                         # True or False, enables a generic landing gear
                         'landinggear': False,
                         # True or False, enables calculation of rigid and elastic derivatives
                         'derivatives': False,
                         # True or False, enables flutter check with k, ke or pk method
                         'flutter': False,
                         },
                        {'dt': 0.01,  # Time step size of the output in [s]
                         # Final simulation time  in [s]
                         't_final': 3.0,
                         # True or False, enables 1-cosine gust according to CS-25
                         'gust': True,
                         # Gust gradient H (half gust length) in [m]
                         'gust_gradient': 16.0,
                         # Orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top,
                         # 90 = gust from the right, 270 = gust from the left, arbitrary values possible
                         # (rotation of gust direction vector about Nastran's x-axis pointing backwards)
                         'gust_orientation': 0,
                         # Gust parameters according to CS-25 to calculate the gust velocity
                         # MD is the dive speed in Mach
                         'gust_para': {'Z_mo': 8046.72, 'MLW': 11793.40, 'MTOW': 11883.98, 'MZFW': 10594.47,
                                       'MD': 0.334,
                                       'T1': 0.0},
                         # Alternatively, give gust velocity / Vtas directly
                         # 'WG_TAS': 0.1,
                         # True or False, enables continuous turbulence excitation
                         'turbulence': False,
                         # True or False, calculates limit turbulence according to CS-25
                         'limit_turbulence': False,
                         # True or False, enables playback of control surface signals via efcs
                         'cs_signal': False,
                         # True or False, enables a generic controller e.g. to maintain p, q and r
                         'controller': False,
                         # True or False, enables a generic landing gear
                         'landinggear': False,
                         # True or False, enables calculation of rigid and elastic derivatives
                         'derivatives': False,
                         # True or False, enables flutter check with k, ke or pk method
                         'flutter': False,
                         },
                        {'dt': 0.01,  # Time step size of the output in [s]
                         # Final simulation time  in [s]
                         't_final': 3.0,
                         # True or False, enables 1-cosine gust according to CS-25
                         'gust': True,
                         # Gust gradient H (half gust length) in [m]
                         'gust_gradient': 23.0,
                         # Orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top,
                         # 90 = gust from the right, 270 = gust from the left, arbitrary values possible
                         # (rotation of gust direction vector about Nastran's x-axis pointing backwards)
                         'gust_orientation': 0,
                         # Gust parameters according to CS-25 to calculate the gust velocity
                         # MD is the dive speed in Mach
                         'gust_para': {'Z_mo': 8046.72, 'MLW': 11793.40, 'MTOW': 11883.98, 'MZFW': 10594.47,
                                       'MD': 0.334,
                                       'T1': 0.0},
                         # Alternatively, give gust velocity / Vtas directly
                         # 'WG_TAS': 0.1,
                         # True or False, enables continuous turbulence excitation
                         'turbulence': False,
                         # True or False, calculates limit turbulence according to CS-25
                         'limit_turbulence': False,
                         # True or False, enables playback of control surface signals via efcs
                         'cs_signal': False,
                         # True or False, enables a generic controller e.g. to maintain p, q and r
                         'controller': False,
                         # True or False, enables a generic landing gear
                         'landinggear': False,
                         # True or False, enables calculation of rigid and elastic derivatives
                         'derivatives': False,
                         # True or False, enables flutter check with k, ke or pk method
                         'flutter': False,
                         },
                        {'dt': 0.01,  # Time step size of the output in [s]
                         # Final simulation time  in [s]
                         't_final': 3.0,
                         # True or False, enables 1-cosine gust according to CS-25
                         'gust': True,
                         # Gust gradient H (half gust length) in [m]
                         'gust_gradient': 30.0,
                         # Orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top,
                         # 90 = gust from the right, 270 = gust from the left, arbitrary values possible
                         # (rotation of gust direction vector about Nastran's x-axis pointing backwards)
                         'gust_orientation': 0,
                         # Gust parameters according to CS-25 to calculate the gust velocity
                         # MD is the dive speed in Mach
                         'gust_para': {'Z_mo': 8046.72, 'MLW': 11793.40, 'MTOW': 11883.98, 'MZFW': 10594.47,
                                       'MD': 0.334,
                                       'T1': 0.0},
                         # Alternatively, give gust velocity / Vtas directly
                         # 'WG_TAS': 0.1,
                         # True or False, enables continuous turbulence excitation
                         'turbulence': False,
                         # True or False, calculates limit turbulence according to CS-25
                         'limit_turbulence': False,
                         # True or False, enables playback of control surface signals via efcs
                         'cs_signal': False,
                         # True or False, enables a generic controller e.g. to maintain p, q and r
                         'controller': False,
                         # True or False, enables a generic landing gear
                         'landinggear': False,
                         # True or False, enables calculation of rigid and elastic derivatives
                         'derivatives': False,
                         # True or False, enables flutter check with k, ke or pk method
                         'flutter': False,
                         },
                        {'dt': 0.01,  # Time step size of the output in [s]
                         # Final simulation time  in [s]
                         't_final': 3.0,
                         # True or False, enables 1-cosine gust according to CS-25
                         'gust': True,
                         # Gust gradient H (half gust length) in [m]
                         'gust_gradient': 37.0,
                         # Orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top,
                         # 90 = gust from the right, 270 = gust from the left, arbitrary values possible
                         # (rotation of gust direction vector about Nastran's x-axis pointing backwards)
                         'gust_orientation': 0,
                         # Gust parameters according to CS-25 to calculate the gust velocity
                         # MD is the dive speed in Mach
                         'gust_para': {'Z_mo': 8046.72, 'MLW': 11793.40, 'MTOW': 11883.98, 'MZFW': 10594.47,
                                       'MD': 0.334,
                                       'T1': 0.0},
                         # Alternatively, give gust velocity / Vtas directly
                         # 'WG_TAS': 0.1,
                         # True or False, enables continuous turbulence excitation
                         'turbulence': False,
                         # True or False, calculates limit turbulence according to CS-25
                         'limit_turbulence': False,
                         # True or False, enables playback of control surface signals via efcs
                         'cs_signal': False,
                         # True or False, enables a generic controller e.g. to maintain p, q and r
                         'controller': False,
                         # True or False, enables a generic landing gear
                         'landinggear': False,
                         # True or False, enables calculation of rigid and elastic derivatives
                         'derivatives': False,
                         # True or False, enables flutter check with k, ke or pk method
                         'flutter': False,
                         },
                        {'dt': 0.01,  # Time step size of the output in [s]
                         # Final simulation time  in [s]
                         't_final': 3.0,
                         # True or False, enables 1-cosine gust according to CS-25
                         'gust': True,
                         # Gust gradient H (half gust length) in [m]
                         'gust_gradient': 51.0,
                         # Orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top,
                         # 90 = gust from the right, 270 = gust from the left, arbitrary values possible
                         # (rotation of gust direction vector about Nastran's x-axis pointing backwards)
                         'gust_orientation': 0,
                         # Gust parameters according to CS-25 to calculate the gust velocity
                         # MD is the dive speed in Mach
                         'gust_para': {'Z_mo': 8046.72, 'MLW': 11793.40, 'MTOW': 11883.98, 'MZFW': 10594.47,
                                       'MD': 0.334,
                                       'T1': 0.0},
                         # Alternatively, give gust velocity / Vtas directly
                         # 'WG_TAS': 0.1,
                         # True or False, enables continuous turbulence excitation
                         'turbulence': False,
                         # True or False, calculates limit turbulence according to CS-25
                         'limit_turbulence': False,
                         # True or False, enables playback of control surface signals via efcs
                         'cs_signal': False,
                         # True or False, enables a generic controller e.g. to maintain p, q and r
                         'controller': False,
                         # True or False, enables a generic landing gear
                         'landinggear': False,
                         # True or False, enables calculation of rigid and elastic derivatives
                         'derivatives': False,
                         # True or False, enables flutter check with k, ke or pk method
                         'flutter': False,
                         },
                        {'dt': 0.01,  # Time step size of the output in [s]
                         # Final simulation time  in [s]
                         't_final': 3.0,
                         # True or False, enables 1-cosine gust according to CS-25
                         'gust': True,
                         # Gust gradient H (half gust length) in [m]
                         'gust_gradient': 65.0,
                         # Orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top,
                         # 90 = gust from the right, 270 = gust from the left, arbitrary values possible
                         # (rotation of gust direction vector about Nastran's x-axis pointing backwards)
                         'gust_orientation': 0,
                         # Gust parameters according to CS-25 to calculate the gust velocity
                         # MD is the dive speed in Mach
                         'gust_para': {'Z_mo': 8046.72, 'MLW': 11793.40, 'MTOW': 11883.98, 'MZFW': 10594.47,
                                       'MD': 0.334,
                                       'T1': 0.0},
                         # Alternatively, give gust velocity / Vtas directly
                         # 'WG_TAS': 0.1,
                         # True or False, enables continuous turbulence excitation
                         'turbulence': False,
                         # True or False, calculates limit turbulence according to CS-25
                         'limit_turbulence': False,
                         # True or False, enables playback of control surface signals via efcs
                         'cs_signal': False,
                         # True or False, enables a generic controller e.g. to maintain p, q and r
                         'controller': False,
                         # True or False, enables a generic landing gear
                         'landinggear': False,
                         # True or False, enables calculation of rigid and elastic derivatives
                         'derivatives': False,
                         # True or False, enables flutter check with k, ke or pk method
                         'flutter': False,
                         },
                        {'dt': 0.01,  # Time step size of the output in [s]
                         # Final simulation time  in [s]
                         't_final': 3.0,
                         # True or False, enables 1-cosine gust according to CS-25
                         'gust': True,
                         # Gust gradient H (half gust length) in [m]
                         'gust_gradient': 79.0,
                         # Orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top,
                         # 90 = gust from the right, 270 = gust from the left, arbitrary values possible
                         # (rotation of gust direction vector about Nastran's x-axis pointing backwards)
                         'gust_orientation': 0,
                         # Gust parameters according to CS-25 to calculate the gust velocity
                         # MD is the dive speed in Mach
                         'gust_para': {'Z_mo': 8046.72, 'MLW': 11793.40, 'MTOW': 11883.98, 'MZFW': 10594.47,
                                       'MD': 0.334,
                                       'T1': 0.0},
                         # Alternatively, give gust velocity / Vtas directly
                         # 'WG_TAS': 0.1,
                         # True or False, enables continuous turbulence excitation
                         'turbulence': False,
                         # True or False, calculates limit turbulence according to CS-25
                         'limit_turbulence': False,
                         # True or False, enables playback of control surface signals via efcs
                         'cs_signal': False,
                         # True or False, enables a generic controller e.g. to maintain p, q and r
                         'controller': False,
                         # True or False, enables a generic landing gear
                         'landinggear': False,
                         # True or False, enables calculation of rigid and elastic derivatives
                         'derivatives': False,
                         # True or False, enables flutter check with k, ke or pk method
                         'flutter': False,
                         },
                        {'dt': 0.01,  # Time step size of the output in [s]
                         # Final simulation time  in [s]
                         't_final': 3.0,
                         # True or False, enables 1-cosine gust according to CS-25
                         'gust': True,
                         # Gust gradient H (half gust length) in [m]
                         'gust_gradient': 93.0,
                         # Orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top,
                         # 90 = gust from the right, 270 = gust from the left, arbitrary values possible
                         # (rotation of gust direction vector about Nastran's x-axis pointing backwards)
                         'gust_orientation': 0,
                         # Gust parameters according to CS-25 to calculate the gust velocity
                         # MD is the dive speed in Mach
                         'gust_para': {'Z_mo': 8046.72, 'MLW': 11793.40, 'MTOW': 11883.98, 'MZFW': 10594.47,
                                       'MD': 0.334,
                                       'T1': 0.0},
                         # Alternatively, give gust velocity / Vtas directly
                         # 'WG_TAS': 0.1,
                         # True or False, enables continuous turbulence excitation
                         'turbulence': False,
                         # True or False, calculates limit turbulence according to CS-25
                         'limit_turbulence': False,
                         # True or False, enables playback of control surface signals via efcs
                         'cs_signal': False,
                         # True or False, enables a generic controller e.g. to maintain p, q and r
                         'controller': False,
                         # True or False, enables a generic landing gear
                         'landinggear': False,
                         # True or False, enables calculation of rigid and elastic derivatives
                         'derivatives': False,
                         # True or False, enables flutter check with k, ke or pk method
                         'flutter': False,
                         },
                        {'dt': 0.01,  # Time step size of the output in [s]
                         # Final simulation time  in [s]
                         't_final': 3.0,
                         # True or False, enables 1-cosine gust according to CS-25
                         'gust': True,
                         # Gust gradient H (half gust length) in [m]
                         'gust_gradient': 107.0,
                         # Orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top,
                         # 90 = gust from the right, 270 = gust from the left, arbitrary values possible
                         # (rotation of gust direction vector about Nastran's x-axis pointing backwards)
                         'gust_orientation': 0,
                         # Gust parameters according to CS-25 to calculate the gust velocity
                         # MD is the dive speed in Mach
                         'gust_para': {'Z_mo': 8046.72, 'MLW': 11793.40, 'MTOW': 11883.98, 'MZFW': 10594.47,
                                       'MD': 0.334,
                                       'T1': 0.0},
                         # Alternatively, give gust velocity / Vtas directly
                         # 'WG_TAS': 0.1,
                         # True or False, enables continuous turbulence excitation
                         'turbulence': False,
                         # True or False, calculates limit turbulence according to CS-25
                         'limit_turbulence': False,
                         # True or False, enables playback of control surface signals via efcs
                         'cs_signal': False,
                         # True or False, enables a generic controller e.g. to maintain p, q and r
                         'controller': False,
                         # True or False, enables a generic landing gear
                         'landinggear': False,
                         # True or False, enables calculation of rigid and elastic derivatives
                         'derivatives': False,
                         # True or False, enables flutter check with k, ke or pk method
                         'flutter': False,
                         }
                        ]
        # End
