"""
Job Control File documentation
The Job Control (jcl) is a python class which defines the model and and the simulation and is imported at
the beginning of every simulation. Unlike a conventional parameter file, this allows scripting/programming
of the input, e.g. to convert units, generate mutiple load cases, etc.
Note that this documentation of parameters is comprehensive, but a) not all parameters are necessary for
every kind of simulation and b) some parameters are for experts only --> your JCL might be much smaller.
"""
import numpy as np
from loadskernel.units import ft2m


class jcl:

    def __init__(self):
        # Give your aircraft a name and set some general parameters
        self.general = {'aircraft': 'DLR F-19-S',
                        # Reference span width (from tip to tip)
                        'b_ref': 15.375,
                        # Reference chord length
                        'c_ref': 4.79,
                        # Reference area
                        'A_ref': 77,
                        # Mean aerodynamic center, also used as moments reference point
                        'MAC_ref': [6.0, 0.0, 0.0],
                        }
        """
        The electronic flight control system (EFCS) provides the "wireing" of the pilot commands
        xi, eta and zeta with the control surface deflections. This is aicraft-specific and needs
        to be implemented as a python module.
        """
        # Electronic flight control system
        self.efcs = {'version': 'mephisto', # Name of the corresponding python module
                     # Path where to find the EFCS module
                     'path': '/path/to/EFCS',
                     }
        # Read the structural geometry
        self.geom = {'method': 'mona',  # ModGen and/or Nastran (mona) BDFs
                     # bdf file(s) with GRIDs and CORDs (CORD1R and CORD2R)
                     'filename_grid': ['grids.bdf'],
                     # bdf file(s) with CQUADs and CTRIAs, for visualization only, e.g. outer skin on the aircraft
                     'filename_shell': ['shells.bdf'],
                     # bdf file(s) with MONPNT-cards
                     'filename_monpnt': 'monpnt.bdf',
                     # Alternative way to define monitoring stations:
                     # bdf file with GRID-cards, one monitoring station is created at each GRID point
                     # 1st GRID point -> 1st monstation
                     'filename_mongrid': 'monstations_grids.bdf',
                     # additional CORDs for monitoring stations
                     'filename_moncoord': 'monstations_coords.bdf',
                     # bdf file with GRID-cards, 1st file -> 1st monstation
                     'filename_monstations': ['monstation_MON1.bdf', 'monstation_MON2.bdf'],
                     # The following matrices are required for some mass methods. However, the stiffness is geometry
                     # and not mass dependent. Overview:
                     # KGG via DMAP Alter (.op4 or .h5)      - required for mass method = 'modalanalysis', 'guyan' or 'B2000'
                     # GM via DMAP Alter (.op4 or .h5)       - required for mass method = 'modalanalysis', 'guyan'
                     # USET via DMAP Alter and OP2           - required for mass method = 'modalanalysis', 'guyan'
                     # bdf file(s) with ASET1-card           - required for mass method = 'guyan'
                     # matrix R_trans from B2000             - required for mass method = 'B2000'
                     # The HDF5 file format is preferred over OP4 due to better performance and higher precision. Because the
                     # uset is a table, not a matrix, it can't be included in the HDF5 file and still needs to be given as OP2.
                     'filename_h5': 'SOL103.mtx.h5',
                     'filename_KGG': 'KGG.dat',
                     'filename_GM': 'GM.dat',
                     'filename_uset': 'uset.op2',
                     'filename_aset': 'aset.bdf',
                     'filename_Rtrans': 'Rtrans.csv',
                     }
        # Settings for the aerodynamic model
        self.aero = {'method': 'mona_steady',
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
                     'filename_caero_bdf': ['CAERO1_bdf'],
                     # DMI Matrix for camber and twist correction. Same order as the aerogrid.
                     'filename_DMI_W2GJ': [],
                     # bdf file(s) with AESURF-cards
                     'filename_aesurf': ['filename.AESURF'],
                     # bdf file(s) with AELIST-cards
                     'filename_aelist': ['filename.AELIST'],
                     # The hingeline of a CS is given by a CORD. Either the y- or the z-axis is taken as hingeline. 'y', 'z'
                     'hingeline': 'z',
                     # 'vlm' (panel-aero), 'dlm' (panel-aero) or 'nastran' (external form matrices)
                     'method_AIC': 'vlm',
                     'key': ['VC', 'MC'],
                     'Ma': [0.8, 0.9],
                     # provide OP4 files with AICs if method_AIC = 'nastran'
                     'filename_AIC': ['AIC_VC.dat', 'AIC_MC.dat'],
                     # reduced frequencies for DLM, Nastran Definition!
                     'k_red': [0.001, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 1.5],
                     # number of poles for rational function approximation (RFA)
                     'n_poles': 4,
                     # Additional parameters for CFD
                     'para_path': '/scratch/tau/',
                     'para_file': 'para',
                     # Currently implemented interfaces: 'tau' or 'su2'
                     'cfd_solver': 'tau',
                     'tau_solver': 'el',
                     'tau_cores': 16,
                     # --- Start of experimental section, only for special cases ---
                     # Correction coefficient at CG, negativ = destabilizing
                     'Cn_beta_corr': [-0.012],
                     # Correction coefficient at CG, positiv = destabilizing
                     'Cm_alpha_corr': [0.22],
                     # Correction coefficient at MAC, Cd = Cd0 + dCd/dalpha^2 * alpha^2
                     'viscous_drag': 'coefficients',
                     'Cd_0': 0.005,
                     'Cd_alpha^2': 0.018 * 6.28 ** 2.0,
                     # True or False, calculates local induced drag e.g. for roll-yaw-coupling
                     'induced_drag': False,
                     # Symmetry about xz-plane: Only the right hand side on the aero mesh is give.
                     # The (missing) left hand side is created virtually by mirroring.
                     'xz_symmetry': False,
                     # --- End of experimental section ---
                     }
        # General CFD surface mesh information
        self.meshdefo = {'surface': {'fileformat': 'netcdf', # implemented file formats: 'cgns', 'netcdf', 'su2'
                                     # file name of the CFD mesh
                                     'filename_grid': 'tau.grid',
                                     # list of markers [1, 2, ...] or ['upper', 'lower', ...] of surfaces to be included in
                                     # deformation
                                     'markers': [1, 3],
                                     },
                         # Volume mesh information, currently unused
                         'volume': {},
                         }
        # Set the was in which the aerodynamic forces are applied to the structure.
        self.spline = {'method': 'nearest_neighbour',  # Options: 'nearest_neighbour', 'rbf' or 'nastran'
                       # The nastran spline matrix is written to .f06-file with PARAM    OPGTKG   1
                       'filename_f06': 'filename.f06',
                       # Possibility to use only a subset of the structural grid for splining. True or False
                       'splinegrid': True,
                       # bdf file(s) with GRIDs to ne used
                       'filename_splinegrid': ['splinegrid.bdf']
                       }
        # Settings for the structural dynamics.
        self.mass = {'method': 'modalanalysis', # Inplemented interfaces: 'f06', 'modalanalysis', 'guyan', 'CoFE', 'B2000'
                     'key': ['M1', 'M2'],
                     # MGG via DMAP Alter and HDF5
                     'filename_h5': ['SOL103_M1.mtx.h5', 'SOL103_M2.mtx.h5'],
                     # MGG via DMAP Alter and OP4
                     'filename_MGG': ['MGG_M1.dat', 'MGG_M2.dat'],
                     # eigenvalues and eigenvectors from .f06-file - required for 'mona'
                     'filename_S103': ['SOL103_M1.f06', 'SOL103_M1.f06'],
                     # eigenvalues and eigenvectors from .f06-file - required for 'mona'
                     'filename_CoFE': ['M1.mat', 'M2.mat'],
                     # True or False, omits first six modes
                     'omit_rb_modes': False,
                     # list(s) of modes to use
                     'modes': [np.arange(1, 13), np.arange(1, 16)],
                     }
        # Modal damping can be applied as a factor of the stiffness matrix.
        self.damping = {'method': 'modal',
                        'damping': 0.02,
                        }
        # The international standard atmosphere (ISA)
        self.atmo = {'method': 'ISA',
                     'key': ['FL000', 'FL055', 'FL075', 'FL200', 'FL300', 'FL450'],
                     # Altitude in meters
                     'h': ft2m([0, 5500, 7500, 20000, 30000, 45000]),
                     }
        # Setting of the rigid body equations of motion
        self.eom = {'version': 'linear'}  # 'linear' or 'waszak'

        # --- Start of experimental section, only for special cases ---
        # Parameters for generic landing gear, see PhD Thesis of Wolf Krueger and Sunpeth Cumnuantip
        para_LG = {'stroke_length': 0.3,  # m
                   'fitting_length': 0.72,  # m
                   'n': 1.4,
                   'ck': 1.0,
                   'd2': 85000.0,  # N/(m/s)^2
                   'F_static': 53326.0,  # N
                   'r_tire': 0.28,  # m
                   'c1_tire': 832000.0,  # N/m
                   'd1_tire': 4500.0,  # N/(m/s)^2
                   'm_tire': 58.0,  # kg
                   }
        # Activates a generic landing gear model during time simulation
        self.landinggear = {'method': 'generic',
                            'key': ['MLG1', 'MLG2', 'NLG'],
                            # IDs of FE attachment nodes
                            'attachment_point': [800002, 800003, 800001],
                            # Parameters for generic landing gear module, see above
                            'para': [para_LG, para_LG, para_LG],
                            }
        # Activates an engine model
        self.engine = {'method': 'thrust_only',  # Engine models: 'thrust_only', 'propellerdisk', 'PyPropMat' or 'VLM4Prop'
                       # Note: 'PyPropMAt' and 'VLM4Prop' require sensors with the same key to measure the local onflow.
                       'key': ['E-P', 'E-S'],
                       # IDs of FE attachment nodes
                       'attachment_point': [54100003, 64100003],
                       # Thrust orientation vector in body coordinate system
                       'thrust_vector': [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                       # Ratational axis of a propeller in body coordinate system
                       'rotation_vector': [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
                       # Roational inertia in Nms^2
                       'rotation_inertia': [0.18, 0.18],
                       # Propeller diamater in m
                       'diameter': [1.7, 1.7],
                       # Number of blades
                       'n_blades': [2, 2],
                       # Mach number for VLM4Prop
                       'Ma': [0.25],
                       # Input-file ('.yaml') for PyPropMAt and VLM4Prop
                       'propeller_input_file': 'HAP_O6_PROP_pitch.yaml',
                       }
        # CFD-specific
        # In case a pressure inlet boundary is modeled in the CFD mesh, the boundary condition will be
        # updated with the ambient pressure and temperature. Possible application: engine exhaust
        # without thrust.
        self.pressure_inlet = {'marker': 'exhaust',
                               'flow_direction': [1.0, 0.0, 0.0],
                               }
        # --- End of experimental section ---
        """
        Individual FE nodes can be defiend as sensors, e.g. to "measure" accelerations.
        Because the data is calculated during the simulations, these sensors may be used as input for a flight controller or
        similar. In case a wind sensor is specified here, this sensor is used to "measure" alpha and beta.
        """
        self.sensor = {'key': ['wind', 'E-P', 'E-S'],
                       # IDs of FE attachment nodes
                       'attachment_point': [200013, 54100003, 64100003],
                       }
        """
        This section controls the automatic plotting and selection of dimensioning load cases.
        Simply put a list of names of the monitoring stations (e.g. ['MON1', 'MON2',...]) into the dictionary
        of possible load plots listed below. This will generate a pdf document and nastran force and moment
        cards for the dimensioning load cases.
        """
        self.loadplots = {'potatos_fz_mx': ['MON5'],
                          'potatos_mx_my': ['MON1', 'MON2', 'MON3', 'MON4', 'MON334'],
                          'potatos_fz_my': [],
                          'potatos_fy_mx': [],
                          'potatos_mx_mz': ['MON324'],
                          'potatos_my_mz': [],
                          'cuttingforces_wing': ['MON1', 'MON2', 'MON3', 'MON4'],
                          }
        """
        The trimcase defines the maneuver load case, one dictionary per load case.
        There may be hundreds or thousands of load cases, so at some point it might be beneficial to script this section or
        import an excel sheet.
        """
        self.trimcase = [{'desc': 'CC.BFDM.OVCFL000.Maneuver_xyz', # Descriptive string of the maneuver case
                          # Kind of trim condition, blank for trim about all three axes, for more trim conditions see
                          # trim_conditions.py
                          'maneuver': '',
                          # Subcase ID number, for Nastran in acending order
                          'subcase': 53,
                          # Setting of the operational point
                          # The flight speed is given by the Mach number
                          'Ma': 0.8,
                          # Aero key
                          'aero': 'VC',
                          # Atmo key
                          'altitude': 'FL000',
                          # Mass key
                          'mass': 'BFDM',
                          # Load factor Nz
                          'Nz': 5.0,
                          # Velocities and accelerations given in ISO 9300 coordinate system (right-handed, forward-right-down)
                          # Roll rate in rad/s
                          'p': 34.3 / 180.0 * np.pi,
                          # Pitch rate in rad/s
                          'q': 28.6 / 180.0 * np.pi,
                          # Yaw rate in rad/s
                          'r': 0.0,
                          # Roll acceleration in rad/s^2
                          'pdot': -286.5 / 180.0 * np.pi,
                          # Pitch acceleration in rad/s^2
                          'qdot': 0.0,
                          # Yaw acceleration in rad/s^2
                          'rdot': 0.0,
                          # --- Start of experimental section, only for special cases ---
                          # List of DoF to be constrained
                          'support': [0, 1, 2, 3, 4, 5],
                          # Thrust per engine in N or 'balanced'
                          'thrust': 'balanced',
                          # Euler angle Phi in rad
                          'phi': 0.0 / 180.0 * np.pi,
                          # Euler angle Theta in rad
                          'theta': 0.0 / 180.0 * np.pi,
                          # Pilot command Xi in rad
                          'command_xi': 0.0 / 180.0 * np.pi,
                          # Pilot command Eta in rad
                          'command_eta': 0.0 / 180.0 * np.pi,
                          # Pilot command Zeta in rad
                          'command_zeta': 0.0 / 180.0 * np.pi,
                          # --- End of experimental section ---
                          }]
        """
        For every trimcase, a corresponding simcase is required. For maneuvers, it may be empty self.simcase = [{}].
        A time simulation is triggered if the simcase contains at least 'dt' and 't_final'
        """
        self.simcase = [{'dt': 0.01,  # Time step size of the output in [s]
                         # Time step size for the integration scheme, only applicable in case of unsteady cfd simulation
                         'dt_integration': 0.001,
                         # Final simulation time  in [s]
                         't_final': 2.0,
                         # True or False, enables 1-cosine gust according to CS-25
                         'gust': False,
                         # Gust gradient H (half gust length) in [m]
                         'gust_gradient': 9.0,
                         # Orientation of the gust in [deg], 0/360 = gust from bottom, 180 = gust from top,
                         # 90 = gust from the right, 270 = gust from the left, arbitrary values possible
                         # (rotation of gust direction vector about Nastran's x-axis pointing backwards)
                         'gust_orientation': 0,
                         # Gust parameters according to CS-25 to calculate the gust velocity
                         'gust_para': {'Z_mo': 12500.0, 'MLW': 65949.0, 'MTOW': 73365.0, 'MZFW': 62962.0, 'MD': 0.87,
                                       'T1': 0.00},
                         # Alternatively, give gust velocity / Vtas directly
                         'WG_TAS': 0.1,
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
                         # List of DoF to be constrained
                         'support': [0, 1, 2, 3, 4, 5],
                         # True or False, enables flutter check with k, ke or pk method
                         'flutter': False,
                         # flutter parameters for k and ke method
                         'flutter_para': {'method': 'k', 'k_red': np.linspace(2.0, 0.001, 1000)},
                         # flutter parameters for pk method
                         # 'flutter_para': {'method': 'pk', 'Vtas': np.linspace(100.0, 500.0, 100)},
                         },
                        ]
        # End
