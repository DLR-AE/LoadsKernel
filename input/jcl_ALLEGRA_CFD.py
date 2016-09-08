# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 10:46:53 2014

@author: voss_ar
"""

# JCL - Job Control File
#
# ToDo: more than one trim case

import numpy as np
from units import *

class jcl:
    def __init__(self):
        self.general = {'aircraft':'ALLEGRA',
                        'b_ref': 35.68,
                        'c_ref': 4.01,
                        'A_ref': 132.0,
                        'MAC_ref': [0.0, 0.0, 0.0],
                       }
        self.geom = {'method': 'mona',
                     'filename_grid':['/work/hand_ve/Transfer/Arne/ALLEGRA/allegra-s_c05.bdf', ],
                     'filename_KGG':'/work/hand_ve/Transfer/Arne/ALLEGRA/KGG_MTFFa.dat',
                     'filename_monpnt': '/work/hand_ve/Transfer/Arne/ALLEGRA/allegra-s_monpnt1_vert6.bdf',
                     'filename_mongrid': '', # if no MONPNTs are available: file with GRIDs where to create MONPNTs
                     'filename_moncoord':'', # if no MONPNTs are available: additional coordiante systems
                     'filename_monstations': [], # if no MONPNTs are available: one file per MONPNT containing all GRIDs to be integrated
                    }
        self.aero = {'method': 'hybrid', # 'mona_steady', 'hybrid'
                     'flex': True,
                     'key':['MC',],
                     'Ma': [0.7966,],
                     'method_caero': 'CAERO1', # 'CAERO1', 'CQUAD4'
                     'filename_caero_bdf': ['/work/hand_ve/Transfer/Arne/ALLEGRA/allegra-s_CAERO1_1g-flight-shape_with_1fuse.bdf'],
                     'filename_deriv_4_W2GJ': [],
                     'filename_aesurf': ['/work/hand_ve/Transfer/Arne/ALLEGRA/allegra-s_CAERO1_1g-flight-shape_with_1fuse.bdf'],
                     'filename_aelist': ['/work/hand_ve/Transfer/Arne/ALLEGRA/allegra-s_CAERO1_1g-flight-shape_with_1fuse.bdf'],
                     'method_AIC': 'nastran',
                     'filename_AIC': ['/work/hand_ve/Transfer/Arne/ALLEGRA/OC230/AJJ01.dat', ],
                    }
        self.efcs = {'version': 'allegra'}
                    
        self.matrix_aerodb = {}
        self.matrix_aerodb['alpha'] = {}
        self.matrix_aerodb['alpha']['MC'] = {   'markers': 'all', # 'all' or list of markers [1, 2, ...]
                                                'filename_grid': '', # only if special markers are requested
                                                'q_dyn': 0.5*0.5888*248.667589877509**2,                                
                                                'values': [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                                'filenames_surface_pval': ['/work/hand_ve/Transfer/Arne/ALLEGRA/CFD/Allegra_v11_fl230ma08alp-4/Allegra_v11_fl230ma08.surface.pval.10000',
                                                                           '/work/hand_ve/Transfer/Arne/ALLEGRA/CFD/Allegra_v11_fl230ma08alp-3/Allegra_v11_fl230ma08.surface.pval.3395',
                                                                           '/work/hand_ve/Transfer/Arne/ALLEGRA/CFD/Allegra_v11_fl230ma08alp-2/Allegra_v11_fl230ma08.surface.pval.3519',
                                                                           '/work/hand_ve/Transfer/Arne/ALLEGRA/CFD/Allegra_v11_fl230ma08alp-1/Allegra_v11_fl230ma08.surface.pval.3362',
                                                                           '/work/hand_ve/Transfer/Arne/ALLEGRA/CFD/Allegra_v11_fl230ma08alp0/Allegra_v11_fl230ma08.surface.pval.3552',
                                                                           '/work/hand_ve/Transfer/Arne/ALLEGRA/CFD/Allegra_v11_fl230ma08alp1/Allegra_v11_fl230ma08.surface.pval.3593',
                                                                           '/work/hand_ve/Transfer/Arne/ALLEGRA/CFD/Allegra_v11_fl230ma08alp2/Allegra_v11_fl230ma08.surface.pval.3540',
                                                                           '/work/hand_ve/Transfer/Arne/ALLEGRA/CFD/Allegra_v11_fl230ma08alp3/Allegra_v11_fl230ma08.surface.pval.3576',
                                                                           '/work/hand_ve/Transfer/Arne/ALLEGRA/CFD/Allegra_v11_fl230ma08alp4/Allegra_v11_fl230ma08.surface.pval.3590',
                                                                           '/work/hand_ve/Transfer/Arne/ALLEGRA/CFD/Allegra_v11_fl230ma08alp5/Allegra_v11_fl230ma08.surface.pval.3706',
                                                                           '/work/hand_ve/Transfer/Arne/ALLEGRA/CFD/Allegra_v11_fl230ma08alp6/Allegra_v11_fl230ma08.surface.pval.4008',],
                                            }                            
        self.spline = {'method': 'nastran', # 'nearest_neighbour', 'rbf', 'nastran'
                       'filename_f06': '/work/hand_ve/Transfer/Arne/ALLEGRA/main_c05_sol144.f06',
                       'splinegrid': False, # if true, provide filename_grid, not valid and ignored when spline method = 'nastran'
                       'filename_splinegrid': ''
                      }
        self.mass = {'method': 'mona',
                       'key': ['MTFFa',],
                       'filename_MGG':['/work/hand_ve/Transfer/Arne/ALLEGRA/MGG_MTFFa.dat',],
                       'filename_S103':['/work/hand_ve/Transfer/Arne/ALLEGRA/allegra-s_fem_c05.f06',], 
                       'omit_rb_modes': True, 
                       'modes':[np.arange(1,56)]           
                      }
        self.atmo = {'method':'ISA', 
                     'key':['FL000', 'FL055', 'FL075', 'FL200', 'FL230', 'FL300', 'FL450'],
                     'h': ft2m([0, 5500, 7500, 20000, 23000, 30000, 45000]),
                    }
        self.simcase = [{}, {}, {}, {}, {}]
        self.trimcase = [{'desc': '#1019', 
                          'manoeuver': 'pitch', 
                          'subcase': 1019,
                          'Ma': 0.7966, 
                          'aero': 'MC', 
                          'altitude': 'FL230', 
                          'mass': 'MTFFa',
                          'Nz': 1.0, 
                          'p': 0.0,
                          'q': 0.0, 
                          'r': 0.0,
                          'pdot': 0.0, 
                          'qdot': 0.0, 
                          'rdot': 0.0,
                         },
                         {'desc': '#1119', 
                          'manoeuver': 'pitch', 
                          'subcase': 1119,
                          'Ma': 0.7966, 
                          'aero': 'MC', 
                          'altitude': 'FL230', 
                          'mass': 'MTFFa',
                          'Nz': 2.5, 
                          'p': 0.0,
                          'q': 0.0, 
                          'r': 0.0,
                          'pdot': 0.0, 
                          'qdot': 0.0, 
                          'rdot': 0.0,
                         },
                         {'desc': '#1219', 
                          'manoeuver': 'pitch', 
                          'subcase': 1219,
                          'Ma': 0.7966, 
                          'aero': 'MC', 
                          'altitude': 'FL230', 
                          'mass': 'MTFFa',
                          'Nz': 2.5, 
                          'p': 0.0,
                          'q': 0.059167, 
                          'r': 0.0,
                          'pdot': 0.0, 
                          'qdot': 0.0, 
                          'rdot': 0.0,
                         },
                         {'desc': '#1319', 
                          'manoeuver': 'pitch', 
                          'subcase': 1319,
                          'Ma': 0.7966, 
                          'aero': 'MC', 
                          'altitude': 'FL230', 
                          'mass': 'MTFFa',
                          'Nz': -1.0, 
                          'p': 0.0,
                          'q': 0.0, 
                          'r': 0.0,
                          'pdot': 0.0, 
                          'qdot': 0.0, 
                          'rdot': 0.0,
                         },
                         {'desc': '#1419', 
                          'manoeuver': 'pitch', 
                          'subcase': 1419,
                          'Ma': 0.7966, 
                          'aero': 'MC', 
                          'altitude': 'FL230', 
                          'mass': 'MTFFa',
                          'Nz': -1.0, 
                          'p': 0.0,
                          'q': -.078890, 
                          'r': 0.0,
                          'pdot': 0.0, 
                          'qdot': 0.0, 
                          'rdot': 0.0,
                         },
                        ]
                        
#        from numpy import array
#        with open('/scratch/kernel_pre_20150821/input/jcl_DLR_F19_CFD.trimcase_dict', 'r') as fid:
#            trimcase_str = fid.read()
#        self.trimcase = eval(trimcase_str)
        # End

    
    
    