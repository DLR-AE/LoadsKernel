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
        self.aero = {'method': 'mona_steady', # 'mona_steady', 'hybrid'
                     'flex': True,
                     'key':['MC',],
                     'Ma': [0.7966,],
                     'method_caero': 'CAERO1', # 'CAERO1', 'CQUAD4'
                     'filename_caero_bdf': ['/work/hand_ve/Transfer/Arne/ALLEGRA/allegra-s_CAERO1_1g-flight-shape_with_1fuse.bdf'],
                     'filename_deriv_4_W2GJ': [],
                     'filename_aesurf': ['/work/hand_ve/Transfer/Arne/ALLEGRA/allegra-s_CAERO1_1g-flight-shape_with_1fuse.bdf'],
                     'filename_aelist': ['/work/hand_ve/Transfer/Arne/ALLEGRA/allegra-s_CAERO1_1g-flight-shape_with_1fuse.bdf'],
                     'filename_AIC': ['/work/hand_ve/Transfer/Arne/ALLEGRA/OC230/AJJ01.dat', ],
                    }
        self.efcs = {'version': 'allegra'}
                    
        self.matrix_aerodb = {}
                         
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
        self.trimcase = [{'desc': '#1019', 
                          'manoeuver': 'LLFLevel', 
                          'subcase': 1,
                          'Ma': 0.7966, 
                          'aero': 'MC', 
                          'altitude': 'FL230', 
                          'mass': 'MTFFa',
                          'Nz': 1.0, 
                          'p': 0.0,
                          'q': 0.0, 
                          'pdot': 0.0, 
                          'qdot': 0.0, 
                         },
                        ] 
        gust_para = {'Z_mo': ft2m(45000), 'MLW': 100000, 'MTOW': 120000, 'MZFW': 50000, 'MD': 0.95, 'T1': 0.02}
        self.simcase = [{'dt': 0.01, 
                         't_final': 2.0,
                         'gust': False, 
                         'gust_gradient': 35,
                         'gust_orientation': 360, # degree, 0/360° = gust from bottom
                         'gust_para':gust_para,
                        }]

                        
#        from numpy import array
#        with open('/scratch/kernel_pre_20150821/input/jcl_DLR_F19_CFD.trimcase_dict', 'r') as fid:
#            trimcase_str = fid.read()
#        self.trimcase = eval(trimcase_str)
        # End

    
    
    