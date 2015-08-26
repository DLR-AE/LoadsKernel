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
        self.general = {'aircraft':'DLR F-19-S',
                        'b_ref': 15.375,
                        'c_ref': 4.79,
                        'A_ref': 77,
                        'MAC_ref': [0.0, 0.0, 0.0],
                       }
        self.geom = {'method': 'mona',
                     'filename_grid':['/scratch/DLR-F19-S_150217_work/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.GRID',
                                      '/scratch/DLR-F19-S_150217_work/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.GRID',
                                      '/scratch/DLR-F19-S_150217_work/mg03_DLR-F19-S_Klappen/output/Klappen_innen.GRID',
                                      '/scratch/DLR-F19-S_150217_work/mg03_DLR-F19-S_Klappen/output/Klappen_aussen.GRID',
                                      '/scratch/DLR-F19-S_150217_work/mg03_DLR-F19-S_Klappen_LinkeSeite/output/Klappen_innen.GRID',
                                      '/scratch/DLR-F19-S_150217_work/mg03_DLR-F19-S_Klappen_LinkeSeite/output/Klappen_aussen.GRID',
                                      '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/massconfig/special.rbe3',
                                      '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/massconfig/mg02_DLR-F19-S_baseline.RBE3_SUBSEG2_mod',
                                      '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/massconfig/mg05_DLR-F19-S_baseline.RBE3_SUBSEG2_mod',
                                      '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/Klappenanbindung.bdf',
                                      ],
                     'filename_KGG':'/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/KGG_dim.dat',
                     'filename_mongrid': '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_grids.bdf',
                     'filename_monstations': ['/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon08.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon01.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon02.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon03.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon04.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon05.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon06.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon07.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon20.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon22.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon23.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon24.bdf',
                                                 ],
                    }
        self.aero = {'method': 'mona_steady_corrected', # mona_steady, mona_steady_corrected
                     'key':['VC', 'MC'],
                     'Ma': [0.8, 0.9],
                     'filename_caero_bdf': ['/scratch/DLR-F19-S_150217_work/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.CAERO1_bdf', 
                                            '/scratch/DLR-F19-S_150217_work/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.CAERO1_bdf'],
                     'filename_deriv_4_W2GJ': ['/scratch/DLR-F19-S_150217_work/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.deriv_4_W2GJ', 
                                               '/scratch/DLR-F19-S_150217_work/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.deriv_4_W2GJ'],
                     'filename_aesurf': ['/scratch/DLR-F19-S_150217_work/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.AESURF', 
                                         '/scratch/DLR-F19-S_150217_work/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.AESURF'],
                     'filename_aelist': ['/scratch/DLR-F19-S_150217_work/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.AELIST', 
                                         '/scratch/DLR-F19-S_150217_work/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.AELIST'],
                     'filename_AIC': ['/scratch/DLR-F19-S_150217_work/trim_DLR-F19-S/aic/AJJ01.dat', \
                                      '/scratch/DLR-F19-S_150217_work/trim_DLR-F19-S/aic/AJJ02.dat', \
                                     ],
                     'filename_correction': ['/scratch/DLR-F19_tau/downwash_corr.pickle',\
                                             '/scratch/DLR-F19_tau/downwash_corr.pickle',\
                                            ]
                    }
        self.spline = {'method': 'nearest_neighbour', # 'nearest_neighbour', 'rbf', 'nastran'
                       'filename_f06': '/scratch/DLR-F19-S_150217_work/trim_DLR-F19-S/test_trim/test_trim_BFDM_loop3.f06',
                       'splinegrid': True, # if true, provide filename_grid, not valid when spline method = 'nastran'
                       'filename_splinegrid': '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/SplineKnoten/splinegrid.bdf'
                      }
        self.mass = {'method': 'mona',
                       'key': ['BFDM'],
                       'filename_MGG':['/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MGG_BFDM_dim.dat'],
                       'filename_S103':['/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/dim_crosscheck_SOL103_BFDM.f06',
                                       ], 
                       'omit_rb_modes': True, 
                       'modes':[np.arange(1,11), np.arange(1,15)]    # 15           
                      }
        self.atmo = {'method':'ISA', 
                     'key':['FL000','FL055', 'FL075', 'FL300', 'FL450'],
                     'h': ft2m([0, 5500, 7500, 30000, 45000]),
                    }
        self.trimcase = [{'desc': 'CC.BFDM.OVCFL000.LLFPUNz25', 
                          'manoeuver': 'LLFPUNz25', 
                          'Ma': 0.8, 
                          'aero': 'VC', 
                          'altitude': 'FL000', 
                          'mass': 'BFDM',
                          'Nz': 2.5, 
                          'p': 0.0,
                          'q': 0.055, 
                          'pdot': 0.0, 
                          'qdot': 0.0, 
                         },
                        ]
                        
#        from numpy import array
#        fid = open('/scratch/DLR-F19-S_150217_work/trim_DLR-F19-S/test_trim.trimcase_dict', 'r')
#        trimcase_str = fid.read()
#        fid.close()
#        self.trimcase = eval(trimcase_str)
        
        # End

    
    
    