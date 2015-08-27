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
                                      '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/Einspannung.bdf',
                                      ],
                     'filename_KGG':'/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/KGG_dim.dat',
                     'filename_mongrid': '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_grids.bdf',
                     'filename_moncoord':'/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_coords.bdf',
                     'filename_monstations': ['/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon09.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon08.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon01.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon02.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon03.bdf',
                                                 '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/monstations/monstations_Mon33.bdf',
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
        self.aero = {'method': 'hybrid', # 'mona_steady', 'hybrid'
                     'key':['VC', 'MC', 'VD', 'MD'],
                     'Ma': [0.8, 0.9, 0.89, 0.97],
                     'filename_caero_bdf': ['/scratch/DLR-F19-S_150217_work/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.CAERO1_bdf', 
                                            '/scratch/DLR-F19-S_150217_work/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.CAERO1_bdf'],
                     'filename_deriv_4_W2GJ': ['/scratch/DLR-F19-S_150217_work/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.deriv_4_W2GJ', 
                                               '/scratch/DLR-F19-S_150217_work/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.deriv_4_W2GJ'],
                     'filename_aesurf': ['/scratch/DLR-F19-S_150217_work/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.AESURF', 
                                         '/scratch/DLR-F19-S_150217_work/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.AESURF'],
                     'filename_aelist': ['/scratch/DLR-F19-S_150217_work/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.AELIST', 
                                         '/scratch/DLR-F19-S_150217_work/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.AELIST'],
                     'filename_AIC': ['/scratch/DLR-F19-S_150217_work/manloads_starr_DLR-F19-S/aio/AJJ01.dat', \
                                      '/scratch/DLR-F19-S_150217_work/manloads_starr_DLR-F19-S/aio/AJJ02.dat', \
                                      '/scratch/DLR-F19-S_150217_work/manloads_starr_DLR-F19-S/aio/AJJ03.dat', \
                                      '/scratch/DLR-F19-S_150217_work/manloads_starr_DLR-F19-S/aio/AJJ04.dat', \
                                     ],
                    }
                    
        self.matrix_aerodb = {}
        self.matrix_aerodb['alpha'] = {}
        self.matrix_aerodb['alpha']['VC'] = {   'markers': 'all', # 'all' or list of markers [1, 2, ...]
                                                'filename_grid': '/scratch/DLR-F19_centaur_euler2/DLR-F19_AIL.grid', # only if special markers are requested
                                                'q_dyn': 0.5*1.225*265.030316756404**2,                                
                                                'values': [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                                                'filenames_surface_pval': ['/marvin/work/92240-F19-1447591004/alpha/sol/sol_polar_AoA-2.0.surface.pval.1353',
                                                                           '/marvin/work/92240-F19-1447591004/alpha/sol/sol_polar_AoA-1.0.surface.pval.851',
                                                                           '/marvin/work/92240-F19-1447591004/alpha/sol/sol_steady.surface.pval.551',
                                                                           '/marvin/work/92240-F19-1447591004/alpha/sol/sol_polar_AoA1.0.surface.pval.568',
                                                                           '/marvin/work/92240-F19-1447591004/alpha/sol/sol_polar_AoA2.0.surface.pval.538',
                                                                           '/marvin/work/92240-F19-1447591004/alpha/sol/sol_polar_AoA3.0.surface.pval.536',
                                                                           '/marvin/work/92240-F19-1447591004/alpha/sol/sol_polar_AoA4.0.surface.pval.770',
                                                                           '/marvin/work/92240-F19-1447591004/alpha/sol/sol_polar_AoA5.0.surface.pval.1217',
                                                                           '/marvin/work/92240-F19-1447591004/alpha/sol/sol_polar_AoA6.0.surface.pval.1629'],
                                            }
        self.matrix_aerodb['AIL-S1'] = {}
        self.matrix_aerodb['AIL-S1']['VC'] = {  'markers': 'all', # 'all' or list of markers [1, 2, ...]
                                                'filename_grid': '', # only if special markers are requested
                                                'q_dyn': 0.5*1.225*265.030316756404**2,                                
                                                'values': [-5.0, 0.0, 5.0 ],
                                                'filenames_surface_pval': ['/marvin/work/92240-F19-1447591004/eta/sol/sol_mesh_state1.surface.pval.820',
                                                                           '/marvin/work/92240-F19-1447591004/eta/sol/sol_steady.surface.pval.837',
                                                                           '/marvin/work/92240-F19-1447591004/eta/sol/sol_mesh_state2.surface.pval.865'],
                                             }
        self.matrix_aerodb['AIL-S2'] = {}
        self.matrix_aerodb['AIL-S2']['VC'] = {  'markers': 'all', # 'all' or list of markers [1, 2, ...]
                                                'filename_grid': '', # only if special markers are requested
                                                'q_dyn': 0.5*1.225*265.030316756404**2,                                
                                                'values': [-5.0, 0.0, 5.0 ],
                                                'filenames_surface_pval': ['/marvin/work/92240-F19-1447591004/eta/sol/sol_mesh_state3.surface.pval.839',
                                                                           '/marvin/work/92240-F19-1447591004/eta/sol/sol_steady.surface.pval.837',
                                                                           '/marvin/work/92240-F19-1447591004/eta/sol/sol_mesh_state4.surface.pval.867'],
                                             }
        self.matrix_aerodb['AIL-S3'] = {}
        self.matrix_aerodb['AIL-S3']['VC'] = {  'markers': 'all', # 'all' or list of markers [1, 2, ...]
                                                'filename_grid': '', # only if special markers are requested
                                                'q_dyn': 0.5*1.225*265.030316756404**2,                                
                                                'values': [-5.0, 0.0, 5.0 ],
                                                'filenames_surface_pval': ['/marvin/work/92240-F19-1447591004/eta/sol/sol_mesh_state5.surface.pval.823',
                                                                           '/marvin/work/92240-F19-1447591004/eta/sol/sol_steady.surface.pval.837',
                                                                           '/marvin/work/92240-F19-1447591004/eta/sol/sol_mesh_state6.surface.pval.842'],
                                             }
        self.matrix_aerodb['AIL-S4'] = {}
        self.matrix_aerodb['AIL-S4']['VC'] = {  'markers': 'all', # 'all' or list of markers [1, 2, ...]
                                                'filename_grid': '', # only if special markers are requested
                                                'q_dyn': 0.5*1.225*265.030316756404**2,                                
                                                'values': [-5.0, 0.0, 5.0 ],
                                                'filenames_surface_pval': ['/marvin/work/92240-F19-1447591004/eta/sol/sol_mesh_state7.surface.pval.821',
                                                                           '/marvin/work/92240-F19-1447591004/eta/sol/sol_steady.surface.pval.837',
                                                                           '/marvin/work/92240-F19-1447591004/eta/sol/sol_mesh_state8.surface.pval.395'],
                                             }
                            
        self.spline = {'method': 'nearest_neighbour', # 'nearest_neighbour', 'rbf', 'nastran'
                       'filename_f06': '/scratch/DLR-F19-S_150217_work/manloads_starr_DLR-F19-S/aio/trim_matrices_aio.f06',
                       'splinegrid': True, # if true, provide filename_grid, not valid and ignored when spline method = 'nastran'
                       'filename_splinegrid': '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/SplineKnoten/splinegrid.bdf'
                      }
        self.mass = {'method': 'mona',
                       'key': ['M', 'MT1rT2rT1lT2l', 'MT1rT2rPrT1lT2lPl', 'MPrPl', 'BFDM'],
                       'filename_MGG':['/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MGG_M_clamped.dat',
                                       '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MGG_MT1rT2rT1lT2l_clamped.dat',
                                       '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MGG_MT1rT2rPrT1lT2lPl_cla.dat',
                                       '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MGG_MPrPl_clamped.dat',
                                       '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MGG_BFDM_clamped.dat'],
                       'filename_S103':['/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/dim_crosscheck_SOL103_M_clamped.f06', 
                                        '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/dim_crosscheck_SOL103_MT1rT2rT1lT2l_clamped.f06',
                                        '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/dim_crosscheck_SOL103_MT1rT2rPrT1lT2lPl_clamped.f06',
                                        '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/dim_crosscheck_SOL103_MPrPl_clamped.f06',
                                        '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/dim_crosscheck_SOL103_BFDM_clamped.f06',
                                       ], 
                       'omit_rb_modes': False, 
                       'modes':[np.arange(1,13), np.arange(1,16), np.arange(1,15), np.arange(1,12), np.arange(1,12)]           
                      }
        self.atmo = {'method':'ISA', 
                     'key':['FL000','FL055', 'FL075', 'FL200', 'FL300', 'FL450'],
                     'h': ft2m([0, 5500, 7500, 20000, 30000, 45000]),
                    }
        self.trimcase = [{'desc': 'CC.BFDM.OVCFL000.LLFPUNz25', 
                          'manoeuver': 'LLFPUNz25', 
                          'subcase': 1,
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
                        
        
        # End

    
    
    