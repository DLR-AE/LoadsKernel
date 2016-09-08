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
                        'MAC_ref': [6.0, 0.0, 0.0],
                       }
        self.efcs = {'version': 'mephisto'}
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
                     'filename_KGG':'',
                     'filename_KFF':'/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/KAA_clamped.dat',
                     'filename_uset': '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/uset_clamped.op2',
                     'filename_GM': '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/GM_clamped.dat',
                     'filename_aset': '',
                     'filename_monpnt': '',
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
        self.aero = {'method': 'mona_steady',
                     'flex': True,
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
                     'method_AIC': 'ae', # 'nastran', 'ae' - provide 'filename_AIC' with OP4 files if method = 'nastran'
                     'filename_AIC': ['/scratch/DLR-F19-S_150217_work/manloads_starr_DLR-F19-S/aio/AJJ01.dat', \
                                      '/scratch/DLR-F19-S_150217_work/manloads_starr_DLR-F19-S/aio/AJJ02.dat', \
                                      '/scratch/DLR-F19-S_150217_work/manloads_starr_DLR-F19-S/aio/AJJ03.dat', \
                                      '/scratch/DLR-F19-S_150217_work/manloads_starr_DLR-F19-S/aio/AJJ04.dat', \
                                     ],
                    }
        self.spline = {'method': 'nearest_neighbour', # 'nearest_neighbour', 'rbf', 'nastran'
                       'filename_f06': '/scratch/DLR-F19-S_150217_work/manloads_starr_DLR-F19-S/aio/trim_matrices_aio.f06',
                       'splinegrid': True, # if true, provide filename_grid, not valid and ignored when spline method = 'nastran'
                       'filename_splinegrid': '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/SplineKnoten/splinegrid.bdf'
                      }
        self.mass = {'method': 'modalanalysis', # 'mona', 'modalanalysis'
                       'key': ['M', 'MT1rT2rT1lT2l', 'MT1rT2rPrT1lT2lPl', 'MPrPl', 'BFDM'],
                       'filename_MGG':['/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MGG_M_clamped.dat',
                                       '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MGG_MT1rT2rT1lT2l_clamped.dat',
                                       '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MGG_MT1rT2rPrT1lT2lPl_cla.dat',
                                       '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MGG_MPrPl_clamped.dat',
                                       '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MGG_BFDM_clamped.dat'],
                       'filename_MFF':['/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MAA_M_clamped.dat',
                                       '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MAA_MT1rT2rT1lT2l_clamped.dat',
                                       '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MAA_MT1rT2rPrT1lT2lPl_cla.dat',
                                       '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MAA_MPrPl_clamped.dat',
                                       '/scratch/DLR-F19-S_150217_work/assembly_DLR-F-19-S/nastran/MAA_BFDM_clamped.dat'],
                       'filename_S103':[], 
                       'omit_rb_modes': False, 
                       'modes':[np.arange(1,13), np.arange(1,16), np.arange(1,15), np.arange(1,12), np.arange(1,12)]           
                      }
        self.atmo = {'method':'ISA', 
                     'key':['FL000','FL055', 'FL075', 'FL200', 'FL300', 'FL450'],
                     'h': ft2m([0, 5500, 7500, 20000, 30000, 45000]),
                    }
        self.simcase = [{}]
        self.trimcase = [{'desc': 'CC.BFDM.OVCFL000.Vergleichsfall53', 
                          'manoeuver': 'Vergleichsfall53', 
                          'subcase': 53,
                          'Ma': 0.8, 
                          'aero': 'VC', 
                          'altitude': 'FL000', 
                          'mass': 'BFDM',
                          'Nz': 5.0, 
                          'p': 34.3/180.0*np.pi,
                          'q': 28.6/180.0*np.pi, 
                          'pdot': -286.5/180.0*np.pi, 
                          'qdot': 0.0, 
                         },
                        ]

        # End

    
    
    