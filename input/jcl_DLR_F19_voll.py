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
        self.geom = {'method': 'mona',
                     'filename_grid':['/scratch/DLR-F19-S_150126/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.GRID',
                                      '/scratch/DLR-F19-S_150126/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.GRID',
                                      '/scratch/DLR-F19-S_150126/mg03_DLR-F19-S_Klappen/output/Klappen_innen.GRID',
                                      '/scratch/DLR-F19-S_150126/mg03_DLR-F19-S_Klappen/output/Klappen_aussen.GRID',
                                      '/scratch/DLR-F19-S_150126/mg03_DLR-F19-S_Klappen_LinkeSeite/output/Klappen_innen.GRID',
                                      '/scratch/DLR-F19-S_150126/mg03_DLR-F19-S_Klappen_LinkeSeite/output/Klappen_aussen.GRID'],
                     #'filename_KAA':'/scratch/DLR-F19-S_150126/assembly_DLR-F-19-S/matrix_export/KAA.dat',
                    }
        self.aero = {'method': 'mona_steady',
                     'key':['VC_FL000', 'MC', 'VD_FL000', 'MD'],
                     'Ma': [0.8, 0.9, 0.89, 0.97],
                     'filename_caero_bdf': ['/scratch/DLR-F19-S_150126/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.CAERO1_bdf', 
                                            '/scratch/DLR-F19-S_150126/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.CAERO1_bdf'],
                     'filename_deriv_4_W2GJ': ['/scratch/DLR-F19-S_150126/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.deriv_4_W2GJ', 
                                               '/scratch/DLR-F19-S_150126/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.deriv_4_W2GJ'],
                     'filename_aesurf': ['/scratch/DLR-F19-S_150126/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.AESURF', 
                                         '/scratch/DLR-F19-S_150126/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.AESURF'],
                     'filename_aelist': ['/scratch/DLR-F19-S_150126/mg02_DLR-F19-S/output/mg02_DLR-F19-S_baseline.AELIST', 
                                         '/scratch/DLR-F19-S_150126/mg05_DLR-F19-S_LinkeSeite/output/mg05_DLR-F19-S_baseline.AELIST'],
                     'filename_AIC': ['/scratch/DLR-F19-S_150126/trim_DLR-F19-S/aic/AJJ01.dat', \
                                      '/scratch/DLR-F19-S_150126/trim_DLR-F19-S/aic/AJJ02.dat', \
                                      '/scratch/DLR-F19-S_150126/trim_DLR-F19-S/aic/AJJ03.dat', \
                                      '/scratch/DLR-F19-S_150126/trim_DLR-F19-S/aic/AJJ04.dat', \
                                     ],
                    }
        self.mass = {'method': 'mona',
                       'key': ['M'],
                       #'filename_MAA':['/scratch/DLR-F19-S_150126/assembly_DLR-F-19-S/matrix_export/MAA_M.dat'],
                       'filename_S103':['/scratch/DLR-F19-S_150126/assembly_DLR-F-19-S/nastran/dim_crosscheck_SOL103_M.f06'], 
                       'omit_rb_modes': True, 
                       'modes':np.arange(1,11),                        
                      }
        self.atmo = {'method':'ISA', 
                     'key':['FL000','FL055', 'FL075', 'FL300', 'FL450'],
                     'h': ft2m([0, 5500, 7500, 30000, 45000]),
                    }
        self.trimcase = {'altitude': 'FL450',
                         'Ma': 0.9, # cas2Ma(cas, altitude)
                         'aero':'MC',
                         'manoeuver':'PU',
                         'Nz': 2.5,
                         'Cl_max':1.5,
                         'mass':'M',
                        }
        # End

    
    
    