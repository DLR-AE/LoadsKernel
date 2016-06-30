# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 10:53:48 2014

@author: voss_ar
"""
import build_mass
import build_aero
import build_aerodb
import spline_rules
import spline_functions
import build_splinegrid
import read_geom
import write_functions
from grid_trafo import grid_trafo
from  atmo_isa import atmo_isa

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cPickle, sys
from oct2py import octave 

class model:
    def __init__(self,jcl):
        self.jcl = jcl
        for dir in sys.path:
            octave.addpath(dir) # add path in octave so the m-file(s) are found
    
    def write_aux_data(self, path_output):
        with open(path_output + 'monitoring_stations.bdf', 'w') as fid:
            fid.write('$ This is a file generated by Loads Kernel.\n')
            fid.write('$ It includes all monitoring stations in the form of Nastran cards.\n')
            write_functions.write_MONPNT1(fid, self.mongrid, self.mongrid_rules)
            
        if self.jcl.spline['method'] in ['rbf', 'nearest_neighbour']:
            with open(path_output + 'splinegrid.set1', 'w') as fid:
                fid.write('$ This is a file generated by Loads Kernel.\n')
                fid.write('$ It includes all grid points of the splinegrid, which is used for aero-structural coupling in the Loads Kernel.\n')
                write_functions.write_SET1(fid,1001, self.splinegrid['ID'])        
        
        
    def build_model(self):
        self.coord = {'ID': [0, 9300],
                      'RID': [0, 0],
                      'dircos': [np.eye(3), np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])],
                      'offset': [np.array([0,0,0]), np.array([0,0,0])],
                     }        
        print 'Building structural model...'
        if self.jcl.geom['method'] == 'mona':
            
            for i_file in range(len(self.jcl.geom['filename_grid'])):
                subgrid = read_geom.Modgen_GRID(self.jcl.geom['filename_grid'][i_file]) 
                if i_file == 0:
                    self.strcgrid = subgrid
                else:
                    self.strcgrid['ID'] = np.hstack((self.strcgrid['ID'],subgrid['ID']))
                    self.strcgrid['CD'] = np.hstack((self.strcgrid['CD'],subgrid['CD']))
                    self.strcgrid['CP'] = np.hstack((self.strcgrid['CP'],subgrid['CP']))
                    self.strcgrid['n'] += subgrid['n']
                    self.strcgrid['set'] = np.vstack((self.strcgrid['set'],subgrid['set']+self.strcgrid['set'].max()+1))
                    self.strcgrid['offset'] = np.vstack((self.strcgrid['offset'],subgrid['offset']))
                    
                self.coord = read_geom.Modgen_CORD2R(self.jcl.geom['filename_grid'][i_file], self.coord, self.strcgrid)
            
            # sort stucture grid to be in accordance with matricies such as Mgg from Nastran
            sort_vector = self.strcgrid['ID'].argsort()
            self.strcgrid['ID'] = self.strcgrid['ID'][sort_vector]
            self.strcgrid['CD'] = self.strcgrid['CD'][sort_vector]
            self.strcgrid['CP'] = self.strcgrid['CP'][sort_vector]
            #self.strcgrid['set'] = self.strcgrid['set'][sort_vector,:]
            self.strcgrid['offset'] = self.strcgrid['offset'][sort_vector,:]
            
            # make sure the strcgrid is in one common coordinate system with ID = 0 (basic system)
            grid_trafo(self.strcgrid, self.coord, 0)

            #self.Kgg = read_geom.Nastran_OP4(self.jcl.geom['filename_KGG'], sparse_output=True, sparse_format=True) 
            
            if not self.jcl.geom['filename_mongrid'] == '':
                print 'Building Monitoring Stations from GRID data...' 
                self.mongrid = read_geom.Modgen_GRID(self.jcl.geom['filename_mongrid']) 
                self.coord = read_geom.Modgen_CORD2R(self.jcl.geom['filename_moncoord'], self.coord)
                rules = spline_rules.monstations_from_bdf(self.mongrid, self.jcl.geom['filename_monstations'])
                self.PHIstrc_mon = spline_functions.spline_rb(self.mongrid, '', self.strcgrid, '', rules, self.coord, sparse_output=True)
                self.mongrid_rules = rules # save rules for optional writing of MONPNT1 cards
                    
            elif not self.jcl.geom['filename_monpnt'] == '':
                print 'Reading Monitoring Stations from MONPNTs...' 
                self.mongrid = read_geom.Nastran_MONPNT1(self.jcl.geom['filename_monpnt']) 
                self.coord = read_geom.Modgen_CORD2R(self.jcl.geom['filename_monpnt'], self.coord)
                rules = spline_rules.monstations_from_aecomp(self.mongrid, self.jcl.geom['filename_monpnt'])
                self.PHIstrc_mon = spline_functions.spline_rb(self.mongrid, '', self.strcgrid, '', rules, self.coord, sparse_output=True)
                self.mongrid_rules = rules # save rules for optional writing of MONPNT1 cards
                #spline_functions.plot_splinerules(self.mongrid, '', self.strcgrid, '', self.mongrid_rules, self.coord) 
            else: 
                print 'Warning: No Monitoring Stations are created!'

        print 'Building atmo model...'
        if self.jcl.atmo['method']=='ISA':
            self.atmo = {'key':[],
                         'h': [], 
                         'p': [],
                         'rho': [],
                         'T': [],
                         'a': [],
                        }
            for i_atmo in range(len(self.jcl.atmo['key'])):
                p, rho, T, a = atmo_isa(self.jcl.atmo['h'][i_atmo])
                self.atmo['key'].append(self.jcl.atmo['key'][i_atmo])
                self.atmo['h'].append(self.jcl.atmo['h'][i_atmo])
                self.atmo['p'].append(p)
                self.atmo['rho'].append(rho)
                self.atmo['T'].append(T)
                self.atmo['a'].append(a)

        else:
            print 'Unknown atmo method: ' + str(self.jcl.aero['method'])
              
        print 'Building aero model...'
        if self.jcl.aero['method'] in [ 'mona_steady', 'mona_unsteady', 'hybrid']:
            # grids
            for i_file in range(len(self.jcl.aero['filename_caero_bdf'])):
                if self.jcl.aero.has_key('method_caero'):
                    subgrid = build_aero.build_aerogrid(self.jcl.aero['filename_caero_bdf'][i_file], method_caero = self.jcl.aero['method_caero']) 
                else: # use default method defined in function
                    subgrid = build_aero.build_aerogrid(self.jcl.aero['filename_caero_bdf'][i_file]) 
                if i_file == 0:
                    self.aerogrid =  subgrid
                else:
                    self.aerogrid['ID'] = np.hstack((self.aerogrid['ID'],subgrid['ID']))
                    self.aerogrid['l'] = np.hstack((self.aerogrid['l'],subgrid['l']))
                    self.aerogrid['A'] = np.hstack((self.aerogrid['A'],subgrid['A']))
                    self.aerogrid['N'] = np.vstack((self.aerogrid['N'],subgrid['N']))
                    self.aerogrid['offset_l'] = np.vstack((self.aerogrid['offset_l'],subgrid['offset_l']))
                    self.aerogrid['offset_k'] = np.vstack((self.aerogrid['offset_k'],subgrid['offset_k']))
                    self.aerogrid['offset_j'] = np.vstack((self.aerogrid['offset_j'],subgrid['offset_j']))
                    self.aerogrid['set_l'] = np.vstack((self.aerogrid['set_l'],subgrid['set_l']+self.aerogrid['set_l'].max()+1))
                    self.aerogrid['set_k'] = np.vstack((self.aerogrid['set_k'],subgrid['set_k']+self.aerogrid['set_k'].max()+1))
                    self.aerogrid['set_j'] = np.vstack((self.aerogrid['set_j'],subgrid['set_j']+self.aerogrid['set_j'].max()+1))
                    self.aerogrid['CD'] = np.hstack((self.aerogrid['CD'],subgrid['CD']))
                    self.aerogrid['CP'] = np.hstack((self.aerogrid['CP'],subgrid['CP']))
                    self.aerogrid['n'] += subgrid['n']
                    self.aerogrid['cornerpoint_panels'] = np.vstack((self.aerogrid['cornerpoint_panels'],subgrid['cornerpoint_panels']))
                    self.aerogrid['cornerpoint_grids'] = np.vstack((self.aerogrid['cornerpoint_grids'],subgrid['cornerpoint_grids']))
                       
            # Correctionfor camber and twist, W2GJ
            if self.jcl.aero['filename_deriv_4_W2GJ']:
                # parsing of several files possible, must be in correct sequence
                for i_file in range(len(self.jcl.aero['filename_deriv_4_W2GJ'])):
                    subgrid = read_geom.Modgen_W2GJ(self.jcl.aero['filename_deriv_4_W2GJ'][i_file]) 
                    if i_file == 0:
                        self.camber_twist =  subgrid
                    else:
                        self.camber_twist['ID'] = np.hstack((self.camber_twist['ID'], subgrid['ID']))
                        self.camber_twist['cam_rad'] = np.hstack((self.camber_twist['cam_rad'], subgrid['cam_rad']))
            else:
                print 'No W2GJ data (correction of camber and twist) given, setting to zero'
                self.camber_twist = {'ID':self.aerogrid['ID'], 'cam_rad':np.zeros(self.aerogrid['ID'].shape)}
            
            # build mac grid from geometry, except other values are given in general section of jcl
            self.macgrid = build_aero.build_macgrid(self.aerogrid, self.jcl.general['b_ref'])
            if self.jcl.general.has_key('A_ref'):
                self.macgrid['A_ref'] = self.jcl.general['A_ref']
            if self.jcl.general.has_key('c_ref'):
                self.macgrid['c_ref'] = self.jcl.general['c_ref']
            if self.jcl.general.has_key('b_ref'):
                self.macgrid['b_ref'] = self.jcl.general['b_ref']
            if self.jcl.general.has_key('MAC_ref'):
                self.macgrid['offset'] = np.array([self.jcl.general['MAC_ref']])
            
            rules = spline_rules.rules_aeropanel(self.aerogrid)
            self.Djk = spline_functions.spline_rb(self.aerogrid, '_k', self.aerogrid, '_j', rules, self.coord)
            self.Dlk = spline_functions.spline_rb(self.aerogrid, '_k', self.aerogrid, '_l', rules, self.coord, sparse_output=True)
            
            rules = spline_rules.rules_point(self.macgrid, self.aerogrid)
            self.Dkx1 = spline_functions.spline_rb(self.macgrid, '', self.aerogrid, '_k', rules, self.coord)
            self.Djx1 = np.dot(self.Djk, self.Dkx1)
            #self.Dlx1 = np.dot(self.Dlk, self.Dkx1)
            
            # control surfaces
            self.x2grid, self.coord = build_aero.build_x2grid(self.jcl.aero, self.aerogrid, self.coord)            
            self.Djx2 = []
            for i_surf in range(len(self.x2grid['ID_surf'])):
                
                hingegrid = {'ID':np.array([self.x2grid['ID_surf'][i_surf]]),
                             'offset':np.array([[0,0,0]]),
                             'CD':np.array([self.x2grid['CID'][i_surf]]),
                             'CP':np.array([self.x2grid['CID'][i_surf]]), 
                             'set':np.array([[0,1,2,3,4,5]]),
                            }
                surfgrid = {'ID':self.x2grid['ID'][i_surf],
                            'offset_j':np.array(self.x2grid['offset_j'][i_surf]),
                            'CD':self.x2grid['CD'][i_surf],
                            'CP':self.x2grid['CP'][i_surf], 
                            'set_j':np.array(self.x2grid['set_j'][i_surf]), 
                           }
                dimensions = [6,6*len(self.aerogrid['ID'])]
                rules = spline_rules.rules_point(hingegrid, surfgrid)
                self.Djx2.append(spline_functions.spline_rb(hingegrid, '', surfgrid, '_j', rules, self.coord, dimensions))
                
#            Djx2_test = self.Djx2[0]
#            Uj = np.dot(Djx2_test,[0,0,0,0,20.0/180*np.pi,0])      
#            Djx2_test = self.Djx2[1]
#            Uj = np.dot(Djx2_test,[0,0,0,0,20.0/180*np.pi,0])  
#            Djx2_test = self.Djx2[2]
#            Uj = np.dot(Djx2_test,[0,0,0,0,20.0/180*np.pi,0])   
#            U_strc_x = self.aerogrid['offset_j'][:,0] + Uj[self.aerogrid['set_j'][:,0]]
#            U_strc_y = self.aerogrid['offset_j'][:,1] + Uj[self.aerogrid['set_j'][:,1]]
#            U_strc_z = self.aerogrid['offset_j'][:,2] + Uj[self.aerogrid['set_j'][:,2]]
#            
#            fig = plt.figure()
#            ax = fig.add_subplot(111, projection='3d')
#            ax.scatter(self.aerogrid['offset_j'][:,0], self.aerogrid['offset_j'][:,1], self.aerogrid['offset_j'][:,2], color='g', marker='.' )
#            ax.scatter(U_strc_x, U_strc_y, U_strc_z, color='r', marker='.' )
#            ax.set_xlabel('x')
#            ax.set_ylabel('y')
#            ax.set_zlabel('z')
#            ax.auto_scale_xyz([0, 50], [-25, 25], [0, 50])
#            plt.show()
            
        else:
            print 'Unknown aero method: ' + str(self.jcl.aero['method'])
            
        # AIC
        self.aero = {'key':[], 'Qjj':[],'interp_wj_corrfac_alpha': []}
        
        if self.jcl.aero['method_AIC'] == 'nastran':
            for i_aero in range(len(self.jcl.aero['key'])):
                Ajj = read_geom.Nastran_OP4(self.jcl.aero['filename_AIC'][i_aero], sparse_output=False, sparse_format=False)  
                Qjj = np.linalg.inv(Ajj.T)
                self.aero['key'].append(self.jcl.aero['key'][i_aero])
                self.aero['Qjj'].append(Qjj)
        elif self.jcl.aero['method_AIC'] in ['vlm', 'dlm', 'ae']:
            print 'Calculating steady AIC matrices ({} panels, k=0.0) with ae_getaic.m for {} Mach number(s)...'.format( self.aerogrid['n'], len(self.jcl.aero['key']) )
            #AIC = ae_getaic(aerogrid, Mach, k);
            out = octave.ae_getaic(self.aerogrid, self.jcl.aero['Ma'], [0.0])
            self.aero['key'] = self.jcl.aero['key']
            self.aero['Qjj'] = [out[i_aero,0,:,:,] for i_aero in range(len(self.jcl.aero['key']))] # dim: Ma,n,n
        else:
            print 'Unknown AIC method: ' + str(self.jcl.aero['method_AIC'])
        
        if self.jcl.aero['method_AIC'] == 'dlm':
            print 'Calculating unsteady AIC matrices ({} panels, k={} (Nastran Definition!)) with ae_getaic.m for {} Mach number(s)...'.format( self.aerogrid['n'], self.jcl.aero['k_red'], len(self.jcl.aero['key']) )
            # Definitions for reduced frequencies:
            # ae_getaic: k = omega/U 
            # Nastran:   k = 0.5*cref*omega/U
            out = octave.ae_getaic(self.aerogrid, self.jcl.aero['Ma'], np.array(self.jcl.aero['k_red'])/(0.5*self.jcl.general['c_ref']))
            self.aero['Qjj_unsteady'] = out # dim: Ma,k,n,n
            self.aero['k_red'] =  self.jcl.aero['k_red']
            
        if self.jcl.aero['method'] == 'mona_unsteady':
            # perform rfa
            self.aero['B'] = []
            for i_aero in range(len(self.jcl.aero['key'])):
                B, n_poles, betas = build_aero.rfa(self.aero['Qjj_unsteady'][i_aero,:,:,:], self.aero['k_red'])
                self.aero['B'].append(B)
            self.aero['n_poles'] = n_poles
            self.aero['betas'] =  betas
        else:
            self.aero['n_poles'] = 0

        # Aero DB    
        if self.jcl.aero['method'] == 'hybrid':   
            print 'Building aero db...'
            self.aerodb = build_aerodb.process_matrix(self, self.jcl.matrix_aerodb, plot=False)  
            
        # splines 
        # PHIk_strc with 'nearest_neighbour', 'rbf' or 'nastran'
        if self.jcl.spline['method'] in ['rbf', 'nearest_neighbour']:
            if self.jcl.spline['splinegrid'] == True:
                # this optin is only valid if spline['method'] == 'rbf' or 'rb'
                print 'Coupling aerogrid to strcgrid via splinegrid:'
                self.splinegrid = build_splinegrid.build_splinegrid(self.strcgrid, self.jcl.spline['filename_splinegrid'])
                # self.splinegrid = build_splinegrid.grid_thin_out_random(self.splinegrid, 0.05)
                # self.splinegrid = build_splinegrid.grid_thin_out_radius(self.splinegrid, 0.5)
            else:
                print 'Coupling aerogrid directly. Doing cleanup/thin out of strcgrid to avoid singularities (safety first!)'
                self.splinegrid = build_splinegrid.grid_thin_out_radius(self.strcgrid, 0.01)

        if self.jcl.spline['method'] == 'rbf': 
            self.PHIk_strc = spline_functions.spline_rbf(self.splinegrid, '',self.aerogrid, '_k', 'tps', dimensions=[len(self.strcgrid['ID'])*6, len(self.aerogrid['ID'])*6] )
            # rbf-spline not (yet) stable for translation of forces and moments to structure grid, so use rb-spline with nearest neighbour search instead
        elif self.jcl.spline['method'] == 'nearest_neighbour':
            rules = spline_rules.nearest_neighbour(self.splinegrid, '', self.aerogrid, '_k')    
            self.PHIk_strc = spline_functions.spline_rb(self.splinegrid, '', self.aerogrid, '_k', rules, self.coord, dimensions=[len(self.strcgrid['ID'])*6, len(self.aerogrid['ID'])*6])
        elif self.jcl.spline['method'] == 'nastran': 
            self.PHIk_strc = spline_functions.spline_nastran(self.jcl.spline['filename_f06'], self.strcgrid, self.aerogrid)  
        else:
            print 'Unknown spline method.'

        print 'Building mass model...'
        if self.jcl.mass['method'] in ['mona', 'modalanalysis', 'guyan']:
            self.mass = {'key': [],
                         'Mb': [],
                         'MGG': [],
                         'Mfcg': [],  
                         'cggrid': [],
                         'cggrid_norm': [],
                         'PHIstrc_cg': [],
                         'PHImac_cg': [],
                         'PHIcg_mac': [],
                         'PHInorm_cg': [],
                         'PHIcg_norm': [],
                         'PHIf_strc': [],
                         'PHIjf': [],
                         'PHIkf': [],
                         'Mff': [],
                         'Kff': [],
                         'Dff': [],
                         'n_modes': []
                        }   
            bm = build_mass.build_mass(self.jcl, self.strcgrid, self.coord, octave)
            
            if self.jcl.mass['method'] == 'modalanalysis': 
                bm.init_modalanalysis()
            elif self.jcl.mass['method'] == 'guyan':
                bm.init_modalanalysis()
                bm.init_guyanreduction()
            
            # loop over mass configurations
            for i_mass in range(len(self.jcl.mass['key'])):
                print 'Mass configuration {} of {}: {} '.format(i_mass+1, len(self.jcl.mass['key']), self.jcl.mass['key'][i_mass])
                MGG = read_geom.Nastran_OP4(self.jcl.mass['filename_MGG'][i_mass], sparse_output=True, sparse_format=True) 
                
                if self.jcl.mass['method'] == 'mona': 
                    Mff, Kff, Dff, PHIf_strc, Mb, cggrid, cggrid_norm = bm.mass_from_SOL103(i_mass)
                elif self.jcl.mass['method'] == 'modalanalysis': 
                    Mb, cggrid, cggrid_norm = bm.calc_cg(i_mass, MGG)
                    # a-set is equal to f-set, no further reduction
                    MFF = read_geom.Nastran_OP4(self.jcl.mass['filename_MFF'][i_mass], sparse_output=True, sparse_format=True) 
                    Mff, Kff, Dff, PHIf_strc = bm.modalanalysis(i_mass, MFF)
                elif self.jcl.mass['method'] == 'guyan': 
                    Mb, cggrid, cggrid_norm = bm.calc_cg(i_mass, MGG)
                    MFF = read_geom.Nastran_OP4(self.jcl.mass['filename_MFF'][i_mass], sparse_output=True, sparse_format=True) 
                    Mff, Kff, Dff, PHIf_strc, Maa = bm.guyanreduction(i_mass, MFF)              
                    # Vergleich mit SOL103:
                    # Mff2, Kff2, Dff2, PHIf_strc2, Mb2, cggrid2, cggrid_norm2 = bm.mass_from_SOL103(i_mass)
                    # MAC, plt = bm.calc_MAC( PHIf_strc.T, PHIf_strc2.T)
                    # plt.title('MAC bm guyan vs. SOL103 aset')
                    # MAC, plt = bm.calc_MAC( PHIf_strc.T, PHIf_strc2.T)
                    # plt.title('Auto-MAC bm guyan')
                    # plt.show()

                rules = spline_rules.rules_point(cggrid, self.strcgrid)
                PHIstrc_cg = spline_functions.spline_rb(cggrid, '', self.strcgrid, '', rules, self.coord)
                
                rules = spline_rules.rules_point(cggrid, self.macgrid)
                PHImac_cg = spline_functions.spline_rb(cggrid, '', self.macgrid, '', rules, self.coord)    
                rules = spline_rules.rules_point(self.macgrid, cggrid)
                PHIcg_mac = spline_functions.spline_rb( self.macgrid, '', cggrid, '', rules, self.coord)      
                
                rules = spline_rules.rules_point(cggrid, cggrid_norm)
                PHInorm_cg = spline_functions.spline_rb(cggrid, '', cggrid_norm, '', rules, self.coord)   
                rules = spline_rules.rules_point(cggrid_norm, cggrid)
                PHIcg_norm = spline_functions.spline_rb(cggrid_norm, '', cggrid, '', rules, self.coord) 
                
                PHIjf = np.dot(self.Djk, np.dot(self.PHIk_strc, PHIf_strc.T))                
                PHIkf = np.dot(self.PHIk_strc, PHIf_strc.T)

                Mfcg=PHIf_strc.dot(-MGG.dot(PHIstrc_cg))
        
                # save all matrices to data structure
                self.mass['key'].append(self.jcl.mass['key'][i_mass])
                self.mass['Mb'].append(Mb)
                self.mass['MGG'].append(MGG)
                self.mass['Mfcg'].append(Mfcg)
                self.mass['cggrid'].append(cggrid)
                self.mass['cggrid_norm'].append(cggrid_norm)
                self.mass['PHIstrc_cg'].append(PHIstrc_cg)
                self.mass['PHImac_cg'].append(PHImac_cg)
                self.mass['PHIcg_mac'].append(PHIcg_mac)
                self.mass['PHIcg_norm'].append(PHIcg_norm)
                self.mass['PHInorm_cg'].append(PHInorm_cg)
                self.mass['PHIf_strc'].append(PHIf_strc) 
                self.mass['PHIjf'].append(PHIjf)
                self.mass['PHIkf'].append(PHIkf)
                self.mass['Mff'].append(Mff) 
                self.mass['Kff'].append(Kff) 
                self.mass['Dff'].append(Dff) 
                self.mass['n_modes'].append(len(self.jcl.mass['modes'][i_mass]))
                
        else:
            print 'Unknown mass method: ' + str(self.jcl.mass['method'])
            
            
        octave.exit() # closes and cleans up octave session