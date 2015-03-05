# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 10:53:48 2014

@author: voss_ar
"""
import build_aero
import spline_rules
import spline_functions
import read_geom
from  atmo_isa import atmo_isa
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class model:
    def __init__(self,jcl):
        self.jcl = jcl
            
    def build_model(self):
        self.coord = {'ID': [0],
                      'RID': [0],
                      'dircos': [np.eye(3)],
                      'offset': [np.array([0,0,0])],
                     }        
        print 'Building structure model...'
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
            #self.KAA = read_geom.Nastran_OP4(self.jcl.geom['filename_KAA'], sparse_output=True, sparse_format=True) 

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
            print 'Unknown atmo method: ' + str(jcl.aero['method'])
              
        print 'Building aero model...'
        if self.jcl.aero['method'] == 'mona_steady':
            # grids
            for i_file in range(len(self.jcl.aero['filename_caero_bdf'])):
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
            
            # Correctionfor camber and twist, W2GJ
            for i_file in range(len(self.jcl.aero['filename_deriv_4_W2GJ'])):
                subgrid = read_geom.Modgen_W2GJ(self.jcl.aero['filename_deriv_4_W2GJ'][i_file]) 
                if i_file == 0:
                    self.camber_twist =  subgrid
                else:
                    self.camber_twist['ID'] = np.hstack((self.camber_twist['ID'], subgrid['ID']))
                    self.camber_twist['cam_rad'] = np.hstack((self.camber_twist['cam_rad'], subgrid['cam_rad']))

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
#            ax.set_xlim([-1, 15])
#            ax.set_ylim([-1, 15])
#            ax.set_zlim([-8, 8])
#            plt.show()
            
            
            # splines
            
            #self.PHIk_strc = spline_functions.spline_rbf(self.strcgrid, '',self.aerogrid, '_k', 'tps' )
            # rbf-spline not (yet) stable for translation of forces and moments to structure grid, so use rb-spline with nearest neighbour search instead
            rules = spline_rules.nearest_neighbour(self.strcgrid, self.aerogrid)    
            self.PHIk_strc = spline_functions.spline_rb(self.strcgrid, '', self.aerogrid, '_k', rules, self.coord, dimensions=[len(self.strcgrid['ID'])*6, len(self.aerogrid['ID'])*6])

            rules = spline_rules.rules_aeropanel(self.aerogrid)
            self.Djk = spline_functions.spline_rb(self.aerogrid, '_k', self.aerogrid, '_j', rules, self.coord)
            self.Dlk = spline_functions.spline_rb(self.aerogrid, '_k', self.aerogrid, '_l', rules, self.coord)
            
            rules = spline_rules.rules_point(self.macgrid, self.aerogrid)
            self.Dkx1 = spline_functions.spline_rb(self.macgrid, '', self.aerogrid, '_k', rules, self.coord)
            self.Djx1 = np.dot(self.Djk, self.Dkx1)
            
            # AIC
            self.aero = {'key':[],
                         'Qjj':[],
                        }
            for i_aero in range(len(self.jcl.aero['key'])):
                Ajj = read_geom.Nastran_OP4(self.jcl.aero['filename_AIC'][i_aero], sparse_output=False, sparse_format=False)  
                Qjj = np.linalg.inv(Ajj.T)
                self.aero['key'].append(self.jcl.aero['key'][i_aero])
                self.aero['Qjj'].append(Qjj)
        else:
            print 'Unknown aero method: ' + str(self.jcl.aero['method'])
            
        print 'Building mass model...'
        if self.jcl.mass['method'] == 'mona':
            self.mass = {'key': [],
                         'Mb': [],
                         'MAA': [],   
                         'cggrid': [],
                         'PHIstrc_cg': [],
                         'PHImac_cg': [],
                         'PHIf_strc': [],
                         'PHIjf': [],
                         'Mff': [],
                         'Kff': [],
                         'Dff': [],
                         'n_modes': []
                        }   
            
            for i_mass in range(len(self.jcl.mass['key'])):
                # Mff, Kff and PHIstrc_f
                eigenvalues, eigenvectors, node_ids_all = read_geom.NASTRAN_f06_modal(self.jcl.mass['filename_S103'][i_mass])
                nodes_selection = self.strcgrid['ID']
                modes_selection = self.jcl.mass['modes']            
                if self.jcl.mass['omit_rb_modes']:
                    modes_selection += 6
                eigenvalues, eigenvectors = read_geom.reduce_modes(eigenvalues, eigenvectors, nodes_selection, modes_selection)
                Mff = np.eye(len(self.jcl.mass['modes'])) * eigenvalues['GeneralizedMass']
                Kff = np.eye(len(self.jcl.mass['modes'])) * eigenvalues['GeneralizedStiffness']
                Dff = Kff * 0.0
                PHIf_strc = np.zeros((len(self.jcl.mass['modes']), len(self.strcgrid['ID'])*6))
                for i_mode in range(len(modes_selection)):
                    eigenvector = eigenvectors[str(modes_selection[i_mode])][:,1:]
                    PHIf_strc[i_mode,:] = eigenvector.reshape((1,-1))[0]
                # MAA
                #MAA = read_geom.Nastran_OP4(self.jcl.mass['filename_MAA'][i_mass], sparse_output=True, sparse_format=True) 
                
                # Mb        
                massmatrix_0, inertia, offset_cg, CID = read_geom.Nastran_weightgenerator(self.jcl.mass['filename_S103'][i_mass])  
                cggrid = {"ID": np.array([9000+i_mass]),
                          "offset": np.array([offset_cg]),
                          "set": np.array([[0, 1, 2, 3, 4, 5]]),
                          'CD': np.array([CID]),
                          'CP': np.array([CID]),
                          'coord_desc': 'bodyfixed',
                          }
                # assemble mass matrix about center of gravity, relativ to the axis of the basic coordinate system
                Mb = np.zeros((6,6))
                Mb[0,0] = massmatrix_0[0,0]
                Mb[1,1] = massmatrix_0[0,0]
                Mb[2,2] = massmatrix_0[0,0]
                Mb[3:6,3:6] = inertia
                
                rules = spline_rules.rules_point(cggrid, self.strcgrid)
                PHIstrc_cg = spline_functions.spline_rb(cggrid, '', self.strcgrid, '', rules, self.coord)
                
                rules = spline_rules.rules_point(cggrid, self.macgrid)
                PHImac_cg = spline_functions.spline_rb(cggrid, '', self.macgrid, '', rules, self.coord)          
                
                PHIjf = np.dot(self.Djk, np.dot(self.PHIk_strc, PHIf_strc.T))                
                
                # save all matrices to data structure
                self.mass['key'].append(self.jcl.mass['key'][i_mass])
                self.mass['Mb'].append(Mb)
                #self.mass['MAA'].append(MAA)
                self.mass['cggrid'].append(cggrid)
                self.mass['PHIstrc_cg'].append(PHIstrc_cg)
                self.mass['PHImac_cg'].append(PHImac_cg)
                self.mass['PHIf_strc'].append(PHIf_strc) 
                self.mass['PHIjf'].append(PHIjf)
                self.mass['Mff'].append(Mff) 
                self.mass['Kff'].append(Kff) 
                self.mass['Dff'].append(Dff) 
                self.mass['n_modes'].append(len(modes_selection))

            