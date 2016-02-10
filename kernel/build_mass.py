
import read_geom, spline_rules, spline_functions

import scipy
from scipy import sparse
from scipy.sparse import linalg
import numpy as np
import sys

class build_mass:
    
    def __init__(self, jcl, strcgrid, coord):
        self.jcl = jcl
        self.strcgrid = strcgrid
        self.coord = coord
        
    def mass_from_SOL103(self, i_mass):
          # Mff, Kff and PHIstrc_f
          eigenvalues, eigenvectors, node_ids_all = read_geom.NASTRAN_f06_modal(self.jcl.mass['filename_S103'][i_mass])
          nodes_selection = self.strcgrid['ID']
          modes_selection = self.jcl.mass['modes'][i_mass]           
          if self.jcl.mass['omit_rb_modes']:
              modes_selection += 6
          eigenvalues, eigenvectors = read_geom.reduce_modes(eigenvalues, eigenvectors, nodes_selection, modes_selection)
          Mff = np.eye(len(self.jcl.mass['modes'][i_mass])) * eigenvalues['GeneralizedMass']
          Kff = np.eye(len(self.jcl.mass['modes'][i_mass])) * eigenvalues['GeneralizedStiffness']
          Dff = Kff * 0.0
          PHIf_strc = np.zeros((len(self.jcl.mass['modes'][i_mass]), len(self.strcgrid['ID'])*6))
          for i_mode in range(len(modes_selection)):
              eigenvector = eigenvectors[str(modes_selection[i_mode])][:,1:]
              PHIf_strc[i_mode,:] = eigenvector.reshape((1,-1))[0]
          
          # Mb        
          massmatrix_0, inertia, offset_cg, CID = read_geom.Nastran_weightgenerator(self.jcl.mass['filename_S103'][i_mass])  
          cggrid = {"ID": np.array([9000+i_mass]),
                    "offset": np.array([offset_cg]),
                    "set": np.array([[0, 1, 2, 3, 4, 5]]),
                    'CD': np.array([CID]),
                    'CP': np.array([CID]),
                    'coord_desc': 'bodyfixed',
                    }
          cggrid_norm = {"ID": np.array([9300+i_mass]),
                    "offset": np.array([[-offset_cg[0], offset_cg[1], -offset_cg[2]]]),
                    "set": np.array([[0, 1, 2, 3, 4, 5]]),
                    'CD': np.array([9300]),
                    'CP': np.array([9300]),
                    'coord_desc': 'bodyfixed_DIN9300',
                    } 
    
          # assemble mass matrix about center of gravity, relativ to the axis of the basic coordinate system
          # DO NOT switch signs for coupling terms of I to suite EoMs, Nastran already did this!
          Mb = np.zeros((6,6))
          Mb[0,0] = massmatrix_0[0,0]
          Mb[1,1] = massmatrix_0[0,0]
          Mb[2,2] = massmatrix_0[0,0]
          Mb[3:6,3:6] = inertia #np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]]) * inertia
          
          return Mff, Kff, Dff, PHIf_strc, Mb, cggrid, cggrid_norm 



    def init_modalanalysis(self, uset, GM, Kaa, Kgg):
        self.uset = uset
        self.GM = GM
        self.Kaa = Kaa
        self.Kgg = Kgg
        
        # Annahmen
        # - es gibt das a-, s- & m-set
        # - jeder Punkt gehoert mit allen 6 DoFs nur einem Set an
        bitposes = self.uset['bitpos'] #.reshape((-1,6))
        i = 0
        self.pos_a = []
        self.pos_s = []
        self.pos_m = []
        for bitpos in bitposes:
            if bitpos==31: # 'S'
                self.pos_a.append(i)
            elif bitpos==22: # 'SB'
                self.pos_s.append(i)
            elif bitpos==32: # 'M'
                self.pos_m.append(i)
            else:
                print 'Error: Unknown set of grid point {}'.format(bitpos)
            i += 1
        self.pos_n = np.sort(np.hstack((self.pos_s, self.pos_a)))
      
    def modalanalysis(self, i_mass, Maa):
        modes_selection = self.jcl.mass['modes'][i_mass]           
        if self.jcl.mass['omit_rb_modes']:
              modes_selection += 6
        n_modes = modes_selection.max()
        # perform modal analysis on a-set
        print 'Modal analysis for first {} modes...'.format( n_modes )
        eigenvalue, eigenvector = scipy.sparse.linalg.eigs(A=self.Kaa, M=Maa , k=n_modes, sigma=0) 
        print 'Found {} modes with the following frequencies [Hz]:'.format(len(eigenvalue))
        print np.real(eigenvalue)**0.5 /2/np.pi
        print 'From these {} modes, the following {} modes are selected: {}'.format(n_modes, len(modes_selection), modes_selection)
        
        PHIf_strc = np.zeros((6*self.strcgrid['n'], len(modes_selection)))
        for i_mode in modes_selection - 1:
            # deformation of a-set due to i_mode is the ith column of the eigenvector
            Ua = eigenvector[:,i_mode].real.reshape((-1,1))
            # deflection of s-set is zero, because that's the sense of an SPC ...
            Us = np.zeros((len(self.pos_s),1))
            # assemble deflections of s and a to n
            Un = np.zeros((6*self.strcgrid['n'],1))
            Un[self.pos_a] = Ua
            Un[self.pos_s] = Us
            Un = Un[self.pos_n]
            # calc remaining deflections with GM
            Um = self.GM.T.dot(Un)
            # assemble everything to Ug
            Ug = np.zeros((6*self.strcgrid['n'],1))
            Ug[self.pos_a] = Ua
            Ug[self.pos_s] = Us
            Ug[self.pos_m] = Um
            
            PHIf_strc[:,i_mode] = Ug.squeeze()
            
        #Mff = np.dot (PHIf_strc.T, Mgg.dot(PHIf_strc))  
        #Kff = np.dot (PHIf_strc.T, self.Kgg.dot(PHIf_strc))  
        # zur Kontrolle, sollte das Gleiche raus kommen
        Mff = np.dot( eigenvector.real.T, Maa.dot(eigenvector.real) )
        Kff = np.dot( eigenvector.real.T, self.Kaa.dot(eigenvector.real) )
        Dff = Kff * 0.0
        return Mff, Kff, Dff, PHIf_strc.T
          
    def calc_cg(self, i_mass, Mgg):
        print 'Calculate center of gravity, mass and inertia (GRDPNT)...'
        # First step: calculate M0
        m0grid = {"ID": np.array([999999]),
                  "offset": np.array([[0,0,0]]),
                  "set": np.array([[0, 1, 2, 3, 4, 5]]),
                  'CD': np.array([0]),
                  'CP': np.array([0]),
                 }
        rules = spline_rules.rules_point(m0grid, self.strcgrid)
        PHIstrc_m0 = spline_functions.spline_rb(m0grid, '', self.strcgrid, '', rules, self.coord)                    
        m0 = PHIstrc_m0.T.dot(Mgg.dot(PHIstrc_m0))
        m = m0[0,0]
        # Second step: calculate CG location with the help of the inertia moments 
        offset_cg = [m0[1,5],m0[2,3],m0[0,4]]/m
        
        cggrid = {"ID": np.array([9000+i_mass]),
                  "offset": np.array([offset_cg]),
                  "set": np.array([[0, 1, 2, 3, 4, 5]]),
                  'CD': np.array([0]),
                  'CP': np.array([0]),
                  'coord_desc': 'bodyfixed',
                    }
        cggrid_norm = {"ID": np.array([9300+i_mass]),
                    "offset": np.array([[-offset_cg[0], offset_cg[1], -offset_cg[2]]]),
                    "set": np.array([[0, 1, 2, 3, 4, 5]]),
                    'CD': np.array([9300]),
                    'CP': np.array([9300]),
                    'coord_desc': 'bodyfixed_DIN9300',
                    } 
        # Third step: calculate Mb
        rules = spline_rules.rules_point(cggrid, self.strcgrid)
        PHIstrc_mb = spline_functions.spline_rb(cggrid, '', self.strcgrid, '', rules, self.coord)                    
        mb = PHIstrc_mb.T.dot(Mgg.dot(PHIstrc_mb))
        # switch signs for coupling terms of I to suite EoMs
        Mb = np.zeros((6,6))
        Mb[0,0] = m
        Mb[1,1] = m
        Mb[2,2] = m
        Mb[3:6,3:6] = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]]) * mb[3:6,3:6]
        
        print 'Mass: {}'.format(Mb[0,0])
        print 'CG at: {}'.format(offset_cg)
        print 'Inertia: \n{}'.format(Mb[3:6,3:6])
        
        return Mb, cggrid, cggrid_norm
    
    
    
    
    