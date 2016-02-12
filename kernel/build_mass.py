
import read_geom, spline_rules, spline_functions

import scipy
from scipy import sparse
from scipy.sparse import linalg
import numpy as np
import sys

class build_mass:
    
    def __init__(self, jcl, strcgrid, coord, octave):
        self.jcl = jcl
        self.strcgrid = strcgrid
        self.coord = coord
        self.octave = octave
        
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

    def init_guyanreduction(self):
        # In a first step, the positions of the a- and o-set DoFs are prepared.
        # Then the equations are solved for the stiffness matrix.
        # References: 
        # R. Guyan, "Reduction of Stiffness and Mass Matrices," AIAA Journal, vol. 3, no. 2, p. 280, 1964.
        # MSC.Software Corporation, "Matrix Operations," in MSC Nastran Linear Static Analysis User's Guide, vol. 2003, D. M. McLean, Ed. 2003, p. 473.

        print "Guyan reduction of stiffness matrix Kff --> Kaa ..."
        self.aset = read_geom.Nastran_SET1(self.jcl.geom['filename_aset'])
        id_a = [np.where(self.strcgrid['ID'] == x)[0][0] for x in set(self.aset['values'][0])] # take the set of the b-set because IDs might be given repeatedly
        pos_a = self.strcgrid['set'][id_a,:].reshape((1,-1))[0]
        self.pos_a = np.intersect1d(self.pos_f, pos_a) # make sure DoFs of a-set are really in f-set (e.g. due to faulty user input)
        self.pos_o = np.setdiff1d(self.pos_f, self.pos_a) # the remainders will be omitted
        print ' - prepare a-set ({} DoFs) and o-set ({} DoFs)'.format(len(self.pos_a), len(self.pos_o) )
        self.pos_f2a = [np.where(self.pos_f == x)[0][0] for x in self.pos_a]
        self.pos_f2o = [np.where(self.pos_f == x)[0][0] for x in self.pos_o] # takes much time if o-set is big... is there anything faster??
        print ' - splitting Kff'
        K = {}
        K['A']       = self.KFF[self.pos_f2a,:][:,self.pos_f2a]
        K['B']       = self.KFF[self.pos_f2a,:][:,self.pos_f2o]
        K['B_trans'] = self.KFF[self.pos_f2o,:][:,self.pos_f2a]
        K['C']       = self.KFF[self.pos_f2o,:][:,self.pos_f2o]
        # nach Nastran
        # Anstelle die Inverse zu bilden, wird ein Gleichungssystem geloest. Das ist schneller!
        print " - solve for Goa = C^-1 * B'"
        self.Goa = - scipy.sparse.linalg.spsolve(K['C'], K['B_trans'])
        self.Kaa = K['A'] + K['B'].dot(self.Goa)
        
    def guyanreduction(self, i_mass, MFF):
        # First, Guyan's equations are solved for the current mass matrix.
        # In a second step, the eigenvalue-eigenvector problem is solved. According to Guyan, the solution is closely  but not exactly preserved.
        # Next, the eigenvector for the g-set/strcgrid is reconstructed.
        # In a final step, the generalized mass and stiffness matrices are calculated.
        # The nomenclature might be a little confusing at this point, because the flexible mode shapes and Nastran's free DoFs (f-set) are both labeled with the subscript 'f'...
        print "Guyan reduction of mass matrix Mff --> Maa ..."
        M = {}
        M['A']       = MFF[self.pos_f2a,:][:,self.pos_f2a]
        M['B']       = MFF[self.pos_f2a,:][:,self.pos_f2o]
        M['B_trans'] = MFF[self.pos_f2o,:][:,self.pos_f2a]
        M['C']       = MFF[self.pos_f2o,:][:,self.pos_f2o]
        Maa = M['A'] - M['B'].dot(self.Goa) - self.Goa.T.dot( M['B_trans'] - M['C'].dot(self.Goa) )
        
        modes_selection = self.jcl.mass['modes'][i_mass]           
        if self.jcl.mass['omit_rb_modes']: 
              modes_selection += 6
        eigenvalue, eigenvector = self.calc_modes(self.Kaa, Maa, modes_selection.max())
        print 'From these {} modes, the following {} modes are selected: {}'.format(modes_selection.max(), len(modes_selection), modes_selection)
        # reconstruct modal matrix for g-set / strcgrid
        PHIf_strc = np.zeros((6*self.strcgrid['n'], len(modes_selection)))
        for i_mode in modes_selection - 1:
            # deformation of a-set due to i_mode is the ith column of the eigenvector
            Ua = eigenvector[:,i_mode].real.reshape((-1,1))
            # calc ommitted grids with Guyan
            Uo = self.Goa.dot(Ua)
            # assemble deflections of a and o to f
            Uf = np.zeros((len(self.pos_f),1))
            Uf[self.pos_f2a] = Ua
            Uf[self.pos_f2o] = Uo
            # deflection of s-set is zero, because that's the sense of an SPC ...
            Us = np.zeros((len(self.pos_s),1))
            # assemble deflections of s and f to n
            Un = np.zeros((6*self.strcgrid['n'],1))
            Un[self.pos_f] = Uf
            Un[self.pos_s] = Us
            Un = Un[self.pos_n]
            # calc remaining deflections with GM
            Um = self.GM.T.dot(Un)
            # assemble everything to Ug
            Ug = np.zeros((6*self.strcgrid['n'],1))
            Ug[self.pos_f] = Uf
            Ug[self.pos_s] = Us
            Ug[self.pos_m] = Um
            # store vector in modal matrix
            PHIf_strc[:,i_mode] = Ug.squeeze()
        # calc modal mass and stiffness
        Mff = np.dot( eigenvector.real.T, Maa.dot(eigenvector.real) )
        Kff = np.dot( eigenvector.real.T, self.Kaa.dot(eigenvector.real) )
        Dff = Kff * 0.0
        return Mff, Kff, Dff, PHIf_strc.T, Maa
        
        

    def init_modalanalysis(self):
        # KFF, GM and uset are actually geometry dependent and should go into the geometry section.
        # However, the they are only required for modal analysis...
        self.KFF = read_geom.Nastran_OP4(self.jcl.geom['filename_KFF'], sparse_output=True, sparse_format=True)
        self.GM  = read_geom.Nastran_OP4(self.jcl.geom['filename_GM'],  sparse_output=True, sparse_format=True) 
        print 'Read USET from OP2-file {} with get_uset.m ...'.format( self.jcl.geom['filename_uset'] )
        self.uset = self.octave.get_uset(self.jcl.geom['filename_uset'])
        # Annahme: es gibt (nur) das a-, s- & m-set
        bitposes = self.uset['bitpos'] #.reshape((-1,6))
        i = 0
        self.pos_f = []
        self.pos_s = []
        self.pos_m = []
        for bitpos in bitposes:
            if bitpos==31: # 'S'
                self.pos_f.append(i)
            elif bitpos==22: # 'SB'
                self.pos_s.append(i)
            elif bitpos==32: # 'M'
                self.pos_m.append(i)
            else:
                print 'Error: Unknown set of grid point {}'.format(bitpos)
            i += 1
        self.pos_n = np.sort(np.hstack((self.pos_s, self.pos_f)))

    def modalanalysis(self, i_mass, MFF):
        modes_selection = self.jcl.mass['modes'][i_mass]           
        if self.jcl.mass['omit_rb_modes']: 
              modes_selection += 6
        eigenvalue, eigenvector = self.calc_modes(self.KFF, MFF, modes_selection.max())
        print 'From these {} modes, the following {} modes are selected: {}'.format(modes_selection.max(), len(modes_selection), modes_selection)
        # reconstruct modal matrix for g-set / strcgrid
        PHIf_strc = np.zeros((6*self.strcgrid['n'], len(modes_selection)))
        for i_mode in modes_selection - 1:
            # deformation of f-set due to i_mode is the ith column of the eigenvector
            Uf = eigenvector[:,i_mode].real.reshape((-1,1))
            # deflection of s-set is zero, because that's the sense of an SPC ...
            Us = np.zeros((len(self.pos_s),1))
            # assemble deflections of s and f to n
            Un = np.zeros((6*self.strcgrid['n'],1))
            Un[self.pos_f] = Uf
            Un[self.pos_s] = Us
            Un = Un[self.pos_n]
            # calc remaining deflections with GM
            Um = self.GM.T.dot(Un)
            # assemble everything to Ug
            Ug = np.zeros((6*self.strcgrid['n'],1))
            Ug[self.pos_f] = Uf
            Ug[self.pos_s] = Us
            Ug[self.pos_m] = Um
            # store vector in modal matrix
            PHIf_strc[:,i_mode] = Ug.squeeze()
        # calc modal mass and stiffness
        Mff = np.dot( eigenvector.real.T, MFF.dot(eigenvector.real) )
        Kff = np.dot( eigenvector.real.T, self.KFF.dot(eigenvector.real) )
        Dff = Kff * 0.0
        return Mff, Kff, Dff, PHIf_strc.T
    
    def calc_modes(self, K, M, n_modes):
        # perform modal analysis on a-set
        print 'Modal analysis for first {} modes...'.format( n_modes )
        eigenvalue, eigenvector = scipy.sparse.linalg.eigs(A=K, M=M , k=n_modes, sigma=0) 
        idx_sort = np.argsort(eigenvalue) # sort result by eigenvalue
        eigenvalue = eigenvalue[idx_sort]
        eigenvector = eigenvector[:,idx_sort]
        print 'Found {} modes with the following frequencies [Hz]:'.format(len(eigenvalue))
        print np.real(eigenvalue)**0.5 /2/np.pi
        return eigenvalue, eigenvector
    
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
    
    def calc_MAC(self, X, Y):
        MAC = np.zeros((X.shape[1],Y.shape[1]))
        for jj in range(Y.shape[1]):
            for ii in range(X.shape[1]):
                q1 = np.dot(np.conj(X[:,ii].T), X[:,ii])
                q2 = np.dot(np.conj(Y[:,jj].T), Y[:,jj])
                q3 = np.dot(np.conj(X[:,ii].T), Y[:,jj])
                MAC[ii,jj]  = np.conj(q3)*q3/q1/q2
        MAC = np.abs(MAC)
        return MAC
    
    
    