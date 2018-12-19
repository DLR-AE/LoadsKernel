
import read_geom, spline_rules, spline_functions, read_op2

import scipy
from scipy import sparse
from scipy.sparse import linalg
import numpy as np
import sys, copy, logging
import matplotlib.pyplot as plt


class BuildMass:
    
    def __init__(self, jcl, strcgrid, coord, octave=None):
        self.jcl = jcl
        self.strcgrid = strcgrid
        self.coord = coord
        self.octave = octave
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams.update({'font.size': 16})
        
    def mass_from_SOL103(self, i_mass):
          # Mff, Kff and PHIstrc_f
          eigenvalues, eigenvectors, node_ids_all = read_geom.NASTRAN_f06_modal(self.jcl.mass['filename_S103'][i_mass])
          nodes_selection = self.strcgrid['ID']
          modes_selection = copy.deepcopy(self.jcl.mass['modes'][i_mass])
          if self.jcl.mass['omit_rb_modes']:
              modes_selection += 6
          eigenvalues, eigenvectors = read_geom.reduce_modes(eigenvalues, eigenvectors, nodes_selection, modes_selection)
          Mff = np.eye(len(self.jcl.mass['modes'][i_mass])) * eigenvalues['GeneralizedMass']
          Kff = np.eye(len(self.jcl.mass['modes'][i_mass])) * eigenvalues['GeneralizedStiffness']
          Dff = self.calc_damping(np.array(eigenvalues['Eigenvalue']).real)
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
        # MSC.Software Corporation, "Theoretical Basis for Reduction Methods," in MSC.Nastran Version 70 Advanced Dynamic Analysis User's Guide, Version 70., H. David N., Ed. p. 63.

        logging.info( "Guyan reduction of stiffness matrix Kff --> Kaa ...")
        self.aset = read_geom.Nastran_SET1(self.jcl.geom['filename_aset'])
        self.aset['values_unique'] = np.unique(self.aset['values'][0]) # take the set of the a-set because IDs might be given repeatedly
        id_g2a = [np.where(self.strcgrid['ID'] == x)[0][0] for x in self.aset['values_unique']] 
        pos_a = self.strcgrid['set'][id_g2a,:].reshape((1,-1))[0]
        self.pos_a = pos_a[np.in1d(pos_a, self.pos_f)] # make sure DoFs of a-set are really in f-set (e.g. due to faulty user input)
        self.pos_o = np.setdiff1d(self.pos_f, self.pos_a) # the remainders will be omitted
        logging.info( ' - prepare a-set ({} DoFs) and o-set ({} DoFs)'.format(len(self.pos_a), len(self.pos_o) ))
        # Convert to ndarray and then use list comprehension. This is the fastest way of finding indices.
        pos_f_ndarray = np.array(self.pos_f) 
        self.pos_f2a = [np.where(pos_f_ndarray == x)[0][0] for x in self.pos_a]
        self.pos_f2o = [np.where(pos_f_ndarray == x)[0][0] for x in self.pos_o]
        logging.info( ' - partitioning')
        K = {}
        K['A']       = self.KFF[self.pos_f2a,:][:,self.pos_f2a]
        K['B']       = self.KFF[self.pos_f2a,:][:,self.pos_f2o]
        K['B_trans'] = self.KFF[self.pos_f2o,:][:,self.pos_f2a]
        K['C']       = self.KFF[self.pos_f2o,:][:,self.pos_f2o]
        # nach Nastran
        # Anstelle die Inverse zu bilden, wird ein Gleichungssystem geloest. Das ist schneller!
        logging.info( " - solving")
        self.Goa = - scipy.sparse.linalg.spsolve(K['C'], K['B_trans'])
        self.Goa = self.Goa.toarray() # Sparse format is no longer required as Goa is dense anyway!
        self.Kaa = K['A'].toarray() + K['B'].dot(self.Goa) # make sure the output is an ndarray
        
    def guyanreduction(self, i_mass, MFF, plot=False):
        # First, Guyan's equations are solved for the current mass matrix.
        # In a second step, the eigenvalue-eigenvector problem is solved. According to Guyan, the solution is closely  but not exactly preserved.
        # Next, the eigenvector for the g-set/strcgrid is reconstructed.
        # In a final step, the generalized mass and stiffness matrices are calculated.
        # The nomenclature might be a little confusing at this point, because the flexible mode shapes and Nastran's free DoFs (f-set) are both labeled with the subscript 'f'...
        logging.info( "Guyan reduction of mass matrix Mff --> Maa ...")
        logging.info( ' - partitioning')
        M = {}
        M['A']       = MFF[self.pos_f2a,:][:,self.pos_f2a]
        M['B']       = MFF[self.pos_f2a,:][:,self.pos_f2o]
        M['B_trans'] = MFF[self.pos_f2o,:][:,self.pos_f2a]
        M['C']       = MFF[self.pos_f2o,:][:,self.pos_f2o]
        logging.info( " - solving")
        # a) original formulation according to R. Guyan
        # Maa = M['A'] - M['B'].dot(self.Goa) - self.Goa.T.dot( M['B_trans'] - M['C'].dot(self.Goa) )
        
        # b) General Dynamic Reduction as implemented in Nastran (signs are switched!)
        Maa = M['A'].toarray() + M['B'].dot(self.Goa) + self.Goa.T.dot( M['B_trans'].toarray() + M['C'].dot(self.Goa) ) # make sure the output is an ndarray
        
        modes_selection = copy.deepcopy(self.jcl.mass['modes'][i_mass])         
        if self.jcl.mass['omit_rb_modes']: 
              modes_selection += 6
        eigenvalue, eigenvector = self.calc_modes(self.Kaa, Maa, modes_selection.max())
        logging.info( 'From these {} modes, the following {} modes are selected: {}'.format(modes_selection.max(), len(modes_selection), modes_selection))
        # reconstruct modal matrix for g-set / strcgrid
        PHIf_strc = np.zeros((6*self.strcgrid['n'], len(modes_selection)))
        i = 0 # counter selected modes
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
            PHIf_strc[:,i] = Ug.squeeze()
            i += 1
            if plot:
                from mayavi import mlab
                Ugx = self.strcgrid['offset'][:,0] + Ug[self.strcgrid['set'][:,0]].T * 10.0
                Ugy = self.strcgrid['offset'][:,1] + Ug[self.strcgrid['set'][:,1]].T * 10.0
                Ugz = self.strcgrid['offset'][:,2] + Ug[self.strcgrid['set'][:,2]].T * 10.0
                mlab.figure(101+i_mode)
                mlab.points3d(self.strcgrid['offset'][:,0], self.strcgrid['offset'][:,1], self.strcgrid['offset'][:,2], scale_factor=0.05)
                mlab.points3d(Ugx, Ugy, Ugz, scale_factor=0.05, color=(0,1,0))
        if plot:
            mlab.show()
        # calc modal mass and stiffness
        Mff = np.dot( eigenvector[:,modes_selection - 1].real.T,      Maa.dot(eigenvector[:,modes_selection - 1].real) )
        Kff = np.dot( eigenvector[:,modes_selection - 1].real.T, self.Kaa.dot(eigenvector[:,modes_selection - 1].real) )
        #Dff = Kff * 0.0
        Dff = self.calc_damping(eigenvalue[modes_selection - 1].real)
        return Mff, Kff, Dff, PHIf_strc.T, Maa
        
    def get_bitposes(self, x_dec):
        bitposes = []
        for x in x_dec:
            binstring = np.binary_repr(x, width=32)
            bitposes.append( binstring.index('1')+1 ) # +1 as python starts counting with 0
        return bitposes

    def init_modalanalysis(self):
        # Prepare some data required for modal analysis which is not mass case dependent. 
        # KFF, GM and uset are actually geometry dependent and should go into the geometry section.
        # However, the they are only required for modal analysis...
        self.KFF = read_geom.Nastran_OP4(self.jcl.geom['filename_KFF'], sparse_output=True, sparse_format=True)
        self.GM  = read_geom.Nastran_OP4(self.jcl.geom['filename_GM'],  sparse_output=True, sparse_format=True) 
        logging.info( 'Read USET from OP2-file {} ...'.format( self.jcl.geom['filename_uset'] ))
        #self.uset = self.octave.get_uset(self.jcl.geom['filename_uset'])
        #bitposes = self.uset['bitpos'] #.reshape((-1,6))
        op2_data = read_op2.read_post_op2(self.jcl.geom['filename_uset'], verbose=True)
        if op2_data['uset'] is None:
            logging.error( 'No USET found in OP2-file {} !'.format( self.jcl.geom['filename_uset'] ))
        bitposes = self.get_bitposes(op2_data['uset'])
        
        # Reference:
        # National Aeronautics and Space Administration, The Nastran Programmer's Manual, NASA SP-223(01). Washington, D.C.: COSMIC, 1972.
        # Section 2.3.13.3 USET (TABLE), page 2.3-61
        # Annahme: es gibt (nur) das a-, s- & m-set

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
                logging.error( 'Unknown set of grid point {}'.format(bitpos))
            i += 1
        self.pos_n = self.pos_s + self.pos_f
        self.pos_n.sort()
        
    def modalanalysis(self, i_mass, MFF, plot=False):
        modes_selection = copy.deepcopy(self.jcl.mass['modes'][i_mass])
        if self.jcl.mass['omit_rb_modes']: 
              modes_selection += 6
        eigenvalue, eigenvector = self.calc_modes(self.KFF, MFF, modes_selection.max())
        logging.info( 'From these {} modes, the following {} modes are selected: {}'.format(modes_selection.max(), len(modes_selection), modes_selection))
        # reconstruct modal matrix for g-set / strcgrid
        PHIf_strc = np.zeros((6*self.strcgrid['n'], len(modes_selection)))
        i = 0 # counter selected modes
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
            PHIf_strc[:,i] = Ug.squeeze()
            i += 1
            if plot:
                from mayavi import mlab
                Ugx = self.strcgrid['offset'][:,0] + Ug[self.strcgrid['set'][:,0]].T * 10.0
                Ugy = self.strcgrid['offset'][:,1] + Ug[self.strcgrid['set'][:,1]].T * 10.0
                Ugz = self.strcgrid['offset'][:,2] + Ug[self.strcgrid['set'][:,2]].T * 10.0
                mlab.figure(1+i_mode)
                mlab.points3d(self.strcgrid['offset'][:,0], self.strcgrid['offset'][:,1], self.strcgrid['offset'][:,2], scale_factor=0.05)
                mlab.points3d(Ugx, Ugy, Ugz, scale_factor=0.05, color=(0,0,1))
        if plot:
            mlab.show()
        # calc modal mass and stiffness
        Mff = np.dot( eigenvector[:,modes_selection - 1].real.T,      MFF.dot(eigenvector[:,modes_selection - 1].real) )
        Kff = np.dot( eigenvector.real[:,modes_selection - 1].T, self.KFF.dot(eigenvector[:,modes_selection - 1].real) )
        #Dff = Kff * 0.0
        Dff = self.calc_damping(eigenvalue[modes_selection - 1].real)
        return Mff, Kff, Dff, PHIf_strc.T
    
    def calc_modes(self, K, M, n_modes):
        # perform modal analysis on a-set
        logging.info( 'Modal analysis for first {} modes...'.format( n_modes ))
        eigenvalue, eigenvector = scipy.sparse.linalg.eigs(A=K, M=M , k=n_modes, sigma=0) 
        idx_sort = np.argsort(eigenvalue) # sort result by eigenvalue
        eigenvalue = eigenvalue[idx_sort]
        eigenvector = eigenvector[:,idx_sort]
        logging.info( 'Found {} modes with the following frequencies [Hz]:'.format(len(eigenvalue)))
        logging.info( np.real(eigenvalue)**0.5 /2/np.pi)
        return eigenvalue, eigenvector
    
    def calc_damping(self, eigenvalues):
        # Currently, only modal damping is implemented. See Bianchi et al.,
        # "Using modal damping for full model transient analysis. Application to
        # pantograph/catenary vibration", presented at the ISMA, 2010.
        n = len(eigenvalues)
        Dff = np.zeros((n,n))
        if hasattr(self.jcl, 'damping') and self.jcl.damping.has_key('method') and self.jcl.damping['method'] == 'modal':
            logging.info( 'Damping: modal damping of {}'.format( self.jcl.damping['damping'] ))
            d = eigenvalues**0.5 * 2.0 * self.jcl.damping['damping']
            np.fill_diagonal(Dff, d) # matrix Dff is modified in-place, no return value
        elif hasattr(self.jcl, 'damping'):
            logging.warning( 'Damping method not implemented: {}'.format( str(self.jcl.damping['method']) ))
            logging.warning( 'Damping: assuming zero damping.' )
        else:
            logging.info( 'Damping: assuming zero damping.' )
        return Dff
        
    def calc_cg(self, i_mass, Mgg):
        logging.info( 'Calculate center of gravity, mass and inertia (GRDPNT)...')
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
        
        logging.info( 'Mass: {}'.format(Mb[0,0]))
        logging.info( 'CG at: {}'.format(offset_cg))
        logging.info( 'Inertia: \n{}'.format(Mb[3:6,3:6]))
        
        return Mb, cggrid, cggrid_norm
    
    def calc_MAC(self, X, Y, plot=True):
        MAC = np.zeros((X.shape[1],Y.shape[1]))
        for jj in range(Y.shape[1]):
            for ii in range(X.shape[1]):
                q1 = np.dot(np.conj(X[:,ii].T), X[:,ii])
                q2 = np.dot(np.conj(Y[:,jj].T), Y[:,jj])
                q3 = np.dot(np.conj(X[:,ii].T), Y[:,jj])
                MAC[ii,jj]  = np.conj(q3)*q3/q1/q2
        MAC = np.abs(MAC)
        
        if plot:
            plt.figure()
            plt.pcolor(MAC, cmap='hot_r')
            plt.colorbar()
            plt.grid('on')
            
            return MAC, plt
        else:   
            return MAC
    
    def plot_masses(self, MGG, Mb, cggrid, filename):
        try:
            from mayavi import mlab
        except:
            logging.warning('Could not import mayavi. Abort plotting of mass configurations.')
            return
        # get nodal masses
        m_cg = Mb[0,0]
        m = MGG.diagonal()[0::6]
        
        radius_mass_cg = ((m_cg*3.)/(4.*2700.0*np.pi))**(1./3.) 
        radius_masses = ((m*3.)/(4.*2700.0*np.pi))**(1./3.) #/ radius_mass_cg
        #radius_masses = radius_masses/radius_masses.max()
       
        mlab.options.offscreen = True
        mlab.figure(bgcolor=(1,1,1))
        mlab.points3d(self.strcgrid['offset'][:,0], self.strcgrid['offset'][:,1], self.strcgrid['offset'][:,2], radius_masses, scale_mode='scalar', scale_factor = 1.0, color=(1,0.7,0), resolution=32)
        mlab.points3d(cggrid['offset'][0,0],        cggrid['offset'][0,1],        cggrid['offset'][0,2],       radius_mass_cg, scale_mode='scalar', scale_factor = 1.0, color=(1,1,0), opacity=0.3, resolution=64)
        mlab.orientation_axes()
        mlab.savefig(filename, size=(1920,1080))
        mlab.close()
        logging.info('Saving plot of nodal masses to: {}'.format(filename))
        
    
    