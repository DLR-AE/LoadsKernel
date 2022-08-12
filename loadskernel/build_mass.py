
import scipy
import numpy as np
import sys, copy, logging
import matplotlib.pyplot as plt

import loadskernel.read_geom as read_geom
import loadskernel.spline_rules as spline_rules
import loadskernel.spline_functions as spline_functions
import loadskernel.read_op2 as read_op2


class BuildMass:
    
    def __init__(self, jcl, strcgrid, coord, KGG, GM):
        self.jcl = jcl
        self.strcgrid = strcgrid
        self.coord = coord
        self.KGG = KGG
        self.GM = GM
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams.update({'font.size': 16})
        
    def modes_from_SOL103(self, i_mass):
        # Mff, Kff and PHIstrc_f
        eigenvalues, eigenvectors, node_ids_all = read_geom.NASTRAN_f06_modal(self.jcl.mass['filename_S103'][i_mass])
        nodes_selection = self.strcgrid['ID']
        modes_selection = copy.deepcopy(self.jcl.mass['modes'][i_mass])
        if self.jcl.mass['omit_rb_modes']:
            modes_selection += 6
        eigenvalues, eigenvectors = read_geom.reduce_modes(eigenvalues, eigenvectors, nodes_selection, modes_selection)
        PHIf_strc = np.zeros((len(self.jcl.mass['modes'][i_mass]), len(self.strcgrid['ID'])*6))
        for i_mode in range(len(modes_selection)):
            eigenvector = eigenvectors[str(modes_selection[i_mode])][:,1:]
            PHIf_strc[i_mode,:] = eigenvector.reshape((1,-1))[0]
        self.PHIstrc_f = PHIf_strc.T
        self.eigenvalues_f = np.array(eigenvalues['GeneralizedStiffness'])
          
    def cg_from_SOL103(self, i_mass):   
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

        self.cggrid = cggrid # store for later internal use
        self.cggrid_norm = cggrid_norm # store for later internal use
  
        return Mb, cggrid, cggrid_norm 

    def prepare_stiffness_matrices_for_guyan(self):
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
        
    def guyanreduction(self, i_mass):
        # First, Guyan's equations are solved for the current mass matrix.
        # In a second step, the eigenvalue-eigenvector problem is solved. According to Guyan, the solution is closely  but not exactly preserved.
        # Next, the eigenvector for the g-set/strcgrid is reconstructed.
        # In a final step, the generalized mass and stiffness matrices are calculated.
        # The nomenclature might be a little confusing at this point, because the flexible mode shapes and Nastran's free DoFs (f-set) are both labeled with the subscript 'f'...
        logging.info( "Guyan reduction of mass matrix Mff --> Maa ...")
        logging.info( ' - partitioning')
        M = {}
        M['A']       = self.MFF[self.pos_f2a,:][:,self.pos_f2a]
        M['B']       = self.MFF[self.pos_f2a,:][:,self.pos_f2o]
        M['B_trans'] = self.MFF[self.pos_f2o,:][:,self.pos_f2a]
        M['C']       = self.MFF[self.pos_f2o,:][:,self.pos_f2o]
        logging.info( " - solving")
        # a) original formulation according to R. Guyan
        # Maa = M['A'] - M['B'].dot(self.Goa) - self.Goa.T.dot( M['B_trans'] - M['C'].dot(self.Goa) )
        
        # b) General Dynamic Reduction as implemented in Nastran (signs are switched!)
        Maa = M['A'].toarray() + M['B'].dot(self.Goa) + self.Goa.T.dot( M['B_trans'].toarray() + M['C'].dot(self.Goa) ) # make sure the output is an ndarray
        
        modes_selection = copy.deepcopy(self.jcl.mass['modes'][i_mass])         
        if self.jcl.mass['omit_rb_modes']: 
              modes_selection += 6
        eigenvalue, eigenvector = self.calc_elastic_modes(self.Kaa, Maa, modes_selection.max())
        logging.info( 'From these {} modes, the following {} modes are selected: {}'.format(modes_selection.max(), len(modes_selection), modes_selection))
        self.eigenvalues_f = eigenvalue[modes_selection - 1]
        # reconstruct modal matrix for g-set / strcgrid
        self.PHIstrc_f = np.zeros((6*self.strcgrid['n'], len(modes_selection)))
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
            self.PHIstrc_f[:,i] = Ug.squeeze()
            i += 1
        return
        
    def get_sets_from_bitposes(self, x_dec):
        # Reference:
        # National Aeronautics and Space Administration, The Nastran Programmer's Manual, NASA SP-223(01). Washington, D.C.: COSMIC, 1972.
        # Section 2.3.13.3 USET (TABLE), page 2.3-61
        # Assumption: There is (only) the f-, s- & m-set
        # The DoFs of f-, s- and m-set are indexed with respect to g-set
        logging.info('Extracting bit positions from USET to determine DoFs')
        i = 0
        self.pos_f = []
        self.pos_s = []
        self.pos_m = []
        for x in x_dec:
            binstring = np.binary_repr(x, width=32)
            bitpos = binstring.index('1')+1 # +1 as python starts counting with 0
            if bitpos in [31, 30, 25]: # 'S', 'O' and 'A'
                self.pos_f.append(i)
            elif bitpos==22: # 'SB'
                self.pos_s.append(i)
            elif bitpos==32: # 'M'
                self.pos_m.append(i)
            else:
                logging.error( 'Unknown set of grid point {}'.format(bitpos))
            i += 1
        # The n-set is the sum of s-set and f-set
        self.pos_n = self.pos_s + self.pos_f
        self.pos_n.sort()
        # Free DoFs (f-set) indexed with respect to n-set
        self.pos_fn = [self.pos_n.index(i) for i in self.pos_f]

    def get_dofs_from_uset(self):
        # Prepare some data required for modal analysis which is not mass case dependent. 
        # The uset is actually geometry dependent and should go into the geometry section.
        # However, it is only required for modal analysis...
        logging.info( 'Read USET from OP2-file {} ...'.format( self.jcl.geom['filename_uset'] ))
        op2_data = read_op2.read_post_op2(self.jcl.geom['filename_uset'], verbose=True)
        if op2_data['uset'] is None:
            logging.error( 'No USET found in OP2-file {} !'.format( self.jcl.geom['filename_uset'] ))
        self.get_sets_from_bitposes(op2_data['uset'])
    
    def get_dofs_from_CoFE(self):
        # Prepare some data required for modal analysis which is not mass case dependent. 
        with open(self.jcl.geom['filename_CoFE']) as fid: CoFE_data = scipy.io.loadmat(fid)
                
        # The DoFs of f-, s- and m-set are indexed with respect to g-set
        # Convert indexing from Matlab to Python
        self.pos_f = CoFE_data['nf_g'].squeeze()-1
        self.pos_s = CoFE_data['s'].squeeze()-1
        self.pos_m = CoFE_data['m'].squeeze()-1
        self.pos_n = CoFE_data['n'].squeeze()-1
       
        # Free DoFs (f-set) indexed with respect to n-set
        self.pos_fn = CoFE_data['nf_n'].squeeze()-1
    
       
    def prepare_stiffness_matrices_from_B2000(self):
        self.KFF = self.Rtrans.dot(self.KGG).dot(self.Rtrans.T)
        
    def prepare_mass_matrices_from_B2000(self, MGG):
        self.MGG = MGG
        self.MFF = self.Rtrans.dot(MGG).dot(self.Rtrans.T)
        
        
    def modalanalysis_B2000(self, i_mass):
        modes_selection = copy.deepcopy(self.jcl.mass['modes'][i_mass])
        if self.jcl.mass['omit_rb_modes']: 
            modes_selection += 6
        eigenvalue, eigenvector = self.calc_elastic_modes(self.KFF, self.MFF, modes_selection.max())
        logging.info( 'From these {} modes, the following {} modes are selected: {}'.format(modes_selection.max(), len(modes_selection), modes_selection))
        self.eigenvalues_f = eigenvalue[modes_selection - 1]
        # reconstruct modal matrix for g-set / strcgrid
        self.PHIstrc_f = np.zeros((6*self.strcgrid['n'], len(modes_selection)))
        i = 0 # counter selected modes
        for i_mode in modes_selection - 1:
            # deformation of f-set due to i_mode is the ith column of the eigenvector
            Uf = eigenvector[:,i_mode].real.reshape((-1,1))
            Ug = self.Rtrans.T.dot(Uf)
            self.PHIstrc_f[:,i] = Ug.squeeze()
            i += 1
        return
       
    
    def prepare_stiffness_matrices(self):
        logging.info('Prepare stiffness matrices for independent and free DoFs (f-set)')
        # K_Gnn = K_G(n,n) +  K_G(n,m)*Gm + Gm.'* K_G(n,m).' + Gm.'* K_G(m,m)*Gm;
        Knn = self.KGG[self.pos_n, :][:,self.pos_n] \
                + self.KGG[self.pos_n, :][:,self.pos_m].dot(self.GM.T) \
                + self.GM.dot(self.KGG[self.pos_m, :][:,self.pos_n]) \
                + self.GM.dot(self.KGG[self.pos_m, :][:,self.pos_m].dot(self.GM.T))
        self.KFF = Knn[self.pos_fn, :][:,self.pos_fn]
    
    def prepare_mass_matrices(self, MGG):
        logging.info('Prepare mass matrices for independent and free DoFs (f-set)')
        self.MGG = MGG
        Mnn = self.MGG[self.pos_n, :][:,self.pos_n] \
                + self.MGG[self.pos_n, :][:,self.pos_m].dot(self.GM.T) \
                + self.GM.dot(self.MGG[self.pos_m, :][:,self.pos_n]) \
                + self.GM.dot(self.MGG[self.pos_m, :][:,self.pos_m].dot(self.GM.T))
        self.MFF = Mnn[self.pos_fn, :][:,self.pos_fn]
        
    def modalanalysis(self, i_mass):
        modes_selection = copy.deepcopy(self.jcl.mass['modes'][i_mass])
        if self.jcl.mass['omit_rb_modes']: 
            modes_selection += 6
        eigenvalue, eigenvector = self.calc_elastic_modes(self.KFF, self.MFF, modes_selection.max())
        logging.info( 'From these {} modes, the following {} modes are selected: {}'.format(modes_selection.max(), len(modes_selection), modes_selection))
        self.eigenvalues_f = eigenvalue[modes_selection - 1]
        # reconstruct modal matrix for g-set / strcgrid
        self.PHIstrc_f = np.zeros((6*self.strcgrid['n'], len(modes_selection)))
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
            self.PHIstrc_f[:,i] = Ug.squeeze()
            i += 1
        return
    
    def calc_modal_matrices(self):
        # calc modal mass and stiffness
        logging.info( 'Working on f-set')
        Mff = self.PHIstrc_f.T.dot(self.MGG.dot(self.PHIstrc_f))
        Kff = self.PHIstrc_f.T.dot(self.KGG.dot(self.PHIstrc_f))
        Dff = self.calc_damping(self.eigenvalues_f)
        
        logging.info( 'Working on h-set')
        # add synthetic modes for rigid body motion
        eigenvalues_rb, PHIb_strc = self.calc_rbm_modes()
        PHIstrc_h = np.concatenate((PHIb_strc, self.PHIstrc_f), axis=1)
        Mhh = PHIstrc_h.T.dot(self.MGG.dot(PHIstrc_h))
        # switch signs of off-diagonal terms in rb mass matrix
        # Mhh[:6,:6] = self.Mb
        Khh = PHIstrc_h.T.dot(self.KGG.dot(PHIstrc_h))
        # set rigid body stiffness explicitly to zero
        Khh[np.diag_indices(len(eigenvalues_rb))] = eigenvalues_rb
        Dhh = self.calc_damping(np.concatenate((eigenvalues_rb, self.eigenvalues_f)))
        return Mff, Kff, Dff, self.PHIstrc_f.T, Mhh, Khh, Dhh, PHIstrc_h.T
 
    def calc_elastic_modes(self, K, M, n_modes):
        # perform modal analysis on a-set
        logging.info( 'Modal analysis for first {} modes...'.format( n_modes ))
        eigenvalue, eigenvector = scipy.sparse.linalg.eigs(A=K, M=M , k=n_modes, sigma=0) 
        idx_sort = np.argsort(eigenvalue) # sort result by eigenvalue
        eigenvalue = eigenvalue[idx_sort]
        eigenvector = eigenvector[:,idx_sort]
        frequencies = np.real(eigenvalue)**0.5 /2/np.pi
        logging.info( 'Found {} modes with the following frequencies [Hz]:'.format(len(eigenvalue)))
        logging.info( frequencies )
        n_rbm_estimate = np.sum(np.isnan(frequencies) + np.less(frequencies, 0.1))
        if all([n_rbm_estimate < 6, self.jcl.mass['omit_rb_modes']]):
            logging.warning('There are only {} modes < 0.1 Hz! Is the number of rigid body modes correct ??'.format(n_rbm_estimate))
        return eigenvalue.real, eigenvector.real
    
    def calc_rbm_modes(self):
        eigenvalues = np.zeros(5)
        rules = spline_rules.rules_point(self.cggrid, self.strcgrid)
        PHIstrc_cg = spline_functions.spline_rb(self.cggrid, '', self.strcgrid, '', rules, self.coord)
        return eigenvalues, PHIstrc_cg[:,1:]
    
    def calc_damping(self, eigenvalues):
        # Currently, only modal damping is implemented. See Bianchi et al.,
        # "Using modal damping for full model transient analysis. Application to
        # pantograph/catenary vibration", presented at the ISMA, 2010.
        n = len(eigenvalues)
        Dff = np.zeros((n,n))
        if hasattr(self.jcl, 'damping') and 'method' in self.jcl.damping and self.jcl.damping['method'] == 'modal':
            logging.info( 'Damping: modal damping of {}'.format( self.jcl.damping['damping'] ))
            d = eigenvalues**0.5 * 2.0 * self.jcl.damping['damping']
            np.fill_diagonal(Dff, d) # matrix Dff is modified in-place, no return value
        elif hasattr(self.jcl, 'damping'):
            logging.warning( 'Damping method not implemented: {}'.format( str(self.jcl.damping['method']) ))
            logging.warning( 'Damping: assuming zero damping.' )
        else:
            logging.info( 'Damping: assuming zero damping.' )
        return Dff
        
    def calc_cg(self, i_mass):
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
        m0 = PHIstrc_m0.T.dot(self.MGG.dot(PHIstrc_m0))
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
        mb = PHIstrc_mb.T.dot(self.MGG.dot(PHIstrc_mb))
        # switch signs for coupling terms of I to suite EoMs
        Mb = np.zeros((6,6))
        Mb[0,0] = m
        Mb[1,1] = m
        Mb[2,2] = m
        Mb[3:6,3:6] = np.array([[1,-1,-1],[-1,1,-1],[-1,-1,1]]) * mb[3:6,3:6]
        
        logging.info( 'Mass: {}'.format(Mb[0,0]))
        logging.info( 'CG at: {}'.format(offset_cg))
        logging.info( 'Inertia: \n{}'.format(Mb[3:6,3:6]))
        
        self.cggrid = cggrid # store for later internal use
        self.cggrid_norm = cggrid_norm # store for later internal use
        
        return Mb, cggrid, cggrid_norm
    
    def calc_MAC(self, X, Y, plot=True):
        MAC = np.zeros((X.shape[1],Y.shape[1]),)
        for jj in range(Y.shape[1]):
            for ii in range(X.shape[1]):
                q1 = np.dot(X[:,ii].conj().T, X[:,ii])
                q2 = np.dot(Y[:,jj].conj().T, Y[:,jj])
                q3 = np.dot(X[:,ii].conj().T, Y[:,jj])
                MAC[ii,jj]  = np.real( np.abs(q3)**2/q1/q2 )
        
        if plot:
            plt.figure()
            plt.pcolor(MAC, cmap='hot_r')
            plt.colorbar()
            plt.grid('on')
            
            return MAC, plt
        else:   
            return MAC
 
    