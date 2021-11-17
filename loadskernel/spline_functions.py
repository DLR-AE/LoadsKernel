# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:50:21 2014

@author: voss_ar
"""
import numpy as np
import scipy
import scipy.sparse as sp
import time, logging

from loadskernel.read_geom import nastran_number_converter
import loadskernel.grid_trafo as grid_trafo

def spline_nastran(filename, strcgrid, aerogrid):
    logging.info('Read Nastran spline (PARAM    OPGTKG   1) from {}'.format(filename))
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    i_line = 0
    PHI = np.zeros((aerogrid['n']*6, strcgrid['n']*6))
    for line in lines:
        i_line += 1
        if str.find(str.replace(line,' ',''),'GPJK') != -1:
            i_line += 3
            break
        
    while str.find(str.replace(lines[i_line],' ',''),'COLUMN') != -1:
        #print lines[i_line]
        line_split = lines[i_line].split()
        if line_split[3].split('-')[1][:2] == 'T3':
            tmp = 2
        elif line_split[3].split('-')[1][:2] == 'R2':
            tmp = 4
        else:
            logging.error('DOF not implemented!')
        col = aerogrid['set_k'][np.where(np.int(line_split[3].split('-')[0]) == aerogrid['ID'])[0][0],tmp]
        
        i_line += 1
        while True:
            if lines[i_line] == '\n' or lines[i_line][0] == '1':
                i_line += 1
                break
            
            line_split = lines[i_line].split()
            while len(line_split) >= 3:
                if line_split[1] == 'T1':
                    tmp = 0
                elif line_split[1] == 'T2':
                    tmp = 1
                elif line_split[1] == 'T3':
                    tmp = 2
                elif line_split[1] == 'R1':
                    tmp = 3
                elif line_split[1] == 'R2':
                    tmp = 4
                elif line_split[1] == 'R3':
                    tmp = 5
                else:
                    logging.error('DOF not implemented!')                
                row = strcgrid['set'][np.where(np.int(line_split[0]) == strcgrid['ID'])[0][0],tmp]
                PHI[col,row] = nastran_number_converter(line_split[2], 'float')
                
                line_split = line_split[3:]
            i_line += 1
            
    return PHI

def sparse_insert(sparsematrix, submatrix, idx1, idx2):
    # For sparse matrices, "fancy indexing" is not supported / not implemented as of 2014
    # -> the items of a submatrix have to be inserted into the main martix item by item
    # This takes some time, but it is faster for large numbers of DOFs (say >10,000) as 
    # matrix multiplication is faster and memory consumption is much (!!!) lower. 
    for id1 in range(np.shape(submatrix)[0]):
        for id2 in range(np.shape(submatrix)[1]):
            if submatrix[id1,id2] != 0.0:
                sparsematrix[ idx1[id1], idx2[id2] ] = submatrix[id1,id2]
    return sparsematrix 

def sparse_insert_coo(sparsematrix, submatrix, idx1, idx2):
    # For sparse matrices, "fancy indexing" is not supported / not implemented as of 2017
    # -> in case of a coo matrix, row and column based indexing is supported
    row, col = np.meshgrid(idx1, idx2, indexing='ij')
    sparsematrix.row = np.concatenate(( sparsematrix.row, row.reshape(-1) ))
    sparsematrix.col = np.concatenate(( sparsematrix.col, col.reshape(-1) ))
    sparsematrix.data = np.concatenate(( sparsematrix.data, submatrix.reshape(-1) ))
    return sparsematrix

def spline_rbf(grid_i,  set_i,  grid_d, set_d, 
               rbf_type='tps', surface_spline=False,
               dimensions='', support_radius=2.0):
    """
    This is a convenience function that wraps the Spline_rbf class and returns the spline matrix PHI.
    """
    rbf = Spline_rbf(grid_i['offset'+set_i].T, grid_d['offset'+set_d].T, rbf_type, surface_spline, support_radius) 
    rbf.build_M()
    rbf.build_BC()
    rbf.build_splinematrix()
    rbf.expand_splinematrix(grid_i,  set_i,  grid_d, set_d, dimensions)
    return rbf.PHI_expanded

class Spline_rbf:

    def __init__(self, nodes_fe, nodes_cfd, rbf_type, surface_spline, support_radius):
        """
        This class organizes the calculation of the spline matrix in four steps:
        1. build_M()
        2. build_BC()
        3. build_splinematrix()
        4. expand_splinematrix()
        """
        self.surface_spline = surface_spline
        self.rbf_type = rbf_type
        self.rbf_type = 'wendland2'
        self.R = support_radius
        if self.surface_spline:
            logging.debug('Using surface formulation (2D xy surface)')
            self.nodes_cfd = nodes_cfd[0:2,:]
            self.nodes_fe = nodes_fe[0:2,:]
            self.n = 3
        else:
            logging.debug('Using volume formulation (3D)')
            self.nodes_cfd = nodes_cfd
            self.nodes_fe = nodes_fe
            self.n = 4
        self.n_fe = nodes_fe.shape[1]
        self.n_cfd = self.nodes_cfd.shape[1]
        logging.info('Splining (rbf) of {:.0f} points to {:.0f} points...'.format(self.n_cfd , self.n_fe))

    def build_M(self):
        # Nomenklatur nach Neumann & Krueger
        logging.debug(' - building M')
        self.A = np.vstack((np.ones(self.n_fe),self.nodes_fe))
        phi = np.zeros((self.n_fe, self.n_fe))
        for i in range(self.n_fe):
            #for j in range(i+1):
                #r_ij = np.linalg.norm(self.nodes_fe[:,i] - self.nodes_fe[:,j])
            r_ii_vec = self.nodes_fe[:,:i+1] - np.tile(self.nodes_fe[:,i],(i+1,1)).T
            r_ii = np.sum(r_ii_vec**2, axis=0)**0.5
            rbf_values = self.eval_rbf(r_ii)
            phi[i,:i+1] = rbf_values
            phi[:i+1,i] = rbf_values
                
        # Gleichungssystem aufstellen und loesen
        # M * ab = rechte_seite
        # 0   A  * a = 0
        # A' phi   b   y
        
        # linke Seite basteln
        M1 = np.hstack((np.zeros((self.n,self.n)) , self.A))
        M2 = np.hstack(( self.A.transpose(), phi))
        self.M = np.vstack((M1, M2))

    def build_BC(self):
        # Nomenklatur nach Neumann & Krueger
        logging.debug(' - building B and C')
        B = np.vstack((np.ones(self.n_cfd),self.nodes_cfd))
        C = np.zeros((self.n_fe, self.n_cfd))
        for i in range(self.n_fe):
            r_ij_vec = np.tile(self.nodes_fe[:,i], (self.n_cfd,1)).T - self.nodes_cfd[:,:] 
            r_ij = np.sum(r_ij_vec**2, axis=0)**0.5
            rbf_values = self.eval_rbf(r_ij)
            C[i,:] = rbf_values
        self.BC = np.vstack((B, C))
    
    def build_splinematrix(self):
        """
        Now that the system of equations is set-up, we can work on the solution.
        Note 1: Instead of calculating the inverse of M, we solve the system, which 
        can be faster.
        Note 2: In PyCSM_CouplingMatrix_3D.py by Jens Neumann a slightly different 
        approach is used by first reducing the system size (M * E) and the solving 
        the system. However, this requires two more matrix multiplications and showed 
        no advantages in terms of speed, so I decided to stay with my formulation.
        Note 3: The results are numerically equivalent (using np.allclose(self.PHI, self.G))
        to results obtained with PyCSM_CouplingMatrix_3D.py.
        """
        # solve the system
        t_start = time.time()
        logging.debug(' - solving M*H=BC for H')
        H = scipy.linalg.solve(self.M, self.BC).T 
        logging.debug(' - done in {:.2f} sec'.format(time.time() - t_start))
        # finally, extract the complete splining matrix PHI from H
        self.PHI = H[:,self.n:]
    
    def expand_splinematrix(self, grid_i,  set_i,  grid_d, set_d, dimensions):
        """
        This functions does three things:
        a) The spline matrix applies to all three translational degrees of freedom.
        b) The size of the splining matrix is expanded as one might want the matrix to be 
        bigger than actually needed. One example might be the multiplication of the (smaller) 
        x2grid with the (larger) AIC matrix.
        c) Because the spline matrix can contain many zeros, a sparse matrix might be a better 
        choice compared to a full numpy array. 
        """
        if dimensions != '' and len(dimensions) == 2:
            dimensions_i = dimensions[0]
            dimensions_d = dimensions[1]
        else:
            dimensions_i = 6*len(grid_i['set'+set_i])
            dimensions_d = 6*len(grid_d['set'+set_d])
        logging.debug(' - expanding spline matrix to {:.0f} DOFs and {:.0f} DOFs...'.format(dimensions_d , dimensions_i))
        # coo sparse matrices are good for inserting new data
        PHI_coo = sp.coo_matrix((dimensions_d , dimensions_i))
        PHI_coo = sparse_insert_coo(PHI_coo, self.PHI, grid_d['set'+set_d][:,0], grid_i['set'+set_i][:,0])
        PHI_coo = sparse_insert_coo(PHI_coo, self.PHI, grid_d['set'+set_d][:,1], grid_i['set'+set_i][:,1])
        PHI_coo = sparse_insert_coo(PHI_coo, self.PHI, grid_d['set'+set_d][:,2], grid_i['set'+set_i][:,2])
        # better sparse format than coo
        self.PHI_expanded = PHI_coo.tobsr() 
                
    def eval_rbf(self, r):
            
        if self.rbf_type == 'linear':
            return r
            
        if self.rbf_type == 'tps':
            """
            See Harder, R. L., and Desmarais, R. N., “Interpolation using 
            surface splines.,” Journal of Aircraft, vol. 9, no. 2, pp. 189–191, 
            Feb. 1972, https://doi.org/10.2514/3.44330.
            """
            rbf = r**2 * np.log(r**2)
            # Fix singularity when r = 0.0
            rbf[r==0.0] = 0.0
            return rbf
        
        if self.rbf_type == 'wendland1':
            """
            See section 4.1.2 in Neumann, J., “Identifikation radialer Basisfunktionen zur 
            räumlichen Strömungs-Struktur-Kopplung unter Berücksichtigung des Interpolations- 
            und des Lasttransformationsfehlers,” Institute of Aeroelasticity, Göttingen, Germany, 
            Internal Report DLR IB 232-2008 J 01, 2008.
            """
            rbf = np.zeros(r.shape)
            pos = r <= self.R
            rbf[pos] = (1.0-r[pos]/self.R)**2
            return rbf
        
        if self.rbf_type == 'wendland2':
            rbf = np.zeros(r.shape)
            pos = r <= self.R
            rbf[pos] = (1.0-r[pos]/self.R)**4 * (4.0*r[pos]/self.R + 1.0)
            return rbf
        
        else:
            logging.error('Unkown Radial Basis Function!')



def spline_rb(grid_i,  set_i,  grid_d, set_d, splinerules, coord, dimensions='', sparse_output=False):
    
    # Here, the size of the splining matrix is determined. One might want the matrix to be bigger than actually needed.
    # One example might be the multiplication of the (smaller) x2grid with the (larger) AIC matrix.
    if dimensions != '' and len(dimensions) == 2:
        dimensions_i = dimensions[0]
        dimensions_d = dimensions[1]
    else:
        dimensions_i = 6*len(grid_i['set'+set_i])
        dimensions_d = 6*len(grid_d['set'+set_d])
    logging.info('Splining (rigid body) of {:.0f} DOFs to {:.0f} DOFs...'.format(dimensions_d , dimensions_i))
        
    # transfer points into common coord
    offset_dest_i = []
    for i_point in range(len(grid_i['ID'])):
        pos_coord = coord['ID'].index(grid_i['CP'][i_point])
        offset_dest_i.append(np.dot(coord['dircos'][pos_coord],grid_i['offset'+set_i][i_point])+coord['offset'][pos_coord])
    offset_dest_i = np.array(offset_dest_i)
    
    offset_dest_d = []
    for i_point in range(len(grid_d['ID'])):
        pos_coord = coord['ID'].index(grid_d['CP'][i_point])
        offset_dest_d.append(np.dot(coord['dircos'][pos_coord],grid_d['offset'+set_d][i_point])+coord['offset'][pos_coord])
    offset_dest_d = np.array(offset_dest_d)
       
    # T_i and T_d are the translation matrices that do the projection to the coordinate systems of gird_i and grid_d
    T_i, T_d = grid_trafo.calc_transformation_matrix(coord, 
                                                     grid_i, set_i, 'CP',
                                                     grid_d, set_d, 'CP', 
                                                     dimensions)
    
    # In matrix T_di the actual splining of gird_d to grid_i according as defined in splinerules is done.
    # Actually, this is the part that implements the rigid body spline
    # The part above should be generic for different splines and could/should be moved to a different function   
    T_di = sp.lil_matrix( (dimensions_d, dimensions_i) )
    for i_i in range(len(splinerules['ID_i'])):
        for i_d in range(len(splinerules['ID_d'][i_i])):
            try:
                position_i = np.where(grid_i['ID']==splinerules['ID_i'][i_i])[0][0]
                position_d = np.where(grid_d['ID']==splinerules['ID_d'][i_i][i_d])[0][0]
            except:
                raise AssertionError('There is a problem at monitoring station {}, grid {}'.format(splinerules['ID_i'][i_i], splinerules['ID_d'][i_i][i_d]))
            
            T_sub = np.eye(6)

            # Kraefte erzeugen zusaetzliche Momente durch Hebelarm 'r'
            # M = r cross F = sk(r)*F          
            r = offset_dest_d[position_d] - offset_dest_i[position_i]    
            T_sub[0,4] =  r[2]
            T_sub[0,5] = -r[1]
            T_sub[1,3] = -r[2]
            T_sub[1,5] =  r[0]
            T_sub[2,3] =  r[1]
            T_sub[2,4] = -r[0]  
            T_di = sparse_insert(T_di, T_sub, grid_d['set'+set_d][position_d,0:6], grid_i['set'+set_i][position_i,0:6])
            
    splinematrix = T_d.transpose().dot(T_di).dot(T_i)
    if sparse_output:
        return splinematrix.tocsc() # better sparse format than lil_matrix
    else:
        return splinematrix.toarray() 