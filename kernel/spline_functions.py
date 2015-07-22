# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 14:50:21 2014

@author: voss_ar
"""
import numpy as np
import scipy
import scipy.sparse as sp
import time
import string
from read_geom import nastran_number_converter

def spline_nastran(filename, strcgrid, aerogrid):
    print 'Read Nastran spline (PARAM    OPGTKG   1) from {}'.format(filename)
    with open(filename, 'r') as fid:
        lines = fid.readlines()
    i_line = 0
    PHI = np.zeros((aerogrid['n']*6, strcgrid['n']*6))
    for line in lines:
        i_line += 1
        if string.find(string.replace(line,' ',''),'GPJK') != -1:
            i_line += 3
            break
        
    while string.find(string.replace(lines[i_line],' ',''),'COLUMN') != -1:
        #print lines[i_line]
        line_split = lines[i_line].split()
        if line_split[3].split('-')[1][:2] == 'T3':
            tmp = 2
        elif line_split[3].split('-')[1][:2] == 'R2':
            tmp = 4
        else:
            print 'DOF not implemented!'
        col = aerogrid['set_k'][np.where(np.int(line_split[3].split('-')[0]) == aerogrid['ID'])[0][0],tmp]
        
        i_line += 1
        while True:
            if lines[i_line] == '\n' or lines[i_line][0] == '1':
                i_line += 1
                break
            
            line_split = lines[i_line].split()
            while len(line_split) >= 3:

                if line_split[1] == 'T3':
                    tmp = 2
                elif line_split[1] == 'R2':
                    tmp = 4
                else:
                    print 'DOF not implemented!'                    
                row = strcgrid['set'][np.where(np.int(line_split[0]) == strcgrid['ID'])[0][0],tmp]
                PHI[col,row] = nastran_number_converter(line_split[2], 'float')
                
                line_split = line_split[3:]
            i_line += 1
            
    return PHI


def spline_rbf(grid_i,  set_i,  grid_d, set_d, rbf_type, dimensions=''):

    cl_rbf = rbf(grid_i['offset'+set_i].T, grid_d['offset'+set_d].T, rbf_type) 
    cl_rbf.build_M()
    cl_rbf.build_splinematrix()
    PHI_tmp = cl_rbf.H[:,4:]
    # obige Matrix gilt fuer Verschiebung eines DOF
    # PHI soll aber alle 6 DOFs verschieben!
    # Hier koennte auch eine sparse-Matrix genommen werden...
    
    # Here, the size of the splining matrix is determined. One might want the matrix to be bigger than actually needed.
    # One example might be the multiplication of the (smaller) x2grid with the (larger) AIC matrix.
    if dimensions != '' and len(dimensions) == 2:
        dimensions_i = dimensions[0]
        dimensions_d = dimensions[1]
    else:
        dimensions_i = 6*len(grid_i['set'+set_i])
        dimensions_d = 6*len(grid_d['set'+set_d])
    print 'Expanding Spline to {:.0f} DOFs and {:.0f} DOFs...'.format(dimensions_d , dimensions_i)
    PHI = np.zeros((dimensions_d, dimensions_i))
    PHI[np.ix_(grid_d['set'+set_d][:,0], grid_i['set'+set_i][:,0])] = PHI_tmp
    PHI[np.ix_(grid_d['set'+set_d][:,1], grid_i['set'+set_i][:,1])] = PHI_tmp
    PHI[np.ix_(grid_d['set'+set_d][:,2], grid_i['set'+set_i][:,2])] = PHI_tmp
    PHI[np.ix_(grid_d['set'+set_d][:,3], grid_i['set'+set_i][:,3])] = PHI_tmp
    PHI[np.ix_(grid_d['set'+set_d][:,4], grid_i['set'+set_i][:,4])] = PHI_tmp
    PHI[np.ix_(grid_d['set'+set_d][:,5], grid_i['set'+set_i][:,5])] = PHI_tmp
    print '... Done'
    return PHI

class rbf:

    def build_M(self):
        # Nomenklatur nach Neumann & Krueger
        print ' - building M'
        self.A = np.vstack((np.ones(self.n_fe),self.nodes_fe))
        self.phi = np.zeros((self.n_fe, self.n_fe))
        for i in range(self.n_fe):
            #for j in range(i+1):
                #r_ij = np.linalg.norm(self.nodes_fe[:,i] - self.nodes_fe[:,j])
            r_ii_vec = self.nodes_fe[:,:i+1] - np.tile(self.nodes_fe[:,i],(i+1,1)).T
            r_ii = np.sum(r_ii_vec**2, axis=0)**0.5
            rbf_values = rbf.eval_rbf(self, r_ii)
            #self.phi.itemset((i,:i+1),rbf_values)
            #self.phi.itemset((:i+1,i),rbf_values)
            self.phi[i,:i+1] = rbf_values
            self.phi[:i+1,i] = rbf_values
                
        # Gleichungssystem aufstellen und loesen
        # M * ab = rechte_seite
        # 0   A  * a = 0
        # A' phi   b   y
        
        # linke Seite basteln
        M1 = np.hstack((np.zeros((self.A.shape[0],self.A.shape[0])) , self.A))
        M2 = np.hstack(( self.A.transpose(), self.phi))
        self.M = np.vstack((M1, M2))

    def build_splinematrix(self):
        # Nomenklatur nach Neumann & Krueger
        print ' - building B and C'
        self.B = np.vstack((np.ones(self.n_cfd),self.nodes_cfd))
        self.C = np.zeros((self.n_fe, self.n_cfd))
        for i in range(self.n_fe):
            #for j in range(self.n_cfd):
                #r_ij = np.linalg.norm(self.nodes_fe[:,i] - self.nodes_cfd[:,j])
            r_ij_vec = np.tile(self.nodes_fe[:,i], (self.n_cfd,1)).T - self.nodes_cfd[:,:] 
            r_ij = np.sum(r_ij_vec**2, axis=0)**0.5
            rbf_values = rbf.eval_rbf(self, r_ij)
            #self.C.itemset((i,j),rbf_value)
            self.C[i,:] = rbf_values
        self.BC = np.vstack((self.B, self.C))
        
        # t_start = time.time()
        # print 'inverting M'
        # M_inv = np.linalg.inv(self.M)
        # print 'calculating H'
        # self.H = np.dot(self.BC.T, M_inv)
        # print str(time.time() - t_start) + 'sec'
        
        t_start = time.time()
        print ' - solving M*H=BC for H'
        self.H= scipy.linalg.solve(self.M, self.BC).T 
        print str(time.time() - t_start) + ' sec'
        
        
    def eval_rbf(self, r):
        if self.rbf_type == 'gauss':
            const = 10.0
            return np.exp(-(const*r)**2)
            
        if self.rbf_type == 'linear':
            return r
            
        if self.rbf_type == 'tps':
#            # sigularitaet bei r = 0 vermeiden
            return r**2 * np.log(r+1e-6)
        
        if self.rbf_type == 'wendlandC0':
            return (1-r)**2
        
        if self.rbf_type == 'wendlandC2':
            return (1-r)**4 + (4*r+1)
        
        else:
            print 'Error: Unkown Radial Basis Function!'
            
    def __init__(self, nodes_fe, nodes_cfd, rbf_type):
        self.nodes_cfd = nodes_cfd
        self.nodes_fe = nodes_fe
        self.n_fe = nodes_fe.shape[1]
        self.n_cfd = self.nodes_cfd.shape[1]
        self.rbf_type = rbf_type
        print 'Splining (rbf) of {:.0f} points to {:.0f} points...'.format(self.n_cfd , self.n_fe)


# Assumptions: 
# - grids have 6 dof

# Usage of sparse matrices from scipy.sparse:
# http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.lil_matrix.html
# lil_matrix((M, N)) is recommended as "an efficient structure for constructing sparse matrices incrementally".

def spline_rb(grid_i,  set_i,  grid_d, set_d, splinerules, coord, dimensions='', sparse_output=False):
    
    # Here, the size of the splining matrix is determined. One might want the matrix to be bigger than actually needed.
    # One example might be the multiplication of the (smaller) x2grid with the (larger) AIC matrix.
    if dimensions != '' and len(dimensions) == 2:
        dimensions_i = dimensions[0]
        dimensions_d = dimensions[1]
    else:
        dimensions_i = 6*len(grid_i['set'+set_i])
        dimensions_d = 6*len(grid_d['set'+set_d])
    print 'Splining (rigid body) of {:.0f} DOFs to {:.0f} DOFs...'.format(dimensions_d , dimensions_i)
        
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
    #T_i = np.zeros((dimensions_i,dimensions_i))    
    T_i = sp.lil_matrix((dimensions_i,dimensions_i))
    for i_i in range(len(grid_i['ID'])):
        pos_coord_i = coord['ID'].index(grid_i['CP'][i_i])
        T_i = sparse_insert( T_i, coord['dircos'][pos_coord_i], grid_i['set'+set_i][i_i,0:3], grid_i['set'+set_i][i_i,0:3] )
        T_i = sparse_insert( T_i, coord['dircos'][pos_coord_i], grid_i['set'+set_i][i_i,3:6], grid_i['set'+set_i][i_i,3:6] )
        #T_i[np.ix_(grid_i['set'+set_i][i_i,0:3], grid_i['set'+set_i][i_i,0:3])] = coord['dircos'][pos_coord_i]
        #T_i[np.ix_(grid_i['set'+set_i][i_i,3:6], grid_i['set'+set_i][i_i,3:6])] = coord['dircos'][pos_coord_i]
        
    #T_d = np.zeros((dimensions_d,dimensions_d))    
    T_d = sp.lil_matrix((dimensions_d,dimensions_d))
    for i_d in range(len(grid_d['ID'])):
        pos_coord_d = coord['ID'].index(grid_d['CP'][i_d])
        T_d = sparse_insert( T_d, coord['dircos'][pos_coord_d], grid_d['set'+set_d][i_d,0:3], grid_d['set'+set_d][i_d,0:3] )
        T_d = sparse_insert( T_d, coord['dircos'][pos_coord_d], grid_d['set'+set_d][i_d,3:6], grid_d['set'+set_d][i_d,3:6] )
        #T_d[np.ix_(grid_d['set'+set_d][i_d,0:3], grid_d['set'+set_d][i_d,0:3])] = coord['dircos'][pos_coord_d]
        #T_d[np.ix_(grid_d['set'+set_d][i_d,3:6], grid_d['set'+set_d][i_d,3:6])] = coord['dircos'][pos_coord_d]
    
    # In matrix T_di the actual splining of gird_d to grid_i according as defined in splinerules is done.
    # Actually, this is the part that implements the rigid body spline
    # The part above should be generic for different splines and could/should be moved to a different function   
    #T_di = np.zeros( (dimensions_d, dimensions_i) )    
    T_di = sp.lil_matrix( (dimensions_d, dimensions_i) )
    for i_i in range(len(splinerules['ID_i'])):
        for i_d in range(len(splinerules['ID_d'][i_i])):
            position_i = np.where(grid_i['ID']==splinerules['ID_i'][i_i])[0][0]
            position_d = np.where(grid_d['ID']==splinerules['ID_d'][i_i][i_d])[0][0]
            
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
        return splinematrix
    else:
        return splinematrix.toarray() 
        
    


def sparse_insert(sparsematrix, submatrix, idx1, idx2):
    # For sparse matrices, "fancy indexing" is not supported / not implemented as of 2014
    # -> the items of a submatrix have to be inserted into the main martix item by item
    # This takes some time, but it is faster for large numbers of DOFs (say >10,000) as 
    # matrix multiplication is faster and memory consumption is much (!!!) lower. 
    for id1 in range(np.shape(submatrix)[0]):
        for id2 in range(np.shape(submatrix)[1]):
            sparsematrix[ idx1[id1], idx2[id2] ] = submatrix[id1,id2]
    return sparsematrix 