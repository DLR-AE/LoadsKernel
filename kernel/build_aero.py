# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 09:07:55 2014

@author: voss_ar
"""

import read_geom
import spline_rules
import spline_functions
import numpy as np

def build_x2grid(jcl_aero, aerogrid, coord):
    
    for i_file in range(len(jcl_aero['filename_aesurf'])):
        sub_aesurf = read_geom.Modgen_AESURF(jcl_aero['filename_aesurf'][i_file])
        if i_file == 0:
            aesurf = sub_aesurf
        else:
            for key in aesurf.keys():
                aesurf[key] += sub_aesurf[key]
                
    for i_file in range(len(jcl_aero['filename_aesurf'])):             
        coord = read_geom.Modgen_CORD2R(jcl_aero['filename_aesurf'][i_file], coord) 
        
    for i_file in range(len(jcl_aero['filename_aelist'])):
        sub_aelist = read_geom.Modgen_AELIST(jcl_aero['filename_aelist'][i_file]) 
        if i_file == 0:
            aelist = sub_aelist
        else:
            aelist['ID'] = np.hstack((aelist['ID'], sub_aelist['ID']))
            aelist['values'] = np.vstack((aelist['values'], sub_aelist['values']))
                
    x2grid = {'ID_surf': aesurf['ID'],
               'CID': aesurf['CID'],
               'key': aesurf['key'],
               'eff': aesurf['eff'],
               'ID': [],
               'CD': [],
               'CP': [], 
               'offset_j': [],
               'set_j': [],
               }    
    for i_surf in range(len(aesurf['ID'])):
        x2grid['CD'].append([])
        x2grid['CP'].append([])
        x2grid['ID'].append([])
        x2grid['offset_j'].append([])
        x2grid['set_j'].append([])
        for i_panel in aelist['values'][i_surf]:
            pos_panel = np.where(aerogrid['ID']==i_panel)[0][0]
            x2grid['ID'][i_surf].append(aerogrid['ID'][pos_panel])
            x2grid['CD'][i_surf].append(aerogrid['CD'][pos_panel])
            x2grid['CP'][i_surf].append(aerogrid['CP'][pos_panel])
            x2grid['offset_j'][i_surf].append(aerogrid['offset_j'][pos_panel])
            x2grid['set_j'][i_surf].append(aerogrid['set_j'][pos_panel])
        
    return x2grid, coord   
   
        

def build_aerogrid(filename_caero_bdf):
    # all corner points are defined as grid points by ModGen
    caero_grid = read_geom.Modgen_GRID(filename_caero_bdf)
    # four grid points are assembled to one panel, this is expressed as CQUAD4s 
    caero_panels = read_geom.Modgen_CQUAD4(filename_caero_bdf)

    ID = []
    l = [] # length of panel
    A = [] # area of one panel
    N = [] # unit normal vector 
    offset_l = [] # 25% point l
    offset_k = [] # 50% point k
    offset_j = [] # 75% downwash control point j
    
    for i_panel in range(len(caero_panels['ID'])):

        #
        #                   l_2
        #             4 o---------o 3
        #               |         |
        #  u -->    b_1 | l  k  j | b_2
        #               |         |
        #             1 o---------o 2
        #         y         l_1
        #         |
        #        z.--- x

        index_1 = np.where(caero_panels['cornerpoints'][i_panel][0]==caero_grid['ID'])[0][0]
        index_2 = np.where(caero_panels['cornerpoints'][i_panel][1]==caero_grid['ID'])[0][0]
        index_3 = np.where(caero_panels['cornerpoints'][i_panel][2]==caero_grid['ID'])[0][0]
        index_4 = np.where(caero_panels['cornerpoints'][i_panel][3]==caero_grid['ID'])[0][0]
        
        l_1 = caero_grid['offset'][index_2] - caero_grid['offset'][index_1]
        l_2 = caero_grid['offset'][index_3] - caero_grid['offset'][index_4]
        b_1 = caero_grid['offset'][index_4] - caero_grid['offset'][index_1]
        b_2 = caero_grid['offset'][index_3] - caero_grid['offset'][index_2]
        l_m = (l_1 + l_2) / 2.0
        b_m = (b_1 + b_2) / 2.0
        
        ID.append(caero_panels['ID'][i_panel])    
        l.append(l_m[0])
        A.append(l_m[0]*b_m[1])
        N.append(np.cross(l_1, b_1)/np.linalg.norm(np.cross(l_1, b_1)))
        offset_l.append(caero_grid['offset'][index_1] + 0.25*l_m + 0.50*b_1)
        offset_k.append(caero_grid['offset'][index_1] + 0.50*l_m + 0.50*b_1)
        offset_j.append(caero_grid['offset'][index_1] + 0.75*l_m + 0.50*b_1)
   
    n = len(ID)
    set_l = np.arange(n*6).reshape((n,6))
    set_k = np.arange(n*6).reshape((n,6))
    set_j = np.arange(n*6).reshape((n,6))
    #set_j = np.zeros((n,6))
    #set_j[:,(2,4)] = np.arange(n*2).reshape((n,2))
    aerogrid = {'ID': np.array(ID),
                'l': np.array(l),
                'A': np.array(A),
                'N': np.array(N),
                'offset_l': np.array(offset_l),
                'offset_k': np.array(offset_k),
                'offset_j': np.array(offset_j),
                'set_l': set_l,
                'set_k': set_k,
                'set_j': set_j,
                'CD': caero_grid['CD'],
                'CP': caero_grid['CP'],
                'coord_desc': 'bodyfixed',
               }     
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(caero_grid['offset'][:,0], caero_grid['offset'][:,1], caero_grid['offset'][:,2], color='b', marker='.', label='caero_grid')
    #ax.scatter(aerogrid['offset_k'][:,0], aerogrid['offset_k'][:,1], aerogrid['offset_k'][:,2], color='r', marker='.', label='k')
    #ax.scatter(aerogrid['offset_j'][:,0], aerogrid['offset_j'][:,1], aerogrid['offset_j'][:,2], color='g', marker='.', label='j')
    #plt.show()    
    return aerogrid

def build_macgrid(aerogrid, b_ref):
    # Assumptions:
    # - the meam aerodynamic center is the 25% point on the mean aerodynamic choord
    # - the mean aerodynamic choord runs through the (geometrical) center of all panels
    A = np.sum(aerogrid['A'])
    mean_aero_choord = A/b_ref
    geo_center_x = np.dot(aerogrid['offset_k'][:,0], aerogrid['A'])/A
    geo_center_y = np.dot(aerogrid['offset_k'][:,1], aerogrid['A'])/A
    geo_center_z = np.dot(aerogrid['offset_k'][:,2], aerogrid['A'])/A
    macgrid = {'ID': np.array([0]),
               'offset': np.array([[geo_center_x-0.25*mean_aero_choord, geo_center_y, geo_center_z]]), 
               "set":np.array([[0, 1, 2, 3, 4, 5]]),
               'CD': np.array([0]),
               'CP': np.array([0]),
               'coord_desc': 'bodyfixed',
               'A_ref': A,
               'b_ref': b_ref,
               'c_ref': mean_aero_choord,
              }
    return macgrid

