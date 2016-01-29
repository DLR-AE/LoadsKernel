# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 09:07:55 2014

@author: voss_ar
"""

import read_geom
import spline_rules
import spline_functions
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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
            for key in aelist.keys():
                aelist[key] += sub_aelist[key]
                
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
        
        for i_panel in aelist['values'][aelist['ID'].index( aesurf['AELIST'][i_surf] )]:
            pos_panel = np.where(aerogrid['ID']==i_panel)[0][0]
            x2grid['ID'][i_surf].append(aerogrid['ID'][pos_panel])
            x2grid['CD'][i_surf].append(aerogrid['CD'][pos_panel])
            x2grid['CP'][i_surf].append(aerogrid['CP'][pos_panel])
            x2grid['offset_j'][i_surf].append(aerogrid['offset_j'][pos_panel])
            x2grid['set_j'][i_surf].append(aerogrid['set_j'][pos_panel])
        
    return x2grid, coord   
     

def build_aerogrid(filename_caero_bdf, method_caero = 'CQUAD4'):
    if method_caero == 'CQUAD4':
        # all corner points are defined as grid points by ModGen
        caero_grid = read_geom.Modgen_GRID(filename_caero_bdf)
        # four grid points are assembled to one panel, this is expressed as CQUAD4s 
        caero_panels = read_geom.Modgen_CQUAD4(filename_caero_bdf)
    elif method_caero == 'CAERO1':
        caero_grid, caero_panels = read_geom.Nastran_CAERO1(filename_caero_bdf)
    else:
        print "Error: Method %s not implemented. Availble options are 'CQUAD4' and 'CAERO1'" % method_caero
    print ' - from corner points and aero panels, constructing aerogrid'
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
        # A.append(l_m[0]*b_m[1])
        A.append(np.linalg.norm(np.cross(l_m, b_m)))
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
                'CD': caero_panels['CD'],
                'CP': caero_panels['CP'],
                'n': n,
                'coord_desc': 'bodyfixed',
                'cornerpoint_panels': caero_panels['cornerpoints'],
                'cornerpoint_grids': np.hstack((caero_grid['ID'][:,None],caero_grid['offset']))
               }   
    #import matplotlib.pyplot as plt
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(caero_grid['offset'][:,0], caero_grid['offset'][:,1], caero_grid['offset'][:,2], color='b', marker='.', label='caero_grid')
    #ax.scatter(aerogrid['offset_k'][:,0], aerogrid['offset_k'][:,1], aerogrid['offset_k'][:,2], color='r', marker='.', label='k')
    #ax.scatter(aerogrid['offset_j'][:,0], aerogrid['offset_j'][:,1], aerogrid['offset_j'][:,2], color='g', marker='.', label='j')
    #plt.scatter( aerogrid['offset_k'][:,0], aerogrid['offset_k'][:,1], color='r', marker='.', label='k')
    #plt.scatter( aerogrid['offset_j'][:,0], aerogrid['offset_j'][:,1], color='g', marker='.', label='j')
    #plt.grid('on')
    #plt.legend()
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

def plot_aerogrid(aerogrid, cp = '', colormap = 'jet', value_min = '', value_max = ''):
    # This function plots aerogrids as used in the Loads Kernel
    # - By default, the panales are plotted as a wireframe.
    # - If a pressure distribution (or any numpy array with n values) is given, 
    #   the panels are colored according to this value.
    # - It is possible to give a min and max value for the color distirbution, 
    #   which is useful to compare severeal plots.  
    
    if len(cp) == aerogrid['n']:
        colors = plt.cm.get_cmap(name=colormap)  
        if value_min == '':
            value_min = cp.min()
        if value_max == '':
            value_max = cp.max()   

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plot evry panel seperatly
    # (plotting all at once is much more complicated!)
    for i_panel in range(aerogrid['n']):
        # construct matrices xx, yy and zz from cornerpoints for each panale
        point0 =  aerogrid['cornerpoint_grids'][aerogrid['cornerpoint_grids'][:,0] == aerogrid['cornerpoint_panels'][i_panel,0],1:]
        point1 =  aerogrid['cornerpoint_grids'][aerogrid['cornerpoint_grids'][:,0] == aerogrid['cornerpoint_panels'][i_panel,1],1:]
        point2 =  aerogrid['cornerpoint_grids'][aerogrid['cornerpoint_grids'][:,0] == aerogrid['cornerpoint_panels'][i_panel,2],1:]
        point3 =  aerogrid['cornerpoint_grids'][aerogrid['cornerpoint_grids'][:,0] == aerogrid['cornerpoint_panels'][i_panel,3],1:]
        xx = np.array(([point0[0,0], point1[0,0]], [point3[0,0], point2[0,0]]))
        yy = np.array(([point0[0,1], point1[0,1]], [point3[0,1], point2[0,1]]))
        zz = np.array(([point0[0,2], point1[0,2]], [point3[0,2], point2[0,2]]))
        # determine the color of the panel according to pressure coefficient
        # (by default, panels are colored according to its z-component)
        if len(cp) == aerogrid['n']:
            color_i = colors(np.int(np.round( colors.N / (value_max - value_min ) * (cp[i_panel] - value_min ) )))
            ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, linewidth=0, color=color_i )
        else:
            ax.plot_wireframe(xx, yy, zz, rstride=1, cstride=1)
            
    if len(cp) == aerogrid['n']:
        # plot one dummy element that is colored by using the colormap
        # (this is required to build a colorbar)
        surf = ax.plot_surface([0],[0],[0], rstride=1, cstride=1, linewidth=0, cmap=colors, vmin=value_min, vmax=value_max)
        fig.colorbar(surf, shrink=0.5) 
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=90, azim=-90) 
    fig.tight_layout()
    
    return ax