#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:37:24 2017

@author: voss_ar
"""

import numpy as np
import cPickle, time, scipy
from scipy import io

def calc_induced_velocities(aerogrid, Ma):
    panels = aerogrid['cornerpoint_panels']
    grids = aerogrid['cornerpoint_grids']
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
    
    # define downwash location (3/4 chord and half span of the aero panel)
    P0 = aerogrid['offset_j']
    # define vortex location points
    P1 = aerogrid['offset_P1']
    P3 = aerogrid['offset_P3']
    #P2 = (P1+P3)/2.0
    n_hat_w  = np.array(aerogrid['N'][:,2], ndmin=2).T.repeat(aerogrid['n'], axis=1)  # normal vector part in vertical direction
    n_hat_wl = np.array(aerogrid['N'][:,1], ndmin=2).T.repeat(aerogrid['n'], axis=1) # normal vector part in lateral direction
    
    # divide x coordinates with beta
    # See Hedman 1965. 
    # However, Hedman divides by beta^2 ... why?? 
    beta = (1-(Ma**2.0))**0.5
    P0[:,0] = P0[:,0]/beta
    P1[:,0] = P1[:,0]/beta
    P3[:,0] = P3[:,0]/beta
    
    # See Katz & Plotkin, Chapter 10.4.5
    # get r1,r2,r0
    r1x = np.array(P0[:,0], ndmin=2).T - np.array(P1[:,0], ndmin=2)
    r1y = np.array(P0[:,1], ndmin=2).T - np.array(P1[:,1], ndmin=2)
    r1z = np.array(P0[:,2], ndmin=2).T - np.array(P1[:,2], ndmin=2)
    
    r2x = np.array(P0[:,0], ndmin=2).T - np.array(P3[:,0], ndmin=2)
    r2y = np.array(P0[:,1], ndmin=2).T - np.array(P3[:,1], ndmin=2)
    r2z = np.array(P0[:,2], ndmin=2).T - np.array(P3[:,2], ndmin=2)
    
    # Step 1
    r1Xr2_x =  r1y*r2z - r1z*r2y
    r1Xr2_y = -r1x*r2z + r1z*r2x # Plus-Zeichen Abweichung zu Katz & Plotkin ??
    r1Xr2_z =  r1x*r2y - r1y*r2x
    mod_r1Xr2 = ( r1Xr2_x**2.0 + r1Xr2_y**2.0 + r1Xr2_z**2.0 )**0.5
    # Step 2
    r1 = ( r1x**2.0 + r1y**2.0 + r1z**2.0 )**0.5
    r2 = ( r2x**2.0 + r2y**2.0 + r2z**2.0 )**0.5
    # Step 4
    r0r1 = (P3[:,0]-P1[:,0])*r1x + (P3[:,1]-P1[:,1])*r1y + (P3[:,2]-P1[:,2])*r1z
    r0r2 = (P3[:,0]-P1[:,0])*r2x + (P3[:,1]-P1[:,1])*r2y + (P3[:,2]-P1[:,2])*r2z
    # Step 5
    gamma = np.ones((aerogrid['n'], aerogrid['n']))
    D1_base = gamma / 4.0 / np.pi / mod_r1Xr2**2.0 * (r0r1/r1 - r0r2/r2)
    D1_u = r1Xr2_x*D1_base
    D1_v = r1Xr2_y*D1_base
    D1_w = r1Xr2_z*D1_base
    # Step 3
    epsilon = 10e-6;
    ind =  np.where(r1<epsilon)[0]
    D1_u[ind] = 0.0
    D1_v[ind] = 0.0
    D1_w[ind] = 0.0
        
    ind =  np.where(r2<epsilon)[0]
    D1_u[ind] = 0.0
    D1_v[ind] = 0.0
    D1_w[ind] = 0.0
        
    ind =  np.where(mod_r1Xr2<epsilon)
    D1_u[ind] = 0.0
    D1_v[ind] = 0.0
    D1_w[ind] = 0.0
        
    # get final D1 matrix 
    # D1 matrix contains the perpendicular component of induced velocities at all panels.
    # For wing panels, it's the z component of induced velocities (D1_w) while for
    # winglets, it's the y component of induced velocities (D1_v)
    D1 = D1_w*n_hat_w + D1_v*n_hat_wl
        
    # induced velocity due to inner semi-infinite vortex line
    d2 = (r1y**2.0 + r1z**2.0)**0.5
    cosBB1 = 1.0
    cosBB2 = -r1x/r1
    cosGamma = r1y/d2
    sinGamma = -r1z/d2
    
    D2_base = -(1.0/(4.0*np.pi))*(cosBB1 - cosBB2)/d2
    D2_u = np.zeros((aerogrid['n'], aerogrid['n']))
    D2_v = sinGamma*D2_base
    D2_w = cosGamma*D2_base
    
    ind =  np.where(r1<epsilon)[0]
    D2_u[ind] = 0.0
    D2_v[ind] = 0.0
    D2_w[ind] = 0.0
    
    D2 = D2_w*n_hat_w + D2_v*n_hat_wl
    
    # induced velocity due to outer semi-infinite vortex line
    d3 = (r2y**2.0 + r2z**2.0)**0.5
    cosBB1 = r2x/r2
    cosBB2 = -1.0
    cosGamma = -r2y/d3
    sinGamma = r2z/d3
    
    D3_base = -(1.0/(4.0*np.pi))*(cosBB1 - cosBB2)/d3
    D3_u = r1Xr2_x*D3_base
    D3_v = sinGamma*D3_base
    D3_w = cosGamma*D3_base
    
    ind =  np.where(r2<epsilon)[0]
    D3_u[ind] = 0.0
    D3_v[ind] = 0.0
    D3_w[ind] = 0.0
    
    D3 = D3_w*n_hat_w + D3_v*n_hat_wl
    
    return D1, D2, D3

def calc_Ajj(aerogrid, Ma):
    D1, D2, D3 = calc_induced_velocities(aerogrid, Ma)
    # define area, chord length and spann of each panel
    A = aerogrid['A']
    chord = aerogrid['l']
    span = A / chord
    # total D
    D = D1 + D2 + D3
    D_final = D*0.5*A/span
    D_induced_drag = (D2 + D3)*0.5*A/span
    return D_final, D_induced_drag

def calc_Qjj(aerogrid, Ma):
    D_final, D_induced_drag = calc_Ajj(aerogrid, Ma)
    Qjj = -np.linalg.inv(D_final)
    Bjj = D_induced_drag
    return Qjj, Bjj

def calc_Gamma(aerogrid, Ma):
    D1, D2, D3 = calc_induced_velocities(aerogrid, Ma)
    # total D
    Gamma = -np.linalg.inv((D1 + D2 + D3))
    return Gamma

# if __name__ == "__main__":
#     with open('/scratch/test/model_jcl_Discus2c_test.pickle', 'r') as f:
#         model = cPickle.load(f)
#     #with open('aerogrid.mat', 'w') as f:
#     #    scipy.io.savemat(f, model['aerogrid'])
#     t_start = time.time()
#     D_final, D_induced_drag = calc_Gamma(aerogrid=model['aerogrid'], Ma=0.15)
#     print( '--> Done in %.2f [sec].' % (time.time() - t_start))
