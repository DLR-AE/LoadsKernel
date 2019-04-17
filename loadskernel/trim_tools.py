# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 17:44:58 2014

@author: voss_ar
"""
import numpy as np
from atmosphere import isa as atmo_isa

def calc_drehmatrix_angular( phi=0.0, theta=0.0, psi=0.0 ):
    # Alle Winkel in [rad] !
    # geo to body
    drehmatrix = np.array(([1., 0. , -np.sin(theta)], [0., np.cos(phi), np.sin(phi)*np.cos(theta)], [0., -np.sin(phi), np.cos(phi)*np.cos(theta)]))
    return drehmatrix

def calc_drehmatrix_angular_inv( phi=0.0, theta=0.0, psi=0.0 ):
    # Alle Winkel in [rad] !
    # body to geo
    drehmatrix = np.array(([1., np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)], [0., np.cos(phi), -np.sin(phi)], [0., np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]))
    return drehmatrix

def calc_drehmatrix( phi=0.0, theta=0.0, psi=0.0 ):
    # Alle Winkel in [rad] !
    drehmatrix_phi = np.array(([1., 0. , 0.], [0., np.cos(phi), np.sin(phi)], [0., -np.sin(phi), np.cos(phi)]))
    drehmatrix_theta  = np.array(([np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]))
    drehematrix_psi = np.array(([np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0],[0, 0, 1]))
    drehmatrix = np.dot(np.dot(drehmatrix_phi, drehmatrix_theta),drehematrix_psi)
    return drehmatrix

def gravitation_on_earth(PHInorm_cg, Tgeo2body):
    g = np.array([0.0, 0.0, 9.8066]) # erdfest, geodetic
    g_cg = np.dot(PHInorm_cg[0:3,0:3], np.dot(Tgeo2body[0:3,0:3],g)) # bodyfixed
    return g_cg

def design_gust_cs_25_341(gust_gradient, altitude, rho, V, Z_mo, V_D, MLW, MTOW, MZFW):
    # Gust Calculation from CS 25.341
    # adapted from matlab-script by Vega Handojo, DLR-AE-LAE, 2015
    
    # convert (possible) integer to float
    gust_gradient = np.float(gust_gradient)
    altitude = np.float(altitude)     # Altitude
    rho = np.float(rho)     # Air density
    V = np.float(V)         # Speed
    Z_mo = np.float(Z_mo)   # Maximum operating altitude
    V_D = np.float(V_D)     # Design speed
    MLW = np.float(MLW)     # Maximum Landing Weight
    MTOW = np.float(MTOW)   # Maximum Take-Off Weight
    MZFW = np.float(MZFW)   # Maximum Zero Fuel Weight

    p0, rho0, T0, a0 = atmo_isa(0.0)
    R1 = MLW/MTOW
    R2 = MZFW/MTOW
    f_gm = (R2*np.tan(np.pi*R1/4))**0.5
    f_gz = 1-Z_mo/76200.0
    
    # flight profile alleviation factor
    fg_sl = 0.5*(f_gz+f_gm) # at sea level
    if altitude == 0:
        fg = fg_sl
    elif altitude == Z_mo:    # at maximum flight-level F_g = 1
        fg = 1.0
    else:                 # between SL and Z_mo increases linearily to 1
        fg = fg_sl + (1-fg_sl)*altitude/Z_mo
        
    # reference gust velocity (EAS) [m/s]
    if altitude <= 4572:
        u_ref = 17.07-(17.07-13.41)*altitude/4572.0
    else:
        u_ref = 13.41-(13.41-6.36)*((altitude-4572.0)/(18288.0-4572.0))
    if V == V_D:
        u_ref = u_ref/2.0

    # design gust velocity (EAS)
    u_ds = u_ref * fg * (gust_gradient/107.0)**(1.0/6.0)
    v_gust = u_ds * (rho0/rho)**0.5 #in TAS
   
    # parameters for Nastran cards
    WG_TAS = v_gust/V #TAS/TAS
    
    return WG_TAS, u_ds, v_gust
