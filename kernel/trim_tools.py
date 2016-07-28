# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 17:44:58 2014

@author: voss_ar
"""
import numpy as np
from  atmo_isa import atmo_isa

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
    
def DesignGust_CS_25_341(gust_gradient, Alt, rho, V, Z_mo, V_D, MLW, MTOW, MZFW):
    # Gust Calculation from CS 25.341
    # adapted from matlab-script by Vega Handojo, DLR-AE-LAE, 2015
    
    # convert (possible) integer to float
    gust_gradient = np.float(gust_gradient)
    Alt = np.float(Alt)
    rho = np.float(rho)
    V = np.float(V)
    Z_mo = np.float(Z_mo)
    V_D = np.float(V_D)
    MLW = np.float(MLW)
    MTOW = np.float(MTOW)
    MZFW = np.float(MZFW)

    p0, rho0, T0, a0 = atmo_isa(0.0)
    #rho0 = 1.225    
    R1 = MLW/MTOW;
    R2 = MZFW/MTOW;
    F_gm = (R2*np.tan(np.pi*R1/4))**0.5;
    F_gz = 1-Z_mo/76200.0;
    
    # flight profile alleviation factor
    Fg_SL = 0.5*(F_gz+F_gm); # at sea level
    if Alt == 0:
        Fg = Fg_SL;
    elif Alt == Z_mo:    # at maximum flight-level F_g = 1
        Fg = 1.0;
    else:                 # between SL and Z_mo increases linearily to 1
        Fg = Fg_SL + (1-Fg_SL)*Alt/Z_mo;  
        
    # reference gust velocity (EAS) [m/s]
    if Alt <= 4572:
      U_ref = 17.07-(17.07-13.41)*Alt/4572.0;
    else:
      U_ref = 13.41-(13.41-6.36)*((Alt-4572.0)/(18288.0-4572.0));
    if V == V_D:
      U_ref = U_ref/2.0;

    # design gust velocity (EAS)
    U_ds = U_ref * Fg * (gust_gradient/107.0)**(1.0/6.0);
    V_gust = U_ds * (rho0/rho)**0.5; #in TAS
   
    # parameters for Nastran cards
    WG_TAS = V_gust/V; #TAS/TAS
    #WG     = WG_TAS;   #EAS/EAS
    #T2 = T1+2*gust_gradient/V;
    #F2 = V/2./gust_gradient;
    
    return WG_TAS, U_ds, V_gust