#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:56:30 2017

@author: voss_ar
"""
import copy
import numpy as np
np.seterr(all='ignore')                 # turn off warnings (divide by zero, multiply NaN, ...) as singularities are expected to occur
import numexpr as ne
n_cores = ne.detect_number_of_cores()   # get number of cores and use all
ne.set_num_threads(n_cores)             # set up numexpr for multithreading

import loadskernel.VLM as VLM


def calc_Qjj(aerogrid, Ma, k):
    # calc steady contributions using VLM
    Ajj_VLM, Bjj = VLM.calc_Ajj(aerogrid=copy.deepcopy(aerogrid), Ma=Ma)
    if k == 0.0:
        # no oscillatory / unsteady contributions at k=0.0
        Ajj_DLM = np.zeros((aerogrid['n'],aerogrid['n']))
    else:
        # calc oscillatory / unsteady contributions using DLM
        Ajj_DLM = calc_Ajj(aerogrid=copy.deepcopy(aerogrid), Ma=Ma, k=k)
    Ajj = Ajj_VLM + Ajj_DLM
    Qjj = -np.linalg.inv(Ajj)

def calc_Qjjs(aerogrid, Ma, k, xz_symmetry=False):
    # allocate memory
    Qjj = np.zeros((len(Ma),len(k),aerogrid['n'],aerogrid['n']), dtype='complex')# dim: Ma,k,n,n
    # Consideration of XZ symmetry like in VLM.
    if xz_symmetry:
        n = aerogrid['n']
        aerogrid = VLM.mirror_aerogrid_xz(aerogrid)
        
    # loop over mach number and freq.
    for im in range(len(Ma)):
        # calc steady contributions using VLM
        Ajj_VLM, Bjj = VLM.calc_Ajj(aerogrid=copy.deepcopy(aerogrid), Ma=Ma[im])
        for ik in range(len(k)):
            if k[ik] == 0.0:
                # no oscillatory / unsteady contributions at k=0.0
                Ajj_DLM = np.zeros((aerogrid['n'],aerogrid['n']))
            else:
                # calc oscillatory / unsteady contributions using DLM
                Ajj_DLM = calc_Ajj(aerogrid=copy.deepcopy(aerogrid), Ma=Ma[im], k=k[ik])
            Ajj = Ajj_VLM + Ajj_DLM
            Ajj_inv = -np.linalg.inv(Ajj)
            if xz_symmetry:
                Qjj[im,ik] = Ajj_inv[0:n,0:n]-Ajj_inv[n:2*n,0:n]
            else:
                Qjj[im,ik] = Ajj_inv
    return Qjj

def calc_Ajj(aerogrid, Ma, k):
    # P0 = downwash recieving location x-y pair (1/2span, 3/4chord)
    # P1 = root doublet location x-y pair (0 span, 1/4chord)
    # P2 = semi-span doublet location x-y pair (1/2span, 1/4chord)
    # P3 = tip doublet location x-y pair (1span, 1/4chord)
    # e = half span length of the aero panel
    # cav = centerline chord of the aero panel
    # k = k1 term from ref 1.: omega/U
    #   omega = frequency of oscillation
    #   U = freestream velocity
    # M = Mach number
    

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
    P2 = (P1+P3)/2.0
    A = aerogrid['A']
    chord = aerogrid['l']
    semispan = 0.5 * A / chord
    
    x01 = np.array(P0[:,0], ndmin=2).T - np.array(P1[:,0], ndmin=2)
    y01 = np.array(P0[:,1], ndmin=2).T - np.array(P1[:,1], ndmin=2)
    z01 = np.array(P0[:,2], ndmin=2).T - np.array(P1[:,2], ndmin=2)
    
    x02 = np.array(P0[:,0], ndmin=2).T - np.array(P2[:,0], ndmin=2)
    y02 = np.array(P0[:,1], ndmin=2).T - np.array(P2[:,1], ndmin=2)
    z02 = np.array(P0[:,2], ndmin=2).T - np.array(P2[:,2], ndmin=2)
    
    x03 = np.array(P0[:,0], ndmin=2).T - np.array(P3[:,0], ndmin=2)
    y03 = np.array(P0[:,1], ndmin=2).T - np.array(P3[:,1], ndmin=2)
    z03 = np.array(P0[:,2], ndmin=2).T - np.array(P3[:,2], ndmin=2)
    
    cosGamma = (P3[:,1] - P1[:,1]) / ( (P3[:,2]-P1[:,2])**2.0 + (P3[:,1]-P1[:,1])**2.0 )**0.5
    sinGamma = (P3[:,2] - P1[:,2]) / ( (P3[:,2]-P1[:,2])**2.0 + (P3[:,1]-P1[:,1])**2.0 )**0.5
                   
    # Kernel function (K) calculation
    # Kappa (defined on page 3, reference 1) is calculated. The steady part of
    # Kappa (i.e. reduced frequency = 0) is subtracted out and later
    # compensated for by adding downwash effects from a VLM code. This ensures 
    # that the doublet lattice code converges to VLM results under steady
    # conditions. (Ref 2, page 3, equation 9)

    Ki_w = getKappa(x01,y01,z01,cosGamma,sinGamma,k,Ma)
    Ki_0 = getKappa(x01,y01,z01,cosGamma,sinGamma,0,Ma)
    Ki = Ki_w - Ki_0
    
    Km_w = getKappa(x02,y02,z02,cosGamma,sinGamma,k,Ma)
    Km_0 = getKappa(x02,y02,z02,cosGamma,sinGamma,0,Ma)
    Km = Km_w - Km_0
    
    K0_w = getKappa(x03,y03,z03,cosGamma,sinGamma,k,Ma)
    K0_0 = getKappa(x03,y03,z03,cosGamma,sinGamma,0,Ma)
    K0 = K0_w - K0_0
    
    # Parabolic approximation of incremental Kernel function (ref 1, equation 7)
    # define terms used in the parabolic approximation
    e1 = np.absolute(np.repeat(np.array(semispan, ndmin=2),aerogrid['n'],axis=0))
    A = (Ki-2.0*Km+K0)/(2.0*e1**2.0)
    B = (K0-Ki)/(2.0*e1)
    C = Km

    # define r1,n0,zeta0
    cosGamma = np.repeat(np.array(cosGamma, ndmin=2),aerogrid['n'],axis=0)
    sinGamma = np.repeat(np.array(sinGamma, ndmin=2),aerogrid['n'],axis=0)
    n0 = (y02*cosGamma) + (z02*sinGamma)
    zeta0 = -(y02*sinGamma) + (z02*cosGamma)
    r2 = ((n0**2.0) + (zeta0**2.0))**0.5
    
    #  normalwash matrix factor I
    I = (A*(2.0*e1))+((0.5*B+n0*A)*np.log((r2**2.0 - 2.0*n0*e1 + e1**2.0)/(r2**2.0 + 2.0*n0*e1 + e1**2.0))) + \
        (((n0**2.0 - zeta0**2.0)*A+n0*B+C)/np.absolute(zeta0)*np.arctan(2.0*e1*np.absolute(zeta0)/(r2**2.0 - e1**2.0)))
    # limit when zeta -> 0
    ind = np.where(zeta0==0)
    I2 = ((A*(2.0*e1))+((0.5*B+n0*A)*np.log(((n0-e1)/(n0+e1))**2.0)) + \
         ((n0**2.0)*A+n0*B+C)*((2.0*e1)/(n0**2.0 - e1**2.0)))
    I[ind] = I2[ind]
    
    # normalwash matrix
    D = np.repeat(np.array(chord, ndmin=2), aerogrid['n'], axis=0)*I/(np.pi*8.0)
    return D


def getKappa(x0,y0,z0,cosGamma,sinGamma,k,M):
    # Function to calculate kappa
    # this function calculates kappa as defined on page 3 of reference 1. The
    # All the formulae are from page 1 of the reference.
    # kappa = (r1^2) * K where K is the incremental Kernel function
    # K = (K1T1 + K2T2)*exp(-jwx0/U)/(r1^2), where w is oscillation frequency
    # variables passed to the function:
    
    # x0 = x - chi (x is location of collocation pt (3/4th chord pt), chi is
    #               location of doublet)
    # y0 = y - eta
    # z0 = z - zeta
    # cosGamma: cosine of panel dihedral
    # sinGamma: sine of panel dihedral
    # k = w/U,  U is freestram velocity
    # M: Mach no.
    
    # Reference papers
    # ref 1: Albano and Rodden - A Doublet-Lattic Method for Calculating 
    #        Lift Distributions on Oscillating Surfaces in Subsonic Flows
    #
    # ref 2: Watkins, C. E., Hunyan, H. L., and Cunningham, H. J., "A Systematic 
    #        Kernel Function Procedure for Determining Aerodynamic Forces on Oscillating 
    #        or Steady Finite Wings at Subsonic Speeds," R-48, 1959, NASA.
    #
    # ref 3: Blair, Max. A Compilation of the mathematics leading to the doublet 
    #        lattice method. No. WL-TR-92-3028. WRIGHT LAB WRIGHT-PATTERSON AFB OH, 1992.
    
    #declare all variables as defined in reference 1, page 1
    #z0 = zeros(size(y0));
    r1 = ((y0**2.0) + (z0**2.0))**0.5
    beta2 = (1-(M**2.0))
    R = ((x0**2.0) + beta2*(r1**2.0))**0.5;
    u1 = ((M*R) - x0) / (beta2*r1)
    k1 = k*r1
    j = 1j
    
    cos1 = np.array(cosGamma, ndmin=2).T.repeat(len(cosGamma), axis=1)
    cos2 = np.array(cosGamma, ndmin=2).repeat(len(cosGamma), axis=0)
    sin1 = np.array(sinGamma, ndmin=2).T.repeat(len(sinGamma), axis=1)
    sin2 = np.array(sinGamma, ndmin=2).repeat(len(sinGamma), axis=0)
    
    T1 = ne.evaluate("cos1*cos2 + sin1*sin2")
    T2 = ne.evaluate("(z0*cos1 - y0*sin1)*(z0*cos2 - y0*sin2)/(r1**2.0)")
    
    I1 = getI1(u1,k1)
    I2 = getI2(u1,k1)
    
    # get kappa_temp
    # evaluate with numpy, slower
#     kappa_temp1 = I1 + ((M*r1)*np.exp(-j*(k1*u1))/(R*(1+(u1**2.0))**0.5))
#     kappa_temp2 =   -3.0*I2 - (j*k1*(M**2.0)*(r1**2.0)*np.exp(-j*k1*u1)/ \
#                     ((R**2.0)*(1+u1**2.0)**0.5)) - (M*r1*((1+u1**2.0)* \
#                     ((beta2*r1**2.0)/R**2.0) + 2.0 + (M*r1*u1/R)))* \
#                     np.exp(-j*k1*u1)/(((1+u1**2.0)**(3.0/2.0))*R)
    
    # evaluate with numexpr, faster
    kappa_temp1_expr = "I1 + ((M*r1)*exp(-j*(k1*u1))/(R*(1+(u1**2.0))**0.5))"
    kappa_temp2_expr = "-3.0*I2 - (j*k1*(M**2.0)*(r1**2.0)*exp(-j*k1*u1)/ \
                    ((R**2.0)*(1+u1**2.0)**0.5)) - (M*r1*((1+u1**2.0)* \
                    ((beta2*r1**2.0)/R**2.0) + 2.0 + (M*r1*u1/R)))* \
                    exp(-j*k1*u1)/(((1+u1**2.0)**(3.0/2.0))*R)"
    kappa_temp1 = ne.evaluate(kappa_temp1_expr)
    kappa_temp2 = ne.evaluate(kappa_temp2_expr)
                
    kappa_temp = kappa_temp1*T1 + kappa_temp2*T2
    
    # Resolve the singularity arising when r1 = 0, ref 2, page 7, Eq 18
    rInd = np.where(r1.ravel()==0)
    kappa_flat = kappa_temp.ravel()
    if np.any(rInd):
        #print 'removing singularities ...'
        kappa_flat[rInd[0][x0.ravel()[rInd]<0.0]]  = 0.0
        kappa_flat[rInd[0][x0.ravel()[rInd]>=0.0]] = 2.0
        kappa_temp = kappa_flat.reshape(kappa_temp.shape)
                        
    kappa = kappa_temp*np.exp(-j*k*x0)
    return kappa
    
def getI1(u1,k1):
    # Function to get I1
    # ref 1: Albano and Rodden - A Doublet-Lattic Method for Calculating 
    #        Lift Distributions on Oscillating Surfaces in Subsonic Flows
    #
    # ref 3: Blair, Max. A Compilation of the mathematics leading to the doublet 
    #        lattice method. No. WL-TR-92-3028. WRIGHT LAB WRIGHT-PATTERSON AFB OH, 1992.
    #
    # I1 described in eqn 3 of page1 of reference 1. Approximated as shown in
    # page 3 of the reference. 
    #
    
    I1          = np.zeros(np.shape(u1), dtype='complex')
    I1_0        = np.zeros(np.shape(u1), dtype='complex')
    I1_neg      = np.zeros(np.shape(u1), dtype='complex')
    u_temp1     = np.zeros(np.shape(u1), dtype='complex')
    u1_temp2    = np.zeros(np.shape(u1), dtype='complex')
    k1_temp1    = np.zeros(np.shape(u1), dtype='complex')
    k1_temp2    = np.zeros(np.shape(u1), dtype='complex')
    # evaluate I1 for u1>0
    ind1 = np.where(u1>=0)
    u_temp1[ind1] = u1[ind1]   # select elements in u1 > 0
    k1_temp1[ind1] = k1[ind1]
    I1_temp1 = getI1pos(u_temp1,k1_temp1)
    I1[ind1] = I1_temp1[ind1]
    j = 1j
    
    # evaluate I1 for u1<0
    # Method taken from ref 3, page 90, eq 275
    ind2 = np.where(u1<0)
    u1_temp2[ind2] = u1[ind2]
    k1_temp2[ind2] = k1[ind2]
    
    I1_0temp = getI1pos(np.zeros(np.shape(u1)),k1_temp2);
    I1_0[ind2] = I1_0temp[ind2]
    I1_negtemp = getI1pos(-u1_temp2,k1_temp2)
    I1_neg[ind2] = I1_negtemp[ind2]
    I1[ind2] = 2.0*np.real(I1_0[ind2]) - np.real(I1_neg[ind2]) + j*np.imag(I1_neg[ind2])
    return I1

def getI1pos(u1,k1):
    # Function to get I1 for positive u1 values
    # ref 1: Albano and Rodden - A Doublet-Lattic Method for Calculating 
    #        Lift Distributions on Oscillating Surfaces in Subsonic Flows
    
    # I1 described in eqn 3 of page1 of reference 1. Approximated as shown in
    # page 3 of the reference. This implementation is only valid for u1>0. For
    # u1<0, this function is still used to obtain I1 in an indirect manner as
    # described in getI1.m, in which this function is called.
    #
    a1 = 0.101
    a2 = 0.899
    a3 = 0.09480933
    b1 = 0.329
    b2 = 1.4067
    b3 = 2.90
    
    # solve for I1
    j = 1j
    
    # evaluate with numpy, slower
#     i1 = a1*np.exp((-b1-(j*k1))*u1)/(b1+(j*k1)) + a2*np.exp((-b2-(j*k1))*u1)/(b2+(j*k1))
#     i2 = (a3/(((b3 + (j*k1))**2.0) + (np.pi**2.0))) * (((b3+(j*k1))*np.sin(np.pi*u1)) + (np.pi*np.cos(np.pi*u1)))*np.exp((-b3-(j*k1))*u1)
#     I1_temp = i1 + i2
#     I1pos = ((1-(u1/(1+u1**2.0)**0.5))*np.exp(-j*k1*u1)) - (j*k1*I1_temp)
    
    # evaluate with numexpr, faster
    pi = np.pi
    i1_expr = "a1*exp((-b1-(j*k1))*u1)/(b1+(j*k1)) + a2*exp((-b2-(j*k1))*u1)/(b2+(j*k1))"
    i2_expr = "(a3/(((b3 + (j*k1))**2.0) + (pi**2.0))) * (((b3+(j*k1))*sin(pi*u1)) + (pi*cos(pi*u1)))*exp((-b3-(j*k1))*u1)"
    I1pos_expr = "((1-(u1/(1+u1**2.0)**0.5))*exp(-j*k1*u1)) - (j*k1*I1_temp)"
      
    i1 = ne.evaluate(i1_expr)
    i2 = ne.evaluate(i2_expr)
    I1_temp = i1 + i2
    I1pos = ne.evaluate(I1pos_expr)
    
    return I1pos

def getI2(u1,k1):
    # Function to get I2
    # ref 1: Albano and Rodden - A Doublet-Lattic Method for Calculating 
    #        Lift Distributions on Oscillating Surfaces in Subsonic Flows
    #
    #ref 3: Blair, Max. A Compilation of the mathematics leading to the doublet 
    #       lattice method. No. WL-TR-92-3028. WRIGHT LAB WRIGHT-PATTERSON AFB OH, 1992.
    #
    # I2 described in eqn 3 of page1 of reference 1. Approximated as mentioned on
    # page 3 of the reference. 
    #
    
    I2          = np.zeros(np.shape(u1), dtype='complex')
    I2_0        = np.zeros(np.shape(u1), dtype='complex')
    I2_neg      = np.zeros(np.shape(u1), dtype='complex')
    u_temp1     = np.zeros(np.shape(u1), dtype='complex')
    u1_temp2    = np.zeros(np.shape(u1), dtype='complex')
    k1_temp1    = np.zeros(np.shape(u1), dtype='complex')
    k1_temp2    = np.zeros(np.shape(u1), dtype='complex')
    # calculate I2
    ind1 = np.where(u1>=0)
    u_temp1[ind1] = u1[ind1]   # select elements in u1 > 0
    k1_temp1[ind1] = k1[ind1]
    I2_temp1 = getI2pos(u_temp1,k1_temp1)
    I2[ind1] = I2_temp1[ind1]
    j = 1j
    # Calulate integral I2(ref 1, page 1, eq 3) for u1<0
    # Method taken from ref 3, page 90, eq 275
    ind2 = np.where(u1<0)
    u1_temp2[ind2] = u1[ind2]
    k1_temp2[ind2] = k1[ind2]
    
    I2_0temp = getI2pos(np.zeros(np.shape(u1)),k1_temp2);
    I2_0[ind2] = I2_0temp[ind2]
    I2_negtemp = getI2pos(-u1_temp2,k1_temp2)
    I2_neg[ind2] = I2_negtemp[ind2]
    I2[ind2] = 2.0*np.real(I2_0[ind2]) - np.real(I2_neg[ind2]) + j*np.imag(I2_neg[ind2])
    return I2

def getI2pos(u1,k1):
    # this function gets I2 integral for non-planar body solutions, for u1>0
    # I2 = I2_1 + I2_2
    # Expressions for I2_1 & I2_2 have been derived using the same
    # approximations as those for I1 
    #
    a1 = 0.101
    a2 = 0.899
    a3 = 0.09480933
    b1 = 0.329
    b2 = 1.4067
    b3 = 2.90
    j = 1j
    pi = np.pi
#     eiku = np.exp(-j*k1*u1)
    eiku = ne.evaluate("exp(-j*k1*u1)")
    
    I2_1 = getI1pos(u1,k1)   
    # evaluate with numpy, slower 
#     I2_2_1 =    a1*np.exp(-(b1+j*k1)*u1)/((b1+j*k1)**2.0) + \
#                 a2*np.exp(-(b2+j*k1)*u1)/((b2+j*k1)**2.0) + \
#                 ((a3*np.exp(-(b3+j*k1)*u1)/(((b3+j*k1)**2.0 + np.pi**2.0)**2.0))*(np.pi*((np.pi*np.sin(np.pi*u1)) - \
#                 ((b3+j*k1)*np.cos(np.pi*u1))) - ((b3+j*k1)*(np.pi*np.cos(np.pi*u1) + ((b3+j*k1)*np.sin(np.pi*u1)))))) 
#     I2_2 =  (eiku*(u1**3.0)/((1+u1**2.0)**(3.0/2.0)) - I2_1 - \
#             eiku*u1/((1+u1**2.0))**0.5)/3.0 - (k1*k1*I2_2_1/3.0)

    # evaluate with numexpr, faster
    I2_2_1_expr = "a1*exp(-(b1+j*k1)*u1)/((b1+j*k1)**2.0) + \
                a2*exp(-(b2+j*k1)*u1)/((b2+j*k1)**2.0) + \
                ((a3*exp(-(b3+j*k1)*u1)/(((b3+j*k1)**2.0 + pi**2.0)**2.0))*(pi*((pi*sin(pi*u1)) - \
                ((b3+j*k1)*cos(pi*u1))) - ((b3+j*k1)*(pi*cos(pi*u1) + ((b3+j*k1)*sin(pi*u1))))))"
    I2_2_expr = "(eiku*(u1**3.0)/((1+u1**2.0)**(3.0/2.0)) - I2_1 - \
            eiku*u1/((1+u1**2.0))**0.5)/3.0 - (k1*k1*I2_2_1/3.0)"
     
    I2_2_1 = ne.evaluate(I2_2_1_expr)  
    I2_2   = ne.evaluate(I2_2_expr)
       
    I2pos  = I2_1 + I2_2
    return I2pos
    
