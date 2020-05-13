#!/usr/bin/env pythoeta2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:56:30 2017

@author: voss_ar
"""
import copy, time
import numpy as np
import logging
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
    # Calculates one unsteady AIC matrix (Qjj = -Ajj^-1) at given Mach number and frequency 
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
    #
    # M = Mach number
    # k = omega/U, the "classical" definition, not Nastran definition!
    # Nomencalture with receiving (r), minus (-e), plus (e), sending (s/0) point and semiwidth e following Rodden 1968 
    Pr = aerogrid['offset_j']   # receiving (r)
    Pm = aerogrid['offset_P1']  # minus (-e)
    Pp = aerogrid['offset_P3']  # plus (e)
    Ps = aerogrid['offset_l']   # sending (s/0)
    A = aerogrid['A']
    e = np.absolute(np.repeat(np.array(0.5 * aerogrid['A'] / aerogrid['l'], ndmin=2),aerogrid['n'],axis=0)) # semiwidth
    e2 = e**2.0
    chord = np.repeat(np.array(aerogrid['l'], ndmin=2), aerogrid['n'], axis=0)
    
    # cartesian coordinates of receiving points relative to sending points
    xsr = np.array(Pr[:,0], ndmin=2).T - np.array(Ps[:,0], ndmin=2)
    ysr = np.array(Pr[:,1], ndmin=2).T - np.array(Ps[:,1], ndmin=2)
    zsr = np.array(Pr[:,2], ndmin=2).T - np.array(Ps[:,2], ndmin=2)
    
    # dihedral angle gamma = arctan(dz/dy) and sweep angle lambda = arctan(dx/dy)
    sinGamma  = (Pp[:,2]-Pm[:,2])/(2.0*e)
    cosGamma  = (Pp[:,1]-Pm[:,1])/(2.0*e)
    tanLambda = (Pp[:,0]-Pm[:,0])/(2.0*e)
    gamma = np.arcsin(sinGamma)
    # relative dihedral angle between receiving point and sending boxes
    gamma_sr = np.array(gamma, ndmin=2) - np.array(gamma, ndmin=2).T
    
    # local coordinates of receiving point relative to sending point
    ybar  = ysr*cosGamma + zsr*sinGamma
    zbar  = zsr*cosGamma - ysr*sinGamma
    
    # call the kernel function
    P1m, P2m = kernelfunction(xsr,ybar,zbar,gamma_sr,tanLambda,-e,k,Ma)
    P1p, P2p = kernelfunction(xsr,ybar,zbar,gamma_sr,tanLambda,+e,k,Ma)
    P1s, P2s = kernelfunction(xsr,ybar,zbar,gamma_sr,tanLambda, 0,k,Ma)
    
    # define terms used in the parabolic approximation
    A1 = (P1m-2.0*P1s+P1p)/(2.0*e2)     # Rodden 1971, eq 28
    B1 = (P1p-P1m)/(2.0*e)              # Rodden 1971, eq 29
    C1 = P1s                            # Rodden 1971, eq 30
    
    A2 = (P2m-2.0*P2s+P2p)/(2.0*e2)     # Rodden 1971, eq 37
    B2 = (P2p-P2m)/(2.0*e)              # Rodden 1971, eq 38
    C2 = P2s                            # Rodden 1971, eq 39


    # pre-calculate some values which will be used a couple of times
    ybar2 = ybar**2.0
    zbar2 = zbar**2.0
    ratio = 2.0*e*zbar/(ybar2 + zbar2 - e2)

    # The "planar" part
    # -----------------
    # Initial values
    F = np.zeros(e.shape)
    # Condition 1, planar
    i0 = zbar==0.0
    F[i0] = (2.0*e[i0])/(ybar2[i0] - e2[i0])
    # Condition 2, co-planar / close-by
    ia = (ratio.__abs__() <= 0.3) & (zbar!=0.0) 
    funny_series = 0.0
    for n in range(2,8): 
        funny_series += (-1.0)**n/(2.0*n-1.0) * ratio[ia]**(2.0*n-4.0)
    alpha = 4.0*e[ia]**4.0/(ybar2[ia] + zbar2[ia]-e2[ia])**2.0 * funny_series                            # Rodden 1971, eq 33 and Rodden 1972, eq 31b
    
    F[ia] = 2.0*e[ia]/(ybar2[ia] + zbar2[ia] - e2[ia])*(1.0-alpha*zbar2[ia]/e2[ia])                      # Rodden 1971, eq 32        
    # Condition 3, the rest / further away
    ir = (ratio.__abs__() > 0.3) & (zbar!=0.0) 
    F[ir] = 1.0/np.abs(zbar[ir])*np.arctan2(2.0*e[ir]*np.abs(zbar[ir]),(ybar2[ir] + zbar2[ir] - e2[ir])) # Rodden 1971, eq 31b
    # check: np.all(i0 + ia + ir) == True
        
    #  normalwash matrix, Rodden 1971, eq 34
    I34 = ((ybar2 - zbar2)*A1 + ybar*B1 + C1) * F \
         + (0.5*B1 + ybar*A1) * np.log( ((ybar-e)**2.0+zbar2)/((ybar+e)**2.0+zbar2) ) \
         + 2.0*e*A1
    D1 = chord/(np.pi*8.0)*I34
    
    # The "nonplanar" part
    # --------------------
    D2 = np.zeros(e.shape, dtype='complex')
    # Condition 1, similar to above but with different boundary, Rodden 1971 eq 40
    # 1/ratio <= 0.1 is equivalent to ratio > 10.0
    ib = (np.abs(1.0/ratio) <= 0.1) #& (zbar!=0.0) 
    I40 = ((ybar2[ib] + zbar2[ib])*A2[ib] + ybar[ib]*B2[ib] + C2[ib])*F[ib] \
        + 1.0/((ybar[ib]+e[ib])**2.0+zbar2[ib]) * ( ((ybar2[ib] + zbar2[ib])*ybar[ib]+(ybar2[ib]-zbar2[ib])*e[ib])*A2[ib] + (ybar2[ib] + zbar2[ib]+ybar[ib]*e[ib])*B2[ib] + (ybar[ib]+e[ib])*C2[ib] ) \
        - 1.0/((ybar[ib]-e[ib])**2.0+zbar2[ib]) * ( ((ybar2[ib] + zbar2[ib])*ybar[ib]+(ybar2[ib]-zbar2[ib])*e[ib])*A2[ib] + (ybar2[ib] + zbar2[ib]-ybar[ib]*e[ib])*B2[ib] + (ybar[ib]-e[ib])*C2[ib] )
    
    D2[ib] = chord[ib]/(16.0*np.pi*zbar2[ib])*I40
    
    # Condition 2, Rodden 1971 eq 41
    ic = (np.abs(1.0/ratio) > 0.1) & (zbar!=0.0) 
    # reconstruct alpha from eq 32, NOT eq 33!
    alpha41 = (1.0 - F[ic] * (ybar2[ic] + zbar2[ic]-e2[ic])/(2.0*e[ic]))/zbar2[ic]*e2[ic]
    I41 = ( 2.0*(ybar2[ic] + zbar2[ic]+e2[ic])*(e2[ic]*A2[ic]+C2[ic])+4.0*ybar[ic]*e2[ic]*B2[ic] ) \
        / ( ((ybar[ic]+e[ic])**2.0+zbar2[ic])*((ybar[ic]-e[ic])**2.0+zbar2[ic]) ) \
        - alpha41/e2[ic] * ( (ybar2[ic] + zbar2[ic])*A2[ic] + ybar[ic]*B2[ic] + C2[ic] )

    D2[ic] = chord[ic]*e[ic]/(8.0*np.pi*(ybar2[ic] + zbar2[ic]-e2[ic]))*I41
    
    # add planar and non-planar parts, # Rodden eq 22
    # the steady part D0 has already been subtracted inside the kernel function
    D = D1 + D2 
    return D


def kernelfunction(xbar,ybar,zbar,gamma_sr,tanLambda,ebar,k,M):
    # This is the function that calculates "the" kernel function(s) of the DLM.
    # K1,2 are reformulated in Rodden 1971 compared to Rodden 1968 and include new 
    # conditions, e.g. for co-planar panels.
    # Note that the signs of K1,2 are switched in Rodden 1971, in this implementation 
    # we stay with the 1968 convention and we don't want to mess with the steady part 
    # from the VLM. Also, we directly subtract the steady parts K10 and K20, as the 
    # steady contribution will be added later from the VLM.
    # Note: Rodden has the habit of leaving out some brackets in his formulas. This   
    # applies to eq 11, 7 and 8 where it is not clear which parts belong to the denominator.
    
    r1 = ((ybar-ebar)**2.0 + zbar**2.0)**0.5                    # Rodden 1971, eq 4
    beta2 = (1.0-(M**2.0))                                      # Rodden 1971, eq 9
    R = ((xbar-ebar*tanLambda)**2.0 + beta2*r1**2.0)**0.5       # Rodden 1971, eq 10
    u1 = (M*R - xbar + ebar*tanLambda) / (beta2*r1)             # Rodden 1971, eq 11
    k1 = k*r1                                                   # Rodden 1971, eq 12 with k = w/U
    j = 1j                                                      # imaginary number
    ejku = np.exp(-j*k1*u1)                                     # pre-multiplication
    
    # direction cosine matrices
    T1 = np.cos(gamma_sr)                                               # Rodden 1971, eq 5
    T2 = zbar*(zbar*np.cos(gamma_sr) + (ybar-ebar)*np.sin(gamma_sr))    # Rodden 1971, eq 21a: T2_new = T2_old*r1^2
    
    # Approximation of intergrals I1,2, Rodden 1971, eq 13+14    
    I1, I2 = get_integrals12(u1, k1)

    # Formulation of K1,2 by Landahl, Rodden 1971, eq 7+8
    K1 = -I1 - ejku*M*r1/R/(1+u1**2.0)**0.5
    K2 = 3.0*I2 - j*k1*ejku*(M**2.0)*(r1**2.0)/(R**2.0)/(1.0+u1**2.0)**0.5 \
            + ejku*M*r1 * ((1.0+u1**2.0)*beta2*r1**2.0 / R**2.0 + 2.0 + M*r1*u1/R)/R/(1.0+u1**2.0)**1.5

    # This is the analytical solution for K1,2 at k=0.0, Rodden 1971, eq 15+16
    K10 = -1.0-(xbar-ebar*tanLambda)/R 
    K20 =  2.0+(xbar-ebar*tanLambda)*(2.0+beta2*r1**2.0/R**2.0)/R 
    
    # Resolve the singularity arising when r1 = 0
    ir0xpos = (r1==0) & (xbar>=0.0)
    ir0xneg = (r1==0) & (xbar<0.0)        
    K1[ir0xpos]=-2.0; K2[ir0xpos]=+4.0
    K1[ir0xneg]=0.0; K2[ir0xneg]=0.0
         
    P1 = -(K1*np.exp(-j*k*(xbar-ebar*tanLambda)) - K10)*T1     # Rodden 1971, eq 27b, check: -K1*np.exp(-j*k*xbar)*T1
    P2 = -(K2*np.exp(-j*k*(xbar-ebar*tanLambda)) - K20)*T2     # Rodden 1971, eq 36b, check: -K2*np.exp(-j*k*xbar)*T2/r1**2.0
    
    return P1, P2

def get_integrals12(u1, k1, method='Laschka'):
    
    I1 = np.zeros(u1.shape, dtype='complex')
    I2 = np.zeros(u1.shape, dtype='complex')
    
    ipos = u1 >= 0.0
    I1[ipos], I2[ipos] = integral_approximations(u1[ipos], k1[ipos], method)

    ineg = u1 < 0.0
    I10, I20 = integral_approximations(0.0*u1[ineg], k1[ineg], method)
    I1n, I2n = integral_approximations(   -u1[ineg], k1[ineg], method)
    I1[ineg] = 2.0*I10.real - I1n.real + 1j*I1n.imag    # Rodden 1971, eq A.5
    I2[ineg] = 2.0*I20.real - I2n.real + 1j*I2n.imag    # Rodden 1971, eq A.9
    return I1, I2

def integral_approximations(u1, k1, method='Laschka'):
    if method == 'Laschka':
        logging.debug('Using Laschka approximation in DLM')
        I1, I2 = laschka_approximation(u1, k1)
    elif method == 'Watkins':
        logging.warning('Using Watkins (not preferred!) approximation in DLM.')
        I1, I2 = watkins_approximation(u1, k1)
    else:
        logging.error('Method {} not implemented!'.format(method))
    return I1, I2
    

def laschka_approximation(u1, k1):
    # Approximate integral I0, Rodden 1971, eq A.4 
    # Approximate integral J0, Rodden 1971, eq A.8
    # These are the coefficients in exponential approximation of u/(1+u**2.0)**0.5
    # The values are difficult to read in Rodden 1971 but are also given in Blair 1992, page 89. 
    a11 = [+0.24186198, -2.7918027, +24.991079, -111.59196, +271.43549, -305.75288, -41.183630, +545.98537, -644.78155, +328.72755, -64.279511]
    c = 0.372
    j = 1j
    ejku = np.exp(-j*k1*u1) # pre-multiplication
    I0 = 0.0
    J0 = 0.0
    for n,a in zip(range(1,12), a11): 
        nck = n**2.0 * c**2.0 + k1**2.0
        I0 += a*np.exp(-n*c*u1) / nck * ( n*c-j*k1 )
        J0 += a*np.exp(-n*c*u1) / nck**2.0 * ( n**2.0*c**2.0 - k1**2.0 + n*c*u1*nck - j*k1*(2.0*n*c + u1*nck) )
    # I1 as in Rodden 1971, eq A.1
    I1 = ( 1.0-u1/(1.0+u1**2.0)**0.5 - 1j*k1*I0 )*ejku 
    # I2 as in Rodden 1971, eq A.6,
    # but divided by 3.0 for compatibility, 
    # and with the square bracket at the correct location ;) 
    I2 = ( (2.0+j*k1*u1)*(1.0-u1/(1.0+u1**2.0)**0.5) - u1/(1.0+u1**2.0)**1.5 - j*k1*I0 + k1**2.0*J0 )*ejku/3.0
    return I1, I2

def watkins_approximation(u1, k1):
    # This is the old/original approximation of integrals I1,2 as in Rodden 1968, page 3.
    # The following code is take from the previous Matlab implementation of the DLM, has been rearranged and is used for comparison.
    a1 = 0.101
    a2 = 0.899
    a3 = 0.09480933
    b1 = 0.329
    b2 = 1.4067
    b3 = 2.90
    j = 1j
    ejku = np.exp(-j*k1*u1) # pre-multiplication
    # evaluate with numpy, slower
    i1 =        a1*np.exp((-b1-(j*k1))*u1)/(b1+(j*k1)) + \
                a2*np.exp((-b2-(j*k1))*u1)/(b2+(j*k1))
    i2 = (a3/(((b3 + (j*k1))**2.0) + (np.pi**2.0))) * (((b3+(j*k1))*np.sin(np.pi*u1)) + (np.pi*np.cos(np.pi*u1)))*np.exp((-b3-(j*k1))*u1)
    I1_temp = i1 + i2
    I1 = ((1-(u1/(1+u1**2.0)**0.5))*ejku) - (j*k1*I1_temp)
    
    I2_1 = I1
    # evaluate with numpy, slower 
    I2_2_1 =    a1*np.exp(-(b1+j*k1)*u1)/((b1+j*k1)**2.0) + \
                a2*np.exp(-(b2+j*k1)*u1)/((b2+j*k1)**2.0) + \
                ((a3*np.exp(-(b3+j*k1)*u1)/(((b3+j*k1)**2.0 + np.pi**2.0)**2.0))*(np.pi*((np.pi*np.sin(np.pi*u1)) - \
                ((b3+j*k1)*np.cos(np.pi*u1))) - ((b3+j*k1)*(np.pi*np.cos(np.pi*u1) + ((b3+j*k1)*np.sin(np.pi*u1)))))) 
    I2_2 =  (ejku*(u1**3.0)/((1+u1**2.0)**(3.0/2.0)) - I2_1 - \
            ejku*u1/((1+u1**2.0))**0.5)/3.0 - (k1*k1*I2_2_1/3.0)
    I2  = I2_1 + I2_2
    return I1, I2

