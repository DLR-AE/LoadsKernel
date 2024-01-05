# -*- coding: utf-8 -*-
import logging
import numpy as np
from loadskernel.atmosphere import isa as atmo_isa


def calc_drehmatrix_angular(phi=0.0, theta=0.0, psi=0.0):
    # Alle Winkel in [rad] !
    # geo to body
    drehmatrix = np.array(([1., 0., -np.sin(theta)],
                           [0., np.cos(phi), np.sin(phi) * np.cos(theta)],
                           [0., -np.sin(phi), np.cos(phi) * np.cos(theta)]))
    return drehmatrix


def calc_drehmatrix_angular_inv(phi=0.0, theta=0.0, psi=0.0):
    # Alle Winkel in [rad] !
    # body to geo
    drehmatrix = np.array(([1., np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                           [0., np.cos(phi), -np.sin(phi)],
                           [0., np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]))
    return drehmatrix


def calc_drehmatrix(phi=0.0, theta=0.0, psi=0.0):
    # Alle Winkel in [rad] !
    drehmatrix_phi = np.array(([1., 0., 0.], [0., np.cos(phi), np.sin(phi)], [0., -np.sin(phi), np.cos(phi)]))
    drehmatrix_theta = np.array(([np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]))
    drehematrix_psi = np.array(([np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]))
    drehmatrix = np.dot(np.dot(drehmatrix_phi, drehmatrix_theta), drehematrix_psi)
    return drehmatrix


def gravitation_on_earth(PHInorm_cg, Tgeo2body):
    g = np.array([0.0, 0.0, 9.8066])  # erdfest, geodetic
    g_cg = np.dot(PHInorm_cg[0:3, 0:3], np.dot(Tgeo2body[0:3, 0:3], g))  # bodyfixed
    return g_cg


def design_gust_cs_25_341(simcase, atmo, V):
    # Gust Calculation from CS 25.341 (a)
    # adapted from matlab-script by Vega Handojo, DLR-AE-LAE, 2015

    # convert (possible) integer to float
    gust_gradient = float(simcase['gust_gradient'])  # Gust gradient / half length
    altitude = float(atmo['h'])  # Altitude
    rho = float(atmo['rho'])  # Air density
    V = float(V)  # Speed
    V_D = float(atmo['a'] * simcase['gust_para']['MD'])  # Design Dive speed

    _, rho0, _, _ = atmo_isa(0.0)

    # Check if flight alleviation factor fg is provided by user as input, else calculate fg according to CS 25.341(a)(6)
    if 'Fg' in simcase['gust_para']:
        fg = float(simcase['gust_para']['Fg'])
    else:
        Z_mo = float(simcase['gust_para']['Z_mo'])  # Maximum operating altitude
        MLW = float(simcase['gust_para']['MLW'])  # Maximum Landing Weight
        MTOW = float(simcase['gust_para']['MTOW'])  # Maximum Take-Off Weight
        MZFW = float(simcase['gust_para']['MZFW'])  # Maximum Zero Fuel Weight
        fg = calc_fg(altitude, Z_mo, MLW, MTOW, MZFW)
    logging.info(
        'CS25_Uds is set up with flight profile alleviation factor Fg = {}'.format(fg))

    # reference gust velocity (EAS) [m/s]
    if altitude <= 4572:
        u_ref = 17.07 - (17.07 - 13.41) * altitude / 4572.0
    else:
        u_ref = 13.41 - (13.41 - 6.36) * ((altitude - 4572.0) / (18288.0 - 4572.0))
    if V == V_D:
        u_ref = u_ref / 2.0

    # design gust velocity (EAS)
    u_ds = u_ref * fg * (gust_gradient / 107.0) ** (1.0 / 6.0)
    v_gust = u_ds * (rho0 / rho) ** 0.5  # in TAS

    # parameters for Nastran cards
    WG_TAS = v_gust / V  # TAS/TAS

    return WG_TAS, u_ds, v_gust


def turbulence_cs_25_341(simcase, atmo, V):
    # Turbulence Calculation from CS 25.341 (b)

    # convert (possible) integers to floats
    altitude = float(atmo['h'])  # Altitude
    Z_mo = float(simcase['gust_para']['Z_mo'])  # Maximum operating altitude
    V = float(V)  # Speed
    V_C = float(atmo['a'] * simcase['gust_para']['MC'])  # Design Cruise speed
    V_D = float(atmo['a'] * simcase['gust_para']['MD'])  # Design Dive speed
    MLW = float(simcase['gust_para']['MLW'])  # Maximum Landing Weight
    MTOW = float(simcase['gust_para']['MTOW'])  # Maximum Take-Off Weight
    MZFW = float(simcase['gust_para']['MZFW'])  # Maximum Zero Fuel Weight

    fg = calc_fg(altitude, Z_mo, MLW, MTOW, MZFW)

    # reference turbulence intensity (TAS) [m/s]
    # CS 25.341 (b)(3): U_sigma_ref is the reference turbulence intensity that...
    if altitude <= 7315.0:
        # ...varies linearly with altitude from 27.43m/s (90 ft/s) (TAS) at sea level
        # to 24.08 m/s (79 ft/s) (TAS) at 7315 m (24000 ft)
        u_ref = 27.43 - (27.43 - 24.08) * altitude / 7315.0
    else:
        # ...and is then constant at 24.08 m/s (79 ft/s) (TAS) up to the altitude of 18288 m (60000 ft)
        u_ref = 24.08

    # limit turbulence intensity (TAS) [m/s]
    u_sigma = u_ref * fg

    # At speed VD: U_sigma is equal to 1/2 the values and
    # At speeds between VC and VD: U_sigma is equal to a value obtained by linear interpolation.
    if V > V_C:
        u_sigma = u_sigma * (1.0 - 0.5 * (V - V_C) / (V_D - V_C))

    return u_sigma


def calc_fg(altitude, Z_mo, MLW, MTOW, MZFW):
    # calculate the flight profile alleviation factor as given in CS 25.341 (a)(6)

    R1 = MLW / MTOW
    R2 = MZFW / MTOW
    f_gm = (R2 * np.tan(np.pi * R1 / 4)) ** 0.5
    f_gz = 1.0 - Z_mo / 76200.0
    # At sea level, the flight profile alleviation factor is determined by
    fg_sl = 0.5 * (f_gz + f_gm)
    # The flight profile alleviation factor, Fg, must be increased linearly from the sea level value
    # to a value of 1.0 at the maximum operating altitude
    if altitude == 0.0:
        fg = fg_sl
    elif altitude == Z_mo:
        fg = 1.0
    else:
        fg = fg_sl + (1.0 - fg_sl) * altitude / Z_mo
    return fg
