# -*- coding: utf-8 -*-
import math
import logging


def isa(h):
    # Calculation of various atmospheric parameters according to the International Standard Atmosphere (ISA)
    # Functions for p and T are derived from the US Standard Atmosphere 1976.
    # Reference:        ISA, ISO 2533, 1975
    #                   US Standard Atmosphere 1976

    # check input
    if h < -5000 or h > 47000:
        logging.error('Altitude h = ' + str(h) + ' m, must be -5000 <= h <= 47000 m.')

    g0 = 9.80665  # acceleration of gravity (sea level)
    # Die US Standard Atmosphere 1976 arbeitet eigentlich mit der universellen
    # Gaskonstante und der molaren Masse.
    # R = R* / M_Luft
    R = 287.0531  # spec. gas constant  [J/kg/K]
    gamma = 1.4  # ratio of specific heats

    # properties of: troposphere, tropopause and stratosphere
    href = [0.0, 11000.0, 20000.0, 32000.0]  # reference altitude   [m]
    Tref = [288.15, 216.65, 216.65, 228.65]  # temperature at href  [K]
    # pressure at href     [Pa]
    pref = [101325.0, 22632.04, 5474.878, 868.0158]
    lambda_ref = [-0.0065, 0.0, 0.001, 0.0028]  # temperature gradient [K/m]

    # find corresponding atmospheric layer for altitude h
    hbounds = [-5001.0, 11000.0, 20000.0, 32000.0, 47000.0]
    for i, hbound in enumerate(hbounds):
        if hbound >= h:
            layer = i - 1
            break

    # formulas apply up to a height of 86km
    # differences in intergartion in case of lambda = 0 lead to a distinction between two formulas for p
    if lambda_ref[layer] == 0:
        p = pref[layer] * math.exp(-g0 * (h - href[layer]) / R / Tref[layer])
    else:
        p = pref[layer] * (Tref[layer] / (Tref[layer] + lambda_ref[layer] * (h - href[layer]))) ** (g0 / R / lambda_ref[layer])

    T = Tref[layer] + lambda_ref[layer] * (h - href[layer])
    rho = p / R / T
    a = (gamma * R * T) ** 0.5

    return p, rho, T, a
