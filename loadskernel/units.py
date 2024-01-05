# -*- coding: utf-8 -*-
import numpy as np
from loadskernel.atmosphere import isa as atmo_isa


def ft2m(length_ft):
    # exactly
    return np.array(length_ft) * 0.3048


def m2ft(length_m):
    # exactly
    return np.array(length_m) / 0.3048


def kn2ms(speed_kn):
    # reference: SI Brochure: The International System of Units (SI) [8th edition, 2006; updated in 2014]
    return np.array(speed_kn) * 1852. / 3600.


def ms2kn(speed_ms):
    return np.array(speed_ms) / 1852. * 3600.


def eas2tas(eas, h):
    _, rho0, _, _ = atmo_isa(0)
    _, rho, _, _ = atmo_isa(h)
    return eas * (rho0 / rho) ** 0.5


def tas2eas(tas, h):
    _, rho0, _, _ = atmo_isa(0)
    _, rho, _, _ = atmo_isa(h)
    return tas / (rho0 / rho) ** 0.5


def tas2cas(tas, h):
    # Like cas2tas but with different procedure to calculate qc.
    # For double-checking, see interactive chart at
    # http://walter.bislins.ch/blog/index.asp?page=Compressibility+Correction+Chart%2C+Verwendung+und+Berechnung
    # In addition, the round-trip Vtas --> Vcas --> Vtas must result in the same Vtas.
    p, rho, _, a = atmo_isa(h)
    p0, rho0, _, _ = atmo_isa(0)
    gamma = 1.4
    qc = p * ((1 + (gamma - 1) / 2 * (tas / a) ** 2) ** (gamma / (gamma - 1)) - 1)
    f = (gamma / (gamma - 1) * p / qc * ((qc / p + 1) ** ((gamma - 1) / gamma) - 1)) ** 0.5
    f0 = (gamma / (gamma - 1) * p0 / qc * ((qc / p0 + 1) ** ((gamma - 1) / gamma) - 1)) ** 0.5
    cas = tas * f0 / f / (rho0 / rho) ** 0.5
    return cas


def cas2tas(cas, h):
    # Reference: NASA RP 1046,Measurement of Aircraft Speed and Altitude, William Gracey, 1980
    p, rho, _, _ = atmo_isa(h)
    p0, rho0, _, _ = atmo_isa(0)
    gamma = 1.4
    qc = p0 * ((1 + (gamma - 1) / (2 * gamma) * rho0 / p0 * cas ** 2) ** (gamma / (gamma - 1)) - 1)
    f = (gamma / (gamma - 1) * p / qc * ((qc / p + 1) ** ((gamma - 1) / gamma) - 1)) ** 0.5
    f0 = (gamma / (gamma - 1) * p0 / qc * ((qc / p0 + 1) ** ((gamma - 1) / gamma) - 1)) ** 0.5
    tas = cas * f / f0 * (rho0 / rho) ** 0.5
    return tas


def cas2Ma(cas, h):
    tas = cas2tas(cas, h)
    _, _, _, a = atmo_isa(h)
    return tas / a


def tas2Ma(tas, h):
    _, _, _, a = atmo_isa(h)
    return tas / a


def eas2Ma(eas, h):
    tas = eas2tas(eas, h)
    return tas2Ma(tas, h)
