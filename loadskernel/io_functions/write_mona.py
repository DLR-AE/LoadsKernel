import logging

import numpy as np


def number_nastran_converter(number):
    if np.abs(number) < 1e-8:
        number = 0.0  # set tiny number to zero
    if number.is_integer():
        number_str = '{:> 7.1f}'.format(number)
    elif 0.0 <= np.log10(number.__abs__()) < 4.0:
        # Here, '{:> 7.6g}' would be nicer, however, trailing zeros '.0' are removed, which leads to an integer, which
        # Nastran doesn't like.
        number_str = '{:> 7.6}'.format(number)
    elif -3.0 <= np.log10(number.__abs__()) < 0.0:
        number_str = '{:> 7.5f}'.format(number)
    else:
        number_str = '{:> 7.3e}'.format(number)
    # try normal formatting
    if len(number_str) <= 8:
        return number_str
    # try to remove 2 characters, works with large numbers
    elif len(number_str.replace('e+0', 'e')) <= 8:
        return number_str.replace('e+0', 'e')
    # try smaller precicion and remove 1 character, works with small numbers
    elif len('{:> 7.2e}'.format(number).replace('e-0', 'e-')) <= 8:
        return '{:> 7.2e}'.format(number).replace('e-0', 'e-')
    elif len('{:> 7.1e}'.format(number)) <= 8:
        return '{:> 7.1e}'.format(number)
    else:
        logging.error('Could not convert number to nastran format: {}'.format(str(number)))


def write_SET1(fid, SID, entrys):
    entries_str = ''
    for entry in entrys:
        entries_str += '{:>8d}'.format(int(entry))
    if len(entries_str) <= 56:
        line = 'SET1    {:>8d}{:s}\n'.format(SID, entries_str)
        fid.write(line)
    else:
        line = 'SET1    {:>8d}{:s}+\n'.format(SID, entries_str[:56])
        entries_str = entries_str[56:]
        fid.write(line)

        while len(entries_str) > 64:
            line = '+       {:s}+\n'.format(entries_str[:64])
            entries_str = entries_str[64:]
            fid.write(line)
        line = '+       {:s}\n'.format(entries_str)
        fid.write(line)


def write_force_and_moment_cards(fid, grid, Pg, SID):
    # FORCE and MOMENT cards with all values equal to zero are ommitted to avoid problems when importing to Nastran.
    for i in range(grid['n']):
        if np.any(np.abs(Pg[grid['set'][i, 0:3]]) >= 1e-8):
            line = 'FORCE   ' + '{:>8d}{:>8d}{:>8d}{:>8.7s}{:>8s}{:>8s}{:>8s}\n'.format(
                SID, int(grid['ID'][i]), int(grid['CD'][i]), str(1.0), number_nastran_converter(Pg[grid['set'][i, 0]]),
                number_nastran_converter(Pg[grid['set'][i, 1]]), number_nastran_converter(Pg[grid['set'][i, 2]]))
            fid.write(line)
        if np.any(np.abs(Pg[grid['set'][i, 3:6]]) >= 1e-8):
            line = 'MOMENT  ' + '{:>8d}{:>8d}{:>8d}{:>8.7s}{:>8s}{:>8s}{:>8s}\n'.format(
                SID, int(grid['ID'][i]), int(grid['CD'][i]), str(1.0), number_nastran_converter(Pg[grid['set'][i, 3]]),
                number_nastran_converter(Pg[grid['set'][i, 4]]), number_nastran_converter(Pg[grid['set'][i, 5]]))
            fid.write(line)


def write_subcases(fid, subcase, desc):
    line = 'SUBCASE {}\n'.format(int(subcase))
    fid.write(line)
    line = '    SUBT={}\n'.format(str(desc))
    fid.write(line)
    line = '    LOAD={}\n'.format(int(subcase))
    fid.write(line)
