import logging
import math
import os

import numpy as np
import scipy.sparse as sp

from loadskernel.io_functions.read_mona import nastran_number_converter


def check_bigmat(n_row):
    if n_row > 65535:
        bigmat = True
        # For BIGMAT matrices, the length L and row position i_row are given directly in record 3.
        # L, IROW
    else:
        bigmat = False
        # Non-BIGMAT matrices have a string header IS, located in record 3, from which the length L and row position i_row
        # are derived by
        # L = INT(IS/65536) - 1
        # IROW = IS - 65536(L + 1)
    return bigmat


def read_op4_header(fid):
    read_string = fid.readline()
    """
    Note: some Nastran versions put a minus sign in front of the matrix dimension.
    So far, I haven't found an explanation or its meaning in any documentation.
    Until we have further information on this, I assume this to be a bug of Nastran, which is ignored but using abs().
    """
    n_col = abs(nastran_number_converter(read_string[0:8], 'int'))
    n_row = abs(nastran_number_converter(read_string[8:16], 'int'))
    # Assumptions:
    # - real values: 16 characters for one value --> five values per line
    # - complex values: 16 characters for real and complex part each --> five values in two line
    # According to the manual, NTYPE 1 and 3 are used for single precision while 2 and 4 are used for double precision.
    # However, no change in string length has been observed.
    # In addition, the format '1P,5E16.9' has 9 digits after the decimal separator, which is double precision, disregarding
    # the information given by NTYPE.
    if nastran_number_converter(read_string[24:32], 'int') in [1, 2]:
        type_real = True
    elif nastran_number_converter(read_string[24:32], 'int') in [3, 4]:
        type_real = False
    else:
        logging.error('Unknown format: ' + read_string[24:32])

    if nastran_number_converter(read_string[24:32], 'int') in [1, 3]:
        type_double = False
    else:
        type_double = True
    return n_col, n_row, type_real, type_double


def read_op4_column(fid, data, i_col, i_row, n_lines, n_items, type_real):
    # read lines of datablock
    row = ''
    for _ in range(n_lines):
        row += fid.readline()[:-1]

    for i_item in range(n_items):
        if type_real:
            data[i_col, i_row + i_item] = nastran_number_converter(row[:16], 'float')
            row = row[16:]
        else:
            data[i_col, i_row + i_item] = np.complex(nastran_number_converter(row[:16], 'float'),
                                                     nastran_number_converter(row[16:32], 'float'))
            row = row[32:]
    return data


def read_op4_sparse(fid, data, n_col, n_row, type_real, type_double):

    bigmat = check_bigmat(n_row)
    while True:
        # read header of data block
        read_string = fid.readline()
        # end of file reached or the last "dummy column" is reached
        if read_string == '' or nastran_number_converter(read_string[0:8], 'int') > n_col:
            break

        i_col = nastran_number_converter(read_string[0:8], 'int') - 1
        n_words = nastran_number_converter(read_string[16:24], 'int')
        while n_words > 0:
            # figure out the row number and number of word to come
            read_string = fid.readline()
            if bigmat:
                L = int(nastran_number_converter(read_string[0:8], 'int') - 1)
                n_words -= L + 2
                i_row = nastran_number_converter(read_string[8:16], 'int') - 1
            else:
                IS = nastran_number_converter(read_string[0:16], 'int')
                L = int(IS / 65536 - 1)
                n_words -= L + 1
                i_row = IS - 65536 * (L + 1) - 1
            # figure out how many lines the datablock will have
            n_items = L  # items that go into the column
            if type_double:
                n_items = int(n_items / 2)
            if type_real:
                n_lines = int(math.ceil(n_items / 5.0))
            else:
                n_items = int(n_items / 2)
                n_lines = int(math.ceil(n_items / 2.5))
            # now read the data
            data = read_op4_column(fid, data, i_col, i_row, n_lines, n_items, type_real)
    return data


def read_op4_dense(fid, data, n_col, type_real):
    while True:
        # read header of data block
        read_string = fid.readline()
        # end of file reached or the last "dummy column" is reached
        if read_string == '' or nastran_number_converter(read_string[0:8], 'int') > n_col:
            break

        i_col = nastran_number_converter(read_string[0:8], 'int') - 1
        i_row = nastran_number_converter(read_string[8:16], 'int') - 1

        # figure out how many lines the datablock will have
        n_items = nastran_number_converter(read_string[16:24], 'int')  # items that go into the column
        if type_real:
            n_lines = int(math.ceil(n_items / 5.0))
        else:
            n_items = int(n_items / 2)
            n_lines = int(math.ceil(n_items / 2.5))
        data = read_op4_column(fid, data, i_col, i_row, n_lines, n_items, type_real)
    return data


def load_matrix(filename, sparse_output=False, sparse_format=False):
    # Assumptions:
    # - only one matrix is give per file
    # - often, matrices have sparse characteristics, but might be given in Non-Sparse format.
    #   -> the reader always assembles the matrix as sparse internally
    # Option:           Value/Type:      Desciption:
    # filename          string          path & filename of OP4 file, e.g. './mg02_DLR-F19-S/nastran/MAA.dat'
    # sparse_output     True/False      decides about output conversion to full matrix for output
    # sparse_format     True/False      True: Sparse and ASCII Format of input file
    #                                   False: Non-Sparse and ASCII Format of input file

    filesize = float(os.stat(filename).st_size)
    logging.info('Read matrix from OP4 file: %s with %.2f MB' % (filename, filesize / 1024 ** 2))

    with open(filename, 'r') as fid:
        # get information from header line
        n_col, n_row, type_real, type_double = read_op4_header(fid)
        if type_real:
            empty_matrix = sp.lil_matrix((n_col, n_row), dtype=float)
        else:
            empty_matrix = sp.lil_matrix((n_col, n_row), dtype=complex)

        if sparse_format:
            data = read_op4_sparse(fid, empty_matrix, n_col, n_row, type_real, type_double)
        else:
            data = read_op4_dense(fid, empty_matrix, n_col, type_real)

    if sparse_output:
        data = data.tocsc()  # better sparse format than lil_matrix
    if not sparse_output:
        data = data.toarray()

    order = int(np.log10(np.abs(data).max()))
    if not type_double and order > 10:
        logging.warning('OP4 format is single precision and largest value is {} orders of magnitude above numerical \
            precision!'.format(order - 10))

    return data
