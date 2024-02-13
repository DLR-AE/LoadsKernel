import logging
import scipy

from loadskernel.io_functions.data_handling import load_hdf5


def load_matrix(filename, name):
    """
    This function reads matrices from Nastran's HDF5 output.
    Usage of this function:
    matrix = load_matrix('/path/to/my_bdf_name.mtx.h5', 'KGG')

    The matrices are exported using the following procedure, tested with Nastran 2023.1:
    In a DMAP alter, any matrices can be send into the HDF5 file with
    CRDB_MTX MGG//'MGG' $
    Then, in the BULK section, the precision is set to 64, compression is deactivated and the
    matrix output is activated  with
    HDF5OUT PRCISION     64 CMPRMTHD   NONE  MTX     YES
    """
    # Open HDF5 file
    hdf5_file = load_hdf5(filename)
    # This is the place/root where the mass and stiffness matrices are stroed by Natsran
    hdf5_group = hdf5_file['NASTRAN/RESULT/MATRIX/GENERAL']
    # Check which matrices are there
    names = [x.decode("utf-8") for x in hdf5_group['IDENTITY']['NAME']]
    # If the matrix is in the file, go ahead, otherwise issue an error message and return an empty 0x0 matrix
    if name in names:
        logging.info('Reading matrix {} from {}'.format(name, filename))
        # Get some meta data given by the 'IDENTITY' table. This is important becasue in case multiple matrices
        # are stored in the same hdf5 file, the data is simply appended. The meta data identifies which pieces
        # of data belong to which matrix.
        i_matrix = names.index(name)
        n_row = hdf5_group['IDENTITY']['ROW'][i_matrix]
        n_col = hdf5_group['IDENTITY']['COLUMN'][i_matrix]
        col_pos = hdf5_group['IDENTITY']['COLUMN_POS'][i_matrix]
        data_pos = hdf5_group['IDENTITY']['DATA_POS'][i_matrix]
        non_zero = hdf5_group['IDENTITY']['NON_ZERO'][i_matrix]
        # Get pointers, indices and data
        indptr = hdf5_group['COLUMN']['POSITION'][col_pos:col_pos + n_col + 1] - data_pos
        indices = hdf5_group['DATA']['ROW'][data_pos:data_pos + non_zero]
        data = hdf5_group['DATA']['VALUE'][data_pos:data_pos + non_zero]
        # Set-up a scipy CSC matrix
        M = scipy.sparse.csc_matrix((data, indices, indptr), shape=(n_row, n_col))
        return M

    logging.error('Matrix {} not found in {}'.format(name, filename))
    return None
