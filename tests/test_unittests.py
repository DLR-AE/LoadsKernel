import numpy as np
import platform
import os
import pytest

from loadskernel.io_functions import read_bdf
from loadskernel.fem_interfaces import fem_helper


@pytest.fixture(scope='class')
def get_test_dir(tmpdir_factory):
    test_dir = tmpdir_factory.mktemp('output')
    return str(test_dir)


def test_dummy():
    print('The dummy test is executed.')
    print('Running on python version {}'.format(platform.python_version()))


def test_bdf_reader_with_includes(get_test_dir):
    """
    Create a parent BDF file that includes two more BDF files.
    The simple case: the include is in the same directory and in one line
    More complex case: the include is in a sub-directory and the filename is distributed over multiple lines
    """
    with open(os.path.join(get_test_dir, 'parent.bdf'), 'w') as fid:
        fid.write("include './a_long_file_name_for_a_grid_point.bdf' \n")
        fid.write("include './one/ \n        more/ \n        grid.bdf' \n")
        fid.write('GRID           1       0     1.0     2.0     3.0       0 \n')

    with open(os.path.join(get_test_dir, 'a_long_file_name_for_a_grid_point.bdf'), 'w') as fid:
        fid.write('GRID           2       0     1.0     2.0     3.0       0')

    os.makedirs(os.path.join(get_test_dir, 'one', 'more'), exist_ok=True)
    with open(os.path.join(get_test_dir, 'one', 'more', 'grid.bdf'), 'w') as fid:
        fid.write('GRID           3       0     1.0     2.0     3.0       0')

    # Read the BDFs created above.
    r = read_bdf.Reader()
    r.process_deck(os.path.join(get_test_dir, 'parent.bdf'))
    # Check if all 3 GRID points are found.
    assert len(r.cards['GRID']) == 3, 'Not all GRID points were found by the BDF reader.'


def test_pole_correlation():
    # Create two artifical arrays of eigenvalues with complex conjugate pairs and different length.
    lam1 = np.array([1.0 + 1.0j, 1.0 - 1.0j, 2.0 + 1.0j, 2.0 - 1.0j, 1.0 + 2.0j, 1.0 - 2.0j])
    lam2 = np.array([1.0 + 1.0j, 1.0 - 1.0j, 2.0 + 1.0j, 2.0 - 1.0j, 1.0 + 2.0j, 1.0 - 2.0j, 0.5 + 1.0j, 0.5 - 1.0j])
    # Provide refence results.
    PCC_ref = np.array([[1.0, 0.33333333, 0.68377223, 0.29289322, 0.75, 0.25, 0.83560101, 0.32216561],
                        [0.33333333, 1.0, 0.29289322, 0.68377223, 0.25, 0.75, 0.32216561, 0.83560101],
                        [0.66666667, 0.25464401, 1.0, 0.36754447, 0.64644661, 0.20943058, 0.50680304, 0.17800506],
                        [0.25464401, 0.66666667, 0.36754447, 1.0, 0.20943058, 0.64644661, 0.17800506, 0.50680304],
                        [0.66666667, 0.0, 0.5527864, 0.0, 1.0, 0.0, 0.63239269, 0.0],
                        [0.0, 0.66666667, 0.0, 0.5527864, 0.0, 1.0, 0.0, 0.63239269]])
    # Calculate the pole correlation criterion (PCC).
    PCC = fem_helper.calc_PCC(lam1, lam2)
    # Check for numerical similarity with reference values.
    assert np.allclose(PCC, PCC_ref, rtol=1e-4, atol=1e-4), "Pole correlation (PCC) does NOT match reference"


def test_hyperbolic_distance_metric():
    # Create two artifical arrays of eigenvalues with complex conjugate pairs and different length.
    lam1 = np.array([1.0 + 1.0j, 1.0 - 1.0j, 2.0 + 1.0j, 2.0 - 1.0j, 1.0 + 2.0j, 1.0 - 2.0j])
    lam2 = np.array([1.0 + 1.0j, 1.0 - 1.0j, 2.0 + 1.0j, 2.0 - 1.0j, 1.0 + 2.0j, 1.0 - 2.0j, 0.5 + 1.0j, 0.5 - 1.0j])
    # Provide refence results.
    HDM_ref = np.array([[1.0, 0.44429156, 0.76909089, 0.6000117, 0.63524992, 0.40062684, 0.69777861, 0.27132177],
                        [0.44429156, 1.0, 0.6000117, 0.76909089, 0.40062684, 0.63524992, 0.27132177, 0.69777861],
                        [0.76909089, 0.6000117, 1.0, 0.80323929, 0.69522953, 0.57169725, 0.50164738, 0.37526785],
                        [0.6000117, 0.76909089, 0.80323929, 1.0, 0.57169725, 0.69522953, 0.37526785, 0.50164738],
                        [0.63524992, 0.40062684, 0.69522953, 0.57169725, 1.0, 0.47973274, 0.44086248, 0.23680655],
                        [0.40062684, 0.63524992, 0.57169725, 0.69522953, 0.47973274, 1.0, 0.23680655, 0.44086248]])
    # Calculate the hyperbolic distance metric (HDM).
    HDM = fem_helper.calc_HDM(lam1, lam2)
    # Check for numerical similarity with reference values.
    assert np.allclose(HDM, HDM_ref, rtol=1e-4, atol=1e-4), "Hyperbolic distance metric (HDM) does NOT match reference"
