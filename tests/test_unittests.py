import platform
import os
import pytest

from loadskernel.io_functions import read_bdf


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
