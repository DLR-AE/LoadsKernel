"""
Comprehensive integration tests are performed based on the tutorials. Currently, these are puerly
functional tests without comparison to reference results.
"""

import logging
import os
import pytest

from loadskernel import program_flow, io_functions
from tests.helper_functions import HelperFunctions


@pytest.fixture(name='tmp_output', scope='class')
def fixture_tmp_output(tmpdir_factory):
    test_dir = tmpdir_factory.mktemp('output')
    test_dir = io_functions.data_handling.check_path(test_dir)
    return str(test_dir)


@pytest.fixture(name='tutorials_repo', scope='session')
def fixture_tutorials_repo():
    # The turorials are part of the repository already, so just point to the right location.
    repo_path = io_functions.data_handling.check_path(os.path.join('.', 'doc', 'tutorials'))
    return repo_path


class TestInfrastructure():

    def test_cloned_repositories(self, tutorials_repo):
        # This test is used to run the fixtures once / check if the cloned repositories are there.
        logging.info('Cloned repositories are here:')
        logging.info(' - %s', tutorials_repo)


class PreMainPostFunctional(HelperFunctions):
    job_name = 'jcl_xyz'
    aircraft_name = 'xyz'

    def test_preprocessing_functional(self, tmp_output, tutorials_repo):
        # Here you launch the Loads Kernel with your job
        k = program_flow.Kernel(self.job_name, pre=True, main=False, post=False,
                                path_input=os.path.join(tutorials_repo, self.aircraft_name, 'JCLs'),
                                path_output=tmp_output)
        k.run()

    def test_mainprocessing_functional(self, tmp_output, tutorials_repo):
        # Here you launch the Loads Kernel with your job
        k = program_flow.Kernel(self.job_name, pre=False, main=True, post=False,
                                path_input=os.path.join(tutorials_repo, self.aircraft_name, 'JCLs'),
                                path_output=tmp_output)
        k.run()

    def test_postprocessing_functional(self, tmp_output, tutorials_repo):
        # Here you launch the Loads Kernel with your job
        k = program_flow.Kernel(self.job_name, pre=False, main=False, post=True,
                                path_input=os.path.join(tutorials_repo, self.aircraft_name, 'JCLs'),
                                path_output=tmp_output)
        k.run()


class TestDC3Trim(PreMainPostFunctional):
    job_name = 'jcl_dc3_trim'
    aircraft_name = 'DC3_model'


class TestDC3Maneuvers(TestDC3Trim):
    job_name = 'jcl_dc3_maneuvers'
    aircraft_name = 'DC3_model'


class TestDC3Gust(TestDC3Trim):
    job_name = 'jcl_dc3_gust_H23'
    aircraft_name = 'DC3_model'


class TestDC3Flutter(TestDC3Trim):
    job_name = 'jcl_dc3_flutter'
    aircraft_name = 'DC3_model'
