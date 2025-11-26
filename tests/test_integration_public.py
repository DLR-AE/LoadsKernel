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


@pytest.fixture(name='reference_repo', scope='session')
def fixture_reference_repo():
    # The reference results repository was cloned by the pipeline, so just check the path.
    repo_path = io_functions.data_handling.check_path(os.path.join('.', 'LoadsKernel-public-reference-results'))
    return repo_path


class TestInfrastructure():

    def test_cloned_repositories(self, tutorials_repo, reference_repo):
        # This test is used to run the fixtures once / check if the cloned repositories are there.
        logging.info('Cloned repositories are here:')
        logging.info(' - %s', tutorials_repo)
        logging.info(' - %s', reference_repo)


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

    def test_preprocessing_results(self, tmp_output, reference_repo):
        # do comparisons
        logging.info('Comparing model with reference')
        model = io_functions.data_handling.load_hdf5(tmp_output + 'model_' + self.job_name + '.hdf5')
        reference_model = io_functions.data_handling.load_hdf5(reference_repo + 'model_' + self.job_name + '.hdf5')
        assert self.compare_dictionaries(model, reference_model), "model does NOT match reference"

    def test_mainprocessing_results(self, tmp_output, reference_repo):
        # do comparisons
        logging.info('Comparing response with reference')
        responses = io_functions.data_handling.load_hdf5_responses(self.job_name, tmp_output)
        reference_responses = io_functions.data_handling.load_hdf5_responses(self.job_name, reference_repo)
        assert self.compare_lists(responses, reference_responses), "response does NOT match reference"

        logging.info('Comparing monstations with reference')
        monstations = io_functions.data_handling.load_hdf5(tmp_output + 'monstations_' + self.job_name + '.hdf5')
        reference_monstations = io_functions.data_handling.load_hdf5(reference_repo + 'monstations_' + self.job_name + '.hdf5')
        assert self.compare_dictionaries(monstations, reference_monstations), "monstations do NOT match reference"

        # do comparisons
        logging.info('Comparing dyn2stat with reference')
        dyn2stat_data = io_functions.data_handling.load_hdf5(tmp_output + 'dyn2stat_' + self.job_name + '.hdf5')
        reference_dyn2stat_data = io_functions.data_handling.load_hdf5(reference_repo + 'dyn2stat_' + self.job_name + '.hdf5')
        assert self.compare_dictionaries(dyn2stat_data, reference_dyn2stat_data), "dyn2stat does NOT match reference"

    def test_postprocessing_results(self, tmp_output, reference_repo):
        # do comparisons
        logging.info('Comparing crit_trimcases with reference')
        with open(tmp_output + 'crit_trimcases_' + self.job_name + '.csv', 'r', encoding="utf-8") as f:
            lines = f.readlines()
        with open(reference_repo + 'crit_trimcases_' + self.job_name + '.csv', 'r', encoding="utf-8") as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "crit_trimcases do NOT match reference"

        logging.info('Comparing subcases with reference')
        with open(tmp_output + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r', encoding="utf-8") as f:
            lines = f.readlines()
        with open(reference_repo + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r', encoding="utf-8") as f:
            reference_lines = f.readlines()
        assert self.compare_lists(
            lines, reference_lines), "subcases do NOT match reference"


class TestDC3Maneuvers(PreMainPostFunctional):
    job_name = 'jcl_dc3_maneuvers'
    aircraft_name = 'DC3_model'


class TestDC3Gust(PreMainPostFunctional):
    job_name = 'jcl_dc3_gust_H23'
    aircraft_name = 'DC3_model'


class TestDC3Flutter(PreMainPostFunctional):
    job_name = 'jcl_dc3_flutter'
    aircraft_name = 'DC3_model'

    def test_preprocessing_results(self, tmp_output, reference_repo):
        # do comparisons
        logging.info('Comparing model with reference')
        model = io_functions.data_handling.load_hdf5(tmp_output + 'model_' + self.job_name + '.hdf5')
        reference_model = io_functions.data_handling.load_hdf5(reference_repo + 'model_' + self.job_name + '.hdf5')
        assert self.compare_dictionaries(model, reference_model), "model does NOT match reference"

    def test_mainprocessing_results(self, tmp_output, reference_repo):
        logging.info('Comparing response with reference')
        responses = io_functions.data_handling.load_hdf5_responses(self.job_name, tmp_output)
        reference_responses = io_functions.data_handling.load_hdf5_responses(self.job_name, reference_repo)

        # For the K and KE method (response[0]), Vtas, damping and frequencies are quantities of interest.
        resp_a = responses[0]
        resp_b = reference_responses[0]
        # Only compare results in an area where the results are meaningful.
        pos_a = (resp_a['Vtas'][()] > 170.0) & (resp_a['Vtas'][()] < 300.0)
        pos_b = (resp_b['Vtas'][()] > 170.0) & (resp_b['Vtas'][()] < 300.0)
        assert self.compare_items(resp_a['Vtas'][pos_a],
                                  resp_b['Vtas'][pos_b], 'Vtas'), "Vtas does NOT match reference"
        assert self.compare_items(resp_a['damping'][pos_a],
                                  resp_b['damping'][pos_b], 'damping'), "damping does NOT match reference"
        assert self.compare_items(resp_a['freqs'][pos_a],
                                  resp_b['freqs'][pos_b], 'freqs'), "freqs do NOT match reference"

        # For the PK method (response[1]), eigenvalues, damping and freqs are quantities of interest.
        resp_a = responses[1]
        resp_b = reference_responses[1]
        # Only compare results in an area where the results are meaningful.
        pos_a = (resp_a['Vtas'][()] > 170.0) & (resp_a['Vtas'][()] < 300.0)
        pos_b = (resp_b['Vtas'][()] > 170.0) & (resp_b['Vtas'][()] < 300.0)
        assert self.compare_items(resp_a['eigenvalues'][pos_a],
                                  resp_b['eigenvalues'][pos_b], 'eigenvalues'), "eigenvalues do NOT match reference"
        assert self.compare_items(resp_a['damping'][pos_a],
                                  resp_b['damping'][pos_b], 'damping'), "damping does NOT match reference"
        assert self.compare_items(resp_a['freqs'][pos_a],
                                  resp_b['freqs'][pos_b], 'freqs'), "freqs do NOT match reference"
