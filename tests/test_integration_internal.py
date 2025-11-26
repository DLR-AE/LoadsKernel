"""
Comprehensive integration tests are performed and compared against long standing reference
results. This is an internal process and the repositories can only be accessed from within DLR.
For the following tests, the loads-kernel-examples and the loads-kernel-reference-results are
used, which are located in dedictaed repositories, which are cloned by the GitLab pipeline.
A tempory directory is used for the outputs in order to avoid pollution of the user's workspace.
"""

import logging
import os
import shlex
import subprocess
import pytest

from loadskernel import program_flow, io_functions
from tests.helper_functions import HelperFunctions


@pytest.fixture(name='tmp_output', scope='class')
def fixture_tmp_output(tmpdir_factory):
    test_dir = tmpdir_factory.mktemp('output')
    test_dir = io_functions.data_handling.check_path(test_dir)
    return str(test_dir)


@pytest.fixture(name='examples_repo', scope='session')
def fixture_examples_repo(tmpdir_factory):
    # The examples repository was cloned by the pipeline already. So just check out the path.
    repo_path = io_functions.data_handling.check_path(os.path.join('.', 'loads-kernel-examples'))
    return repo_path


@pytest.fixture(name='reference_repo', scope='session')
def fixture_reference_repo(tmpdir_factory):
    # The reference results repository was cloned by the pipeline, too.
    repo_path = io_functions.data_handling.check_path(os.path.join('.', 'loads-kernel-reference-results'))
    return repo_path


class TestClonedRepositories():

    def test_cloned_repositories_once(self, examples_repo, reference_repo):
        # This test is used to run the fixtures once / check if the cloned repositories are there.
        logging.info('Cloned repositories are here: ')
        logging.info(' - %s', examples_repo)
        logging.info(' - %s', reference_repo)


class PreMainPostFunctional(HelperFunctions):
    job_name = 'jcl_Discus2c'
    aircraft_name = 'Discus2c'

    def test_preprocessing_functional(self, tmp_output, examples_repo):
        # Here you launch the Loads Kernel with your job
        k = program_flow.Kernel(self.job_name, pre=True, main=False, post=False,
                                path_input=os.path.join(examples_repo, self.aircraft_name, 'JCLs'),
                                path_output=tmp_output)
        k.run()

    def test_mainprocessing_functional(self, tmp_output, examples_repo):
        # Here you launch the Loads Kernel with your job
        k = program_flow.Kernel(self.job_name, pre=False, main=True, post=False,
                                path_input=os.path.join(examples_repo, self.aircraft_name, 'JCLs'),
                                path_output=tmp_output)
        k.run()

    def test_postprocessing_functional(self, tmp_output, examples_repo):
        # Here you launch the Loads Kernel with your job
        k = program_flow.Kernel(self.job_name, pre=False, main=False, post=True,
                                path_input=os.path.join(examples_repo, self.aircraft_name, 'JCLs'),
                                path_output=tmp_output)
        k.run()


class TestDiscus2c(PreMainPostFunctional):
    job_name = 'jcl_Discus2c'
    aircraft_name = 'Discus2c'

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
        with open(tmp_output + 'crit_trimcases_' + self.job_name + '.csv', 'r') as f:
            lines = f.readlines()
        with open(reference_repo + 'crit_trimcases_' + self.job_name + '.csv', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "crit_trimcases do NOT match reference"

        logging.info('Comparing subcases with reference')
        with open(tmp_output + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r') as f:
            lines = f.readlines()
        with open(reference_repo + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(
            lines, reference_lines), "subcases do NOT match reference"


class TestDiscus2cParallelProcessing(HelperFunctions):
    job_name = 'jcl_Discus2c_parallelprocessing'
    aircraft_name = 'Discus2c'

    def test_preprocessing_functional_via_command_line_interface(self, tmp_output, examples_repo):
        # Here we us the command line interface
        args = shlex.split("loads-kernel --job_name {self.job_name} \
            --pre True --main False --post False \
            --path_input {os.path.join(examples_repo, self.aircraft_name, 'JCLs')} \
            --path_output {tmp_output}")
        out = subprocess.run(args, env=os.environ, check=False)
        assert out.returncode == 0, "subprocess failed: " + str(args)

    def test_mainprocessing_functional_via_command_line_interface(self, tmp_output, examples_repo):
        # Here we us the command line interface
        args = shlex.split(f"mpiexec -n 2 loads-kernel --job_name {self.job_name} \
            --pre False --main True --post False \
            --path_input {os.path.join(examples_repo, self.aircraft_name, 'JCLs')} \
            --path_output {tmp_output}")
        out = subprocess.run(args, env=os.environ, check=False)
        assert out.returncode == 0, "subprocess failed: " + str(args)

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


class TestDiscus2cNonlinSteady(TestDiscus2c):
    job_name = 'jcl_Discus2c_nonlin_steady'
    aircraft_name = 'Discus2c'


class TestDiscus2cTimedom(TestDiscus2c):
    job_name = 'jcl_Discus2c_timedom'
    aircraft_name = 'Discus2c'


class TestDiscus2cB2000(TestDiscus2c):
    job_name = 'jcl_Discus2c_B2000'
    aircraft_name = 'Discus2c'


class TestAllegraTimedom(TestDiscus2c):
    job_name = 'jcl_ALLEGRA_timedom'
    aircraft_name = 'Allegra'


class TestAllegraFreqdom(TestDiscus2c):
    job_name = 'jcl_ALLEGRA_freqdom'
    aircraft_name = 'Allegra'


class TestAllegraFlutter(PreMainPostFunctional):
    job_name = 'jcl_ALLEGRA_flutter'
    aircraft_name = 'Allegra'

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

        # Responses 0 and 1: For the K and KE method, Vtas, damping and frequencies are quantities of interest.
        for resp_a, resp_b in zip(responses[:2], reference_responses[:2]):
            # Only compare results in an area where the results are meaningful.
            pos_a = (resp_a['Vtas'][()] > 350.0) & (resp_a['Vtas'][()] < 450.0)
            pos_b = (resp_b['Vtas'][()] > 350.0) & (resp_b['Vtas'][()] < 450.0)
            assert self.compare_items(resp_a['Vtas'][pos_a],
                                      resp_b['Vtas'][pos_b], 'Vtas'), "Vtas does NOT match reference"
            assert self.compare_items(resp_a['damping'][pos_a],
                                      resp_b['damping'][pos_b], 'damping'), "damping does NOT match reference"
            assert self.compare_items(resp_a['freqs'][pos_a],
                                      resp_b['freqs'][pos_b], 'freqs'), "freqs do NOT match reference"

        # Responses 2 and 3: For the PK methods, eigenvalues, damping and freqs are quantities of interest.
        for resp_a, resp_b in zip(responses[2:], reference_responses[2:]):
            # Only compare results in an area where the results are meaningful.
            pos_a = (resp_a['Vtas'][()] > 350.0) & (resp_a['Vtas'][()] < 450.0)
            pos_b = (resp_b['Vtas'][()] > 350.0) & (resp_b['Vtas'][()] < 450.0)
            assert self.compare_items(resp_a['eigenvalues'][pos_a],
                                      resp_b['eigenvalues'][pos_b], 'eigenvalues'), "eigenvalues do NOT match reference"
            assert self.compare_items(resp_a['damping'][pos_a],
                                      resp_b['damping'][pos_b], 'damping'), "damping does NOT match reference"
            assert self.compare_items(resp_a['freqs'][pos_a],
                                      resp_b['freqs'][pos_b], 'freqs'), "freqs do NOT match reference"


class TestAllegraLimitTurbulence(PreMainPostFunctional):
    job_name = 'jcl_ALLEGRA_limitturbulence'
    aircraft_name = 'Allegra'

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


class TestHAPO6Trim(TestDiscus2c):
    job_name = 'jcl_HAP-O6'
    aircraft_name = 'HAP-O6'


class TestHAPO6Derivatives(TestAllegraLimitTurbulence):
    job_name = 'jcl_HAP-O6_derivatives'
    aircraft_name = 'HAP-O6'


class TestHAPO6StateSpaceSystem(TestAllegraLimitTurbulence):
    job_name = 'jcl_HAP-O6_sss'
    aircraft_name = 'HAP-O6'
