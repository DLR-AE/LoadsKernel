
# Launch script for the Loads Kernel
import sys, logging, pytest, subprocess, shlex

# Here you add the location of the Loads Kernel
sys.path.append("../loads-kernel")
sys.path.append("/scratch/panel-aero")

from loadskernel import program_flow
import loadskernel.io_functions as io_functions
from helper_functions import HelperFunctions

@pytest.fixture(scope='class')
def get_test_dir(tmpdir_factory):
    test_dir = tmpdir_factory.mktemp('output')
    test_dir = io_functions.specific_functions.check_path(test_dir)
    return str(test_dir)

class TestDiscus2c(HelperFunctions):
    job_name = 'jcl_Discus2c'
    path_input = '/scratch/loads-kernel-examples/Discus2c/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/reference_output/'

    def test_preprocessing_functional(self, get_test_dir):
        # Here you launch the Loads Kernel with your job
        k = program_flow.Kernel(self.job_name, pre=True, main=False, post=False,
                          path_input=self.path_input,
                          path_output=get_test_dir)
        k.run()

    def test_mainprocessing_functional(self, get_test_dir):
        # Here you launch the Loads Kernel with your job
        k = program_flow.Kernel(self.job_name, pre=False, main=True, post=False,
                          path_input=self.path_input,
                          path_output=get_test_dir)
        k.run()

    def test_postprocessing_functional(self, get_test_dir):
        # Here you launch the Loads Kernel with your job
        k = program_flow.Kernel(self.job_name, pre=False, main=False, post=True,
                          path_input=self.path_input,
                          path_output=get_test_dir)
        k.run()

    def test_preprocessing_results(self, get_test_dir):
        # do comparisons
        logging.info('Comparing model with reference')
        model = io_functions.specific_functions.load_model(self.job_name, get_test_dir)
        reference_model = io_functions.specific_functions.load_model(self.job_name, self.path_reference)
        # Running the test in a temporary directory means the path_output changes constantly and can't be compared.
        del model.path_output
        assert self.compare_dictionaries(model.__dict__, reference_model.__dict__), "model does NOT match reference"

    def test_mainprocessing_results(self, get_test_dir):
        # do comparisons
        logging.info('Comparing response with reference')
        responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, get_test_dir)
        reference_responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, self.path_reference)
        assert self.compare_lists(responses, reference_responses), "response does NOT match reference"

        logging.info('Comparing monstations with reference')
        monstations = io_functions.specific_functions.load_hdf5(get_test_dir + 'monstations_' + self.job_name + '.hdf5')
        reference_monstations = io_functions.specific_functions.load_hdf5(self.path_reference + 'monstations_' + self.job_name + '.hdf5')
        assert self.compare_dictionaries(monstations, reference_monstations), "monstations do NOT match reference"

        # do comparisons
        logging.info('Comparing dyn2stat with reference')
        dyn2stat_data = io_functions.specific_functions.load_hdf5(get_test_dir + 'dyn2stat_' + self.job_name + '.hdf5')
        reference_dyn2stat_data = io_functions.specific_functions.load_hdf5(self.path_reference + 'dyn2stat_' + self.job_name + '.hdf5')
        assert self.compare_dictionaries(dyn2stat_data, reference_dyn2stat_data), "dyn2stat does NOT match reference"

    def test_postprocessing_results(self, get_test_dir):
        # do comparisons
        logging.info('Comparing crit_trimcases with reference')
        with open(get_test_dir + 'crit_trimcases_' + self.job_name + '.csv', 'r') as f:
            lines = f.readlines()
        with open(self.path_reference + 'crit_trimcases_' + self.job_name + '.csv', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "crit_trimcases do NOT match reference"

        logging.info('Comparing subcases with reference')
        with open(get_test_dir + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r') as f:
            lines = f.readlines()
        with open(self.path_reference + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "subcases do NOT match reference"

class TestDiscus2cNonlinSteady(TestDiscus2c):
    job_name = 'jcl_Discus2c_nonlin_steady'
    path_input = '/scratch/loads-kernel-examples/Discus2c/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/reference_output/'

class TestDiscus2cTimedom(TestDiscus2c):
    job_name = 'jcl_Discus2c_timedom'
    path_input = '/scratch/loads-kernel-examples/Discus2c/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/reference_output/'
 
class TestAllegraTimedom(TestDiscus2c):
    job_name = 'jcl_ALLEGRA_timedom'
    path_input = '/scratch/loads-kernel-examples/Allegra/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/reference_output/'
 
class TestAllegraFreqdom(TestDiscus2c):
    job_name = 'jcl_ALLEGRA_freqdom'
    path_input = '/scratch/loads-kernel-examples/Allegra/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/reference_output/'
 
class TestAllegraFlutter(HelperFunctions):
    job_name = 'jcl_ALLEGRA_flutter'
    path_input = '/scratch/loads-kernel-examples/Allegra/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/reference_output/'
     
    def test_preprocessing_functional(self, get_test_dir):
        # Here you launch the Loads Kernel with your job
        k = program_flow.Kernel(self.job_name, pre=True, main=False, post=False,
                          path_input=self.path_input,
                          path_output=get_test_dir)
        k.run()
     
    def test_mainprocessing_functional(self, get_test_dir):
        # Here you launch the Loads Kernel with your job
        k = program_flow.Kernel(self.job_name, pre=False, main=True, post=False,
                          path_input=self.path_input,
                          path_output=get_test_dir)
        k.run()
     
    def test_postprocessing_functional(self, get_test_dir):
        # Here you launch the Loads Kernel with your job
        k = program_flow.Kernel(self.job_name, pre=False, main=False, post=True,
                          path_input=self.path_input,
                          path_output=get_test_dir)
        k.run()
     
    def test_preprocessing_results(self, get_test_dir):
        # do comparisons
        logging.info('Comparing model with reference')
        model = io_functions.specific_functions.load_model(self.job_name, get_test_dir)
        reference_model = io_functions.specific_functions.load_model(self.job_name, self.path_reference)
        # Running the test in a temporary directory means the path_output changes constantly and can't be compared.
        del model.path_output
        assert self.compare_dictionaries(model.__dict__, reference_model.__dict__), "model does NOT match reference"
     
    def test_mainprocessing_results(self, get_test_dir):
        # do comparisons
        logging.info('Comparing response with reference')
        responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, get_test_dir)
        reference_responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, self.path_reference)
        assert self.compare_lists(responses, reference_responses), "response does NOT match reference"

class TestAllegraLimitTurbulence(TestAllegraFlutter):
    job_name = 'jcl_ALLEGRA_limitturbulence'
    path_input = '/scratch/loads-kernel-examples/Allegra/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/reference_output/'

class TestHAPO6Trim(TestDiscus2c):
    job_name = 'jcl_HAP-O6'
    path_input = '/scratch/loads-kernel-examples/HAP-O6/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/reference_output/'

class TestHAPO6Derivatives(TestAllegraFlutter):
    job_name = 'jcl_HAP-O6_derivatives'
    path_input = '/scratch/loads-kernel-examples/HAP-O6/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/reference_output/'

class TestHAPO6StateSpaceSystem(TestAllegraFlutter):
    job_name = 'jcl_HAP-O6_sss'
    path_input = '/scratch/loads-kernel-examples/HAP-O6/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/reference_output/'

class TestDiscus2cParallelProcessing(HelperFunctions):
    job_name = 'jcl_Discus2c_parallelprocessing'
    path_input = '/scratch/loads-kernel-examples/Discus2c/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/reference_output/'
    
    def test_preprocessing_functional_via_command_line_interface(self, get_test_dir):
        # Here we us the command line interface
        args = shlex.split('python ./loadskernel/program_flow.py --job_name %s \
            --pre True --main False --post False \
            --path_input %s --path_output %s' % (self.job_name, self.path_input, get_test_dir))
        returncode = subprocess.call(args)
        assert returncode == 0, "subprocess call failed: " + str(args)

    def test_mainprocessing_functional_via_command_line_interface(self, get_test_dir):
        # Here we us the command line interface
        args = shlex.split('mpiexec -n 2 python ./loadskernel/program_flow.py --job_name %s \
            --pre False --main True --post False \
            --path_input %s --path_output %s' % (self.job_name, self.path_input, get_test_dir))
        returncode = subprocess.call(args)
        assert returncode == 0, "subprocess call failed: " + str(args)
    
    def test_preprocessing_results(self, get_test_dir):
        # do comparisons
        logging.info('Comparing model with reference')
        model = io_functions.specific_functions.load_model(self.job_name, get_test_dir)
        reference_model = io_functions.specific_functions.load_model(self.job_name, self.path_reference)
        # Running the test in a temporary directory means the path_output changes constantly and can't be compared.
        del model.path_output
        assert self.compare_dictionaries(model.__dict__, reference_model.__dict__), "model does NOT match reference"
    
    def test_mainprocessing_results(self, get_test_dir):
        # do comparisons
        logging.info('Comparing response with reference')
        responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, get_test_dir)
        reference_responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, self.path_reference)
        assert self.compare_lists(responses, reference_responses), "response does NOT match reference"

        logging.info('Comparing monstations with reference')
        monstations = io_functions.specific_functions.load_hdf5(get_test_dir + 'monstations_' + self.job_name + '.hdf5')
        reference_monstations = io_functions.specific_functions.load_hdf5(self.path_reference + 'monstations_' + self.job_name + '.hdf5')
        assert self.compare_dictionaries(monstations, reference_monstations), "monstations do NOT match reference"

        # do comparisons
        logging.info('Comparing dyn2stat with reference')
        dyn2stat_data = io_functions.specific_functions.load_hdf5(get_test_dir + 'dyn2stat_' + self.job_name + '.hdf5')
        reference_dyn2stat_data = io_functions.specific_functions.load_hdf5(self.path_reference + 'dyn2stat_' + self.job_name + '.hdf5')
        assert self.compare_dictionaries(dyn2stat_data, reference_dyn2stat_data), "dyn2stat does NOT match reference"
