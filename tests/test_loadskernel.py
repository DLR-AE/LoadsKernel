
# Launch script for the Loads Kernel
import sys, logging, pytest

# Here you add the location of the Loads Kernel
sys.path.append("../../loads-kernel")
sys.path.append("/opt/tau/bin/py_el")

from loadskernel import kernel
import loadskernel.io_functions as io_functions
from tests.helper_functions import HelperFunctions

@pytest.fixture(scope='class')
def _initTestDir(tmpdir_factory):
    testDir = tmpdir_factory.mktemp('output')
    testDir = io_functions.specific_functions.check_path(testDir)
    logging.info("Output directory: %s" % testDir)
    return str(testDir)
    #return '/work/voss_ar/loads-kernel-examples/Discus2c/reference_output_escher/'

@pytest.mark.incremental
class TestDiscus2c(HelperFunctions):
    job_name = 'jcl_Discus2c'
    path_input = '/work/voss_ar/loads-kernel-examples/Discus2c/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/Discus2c/reference_output/'
        
    def test_preprocessing_functional(self, _initTestDir):
        # Here you launch the Loads Kernel with your job
        k = kernel.Kernel(self.job_name, pre=True, main=False, post=False, parallel=False,
                          path_input=self.path_input,
                          path_output=_initTestDir)
        k.run()

    def test_mainprocessing_functional(self, _initTestDir):
        # Here you launch the Loads Kernel with your job
        k = kernel.Kernel(self.job_name, pre=False, main=True, post=False, parallel=False,
                          path_input=self.path_input,
                          path_output=_initTestDir)
        k.run()

    def test_postprocessing_functional(self, _initTestDir):
        # Here you launch the Loads Kernel with your job
        k = kernel.Kernel(self.job_name, pre=False, main=False, post=True, parallel=False,
                          path_input=self.path_input,
                          path_output=_initTestDir)
        k.run()
    
    def test_preprocessing_results(self, _initTestDir):   
        # do comparisons
        logging.info('Comparing model with reference')
        model = io_functions.specific_functions.load_model(self.job_name, _initTestDir)
        del model.path_output # Running the test in a temporary directory means the path_output changes constantly.
        reference_model = io_functions.specific_functions.load_model(self.job_name, self.path_reference)
        assert self.compare_dictionaries(model.__dict__, reference_model.__dict__), "model does NOT match reference"
        
    def test_mainprocessing_results(self, _initTestDir):
        # do comparisons
        logging.info('Comparing response with reference')
        responses = io_functions.specific_functions.load_responses(self.job_name, _initTestDir, sorted=True)
        reference_responses = io_functions.specific_functions.load_responses(self.job_name, self.path_reference, sorted=True)
        assert self.compare_lists(responses, reference_responses), "response does NOT match reference"
        
        logging.info('Comparing monstations with reference')
        with open(_initTestDir + 'monstations_' + self.job_name + '.pickle', 'rb') as f:
            monstations = io_functions.specific_functions.load_pickle(f)
        with open(self.path_reference + 'monstations_' + self.job_name + '.pickle', 'rb') as f:
            reference_monstations = io_functions.specific_functions.load_pickle(f)
        assert self.compare_dictionaries(monstations, reference_monstations), "monstations do NOT match reference"
        
        # do comparisons
        logging.info('Comparing dyn2stat with reference')
        with open(_initTestDir + 'dyn2stat_' + self.job_name + '.pickle', 'rb') as f:
            dyn2stat_data = io_functions.specific_functions.load_pickle(f)
        with open(self.path_reference + 'dyn2stat_' + self.job_name + '.pickle', 'rb') as f:
            reference_dyn2stat_data = io_functions.specific_functions.load_pickle(f)
        assert self.compare_dictionaries(dyn2stat_data, reference_dyn2stat_data), "dyn2stat does NOT match reference"

    def test_postprocessing_results(self, _initTestDir):     
        # do comparisons
        logging.info('Comparing crit_trimcases with reference')
        with open(_initTestDir + 'crit_trimcases_' + self.job_name + '.csv', 'r') as f:
            lines = f.readlines()
        with open(self.path_reference + 'crit_trimcases_' + self.job_name + '.csv', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "crit_trimcases do NOT match reference"
         
        logging.info('Comparing nodalloads with reference')
        with open(_initTestDir + 'nodalloads_' + self.job_name + '.bdf_Pg', 'r') as f:
            lines = f.readlines()
        with open(self.path_reference + 'nodalloads_' + self.job_name + '.bdf_Pg', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "nodalloads do NOT match reference"
         
        logging.info('Comparing subcases with reference')
        with open(_initTestDir + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r') as f:
            lines = f.readlines()
        with open(self.path_reference + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "subcases do NOT match reference"

class TestDiscus2cNonlinSteady(TestDiscus2c):
    job_name = 'jcl_Discus2c_nonlin_steady'
    path_input = '/work/voss_ar/loads-kernel-examples/Discus2c/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/Discus2c/reference_output/'
 
class TestDiscus2cTimedom(TestDiscus2c):
    job_name = 'jcl_Discus2c_timedom'
    path_input = '/work/voss_ar/loads-kernel-examples/Discus2c/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/Discus2c/reference_output/'

class TestAllegraTimedom(TestDiscus2c):
    job_name = 'jcl_ALLEGRA_timedom'
    path_input = '/work/voss_ar/loads-kernel-examples/Allegra/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/Allegra/reference_output/'
        
    def test_postprocessing_results(self, _initTestDir):     
        # do comparisons
        logging.info('Comparing crit_trimcases with reference')
        with open(_initTestDir + 'crit_trimcases_' + self.job_name + '.csv', 'r') as f:
            lines = f.readlines()
        with open(self.path_reference + 'crit_trimcases_' + self.job_name + '.csv', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "crit_trimcases do NOT match reference"

        logging.info('Comparing subcases with reference')
        with open(_initTestDir + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r') as f:
            lines = f.readlines()
        with open(self.path_reference + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "subcases do NOT match reference"
        
class TestAllegraFreqdom(TestAllegraTimedom):
    job_name = 'jcl_ALLEGRA_freqdom'
    path_input = '/work/voss_ar/loads-kernel-examples/Allegra/JCLs/'
    path_reference='/work/voss_ar/loads-kernel-examples/Allegra/reference_output/'
