
# Launch script for the Loads Kernel
import sys, logging, pytest

# Here you add the location of the Loads Kernel
sys.path.append("../../loads-kernel")
sys.path.append("/opt/tau/bin/py_el")

from loadskernel import kernel
import loadskernel.io_functions as io_functions
from tests.helper_functions import HelperFunctions

@pytest.mark.incremental
class TestDiscus2c(HelperFunctions):
    
    @pytest.fixture(scope="session")
    def set_path(self):
        self.job_name = 'jcl_Discus2c'
        self.path_input = '/work/voss_ar/loads-kernel-examples/Discus2c/JCLs/'
        self.path_output='/work/voss_ar/loads-kernel-examples/Discus2c/output/'
        #self.path_output = tmpdir_factory.mktemp("output")
        #print(self.path_output)
        self.path_reference='/work/voss_ar/loads-kernel-examples/Discus2c/reference_output/'
        
    def test_preprocessing(self):
        self.set_path()
        # Here you launch the Loads Kernel with your job
        k = kernel.Kernel(self.job_name, pre=True, main=False, post=False, parallel=False,
                          path_input=self.path_input,
                          path_output=self.path_output)
        k.run()
        
        # do comparisons
        logging.info('Comparing model with reference')
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)
        reference_model = io_functions.specific_functions.load_model(self.job_name, self.path_reference)
        assert self.compare_dictionaries(model.__dict__, reference_model.__dict__), "model does NOT match reference"
        
    
    def test_mainprocessing(self):
        self.set_path()
        # Here you launch the Loads Kernel with your job
        k = kernel.Kernel(self.job_name, pre=False, main=True, post=False, parallel=False,
                          path_input=self.path_input,
                          path_output=self.path_output)
        k.run()
        
        # do comparisons
        logging.info('Comparing response with reference')
        responses = io_functions.specific_functions.load_responses(self.job_name, self.path_output, sorted=True)
        reference_responses = io_functions.specific_functions.load_responses(self.job_name, self.path_reference, sorted=True)
        assert self.compare_lists(responses, reference_responses), "response does NOT match reference"
        
        logging.info('Comparing monstations with reference')
        with open(self.path_output + 'monstations_' + self.job_name + '.pickle', 'rb') as f:
            monstations = io_functions.specific_functions.load_pickle(f)
        with open(self.path_reference + 'monstations_' + self.job_name + '.pickle', 'rb') as f:
            reference_monstations = io_functions.specific_functions.load_pickle(f)
        assert self.compare_dictionaries(monstations, reference_monstations), "monstations do NOT match reference"
        
        # do comparisons
        logging.info('Comparing dyn2stat with reference')
        with open(self.path_output + 'dyn2stat_' + self.job_name + '.pickle', 'rb') as f:
            dyn2stat_data = io_functions.specific_functions.load_pickle(f)
        with open(self.path_reference + 'dyn2stat_' + self.job_name + '.pickle', 'rb') as f:
            reference_dyn2stat_data = io_functions.specific_functions.load_pickle(f)
        assert self.compare_dictionaries(dyn2stat_data, reference_dyn2stat_data), "dyn2stat does NOT match reference"
            
    
    def test_postprocessing(self):
        self.set_path()
        # Here you launch the Loads Kernel with your job
        k = kernel.Kernel(self.job_name, pre=False, main=False, post=True, parallel=False,
                          path_input=self.path_input,
                          path_output=self.path_output)
        k.run()
        
        # do comparisons
        logging.info('Comparing cirt_trimcases with reference')
        with open(self.path_output + 'crit_trimcases_' + self.job_name + '.csv', 'r') as f:
            lines = f.readlines()
        with open(self.path_reference + 'crit_trimcases_' + self.job_name + '.csv', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "crit_trimcases do NOT match reference"
        
        logging.info('Comparing nodalloads with reference')
        with open(self.path_output + 'nodalloads_' + self.job_name + '.bdf_Pg', 'r') as f:
            lines = f.readlines()
        with open(self.path_reference + 'nodalloads_' + self.job_name + '.bdf_Pg', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "nodalloads do NOT match reference"
        
        logging.info('Comparing subcases with reference')
        with open(self.path_output + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r') as f:
            lines = f.readlines()
        with open(self.path_reference + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "subcases do NOT match reference"


