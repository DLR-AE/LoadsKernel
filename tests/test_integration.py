
# Launch script for the Loads Kernel
import sys, logging, pytest, subprocess, shlex, os

"""
This section sets the environment for the gitlab-runner, which uses the functional account 'f_jwsb'.
First, we  add the location of the Loads Kernel to the python path.
Second, for the multiprocessing test case, we add MPI to the environment, which is then handed over in the subprocess call.
"""
my_env = {**os.environ, 
          'PATH': '/work/voss_ar/Software/mpich-3.4.2/bin:' + os.environ['PATH'],
          'LD_LIBRARY_PATH': '/work/voss_ar/Software/mpich-3.4.2/lib:',}

from loadskernel import program_flow
import loadskernel.io_functions as io_functions
from helper_functions import HelperFunctions

"""
For the following tests, the loads-kernel-examples and the loads-kernel-reference-results are 
used, which are located in dedictaed repositories. The examples are cloned by the CI-Pipeline.
The reference results are not updated automatically during testing, this has to be done manually. 
Finally, a tempory directory is used for the outputs in order to avoid pollution of the 
repositories.
"""
path_examples  = './loads-kernel-examples/'
path_reference = '/work/voss_ar/loads-kernel-reference-results/'

@pytest.fixture(scope='class')
def get_test_dir(tmpdir_factory):
    test_dir = tmpdir_factory.mktemp('output')
    test_dir = io_functions.specific_functions.check_path(test_dir)
    return str(test_dir)

class TestDiscus2c(HelperFunctions):
    job_name = 'jcl_Discus2c'
    path_input = os.path.join(path_examples, 'Discus2c', 'JCLs')
 
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
        reference_model = io_functions.specific_functions.load_model(self.job_name, path_reference)
        # Running the test in a temporary directory means the path_output changes constantly and can't be compared.
        del model.path_output
        assert self.compare_dictionaries(model.__dict__, reference_model.__dict__), "model does NOT match reference"
 
    def test_mainprocessing_results(self, get_test_dir):
        # do comparisons
        logging.info('Comparing response with reference')
        responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, get_test_dir)
        reference_responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, path_reference)
        assert self.compare_lists(responses, reference_responses), "response does NOT match reference"
        
        logging.info('Comparing monstations with reference')
        monstations = io_functions.specific_functions.load_hdf5(get_test_dir + 'monstations_' + self.job_name + '.hdf5')
        reference_monstations = io_functions.specific_functions.load_hdf5(path_reference + 'monstations_' + self.job_name + '.hdf5')
        assert self.compare_dictionaries(monstations, reference_monstations), "monstations do NOT match reference"
 
        # do comparisons
        logging.info('Comparing dyn2stat with reference')
        dyn2stat_data = io_functions.specific_functions.load_hdf5(get_test_dir + 'dyn2stat_' + self.job_name + '.hdf5')
        reference_dyn2stat_data = io_functions.specific_functions.load_hdf5(path_reference + 'dyn2stat_' + self.job_name + '.hdf5')
        assert self.compare_dictionaries(dyn2stat_data, reference_dyn2stat_data), "dyn2stat does NOT match reference"
 
    def test_postprocessing_results(self, get_test_dir):
        # do comparisons
        logging.info('Comparing crit_trimcases with reference')
        with open(get_test_dir + 'crit_trimcases_' + self.job_name + '.csv', 'r') as f:
            lines = f.readlines()
        with open(path_reference + 'crit_trimcases_' + self.job_name + '.csv', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "crit_trimcases do NOT match reference"
 
        logging.info('Comparing subcases with reference')
        with open(get_test_dir + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r') as f:
            lines = f.readlines()
        with open(path_reference + 'nodalloads_' + self.job_name + '.bdf_subcases', 'r') as f:
            reference_lines = f.readlines()
        assert self.compare_lists(lines, reference_lines), "subcases do NOT match reference"
 
class TestDiscus2cNonlinSteady(TestDiscus2c):
    job_name = 'jcl_Discus2c_nonlin_steady'
    path_input = os.path.join(path_examples, 'Discus2c', 'JCLs')
 
class TestDiscus2cTimedom(TestDiscus2c):
    job_name = 'jcl_Discus2c_timedom'
    path_input = os.path.join(path_examples, 'Discus2c', 'JCLs')
     
class TestDiscus2cB2000(TestDiscus2c):
    job_name = 'jcl_Discus2c_B2000'
    path_input = os.path.join(path_examples, 'Discus2c', 'JCLs')
  
class TestAllegraTimedom(TestDiscus2c):
    job_name = 'jcl_ALLEGRA_timedom'
    path_input = os.path.join(path_examples, 'Allegra', 'JCLs')
  
class TestAllegraFreqdom(TestDiscus2c):
    job_name = 'jcl_ALLEGRA_freqdom'
    path_input = os.path.join(path_examples, 'Allegra', 'JCLs')
  
class TestAllegraFlutter(HelperFunctions):
    job_name = 'jcl_ALLEGRA_flutter'
    path_input = os.path.join(path_examples, 'Allegra', 'JCLs')
      
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
        reference_model = io_functions.specific_functions.load_model(self.job_name, path_reference)
        # Running the test in a temporary directory means the path_output changes constantly and can't be compared.
        del model.path_output
        assert self.compare_dictionaries(model.__dict__, reference_model.__dict__), "model does NOT match reference"
      
    def test_mainprocessing_results(self, get_test_dir):
        # do comparisons
        logging.info('Comparing response with reference')
        responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, get_test_dir)
        reference_responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, path_reference)
        assert self.compare_lists(responses, reference_responses), "response does NOT match reference"
 
class TestAllegraLimitTurbulence(TestAllegraFlutter):
    job_name = 'jcl_ALLEGRA_limitturbulence'
    path_input = os.path.join(path_examples, 'Allegra', 'JCLs')
 
class TestHAPO6Trim(TestDiscus2c):
    job_name = 'jcl_HAP-O6'
    path_input = os.path.join(path_examples, 'HAP-O6', 'JCLs')
 
class TestHAPO6Derivatives(TestAllegraFlutter):
    job_name = 'jcl_HAP-O6_derivatives'
    path_input = os.path.join(path_examples, 'HAP-O6', 'JCLs')
 
class TestHAPO6StateSpaceSystem(TestAllegraFlutter):
    job_name = 'jcl_HAP-O6_sss'
    path_input = os.path.join(path_examples, 'HAP-O6', 'JCLs')

class TestDiscus2cParallelProcessing(HelperFunctions):
    job_name = 'jcl_Discus2c_parallelprocessing'
    path_input = os.path.join(path_examples, 'Discus2c', 'JCLs')
    
    def test_preprocessing_functional_via_command_line_interface(self, get_test_dir):
        # Here we us the command line interface
        args = shlex.split('python ./loadskernel/program_flow.py --job_name %s \
            --pre True --main False --post False \
            --path_input %s --path_output %s' % (self.job_name, self.path_input, get_test_dir))
        out = subprocess.run(args, env=my_env)
        assert out.returncode == 0, "subprocess failed: " + str(args)

    def test_mainprocessing_functional_via_command_line_interface(self, get_test_dir):
        # Here we us the command line interface
        args = shlex.split('mpiexec -n 2 python ./loadskernel/program_flow.py --job_name %s \
            --pre False --main True --post False \
            --path_input %s --path_output %s' % (self.job_name, self.path_input, get_test_dir))
        out = subprocess.run(args, env=my_env)
        assert out.returncode == 0, "subprocess failed: " + str(args)
    
    def test_preprocessing_results(self, get_test_dir):
        # do comparisons
        logging.info('Comparing model with reference')
        model = io_functions.specific_functions.load_model(self.job_name, get_test_dir)
        reference_model = io_functions.specific_functions.load_model(self.job_name, path_reference)
        # Running the test in a temporary directory means the path_output changes constantly and can't be compared.
        del model.path_output
        assert self.compare_dictionaries(model.__dict__, reference_model.__dict__), "model does NOT match reference"
    
    def test_mainprocessing_results(self, get_test_dir):
        # do comparisons
        logging.info('Comparing response with reference')
        responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, get_test_dir)
        reference_responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, path_reference)
        assert self.compare_lists(responses, reference_responses), "response does NOT match reference"

        logging.info('Comparing monstations with reference')
        monstations = io_functions.specific_functions.load_hdf5(get_test_dir + 'monstations_' + self.job_name + '.hdf5')
        reference_monstations = io_functions.specific_functions.load_hdf5(path_reference + 'monstations_' + self.job_name + '.hdf5')
        assert self.compare_dictionaries(monstations, reference_monstations), "monstations do NOT match reference"

        # do comparisons
        logging.info('Comparing dyn2stat with reference')
        dyn2stat_data = io_functions.specific_functions.load_hdf5(get_test_dir + 'dyn2stat_' + self.job_name + '.hdf5')
        reference_dyn2stat_data = io_functions.specific_functions.load_hdf5(path_reference + 'dyn2stat_' + self.job_name + '.hdf5')
        assert self.compare_dictionaries(dyn2stat_data, reference_dyn2stat_data), "dyn2stat does NOT match reference"
