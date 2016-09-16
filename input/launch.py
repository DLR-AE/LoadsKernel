# Launch script for the Loads Kernel

import sys, os
cwd = os.getcwd()
sys.path.append(cwd)
# Here you add the location of the Loads Kernel 
sys.path.append("../kernel")
import kernel_parallel



# Here you launch the Loads Kernel with your job


kernel_parallel.run_kernel('jcl_MULDICON_gust', pre=False, main=True, post=True, path_output='/scratch/test_parallel')
