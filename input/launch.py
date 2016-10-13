# Launch script for the Loads Kernel

import sys, os
cwd = os.getcwd()
sys.path.append(cwd)
# Here you add the location of the Loads Kernel 
sys.path.append("../kernel")
import kernel



# Here you launch the Loads Kernel with your job
kernel.run_kernel('jcl_your_aircraft', pre=True, main=False, post=False, path_output='/scratch/test', parallel=False)
