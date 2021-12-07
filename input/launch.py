# Launch script for the Loads Kernel

import sys

# Here you add the location of the Loads Kernel, in case you didn't install the code.
#sys.path.append("../../loads-kernel")
#sys.path.append("../../panel-aero")
sys.path.append("/opt/tau/bin/py_el")

from loadskernel import program_flow

# Here you launch the Loads Kernel with your job
k = program_flow.Kernel('jcl_Discus2c', pre=True, main=True, post=True, test=False, parallel=False,
                  path_input='../../loads-kernel-examples/Discus2c/JCLs',
                  path_output='../../loads-kernel-examples/Discus2c/output')
k.run()
