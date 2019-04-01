# Launch script for the Loads Kernel

import sys

# Here you add the location of the Loads Kernel
sys.path.append("../../loads-loadskernel")
sys.path.append("/opt/tau/bin/py_el")

from loadskernel import kernel
from compare import compare
from modelviewer import modelviewer


# Here you launch the Loads Kernel with your job
k = kernel.Kernel('jcl_Discus2c', pre=True, main=True, post=True, test=False, parallel=False,
                  path_input='../../loads-loadskernel-examples/Discus2c/JCLs',
                  path_output='../../loads-loadskernel-examples/Discus2c/output')
k.run()

# Loads Compare
c = compare.Compare()
c.run()

# Model Viewer
m = modelviewer.Modelviewer()
m.run()
