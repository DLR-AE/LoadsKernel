# Launch script for the Loads Kernel

import sys

from kernel import kernel
from compare import compare
from modelviewer import modelviewer

# Here you add the location of the Loads Kernel
sys.path.append("/scratch/loads-kernel")
sys.path.append("/opt/tau/bin/py_el")

# Here you launch the Loads Kernel with your job
k = kernel.Kernel('jcl_Discus2c_CoFE', pre=True, main=True, post=True, test=False, parallel=False,
                  path_input='/scratch/Discus2c/JCLs',
                  path_output='/scratch/test')
k.run()

# Loads Compare
c = compare.Compare()
c.run()

# Model Viewer
m = modelviewer.Modelviewer()
m.run()
