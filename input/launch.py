# Launch script for the Loads Kernel

import sys, os
cwd = os.getcwd()
sys.path.append(cwd)
# Here you add the location of the Loads Kernel 
sys.path.append("../kernel")
import kernel



# Here you launch the Loads Kernel with your job

#kernel.run_kernel('jcl_DLR_F19_singlesubcase',main=True, path_output='/scratch/polar/')
#kernel.run_kernel('jcl_DLR_F19_vlm', pre=False, main=True, post=True, path_output='/scratch/vergleich_aerodb_vlm')
kernel.run_kernel('jcl_DLR_F19_aerodb', pre=True, main=True, post=True, path_output='/scratch/vergleich_aerodb_vlm')
#kernel.run_kernel('jcl_DLR_F19_aerodb_nocs', pre=False, main=True, post=True, path_output='/scratch/vergleich_aerodb_vlm')
#kernel.run_kernel('jcl_ALLEGRA_steady', pre=False, main=False, post=True, path_input=cwd, path_output='/scratch/test_unsteady')
#kernel.run_kernel('jcl_ALLEGRA_unsteady', pre=False, main=True, post=True, path_input=cwd, path_output='/scratch/test_unsteady')
#kernel.run_kernel('jcl_ALLEGRA_unsteady', pre=True, main=False, post=False, path_input=cwd, path_output='/scratch/test_unsteady')
