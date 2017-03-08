# Launch script for the Loads Kernel

import sys, os
cwd = os.getcwd()
sys.path.append(cwd)
# Here you add the location of the Loads Kernel 
sys.path.append("../kernel")
import kernel



# Here you launch the Loads Kernel with your job
#kernel.run_kernel('jcl_MULDICON_maneuver_loadsloop0', pre=False, main=False, post=True, path_input='/scratch/MULDICON_workingcopy/jcl', path_output='/scratch/MULDICON_LoadsKernel', parallel=True)
# kernel.run_kernel('jcl_MULDICON_gust_loadsloop0', pre=False, main=False, post=True, path_input='/scratch/MULDICON_workingcopy/jcl', path_output='/scratch/MULDICON_LoadsKernel', parallel=True)
# kernel.run_kernel('jcl_MULDICON_lg_loadsloop0', pre=False, main=False, post=True, path_input='/scratch/MULDICON_workingcopy/jcl', path_output='/scratch/MULDICON_LoadsKernel', parallel=True)

# kernel.run_kernel('jcl_MULDICON_maneuver_loadsloop1', pre=False, main=False, post=True, path_input='/scratch/MULDICON_workingcopy/jcl', path_output='/scratch/MULDICON_LoadsKernel', parallel=True)
# kernel.run_kernel('jcl_MULDICON_gust_loadsloop1', pre=False, main=False, post=True, path_input='/scratch/MULDICON_workingcopy/jcl', path_output='/scratch/MULDICON_LoadsKernel', parallel=True)
# kernel.run_kernel('jcl_MULDICON_lg_loadsloop1', pre=False, main=False, post=True, path_input='/scratch/MULDICON_workingcopy/jcl', path_output='/scratch/MULDICON_LoadsKernel', parallel=True)

# kernel.run_kernel('jcl_MULDICON_maneuver_loadsloop2', pre=False, main=True, post=True, path_input='/scratch/MULDICON_workingcopy/jcl', path_output='/scratch/MULDICON_LoadsKernel', parallel=6)
# kernel.run_kernel('jcl_MULDICON_gust_loadsloop2', pre=False, main=False, post=True, path_input='/scratch/MULDICON_workingcopy/jcl', path_output='/scratch/MULDICON_LoadsKernel', parallel=True)
# kernel.run_kernel('jcl_MULDICON_lg_loadsloop2', pre=False, main=False, post=True, path_input='/scratch/MULDICON_workingcopy/jcl', path_output='/scratch/MULDICON_LoadsKernel', parallel=4)


# kernel.run_kernel('jcl_MULDICON_ll3_gust', pre=False, main=False, post=True, path_output='/scratch/MULDICON_stability', parallel=4)
kernel.run_kernel('jcl_MULDICON_ll3', pre=False, statespace=True, post=False, path_output='/scratch/MULDICON_stability', parallel=4)


# kernel.run_kernel('jcl_MULDICON_gust_loadsloop0', pre=False, main=True, post=True, path_input='/scratch/test', path_output='/scratch/test', parallel=True)
# kernel.run_kernel('jcl_MULDICON_maneuver_loadsloop0', pre=False, main=False, post=True, path_input='/scratch/test', path_output='/scratch/test', parallel=True)
