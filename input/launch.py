# Launch script for the Loads Kernel

import sys

from loadskernel import kernel
# from compare import compare
# from modelviewer import modelviewer

# Here you add the location of the Loads Kernel
# sys.path.append("/scratch/loads-kernel")
sys.path.append("/opt/tau/bin/py_el")

# Here you launch the Loads Kernel with your job

# k = kernel.Kernel('jcl_XRF1_vlm_ll3_successful', pre=True, main=False, post=False,
#                   path_input='/scratch/XRF1_LoadsKernel/JCLs',
#                   path_output='/scratch/XRF1_LoadsKernel')
# k = kernel.Kernel('jcl_XRF1_cfd_ll3_all', pre=False, main=False, restart=False, post=True,
#                   path_input='/scratch/XRF1_LoadsKernel/JCLs',
#                   path_output='/scratch/XRF1_LoadsKernel')
# k = kernel.Kernel('jcl_XRF1_cfd_ll2_upwind', pre=False, main=False, restart=False, post=True,
#                   path_input='/scratch/XRF1_LoadsKernel/JCLs',
#                   path_output='/scratch/XRF1_LoadsKernel')


# k = kernel.Kernel('jcl_Discus2c_CoFE', pre=True, main=True, post=True, test=False, parallel=False,
#                   path_input='/scratch/Discus2c/JCLs',
#                   path_output='/scratch/test')

# k = kernel.Kernel('jcl_HAP-C1', pre=False, main=False, post=True, test=False, parallel=True,
#                   path_input='/scratch/HAP_workingcopy/JCLs',
#                   path_output='/scratch/HAP_LoadsKernel')

# k = kernel.Kernel('jcl_HAP-C0_gust', pre=True, main=True, post=True, test=False, parallel=False,
#                   path_input='/scratch/HAP_workingcopy/JCLs',
#                   path_output='/scratch/HAP_LoadsKernel')

# k = kernel.Kernel('jcl_HAP-C0_stability', pre=False, statespace=False, main=False, post=True, test=False, parallel=False,
#                   path_input='/scratch/HAP_workingcopy/JCLs',
#                   path_output='/scratch/HAP_LoadsKernel')

# k = kernel.Kernel('jcl_ACFA', pre=False, main=True, post=True,
#                   path_input='/scratch/ACFA_LoadsKernel/input',
#                   path_output='/scratch/ACFA_LoadsKernel')

# k = kernel.Kernel('jcl_MULDICON_ll3_cfd', pre=False, main=False, restart=False, post=False, test=True,
#                   path_input='/scratch/MULDICON_LoadsKernel_cfd/input',
#                   path_output='/scratch/MULDICON_LoadsKernel_cfd')
k = kernel.Kernel('jcl_MULDICON_ll3_rans', pre=True, main=False, restart=False, post=False, test=False,
                  path_input='/scratch/MULDICON_LoadsKernel_cfd/input',
                  path_output='/scratch/MULDICON_LoadsKernel_cfd')

# k = kernel.Kernel('FERMATC6', pre=True, main=False, restart=False, post=False, test=False,
#                   path_input='/scratch/Fermat',
#                   path_output='/scratch/Fermat')
k.run()

# Loads Compare
# c = compare.Compare()
# c.run()

# Model Viewer
# m = modelviewer.Modelviewer()
# m.run()
