# Launch script for the Loads Kernel

import sys, os
cwd = os.getcwd()
sys.path.append(cwd)
# Here you add the location of the Loads Kernel 
sys.path.append("/scratch/loads-kernel/kernel")
import kernel



# Here you launch the Loads Kernel with your job

# kernel.run_kernel('jcl_DLR_F19_ma',pre=False, main=True, post=True, test=True, path_output='/scratch/Vergleichsfall53/')
# kernel.run_kernel('jcl_DLR_F19_manloads_ae', pre=False, main=True, post=True, path_output='/scratch/test')
# kernel.run_kernel('jcl_DLR_F19_aerodb', pre=False, main=False, post=False, test=True, path_output='/scratch/vergleich_aerodb_vlm')
# kernel.run_kernel('jcl_DLR_F19_aerodb_noalpha', pre=False, main=True, post=False, test=False, path_output='/scratch/vergleich_aerodb_vlm')
# kernel.run_kernel('jcl_DLR_F19_aerodb_noalpha_fine', pre=False, main=True, post=True, test=False, path_output='/scratch/vergleich_aerodb_vlm')
# kernel.run_kernel('jcl_DLR_F19_guyan', pre=True, main=False, post=False, path_input=cwd, path_output='/scratch/test/')


# kernel.run_kernel('jcl_ALLEGRA_CFD', pre=False, main=True, post=True, path_input=cwd, path_output='/scratch/test')
# kernel.run_kernel('jcl_ALLEGRA_steady', pre=False, main=True, post=True, path_input=cwd, path_output='/scratch/test_unsteady')
# kernel.run_kernel('jcl_ALLEGRA_unsteady', pre=False, main=False, post=True, path_input=cwd, path_output='/scratch/test_unsteady')
# kernel.run_kernel('jcl_ALLEGRA_unsteady_accu', pre=False, main=True, post=True, path_input=cwd, path_output='/scratch/test_unsteady')
# kernel.run_kernel('jcl_ALLEGRA_unsteady_corrected', pre=True, main=True, post=True, path_input=cwd, path_output='/scratch/test_unsteady')
# kernel.run_kernel('jcl_ALLEGRA_gust_animation', pre=False, main=False, post=True, path_input=cwd, path_output='/scratch/ALLEGRA_gust_animation')


kernel.run_kernel('jcl_Discus2c_unsteady', pre=False, main=True, post=True, test=False, path_input=cwd, path_output='/scratch/Discus2c_LoadsKernel/')
# kernel.run_kernel('jcl_Discus2c_unsteady_corr', pre=False, main=True, post=True, test=False, path_input=cwd, path_output='/scratch/Discus2c_LoadsKernel/')
# kernel.run_kernel('jcl_Discus2c', pre=False, main=True, post=True, test=False, path_input=cwd, path_output='/scratch/Discus2c_LoadsKernel/')
# kernel.run_kernel('jcl_Discus2c_corr', pre=False, main=True, post=True, path_input=cwd, path_output='/scratch/Discus2c_LoadsKernel/')
# kernel.run_kernel('jcl_Discus2c_akro', pre=False, main=True, post=True, path_input=cwd, path_output='/scratch/Discus2c_LoadsKernel/')

# kernel.run_kernel('jcl_MULDICON',       pre=False, main=True, post=True, path_input=cwd, path_output='/scratch/MULDICON_LoadsKernel/')
# kernel.run_kernel('jcl_MULDICON_pratt', pre=False, main=True,  post=True, path_input=cwd, path_output='/scratch/MULDICON_LoadsKernel/')
# kernel.run_kernel('jcl_MULDICON_gust',  pre=False, main=True, post=True, path_input=cwd, path_output='/scratch/MULDICON_LoadsKernel/')
# kernel.run_kernel('jcl_MULDICON_gust_subcase36',  pre=False, main=True, post=True, path_input=cwd, path_output='/scratch/MULDICON_LoadsKernel/')

