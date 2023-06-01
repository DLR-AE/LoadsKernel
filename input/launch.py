from loadskernel import program_flow

# Here you launch the Loads Kernel with your job
k = program_flow.Kernel('jcl_Discus2c', pre=True, main=True, post=True, test=False,
                  path_input='../../loads-kernel-examples/Discus2c/JCLs',
                  path_output='../../loads-kernel-examples/Discus2c/output')
k.run()
