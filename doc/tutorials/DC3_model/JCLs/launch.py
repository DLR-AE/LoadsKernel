from loadskernel import program_flow

# Here you launch the Loads Kernel with your job
k = program_flow.Kernel('jcl_dc3_gust_H23', pre=True, main=True, post=True, test=False,
                        path_input='../JCLs',
                        path_output='../../DC3_results')
k.run()
