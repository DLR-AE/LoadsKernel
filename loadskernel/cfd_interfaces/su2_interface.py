
import numpy as np
import logging, os, subprocess, shlex, sys, platform, time, copy

import loadskernel.cfd_interfaces.meshdefo as meshdefo
import loadskernel.spline_functions as spline_functions
import loadskernel.build_splinegrid as build_splinegrid
from loadskernel.cfd_interfaces.mpi_helper import setup_mpi

try:
    import SU2, pysu2
except:
    pass

class SU2Interface(meshdefo.Meshdefo):
    
    def __init__(self, solution):
        self.model      = solution.model
        self.jcl        = solution.jcl
        self.trimcase   = solution.trimcase
        # get some indices
        self.i_atmo     = self.model.atmo['key'].index(self.trimcase['altitude'])
        self.i_mass     = self.model.mass['key'].index(self.trimcase['mass'])
        # set switch for first execution
        self.first_execution = True
        
        # Check if the correct MPI implementation is used, SU2 requires MPICH
        args_version = shlex.split('mpiexec --help')
        # With Python 3, we can use subprocess.check_output() to get the output from a subprocess
        output = subprocess.check_output(args_version).decode('utf-8')
        if str.find(output, 'mpich.org') == -1:
            logging.error('Wrong MPI implementation detected (SU2 requires MPICH).')
        else:
            self.have_mpi, self.comm, self.myid = setup_mpi(debug=False)
        
        # Check if pysu2 was imported successfully, see try/except statement in the import section.
        if "pysu2" in sys.modules and "SU2" in sys.modules:
            # make sure that all processes are at the same stage
            self.comm.barrier()
            logging.info('Init CFD interface of type "{}" on MPI process {}.'.format(self.__class__.__name__, self.myid))
        else:
            logging.error('pysu2 was/could NOT be imported! Model equations of type "{}" will NOT work.'.format(self.jcl.aero['method']))
        self.FluidSolver = None
        
    def prepare_meshdefo(self, Uf, Ux2):
        """
        In this function, we run all the steps to necessary perform the mesh deformation.
        """
        # There may be CFD partitions which have no deformation markers. In that case, there is nothing to do.
        if self.local_mesh['n'] > 0:
            # Initialize the surface deformation vector with zeros
            self.Ucfd = np.zeros(self.local_mesh['n']*6)
            # These two functions are inherited from the original Meshdefo class
            # Add flexible deformations
            self.Uf(Uf, self.trimcase)
            # Add control surface deformations
            self.Ux2(Ux2)
            # Communicate the deformation of the local mesh to the CFD solver
            self.set_deformations()
        logging.debug('This is process {} and I wait for the mpi barrier in "prepare_meshdefo()"'.format(self.myid))
        self.comm.barrier()
    
    def set_deformations(self):
        """
        Communicate the change of coordinates of the fluid interface to the fluid solver.
        Prepare the fluid solver for mesh deformation.
        """
        logging.info('Sending surface deformations to SU2.')
        solver_all_moving_markers = np.array(self.FluidSolver.GetAllDeformMeshMarkersTag())
        solver_marker_ids = self.FluidSolver.GetAllBoundaryMarkers()
        # The surface marker and the partitioning of the solver usually don't agree.
        # Thus, it is necessary to figure out if the partition of the current mpi process has
        # a node that belongs to a moving surface marker.
        has_moving_marker = [marker in solver_marker_ids.keys() for marker in solver_all_moving_markers]
        # In SU2, markers are tracked by their name, not by their ID, and the ID might differ.
        lk_markers = [cfdgrid['desc'] for cfdgrid in self.model.cfdgrids]
        
        for marker in solver_all_moving_markers[has_moving_marker]:
            solver_marker_id = solver_marker_ids[marker]
            lk_marker_id = lk_markers.index(marker)
            
            # Check:  marker == self.cfdgrids[lk_marker_id]['desc']
            n_vertices = self.FluidSolver.GetNumberVertices(solver_marker_id)
            # Check (only for one domain): n_vertices == self.cfdgrids[lk_marker_id]['n']
            for i_vertex in range(n_vertices):
                GlobalIndex = self.FluidSolver.GetVertexGlobalIndex(solver_marker_id, i_vertex)
                # Check: GlobalIndex in self.cfdgrids[lk_marker_id]['ID']
                pos = self.local_mesh['set'][self.local_mesh['GlobalIndex'].index(GlobalIndex),:3]
                disp_x, disp_y, disp_z = self.Ucfd[pos]
                self.FluidSolver.SetMeshDisplacement(solver_marker_id, i_vertex, disp_x, disp_y, disp_z)
        logging.debug('All surface mesh deformations set.')
    
    def update_para(self, uvwpqr):
        """
        In this section, the parameter file is updated. So far, I haven't found a way to do this via pysu2.
        This also means that the solver must be initialized with the new parameter file every time.
        """
        # derive name of para file for this subcase
        para_filename = self.jcl.aero['para_path']+'para_subcase_{}'.format(self.trimcase['subcase'])
        initialize_su2 = True
        if self.myid == 0:
            # read all existing parameters
            config = SU2.io.Config(para_filename)
            config_old = copy.deepcopy(config)
            if self.first_execution:
                # set general parameters, which don't change over the course of the CFD simulation, so they are only updated 
                # for the first execution
                config['MESH_FILENAME'] = self.jcl.meshdefo['surface']['filename_grid']
                config['RESTART_FILENAME'] = self.jcl.aero['para_path']+'sol/restart_subcase_{}.dat'.format(self.trimcase['subcase'])
                config['SOLUTION_FILENAME'] = self.jcl.aero['para_path']+'sol/restart_subcase_{}.dat'.format(self.trimcase['subcase'])
                config['SURFACE_FILENAME'] = self.jcl.aero['para_path']+'sol/surface_subcase_{}'.format(self.trimcase['subcase'])
                config['VOLUME_FILENAME']  = self.jcl.aero['para_path']+'sol/volume_subcase_{}'.format(self.trimcase['subcase'])
                config['CONV_FILENAME']    = self.jcl.aero['para_path']+'sol/history_subcase_{}'.format(self.trimcase['subcase'])
                # free stream definition
                config['FREESTREAM_TEMPERATURE'] = self.model.atmo['T'][self.i_atmo]
                config['FREESTREAM_DENSITY']     = self.model.atmo['rho'][self.i_atmo]
                config['FREESTREAM_PRESSURE']    = self.model.atmo['p'][self.i_atmo]
                # make sure that the free stream onflow is zero
                config['MACH_NUMBER']    = self.trimcase['Ma']
                # activate grid deformation
                config['DEFORM_MESH'] = 'YES'
                config['MARKER_DEFORM_MESH'] = '( '+', '.join(self.jcl.meshdefo['surface']['markers'])+' )'
                # activate grid movement
                config['GRID_MOVEMENT'] = 'ROTATING_FRAME'
                config['MOTION_ORIGIN'] = '{} {} {}'.format(self.model.mass['cggrid'][self.i_mass]['offset'][0,0],
                                                            self.model.mass['cggrid'][self.i_mass]['offset'][0,1],
                                                            self.model.mass['cggrid'][self.i_mass]['offset'][0,2])
                config['MACH_MOTION'] = self.trimcase['Ma']
                # there is no restart for the first execution
                config['RESTART_SOL'] = 'NO'
            else: 
                config['RESTART_SOL'] = 'YES'
            # update the translational velocities via angle of attack and side slip, given in degree
            # using only the translational velocities resulted in NaNs for the nodal forces
            u, v, w = uvwpqr[:3]
            # with alpha = np.arctan(w/u) and beta = np.arctan(v/u)
            config['AOA']            = np.arctan(w/u)/np.pi*180.0
            # for some reason, the sideslip angle is parsed as as string...
            config['SIDESLIP_ANGLE'] = '{}'.format(np.arctan(v/u)/np.pi*180.0)
            # rotational velocities, given in rad/s in the CFD coordinate system (aft-right-up) ??
            p, q, r = uvwpqr.dot(self.model.mass['PHInorm_cg'][self.i_mass])[3:]
            config['ROTATION_RATE'] = '{} {} {}'.format(p, q, r)
        
            # Find out if the configuration file changed. 
            # If this is the case, then we need to initialize SU2 again.
            parameter_added = [key not in config_old for key in config]
            parameter_changed = [config_old[key] != config[key] for key in config_old]
            if np.any(parameter_added) or np.any(parameter_changed):
                # do the update
                # config.write() maintains the original layout of the file but doesn't add new parameters
                # config.dump() writes all parameters in a weird order, including default values
                config.dump(para_filename)
                logging.info('SU2 parameters updated.')
                initialize_su2 = True
            else:
                logging.info('SU2 parameters are unchanged.')
                initialize_su2 = False
        
        
        # make sure that all process wait until the new parameter file is written
        self.comm.barrier()
        initialize_su2 = self.comm.bcast(initialize_su2, root=0)
        if initialize_su2:
            # then initialize SU2 on all processes
            logging.info('Initializing SU2.')
            self.release_memory()
            self.FluidSolver = pysu2.CSinglezoneDriver(para_filename, 1, self.comm)
            self.get_local_mesh()
        else:
            logging.info('Reusing SU2 instance.')

    def run_solver(self):
        logging.debug('This is process {} and I wait for the mpi barrier in "run_solver()"'.format(self.myid))
        self.comm.barrier()
        logging.info('Launch SU2.')
        # starts timer
        t_start = time.time()
        # run solver
        #self.FluidSolver.StartSolver()
        self.FluidSolver.Preprocess(0)
        self.FluidSolver.Run()
        self.FluidSolver.Postprocess()
        # write outputs and restart file(s)
        self.FluidSolver.Monitor(0) 
        self.FluidSolver.Output(0)
        self.comm.barrier()
        if self.first_execution == True:
            self.first_execution = False
        logging.debug('CFD computation performed in {:.2f} seconds.'.format(time.time() - t_start))
        
    def get_last_solution(self):
        logging.debug('Start recovery of nodal loads from SU2')
        t_start = time.time()
        Pcfd_send = np.zeros(self.model.cfdgrid['n']*6)
        Pcfd_rcv  = np.zeros(( self.comm.Get_size() , self.model.cfdgrid['n']*6))
        solver_all_moving_markers = np.array(self.FluidSolver.GetAllDeformMeshMarkersTag())
        solver_marker_ids = self.FluidSolver.GetAllBoundaryMarkers()
        # The surface marker and the partitioning of the solver usually don't agree.
        # Thus, it is necessary to figure out if the partition of the current mpi process has
        # a node that belongs to a moving surface marker.
        has_moving_marker = [marker in solver_marker_ids.keys() for marker in solver_all_moving_markers]
        
        for marker in solver_all_moving_markers[has_moving_marker]:
            solver_marker_id = solver_marker_ids[marker]
            n_vertices = self.FluidSolver.GetNumberVertices(solver_marker_id)
            for i_vertex in range(n_vertices):
                fxyz = self.FluidSolver.GetFlowLoad(solver_marker_id, i_vertex)
                GlobalIndex = self.FluidSolver.GetVertexGlobalIndex(solver_marker_id, i_vertex)
                pos = self.model.cfdgrid['set'][np.where(GlobalIndex == self.model.cfdgrid['ID'])[0],:3]
                Pcfd_send[pos] += fxyz
        self.comm.barrier()
        self.comm.Allgatherv(Pcfd_send, Pcfd_rcv)
        Pcfd = Pcfd_rcv.sum(axis=0)
        logging.debug('All nodal loads recovered, sorted and gathered in {:.2f} sec.'.format(time.time() - t_start))
        return Pcfd
    
    def prepare_initial_solution(self):   
        pass
    
    def release_memory(self):
        if self.FluidSolver != None:
            self.FluidSolver.Postprocessing()
    
    def get_local_mesh(self):
        """
        Get the local mesh (the partition of this mpi process) of the fluid solver.
        This function is kind of similar to set_deformations() when it comes to looping over all vertices.
        """
        solver_all_moving_markers = np.array(self.FluidSolver.GetAllDeformMeshMarkersTag())
        solver_marker_ids = self.FluidSolver.GetAllBoundaryMarkers()
        # The surface marker and the partitioning of the solver usually don't agree.
        # Thus, it is necessary to figure out if the partition of the current mpi process has
        # a node that belongs to a moving surface marker.
        has_moving_marker = [marker in solver_marker_ids.keys() for marker in solver_all_moving_markers]
        
        # Set-up some helper variables
        tmp_offset = []
        tmp_id = []
        n = 0
        # Loops to get the coordinates of every vertex that belongs the partition of this mpi process
        for marker in solver_all_moving_markers[has_moving_marker]:
            solver_marker_id = solver_marker_ids[marker]
            n_vertices = self.FluidSolver.GetNumberVertices(solver_marker_id)
            n += n_vertices
            for i_vertex in range(n_vertices):
                tmp_id.append( self.FluidSolver.GetVertexGlobalIndex(solver_marker_id, i_vertex) )
                tmp_offset.append( self.FluidSolver.GetInitialMeshCoord(solver_marker_id, i_vertex) )
        
        # Store the local mesh, use a pattern similar to a any other grids
        self.local_mesh = {'GlobalIndex': tmp_id,
                           'offset': np.array(tmp_offset),
                           'set':np.arange(n*6).reshape((n,6)),
                           'n': n
                           }
        logging.debug('This is process {} and my local mesh has a size of {}'.format(self.myid, self.local_mesh['n']))

    def transfer_deformations(self, grid_i, U_i, set_i, rbf_type, surface_spline, support_radius=2.0):
        """
        This function overwrites the original Meshdefo.transfer_deformations().
        This version works on the local mesh of a mpi partition, making the calculation of the 
        mesh deformations faster.
        """
        # build spline matrix
        PHIi_d = spline_functions.spline_rbf(grid_i, set_i, self.local_mesh, '', 
                                             rbf_type=rbf_type, surface_spline=surface_spline, 
                                             support_radius=support_radius, dimensions=[U_i.size, self.local_mesh['n']*6])
        # store deformation of cfdgrid
        self.Ucfd += PHIi_d.dot(U_i)
        del PHIi_d        