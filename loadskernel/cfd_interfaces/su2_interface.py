
import numpy as np
import logging, os, subprocess, shlex, sys, platform, time

import loadskernel.cfd_interfaces.meshdefo as meshdefo
from loadskernel.cfd_interfaces.mpi_helper import setup_mpi

try:
    import SU2, pysu2
except:
    pass

class SU2Interface(object):
    
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

    def prepare_meshdefo(self, Uf, Ux2):
        defo = meshdefo.meshdefo(self.jcl, self.model)
        defo.init_deformations()
        defo.Uf(Uf, self.trimcase)
        defo.Ux2(Ux2)
        defo.set_deformations(self.FluidSolver)
    
    def update_para(self, uvwpqr):
        """
        In this section, the parameter file is updated. So far, I haven't found a way to do this via pysu2.
        This also means that the solver must be initialized with the new parameter file every time.
        """
        # derive name of para file for this subcase
        para_filename = self.jcl.aero['para_path']+'para_subcase_{}'.format(self.trimcase['subcase'])
        if self.myid == 0:
            # read all existing parameters
            config = SU2.io.Config(para_filename)
            if self.first_execution:
                # set general parameters, which don't change over the course of the CFD simulation, so they are only updated 
                # for the first execution
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
                # activate grid movement
                config['GRID_MOVEMENT'] = 'STEADY_TRANSLATION'
                config['MOTION_ORIGIN'] = '{} {} {}'.format(self.model.mass['cggrid'][self.i_mass]['offset'][0,0],
                                                            self.model.mass['cggrid'][self.i_mass]['offset'][0,1],
                                                            self.model.mass['cggrid'][self.i_mass]['offset'][0,2])
                config['MACH_MOTION'] = self.trimcase['Ma']
                # set the translational velocities to zero, just to make sure...
                config['TRANSLATION_RATE'] = '{} {} {}'.format(0.0, 0.0, 0.0)
                # there is no restart for the first execution
                config['RESTART_SOL'] = 'NO'
            else: 
                config['RESTART_SOL'] = 'YES'
            # update the translational velocities via angle of attack and side slip, given in degree
            # using only the translational velocities resulted in NaNs for the nodal forces
            u, v, w = uvwpqr[:3]
            # with alpha = np.arctan(w/u) and beta = np.arctan(v/u)
            config['AOA']            = np.arctan(w/u)/np.pi*180.0
            config['SIDESLIP_ANGLE'] = np.arctan(v/u)/np.pi*180.0
            # rotational velocities, given in rad/s in the CFD coordinate system (aft-right-up) ??
            p, q, r = uvwpqr.dot(self.model.mass['PHInorm_cg'][self.i_mass])[:3]
            config['ROTATION_RATE'] = '{} {} {}'.format(p, q, r)
            # do the update
            # config.write() maintains the original layout of the file but doesn't add new parameters
            # config.dump() writes all parameters in a weird order, including default values
            config.dump(para_filename)
            logging.debug("Parameters updated.")
        # make sure that all process wait until the new parameter file is written
        self.comm.barrier()
        # then initialize SU2 on all processes
        self.FluidSolver = pysu2.CSinglezoneDriver(para_filename, 1, self.comm)
        
        
    def run_solver(self):
        logging.debug('This is process {} an I wait for the barrier in "run_solver()"'.format(self.myid))
        self.comm.barrier()
        logging.debug('Launch SU2 computation...')
        # starts timer
        t_start = time.time()
        # run solver
        self.FluidSolver.Preprocess(0)
        self.FluidSolver.Run()
        self.FluidSolver.Postprocess()
        # write outputs and restart file(s)
        self.FluidSolver.Monitor(0) 
        self.FluidSolver.Output(0)
        self.comm.barrier()
        if self.first_execution == True:
            self.first_execution = False
        logging.debug("Computation successfully performed in {} seconds.".format(time.time() - t_start))
        
    def get_last_solution(self):
        logging.debug('Start recovery of nodal loads from SU2')
        t_start = time.time()
        Pcfd = np.zeros(self.model.cfdgrid['n']*6)
        solver_all_moving_markers = np.array(self.FluidSolver.GetAllDeformMeshMarkersTag())
        solver_marker_ids = self.FluidSolver.GetAllBoundaryMarkers()
        # Die Oberflächenmarker und die Partitionierung stimmen in der Regel nicht überein. 
        # Daher muss geschaut werden, ob der momentane MPI-Knoten überhaupt den Oberflächenmarker enthält.
        has_moving_marker = [marker in solver_marker_ids.keys() for marker in solver_all_moving_markers]
        
        for marker in solver_all_moving_markers[has_moving_marker]:
            solver_marker_id = solver_marker_ids[marker]
            n_vertices = self.FluidSolver.GetNumberVertices(solver_marker_id)
            for i_vertex in range(n_vertices):
                fxyz = self.FluidSolver.GetFlowLoad(solver_marker_id, i_vertex)
                GlobalIndex = self.FluidSolver.GetVertexGlobalIndex(solver_marker_id, i_vertex)
                pos = self.model.cfdgrid['set'][np.where(GlobalIndex == self.model.cfdgrid['ID'])[0],:3]
                Pcfd[pos] += fxyz

        logging.debug('All nodal loads received and sorted in {} sec.'.format(time.time() - t_start))
        
        return Pcfd
    
    def prepare_initial_solution(self):   
        pass