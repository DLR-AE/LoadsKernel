import copy
import logging
import os
import sys
import time
import numpy as np

from loadskernel import spline_functions
from loadskernel.cfd_interfaces import meshdefo
from loadskernel.cfd_interfaces.mpi_helper import setup_mpi
from loadskernel.cfd_interfaces.tau_interface import check_para_path, copy_para_file, check_cfd_folders
from loadskernel.grid_trafo import grid_trafo, vector_trafo
from loadskernel.io_functions.data_handling import load_hdf5_dict
from loadskernel.solution_tools import calc_drehmatrix

try:
    import SU2
    import pysu2
except ImportError:
    pass

"""
There are two ways in which the free stream onflow can be realized:
a) using mesh velocities imposed on each grid point (called motion module in Tau)
b) using the farfield (the classical way with alpha and beta)
The grid velocity approach (parameter GRID_MOVEMENT= ROTATING_FRAME) allows the simulation of the free-flying aircraft
including roll, pitch and yaw rates by imposing a corresponding velocity field in addition to the flight velocities u,
v and w. The farfield onflow is set to zero in this scenario. As a disadvantage, the grid velocity approach is not
implemented for unsteady simulations because additional source terms become necessary (as discussed with the SU2 developers)
and there is an interference with the mesh velocities from the elastic deformations, which make an inplementation tricky,
too trick for me.
The farfield onflow is more straight forward and seems to works properly for unsteady simulations. The rigid body motion
can be handled by moving + rotating the whole aircraft in the elastic mesh, which is acceptable for moderate displacements
expected during e.g. a gust encounter. This approach involves more work on the Loads Kernel side, as the surface, the
surface deformations and the aerodynamic forces need to be translated back and forth.
"""


class SU2InterfaceGridVelocity(meshdefo.Meshdefo):

    def __init__(self, solution):
        self.model = solution.model
        self.jcl = solution.jcl
        self.trimcase = solution.trimcase
        self.simcase = solution.simcase
        # load data from HDF5
        self.mass = load_hdf5_dict(self.model['mass'][self.trimcase['mass']])
        self.atmo = load_hdf5_dict(self.model['atmo'][self.trimcase['altitude']])
        self.cggrid = load_hdf5_dict(self.mass['cggrid'])
        self.cfdgrid = load_hdf5_dict(self.model['cfdgrid'])
        self.strcgrid = load_hdf5_dict(self.model['strcgrid'])
        self.splinegrid = load_hdf5_dict(self.model['splinegrid'])
        self.aerogrid = load_hdf5_dict(self.model['aerogrid'])
        self.x2grid = load_hdf5_dict(self.model['x2grid'])
        self.coord = load_hdf5_dict(self.model['coord'])

        self.Djx2 = self.model['Djx2'][()]

        # set switch for first execution
        self.first_execution = True
        # stepwidth for time domain simulation
        if 'dt_integration' in self.simcase:
            self.stepwidth = self.simcase['dt_integration']
        elif 'dt' in self.simcase:
            self.stepwidth = self.simcase['dt']
        else:
            self.stepwidth = None

        # Initialize and check if MPI can be used, SU2 requires MPICH
        self.have_mpi, self.comm, self.status, self.myid = setup_mpi(debug=False)
        if not self.have_mpi:
            logging.error('No MPI detected (SU2 requires MPI)!')

        # Check if pysu2 was imported successfully, see try/except statement in the import section.
        if "pysu2" in sys.modules and "SU2" in sys.modules:
            # make sure that all processes are at the same stage
            self.comm.barrier()
            logging.info('Init CFD interface of type "{}" on MPI process {}.'.format(self.__class__.__name__, self.myid))
        else:
            logging.error('pysu2 was/could NOT be imported! Model equations of type "{}" will NOT work.'.format(
                self.jcl.aero['method']))
        self.FluidSolver = None

        # Set-up file system structure
        if self.myid == 0:
            check_cfd_folders(self.jcl)
            check_para_path(self.jcl)
            copy_para_file(self.jcl, self.trimcase)
        self.para_filename = self.jcl.aero['para_path'] + 'para_subcase_{}'.format(self.trimcase['subcase'])

        # Storage for the euler transofmation of the unsteady interface
        self.XYZ = None
        self.PhiThetaPsi = None

    def prepare_meshdefo(self, Uf, Ux2):
        """
        In this function, we run all the steps to necessary perform the mesh deformation.
        """
        # There may be CFD partitions which have no deformation markers. In that case, there is nothing to do.
        if self.local_mesh['n'] > 0:
            # Initialize the surface deformation vector with zeros
            self.Ucfd = np.zeros(self.local_mesh['n'] * 6)
            # These two functions are inherited from the original Meshdefo class
            # Add flexible deformations
            self.Uf(Uf)
            # Add control surface deformations
            self.Ux2(Ux2)
            # Communicate the deformation of the local mesh to the CFD solver
            self.set_deformations()

    def set_deformations(self):
        """
        Communicate the change of coordinates of the fluid interface to the fluid solver.
        Prepare the fluid solver for mesh deformation.
        """
        logging.info('Sending surface deformations to SU2.')
        for x in range(self.local_mesh['n']):
            self.FluidSolver.SetMarkerCustomDisplacement(self.local_mesh['MarkerID'][x],
                                                         self.local_mesh['VertexIndex'][x],
                                                         self.Ucfd[self.local_mesh['set'][x, :3]])

    def set_grid_velocities(self, uvwpqr):
        """
        Communicate the translational and rotational velocities to the CFD solver, e.g. before a new iteration step.
        The CFD coordinate system is typically aft-right-up and the values are given m/s and in rad/s.
        """
        # translate the translational and rotational velocities into the right coordinate system
        u, v, w = uvwpqr.dot(self.mass['PHInorm_cg'])[:3]
        p, q, r = uvwpqr.dot(self.mass['PHInorm_cg'])[3:]
        # communicate the new values to the CFD solver
        logging.info('Sending translational and rotational velocities to SU2.')
        self.FluidSolver.SetTranslationRate(xDot=u, yDot=v, zDot=w)
        self.FluidSolver.SetRotationRate(rot_x=p, rot_y=q, rot_z=r)

    def update_general_para(self):
        """
        In this section, the parameter file is updated. So far, I haven't found a way to do this via pysu2 for all parameters.
        This also means that the solver must be initialized with the new parameter file in case the file is updated.
        """
        if self.first_execution and self.myid == 0:
            # read all existing parameters
            config = SU2.io.Config(self.para_filename)
            # set general parameters, which don't change over the course of the CFD simulation, so they are only updated
            # for the first execution
            config['MESH_FILENAME'] = self.jcl.meshdefo['surface']['filename_grid']
            config['RESTART_FILENAME'] = self.jcl.aero['para_path'] + 'sol/restart_subcase_{}.dat'.format(
                self.trimcase['subcase'])
            config['SOLUTION_FILENAME'] = self.jcl.aero['para_path'] + 'sol/restart_subcase_{}.dat'.format(
                self.trimcase['subcase'])
            config['SURFACE_FILENAME'] = self.jcl.aero['para_path'] + 'sol/surface_subcase_{}'.format(
                self.trimcase['subcase'])
            config['VOLUME_FILENAME'] = self.jcl.aero['para_path'] + 'sol/volume_subcase_{}'.format(
                self.trimcase['subcase'])
            config['CONV_FILENAME'] = self.jcl.aero['para_path'] + 'sol/history_subcase_{}'.format(
                self.trimcase['subcase'])
            # free stream definition
            config['FREESTREAM_TEMPERATURE'] = self.atmo['T']
            config['FREESTREAM_DENSITY'] = self.atmo['rho']
            config['FREESTREAM_PRESSURE'] = self.atmo['p']
            # update the pressure inlet
            if hasattr(self.jcl, 'pressure_inlet'):
                config['MARKER_INLET'] = '( ' + ', '.join([self.jcl.pressure_inlet['marker'],
                                                           str(self.atmo['T']),
                                                           str(self.atmo['p']),
                                                           str(self.jcl.pressure_inlet['flow_direction'][0]),
                                                           str(self.jcl.pressure_inlet['flow_direction'][1]),
                                                           str(self.jcl.pressure_inlet['flow_direction'][2]), ]) + ' )'
            # make sure that the free stream onflow is zero
            config['MACH_NUMBER'] = 0.0
            # activate grid deformation
            config['DEFORM_MESH'] = 'YES'
            config['MARKER_DEFORM_MESH'] = '( ' + ', '.join(self.jcl.meshdefo['surface']['markers']) + ' )'
            # activate grid movement
            config['GRID_MOVEMENT'] = 'ROTATING_FRAME'
            config['MOTION_ORIGIN'] = '{} {} {}'.format(self.cggrid['offset'][0, 0],
                                                        self.cggrid['offset'][0, 1],
                                                        self.cggrid['offset'][0, 2])
            config['MACH_MOTION'] = self.trimcase['Ma']
            # there is no restart for the first execution
            config['RESTART_SOL'] = 'NO'
            # detrimine cfd solution output
            config['OUTPUT_FILES'] = ['RESTART', 'RESTART_ASCII', 'TECPLOT', 'SURFACE_TECPLOT']
            # set intemediate outputs
            # config['OUTPUT_WRT_FREQ']= 1

            # do the update
            # config.write() maintains the original layout of the file but doesn't add new parameters
            # config.dump() writes all parameters in a weird order, including default values
            config.dump(self.para_filename)
            logging.info('SU2 parameter file updated.')

    def init_solver(self):
        # make sure that all process wait until the new parameter file is written
        self.comm.barrier()
        if self.first_execution:
            logging.info('Initializing SU2.')
            self.release_memory()
            self.FluidSolver = pysu2.CSinglezoneDriver(self.para_filename, 1, self.comm)
            self.get_local_mesh()

    def run_solver(self, i_timestep=0):
        logging.info('Waiting until all processes are ready to perform a coordinated start...')
        self.comm.barrier()
        logging.info('Launch SU2 for time step {}.'.format(i_timestep))
        # start timer
        t_start = time.time()
        # initialize SU2 if this is the first run.
        if self.first_execution:
            self.prepare_initial_solution()
        # run solver
        # self.FluidSolver.StartSolver()
        self.FluidSolver.Preprocess(i_timestep)
        self.FluidSolver.Run()
        self.FluidSolver.Postprocess()
        self.FluidSolver.Update()
        # write outputs and restart file(s)
        self.FluidSolver.Monitor(i_timestep)
        self.FluidSolver.Output(i_timestep)
        self.comm.barrier()

        logging.debug('CFD computation performed in {:.2f} seconds.'.format(time.time() - t_start))

    def get_last_solution(self):
        return self.Pcfd_global()

    def Pcfd_global(self):
        logging.debug('Start recovery of nodal loads from SU2')
        t_start = time.time()
        Pcfd_send = np.zeros(self.cfdgrid['n'] * 6)
        Pcfd_rcv = np.zeros((self.comm.Get_size(), self.cfdgrid['n'] * 6))
        for x in range(self.local_mesh['n']):
            fxyz = self.FluidSolver.GetMarkerFlowLoad(self.local_mesh['MarkerID'][x], self.local_mesh['VertexIndex'][x])
            Pcfd_send[self.local_mesh['set_global'][x, :3]] += fxyz
        self.comm.barrier()
        self.comm.Allgatherv(Pcfd_send, Pcfd_rcv)
        Pcfd = Pcfd_rcv.sum(axis=0)
        logging.debug('All nodal loads recovered, sorted and gathered in {:.2f} sec.'.format(time.time() - t_start))
        return Pcfd

    def prepare_initial_solution(self):
        """
        So far, starting SU2 with different, more robust parameters was not necessary.
        """
        # Change the first_execution flag
        self.first_execution = False

    def release_memory(self):
        # In case SU2 is re-initialized, release the memory taken by the old instance.
        if self.FluidSolver is not None:
            self.FluidSolver.Finalize()

    def get_local_mesh(self):
        """
        Get the local mesh (the partition of this mpi process) of the fluid solver.
        Because the loops over all grid points and the mapping of the local to the gloabl mesh is
        a little time consuming, the information is stored in the following entries:
        GlobalIndex - the ID of the grid point as defined in the global CFD mesh
        MarkerID    - the ID of the marker the grid point belongs to
        VertexIndex - the ID of the grid point with respect to the marker
        set         - the degrees of freedom of the grid point in the local CFD mesh
        set_global  - the degrees of freedom of the grid point in the global CFD mesh
        """
        solver_all_moving_markers = np.array(self.FluidSolver.GetDeformableMarkerTags())
        solver_marker_ids = self.FluidSolver.GetMarkerIndices()
        # The surface marker and the partitioning of the solver usually don't agree.
        # Thus, it is necessary to figure out if the partition of the current mpi process has
        # a node that belongs to a moving surface marker.
        has_moving_marker = [marker in solver_marker_ids.keys() for marker in solver_all_moving_markers]

        # Set-up some helper variables
        tmp_offset = []
        tmp_set_global = []
        tmp_global_id = []
        tmp_marker_id = []
        tmp_vertex_id = []
        n = 0
        # Loops to get the coordinates of every vertex that belongs the partition of this mpi process
        for marker in solver_all_moving_markers[has_moving_marker]:
            solver_marker_id = solver_marker_ids[marker]
            n_vertices = self.FluidSolver.GetNumberMarkerNodes(solver_marker_id)
            n += n_vertices
            for i_vertex in range(n_vertices):
                i_point = self.FluidSolver.GetMarkerNode(solver_marker_id, i_vertex)
                GlobalIndex = self.FluidSolver.GetNodeGlobalIndex(i_point)
                tmp_global_id.append(GlobalIndex)
                tmp_marker_id.append(solver_marker_id)
                tmp_vertex_id.append(i_vertex)
                tmp_offset.append(self.FluidSolver.InitialCoordinates().Get(i_point))
                tmp_set_global.append(self.cfdgrid['set'][np.where(GlobalIndex == self.cfdgrid['ID'])[0], :])

        # Store the local mesh, use a pattern similar to a any other grids
        self.local_mesh = {'GlobalIndex': tmp_global_id,
                           'MarkerID': tmp_marker_id,
                           'VertexIndex': tmp_vertex_id,
                           'CP': np.repeat(0, n),
                           'CD': np.repeat(0, n),
                           'offset': np.array(tmp_offset),
                           'set': np.arange(n * 6).reshape((n, 6)),
                           'set_global': np.array(tmp_set_global).squeeze(),
                           'n': n
                           }
        logging.debug('This is process {} and my local mesh has a size of {}'.format(self.myid, self.local_mesh['n']))

    def transfer_deformations(self, grid_i, U_i, set_i, rbf_type, surface_spline, support_radius=2.0):
        """
        This function overwrites the original Meshdefo.transfer_deformations().
        This version works on the local mesh of a mpi partition, making the calculation of the
        mesh deformations faster.
        """
        logging.info('Transferring deformations to the local CFD surface with {} nodes.'.format(self.local_mesh['n']))
        # build spline matrix
        PHIi_d = spline_functions.spline_rbf(grid_i, set_i, self.local_mesh, '',
                                             rbf_type=rbf_type, surface_spline=surface_spline,
                                             support_radius=support_radius, dimensions=[U_i.size, self.local_mesh['n'] * 6])
        # store deformation of cfdgrid
        self.Ucfd += PHIi_d.dot(U_i)
        del PHIi_d

    def set_euler_transformation(self, XYZ, PhiThetaPsi):
        self.XYZ = XYZ
        self.PhiThetaPsi = PhiThetaPsi


class SU2InterfaceFarfieldOnflow(SU2InterfaceGridVelocity):

    def set_grid_velocities(self, uvwpqr):
        # Do nothing here.
        pass

    def get_last_solution(self):
        return self.Pcfd_body()

    def Pcfd_body(self):
        Pcfd_global = self.Pcfd_global()

        # translate position and euler angles into body coordinate system
        PHInorm_cg = self.mass['PHInorm_cg']
        PhiThetaPsi_body = PHInorm_cg[0:3, 0:3].dot(self.PhiThetaPsi)

        # setting up coordinate system
        coord_tmp = copy.deepcopy(self.coord)
        coord_tmp['ID'] = np.append(coord_tmp['ID'], [1000000, 1000001])
        coord_tmp['RID'] = np.append(coord_tmp['RID'], [0, 0])
        coord_tmp['dircos'] = np.append(coord_tmp['dircos'], [np.eye(3),
                                                              calc_drehmatrix(PhiThetaPsi_body[0],
                                                                              PhiThetaPsi_body[1],
                                                                              PhiThetaPsi_body[2])], axis=0)

        coord_tmp['offset'] = np.append(coord_tmp['offset'], [np.array([0.0, 0.0, 0.0, ]),
                                                              np.array([0.0, 0.0, 0.0, ])], axis=0)
        cfdgrid_tmp = copy.deepcopy(self.cfdgrid)
        cfdgrid_tmp['CD'] = np.repeat(1000001, self.cfdgrid['n'])

        # transform force vector
        Pcfd_body = vector_trafo(cfdgrid_tmp, coord_tmp, Pcfd_global, dest_coord=1000000)
        return Pcfd_body

    def prepare_meshdefo(self, Uf, Ux2):
        """
        In this function, we run all the steps to necessary perform the mesh deformation.
        """
        # There may be CFD partitions which have no deformation markers. In that case, there is nothing to do.
        if self.local_mesh['n'] > 0:
            # Initialize the surface deformation vector with zeros
            self.Ucfd = np.zeros(self.local_mesh['n'] * 6)
            # These two functions are inherited from the original Meshdefo class
            # Add flexible deformations
            self.Uf(Uf)
            # Add control surface deformations
            self.Ux2(Ux2)
            # Add rigid body rotations
            self.Ucfd_rbm_transformation(self.XYZ, self.PhiThetaPsi)
            # Communicate the deformation of the local mesh to the CFD solver
            self.set_deformations()

    def Ucfd_rbm_transformation(self, XYZ, PhiThetaPsi):
        # translate position and euler angles into body coordinate system
        PHInorm_cg = self.mass['PHInorm_cg']
        PhiThetaPsi_body = PHInorm_cg[0:3, 0:3].dot(PhiThetaPsi)
        XYZ_body = PHInorm_cg[0:3, 0:3].dot(XYZ)
        # remove any translation in x-direction
        XYZ_body[0] = 0.0
        # setting up coordinate system
        coord_tmp = copy.deepcopy(self.coord)
        coord_tmp['ID'] = np.append(coord_tmp['ID'], [1000000, 1000001])
        coord_tmp['RID'] = np.append(coord_tmp['RID'], [0, 0])
        coord_tmp['dircos'] = np.append(coord_tmp['dircos'], [calc_drehmatrix(PhiThetaPsi_body[0],
                                                                              PhiThetaPsi_body[1],
                                                                              PhiThetaPsi_body[2]),
                                                              np.eye(3)], axis=0)
        coord_tmp['offset'] = np.append(coord_tmp['offset'], [self.cggrid['offset'][0] + XYZ_body,
                                                              -self.cggrid['offset'][0]], axis=0)

        # apply transformation to local mesh
        local_mesh_tmp = copy.deepcopy(self.local_mesh)
        local_mesh_tmp['CP'] = np.repeat(1000001, self.local_mesh['n'])
        grid_trafo(local_mesh_tmp, coord_tmp, 1000000)

        # calculate rigid deformation vector from transformed mesh
        Ucfd_rigid = np.zeros(self.local_mesh['n'] * 6)
        Ucfd_rigid[self.local_mesh['set'][:, 0]] = local_mesh_tmp['offset'][:, 0] - self.local_mesh['offset'][:, 0]
        Ucfd_rigid[self.local_mesh['set'][:, 1]] = local_mesh_tmp['offset'][:, 1] - self.local_mesh['offset'][:, 1]
        Ucfd_rigid[self.local_mesh['set'][:, 2]] = local_mesh_tmp['offset'][:, 2] - self.local_mesh['offset'][:, 2]

        # transform existing mesh deformation vector
        Ucfd_trans = vector_trafo(self.local_mesh, coord_tmp, self.Ucfd, dest_coord=1000000)

        # combine transformation vector
        self.Ucfd = Ucfd_rigid + Ucfd_trans

    def update_general_para(self):
        """
        In this section, the parameter file is updated. So far, I haven't found a way to do this via pysu2 for all parameters.
        This also means that the solver must be initialized with the new parameter file in case the file is updated.
        """
        if self.first_execution and self.myid == 0:
            # read all existing parameters
            config = SU2.io.Config(self.para_filename)
            # set general parameters, which don't change over the course of the CFD simulation, so they are only updated
            # for the first execution
            config['MESH_FILENAME'] = self.jcl.meshdefo['surface']['filename_grid']
            config['RESTART_FILENAME'] = self.jcl.aero['para_path'] + 'sol/restart_subcase_{}.dat'.format(
                self.trimcase['subcase'])
            config['SOLUTION_FILENAME'] = self.jcl.aero['para_path'] + 'sol/restart_subcase_{}.dat'.format(
                self.trimcase['subcase'])
            config['SURFACE_FILENAME'] = self.jcl.aero['para_path'] + 'sol/surface_subcase_{}'.format(
                self.trimcase['subcase'])
            config['VOLUME_FILENAME'] = self.jcl.aero['para_path'] + 'sol/volume_subcase_{}'.format(
                self.trimcase['subcase'])
            config['CONV_FILENAME'] = self.jcl.aero['para_path'] + 'sol/history_subcase_{}'.format(
                self.trimcase['subcase'])
            # free stream definition
            config['FREESTREAM_TEMPERATURE'] = self.atmo['T']
            config['FREESTREAM_DENSITY'] = self.atmo['rho']
            config['FREESTREAM_PRESSURE'] = self.atmo['p']
            # update the pressure inlet
            if hasattr(self.jcl, 'pressure_inlet'):
                config['MARKER_INLET'] = '( ' + ', '.join([self.jcl.pressure_inlet['marker'],
                                                           str(self.atmo['T']),
                                                           str(self.atmo['p']),
                                                           str(self.jcl.pressure_inlet['flow_direction'][0]),
                                                           str(self.jcl.pressure_inlet['flow_direction'][1]),
                                                           str(self.jcl.pressure_inlet['flow_direction'][2]), ]) + ' )'
            # set the farfield onflow
            config['MACH_NUMBER'] = self.trimcase['Ma']
            config['AOA'] = 0.0
            config['SIDESLIP_ANGLE'] = 0.0
            # activate grid deformation
            config['DEFORM_MESH'] = 'YES'
            config['MARKER_DEFORM_MESH'] = '( ' + ', '.join(self.jcl.meshdefo['surface']['markers']) + ' )'
            # there is no restart for the first execution
            config['RESTART_SOL'] = 'NO'
            # detrimine cfd solution output
            config['OUTPUT_FILES'] = ['RESTART', 'RESTART_ASCII', 'TECPLOT', 'SURFACE_TECPLOT']
            # set intemediate outputs
            # config['OUTPUT_WRT_FREQ']= 1

            # do the update
            # config.write() maintains the original layout of the file but doesn't add new parameters
            # config.dump() writes all parameters in a weird order, including default values
            config.dump(self.para_filename)
            logging.info('SU2 parameter file updated.')

    def update_timedom_para(self):
        """
        In this section, the time domain-related parameters are updated.
        """
        if self.first_execution and self.myid == 0:
            config = SU2.io.Config(self.para_filename)
            config['TIME_DOMAIN'] = 'YES'
            config['TIME_MARCHING'] = 'DUAL_TIME_STEPPING-2ND_ORDER'
            config['TIME_STEP'] = self.stepwidth
            # estimate the number or steps the integrator will make
            timesteps = np.ceil(self.simcase['t_final'] / self.simcase['dt'])
            iterations_per_timestep = np.ceil(self.simcase['dt'] / self.stepwidth)
            config['TIME_ITER'] = 2 + timesteps * iterations_per_timestep
            config['MAX_TIME'] = (2 + timesteps * iterations_per_timestep) * self.stepwidth

            """
            Perform an unsteady restart from a steady solution currently involves the following steps
            (as discussed here: https://github.com/su2code/SU2/discussions/1964):
            - Link the steady solution twice (e.g. restart_00000.dat and restart_00001.dat)
            - Restart the unsteady solution with RESTART_ITER= 2
            - Because SU2 requires both the .dat and the .csv file, this involves four file operations.
            """
            config['RESTART_SOL'] = 'YES'
            config['RESTART_ITER'] = 2
            # create links for the .dat files...
            filename_steady = self.jcl.aero['para_path'] + 'sol/restart_subcase_{}.dat'.format(
                self.trimcase['subcase'])
            filename_unsteady0 = self.jcl.aero['para_path'] + 'sol/restart_subcase_{}_00000.dat'.format(
                self.trimcase['subcase'])
            filename_unsteady1 = self.jcl.aero['para_path'] + 'sol/restart_subcase_{}_00001.dat'.format(
                self.trimcase['subcase'])
            try:
                os.symlink(filename_steady, filename_unsteady0)
                os.symlink(filename_steady, filename_unsteady1)
            except FileExistsError:
                # Do nothing, the most likely cause is that the file already exists.
                pass
            # ...and for the .csv files
            filename_steady = self.jcl.aero['para_path'] + 'sol/restart_subcase_{}.csv'.format(
                self.trimcase['subcase'])
            filename_unsteady0 = self.jcl.aero['para_path'] + 'sol/restart_subcase_{}_00000.csv'.format(
                self.trimcase['subcase'])
            filename_unsteady1 = self.jcl.aero['para_path'] + 'sol/restart_subcase_{}_00001.csv'.format(
                self.trimcase['subcase'])
            try:
                os.symlink(filename_steady, filename_unsteady0)
                os.symlink(filename_steady, filename_unsteady1)
            except FileExistsError:
                pass

            """
            In the time domain, simply rely on the the density residual to establish convergence.
            This is because SU2 only needs a few inner iterations per time step, which are too few for a meaningful
            cauchy convergence.
            """
            if 'CONV_FIELD' in config:
                config.pop('CONV_FIELD')
            config['INNER_ITER'] = 30
            config['CONV_RESIDUAL_MINVAL'] = -6

            # There is no need for restart solutions, they only take up storage space. Write plotting files only.
            config['OUTPUT_FILES'] = ['TECPLOT', 'SURFACE_TECPLOT']

            # do the update
            config.dump(self.para_filename)
            logging.info('SU2 parameter file updated.')

    def update_gust_para(self, Vtas, Vgust):
        """
        In this section, the gust-related parameters are updated.
        """
        if self.first_execution and self.myid == 0:
            config = SU2.io.Config(self.para_filename)
            config['GRID_MOVEMENT'] = 'GUST'
            config['WIND_GUST'] = 'YES'
            config['GUST_TYPE'] = 'ONE_M_COSINE'
            """
            Establish the gust direction as SU2 can handle gusts either in x, y or z-direction. Arbitrary gust
            orientations are currently not supported. By swithing the sign of the gust velocity, gusts from
            four different directions are possible. This should cover most of our applications, though.
            """
            if self.simcase['gust_orientation'] in [0.0, 360.0]:
                config['GUST_DIR'] = 'Z_DIR'
                config['GUST_AMPL'] = Vgust
            elif self.simcase['gust_orientation'] == 180.0:
                config['GUST_DIR'] = 'Z_DIR'
                config['GUST_AMPL'] = -Vgust
            elif self.simcase['gust_orientation'] == 90.0:
                config['GUST_DIR'] = 'Y_DIR'
                config['GUST_AMPL'] = Vgust
            elif self.simcase['gust_orientation'] == 270.0:
                config['GUST_DIR'] = 'Y_DIR'
                config['GUST_AMPL'] = -Vgust
            else:
                logging.error('Gust orientation {} currently NOT supported by SU2. \
                               Possible values: 0.0, 90.0, 180.0, 270.0 or 360.0 degrees.'.format(
                              self.simcase['gust_orientation']))
            # Note: In SU2 this is the full gust length, not the gust gradient H (half gust length).
            config['GUST_WAVELENGTH'] = 2.0 * self.simcase['gust_gradient']
            config['GUST_PERIODS'] = 1.0
            config['GUST_AMPL'] = Vgust
            config['GUST_BEGIN_TIME'] = 0.0
            config['GUST_BEGIN_LOC'] = -2.0 * self.simcase['gust_gradient'] - self.simcase['gust_para']['T1'] * Vtas

            # do the update
            config.dump(self.para_filename)
            logging.info('SU2 parameter file updated.')
