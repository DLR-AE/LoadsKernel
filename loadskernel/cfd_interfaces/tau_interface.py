"""
Technically, Tau-Python works with both Python 2 and 3. However, the Tau versions on our
linux cluster and on marvinng are compiled with Python 2 and don't work with Python 3.
With the wrong Python version, an error is raised already during the import, which is
handled by the try/except statement below.
"""
try:
    import PyPara
    from tau_python import tau_parallel_end, tau_close
except ImportError:
    pass

import logging
import os
import platform
import shlex
import subprocess
import sys
import shutil
import numpy as np
import scipy.io.netcdf as netcdf

from loadskernel.cfd_interfaces import meshdefo
from loadskernel import spline_functions
from loadskernel.io_functions.data_handling import check_path
from loadskernel.io_functions.data_handling import load_hdf5_dict


def copy_para_file(jcl, trimcase):
    para_path = check_path(jcl.aero['para_path'])
    src = para_path + jcl.aero['para_file']
    dst = para_path + 'para_subcase_{}'.format(trimcase['subcase'])
    shutil.copyfile(src, dst)


def check_para_path(jcl):
    jcl.aero['para_path'] = check_path(jcl.aero['para_path'])


def check_cfd_folders(jcl):
    para_path = check_path(jcl.aero['para_path'])
    # check and create default folders for Tau
    if not os.path.exists(os.path.join(para_path, 'log')):
        os.makedirs(os.path.join(para_path, 'log'))
    if not os.path.exists(os.path.join(para_path, 'sol')):
        os.makedirs(os.path.join(para_path, 'sol'))
    if not os.path.exists(os.path.join(para_path, 'defo')):
        os.makedirs(os.path.join(para_path, 'defo'))
    if not os.path.exists(os.path.join(para_path, 'dualgrid')):
        os.makedirs(os.path.join(para_path, 'dualgrid'))


class TauInterface(meshdefo.Meshdefo):

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
        self.cfdgrids = load_hdf5_dict(self.model['cfdgrids'])
        self.strcgrid = load_hdf5_dict(self.model['strcgrid'])
        self.splinegrid = load_hdf5_dict(self.model['splinegrid'])
        self.aerogrid = load_hdf5_dict(self.model['aerogrid'])
        self.x2grid = load_hdf5_dict(self.model['x2grid'])
        self.coord = load_hdf5_dict(self.model['coord'])

        self.Djx2 = self.model['Djx2'][()]
        # set switch for first execution
        self.first_execution = True
        # Check if Tau-Python was imported successfully, see try/except statement in the import section.
        if "PyPara" in sys.modules:
            logging.info('Init CFD interface of type "{}"'.format(self.__class__.__name__))
        else:
            logging.error('Tau-Python was/could NOT be imported! Model equations of type "{}" will NOT work.'.format(
                self.jcl.aero['method']))

        # Check if the correct MPI implementation is used, Tau requires OpenMPI
        args_version = shlex.split('mpiexec --help')
        process = subprocess.Popen(args_version, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.communicate()
        if str.find(str(output[0]), 'www.open-mpi.org') == -1:
            logging.error('Wrong MPI implementation detected (Tau requires OpenMPI).')
        # Set-up a list of hosts on which MPI shall be executed.
        self.setup_mpi_hosts(n_workers=1)

        # Set-up file system structure
        check_para_path(self.jcl)
        copy_para_file(self.jcl, self.trimcase)
        check_cfd_folders(self.jcl)

    def setup_mpi_hosts(self, n_workers):
        # Set-up a list of hosts on which MPI shall be executed.
        machinefile = self.jcl.machinefile
        n_required = self.jcl.aero['tau_cores'] * n_workers
        if machinefile is None:
            # all work is done on this node
            tau_mpi_hosts = [platform.node()] * n_required
        else:
            tau_mpi_hosts = []
            with open(machinefile) as f:
                lines = f.readlines()
            for line in lines:
                line = line.split(' slots=')
                tau_mpi_hosts += [line[0]] * int(line[1])
        if tau_mpi_hosts.__len__() < n_required:
            logging.error('Number of given hosts ({}) smaller than required hosts ({}). Exit.'.format(
                tau_mpi_hosts.__len__(), n_required))
            sys.exit()
        self.tau_mpi_hosts = tau_mpi_hosts

    def prepare_meshdefo(self, Uf, Ux2):
        self.init_deformations()
        self.Uf(Uf, self.trimcase)
        self.Ux2(Ux2)
        self.write_deformations(self.jcl.aero['para_path'] + './defo/surface_defo_subcase_' + str(self.trimcase['subcase']))

        Para = PyPara.Parafile(self.jcl.aero['para_path'] + 'para_subcase_{}'.format(self.trimcase['subcase']))
        # deformation related parameters
        # it is important to start the deformation always from the undeformed grid !
        para_dict = {'Primary grid filename': self.jcl.meshdefo['surface']['filename_grid'],
                     'New primary grid prefix': './defo/volume_defo_subcase_{}'.format(self.trimcase['subcase'])}
        Para.update(para_dict)
        para_dict = {'RBF basis coordinates and deflections filename': './defo/surface_defo_subcase_{}.nc'.format(
            self.trimcase['subcase']), }
        Para.update(para_dict, 'group end', 0,)
        self.pytau_close()

    def set_grid_velocities(self, uvwpqr):
        Para = PyPara.Parafile(self.jcl.aero['para_path'] + 'para_subcase_{}'.format(self.trimcase['subcase']))
        # set aircraft motion related parameters
        # given in local, body-fixed reference frame, see Tau User Guide Section 18.1 "Coordinate Systems of the TAU-Code"
        # rotations in [deg], translations in grid units
        para_dict = {'Origin of local coordinate system': '{} {} {}'.format(self.cggrid['offset'][0, 0],
                                                                            self.cggrid['offset'][0, 1],
                                                                            self.cggrid['offset'][0, 2]),
                     'Polynomial coefficients for translation x': '0 {}'.format(uvwpqr[0]),
                     'Polynomial coefficients for translation y': '0 {}'.format(uvwpqr[1]),
                     'Polynomial coefficients for translation z': '0 {}'.format(uvwpqr[2]),
                     'Polynomial coefficients for rotation roll': '0 {}'.format(uvwpqr[3] / np.pi * 180.0),
                     'Polynomial coefficients for rotation pitch': '0 {}'.format(uvwpqr[4] / np.pi * 180.0),
                     'Polynomial coefficients for rotation yaw': '0 {}'.format(uvwpqr[5] / np.pi * 180.0),
                     }
        Para.update(para_dict, 'mdf end', 0,)
        logging.debug("Parameters updated.")
        self.pytau_close()

    def update_general_para(self):
        if self.first_execution:
            Para = PyPara.Parafile(self.jcl.aero['para_path'] + 'para_subcase_{}'.format(self.trimcase['subcase']))
            # Para.update(para_dict, block_key, block_id, key, key_value, sub_file, para_replace)

            # set general parameters, which don't change over the course of the CFD simulation, so they are only updated
            # for the first execution
            para_dict = {'Reference Mach number': self.trimcase['Ma'],
                         'Reference temperature': self.atmo['T'],
                         'Reference density': self.atmo['rho'],
                         'Number of domains': self.jcl.aero['tau_cores'],
                         'Number of primary grid domains': self.jcl.aero['tau_cores'],
                         'Output files prefix': './sol/subcase_{}'.format(self.trimcase['subcase']),
                         'Grid prefix': './dualgrid/subcase_{}'.format(self.trimcase['subcase']),
                         }
            Para.update(para_dict)

            logging.debug("Parameters updated.")
            self.pytau_close()

    def update_timedom_para(self):
        pass

    def update_gust_para(self, simcase, v_gust):
        pass

    def pytau_close(self):
        # clean up to avoid trouble at the next run
        tau_parallel_end()
        tau_close()

    def init_solver(self):
        pass

    def run_solver(self):
        tau_mpi_hosts = ','.join(self.tau_mpi_hosts)
        logging.info('Starting Tau deformation, preprocessing and solver on {} hosts ({}).'.format(
            self.jcl.aero['tau_cores'], tau_mpi_hosts))
        old_dir = os.getcwd()
        os.chdir(self.jcl.aero['para_path'])

        args_deform = shlex.split('mpiexec -np {} --host {} deformation para_subcase_{} ./log/log_subcase_{} with mpi'.format(
            self.jcl.aero['tau_cores'], tau_mpi_hosts, self.trimcase['subcase'], self.trimcase['subcase']))
        args_pre = shlex.split('mpiexec -np {} --host {} ptau3d.preprocessing para_subcase_{} ./log/log_subcase_{} \
            with mpi'.format(self.jcl.aero['tau_cores'], tau_mpi_hosts, self.trimcase['subcase'], self.trimcase['subcase']))
        args_solve = shlex.split('mpiexec -np {} --host {} ptau3d.{} para_subcase_{} ./log/log_subcase_{} with mpi'.format(
            self.jcl.aero['tau_cores'], tau_mpi_hosts, self.jcl.aero['tau_solver'], self.trimcase['subcase'],
            self.trimcase['subcase']))

        returncode = subprocess.call(args_deform)
        if returncode != 0:
            raise TauError('Subprocess returned an error from Tau deformation, please see deformation.stdout !')
        returncode = subprocess.call(args_pre)
        if returncode != 0:
            raise TauError('Subprocess returned an error from Tau preprocessing, please see preprocessing.stdout !')

        if self.first_execution == 1:
            self.prepare_initial_solution(args_solve)
            # We are done with the fist run of Tau, the next execution should be the 'real' run
            self.first_execution = False
        else:
            returncode = subprocess.call(args_solve)
            if returncode != 0:
                raise TauError('Subprocess returned an error from Tau solver, please see solver.stdout !')

        logging.info("Tau finished normally.")
        os.chdir(old_dir)

    def get_last_solution(self):
        # get filename of surface solution from para file
        Para = PyPara.Parafile(self.jcl.aero['para_path'] + 'para_subcase_{}'.format(self.trimcase['subcase']))
        filename_surface = self.jcl.aero['para_path'] + Para.get_para_value('Surface output filename')
        self.pytau_close()

        # gather from multiple domains
        old_dir = os.getcwd()
        os.chdir(self.jcl.aero['para_path'])
        # using a individual para file only for gathering allows to gather only the surface, not the volume output, which is
        # faster
        with open('gather_subcase_{}.para'.format(self.trimcase['subcase']), 'w') as fid:
            fid.write('Restart-data prefix : {}'.format(filename_surface))
        subprocess.call(['gather', 'gather_subcase_{}.para'.format(self.trimcase['subcase'])])
        os.chdir(old_dir)

        logging.info('Reading {}'.format(filename_surface))
        ncfile_pval = netcdf.NetCDFFile(filename_surface, 'r')
        global_id = ncfile_pval.variables['global_id'][:].copy()

        # determine the positions of the points in the pval file
        # this could be relevant if not all markers in the pval file are used
        logging.debug('Working on marker {}'.format(self.cfdgrid['desc']))
        # Because our mesh IDs are sorted and the Tau output is sorted, there is no need for an additional sorting.
        # Exception: Additional surface markers are written to the Tau output, which are not used for coupling.
        if global_id.__len__() == self.cfdgrid['n']:
            pos = range(self.cfdgrid['n'])
        else:
            pos = []
            for ID in self.cfdgrid['ID']:
                pos.append(np.where(global_id == ID)[0][0])
        # build force vector from cfd solution self.engine(X)
        Pcfd = np.zeros(self.cfdgrid['n'] * 6)
        Pcfd[self.cfdgrid['set'][:, 0]] = ncfile_pval.variables['x-force'][:][pos].copy()
        Pcfd[self.cfdgrid['set'][:, 1]] = ncfile_pval.variables['y-force'][:][pos].copy()
        Pcfd[self.cfdgrid['set'][:, 2]] = ncfile_pval.variables['z-force'][:][pos].copy()
        return Pcfd

    def prepare_initial_solution(self, args_solve):
        Para = PyPara.Parafile(self.jcl.aero['para_path'] + 'para_subcase_{}'.format(self.trimcase['subcase']))
        # set solution parameters for the initial and following solutions
        para_dicts = [{'Inviscid flux discretization type': 'Upwind',
                       'Order of upwind flux (1-2)': 1.0,
                       'Maximal time step number': 300,
                       },
                      {'Inviscid flux discretization type': Para.get_para_value('Inviscid flux discretization type'),
                       'Order of upwind flux (1-2)': Para.get_para_value('Order of upwind flux (1-2)'),
                       'Maximal time step number': Para.get_para_value('Maximal time step number'),
                       }]
        for para_dict in para_dicts:
            Para.update(para_dict)
            returncode = subprocess.call(args_solve)
            if returncode != 0:
                raise TauError('Subprocess returned an error from Tau solver, please see solver.stdout !')

    def release_memory(self):
        # Because Tau is called via a subprocess, there is no need to release memory manually.
        pass

    def init_deformations(self):
        # create empty deformation vectors for cfdgrids
        self.Ucfd = []
        for marker in self.cfdgrids:
            self.Ucfd.append(np.zeros(self.cfdgrids[marker]['n'][()] * 6))

    def transfer_deformations(self, grid_i, U_i, set_i, rbf_type, surface_spline, support_radius=2.0):
        logging.info('Transferring deformations to the CFD surface mesh.')
        if self.plotting:
            # set-up plot
            from mayavi import mlab
            p_scale = 0.05  # points
            mlab.figure()
            mlab.points3d(grid_i['offset' + set_i][:, 0], grid_i['offset' + set_i][:, 1], grid_i['offset' + set_i][:, 2],
                          scale_factor=p_scale, color=(1, 1, 1))
            mlab.points3d(grid_i['offset' + set_i][:, 0] + U_i[grid_i['set' + set_i][:, 0]],
                          grid_i['offset' + set_i][:, 1] + U_i[grid_i['set' + set_i][:, 1]],
                          grid_i['offset' + set_i][:, 2] + U_i[grid_i['set' + set_i][:, 2]],
                          scale_factor=p_scale, color=(1, 0, 0))
        for marker, Ucfd in zip(self.cfdgrids, self.Ucfd):
            grid_d = load_hdf5_dict(self.cfdgrids[marker])
            logging.debug('Working on marker {}'.format(grid_d['desc']))
            # build spline matrix
            PHIi_d = spline_functions.spline_rbf(grid_i, set_i, grid_d, '',
                                                 rbf_type=rbf_type, surface_spline=surface_spline,
                                                 support_radius=support_radius, dimensions=[U_i.size, grid_d['n'] * 6])
            # store deformation of cfdgrid
            Ucfd += PHIi_d.dot(U_i)
            if self.plotting:
                U_d = PHIi_d.dot(U_i)
                mlab.points3d(grid_d['offset'][:, 0], grid_d['offset'][:, 1], grid_d['offset'][:, 2],
                              color=(0, 0, 0), mode='point')
                mlab.points3d(grid_d['offset'][:, 0] + U_d[grid_d['set'][:, 0]],
                              grid_d['offset'][:, 1] + U_d[grid_d['set'][:, 1]],
                              grid_d['offset'][:, 2] + U_d[grid_d['set'][:, 2]],
                              color=(0, 0, 1), mode='point')
            del PHIi_d
        if self.plotting:
            mlab.show()

    def write_deformations(self, filename_defo):
        self.write_defo_netcdf(filename_defo)

    def write_defo_netcdf(self, filename_defo):
        logging.info('Writing ' + filename_defo + '.nc')
        f = netcdf.netcdf_file(filename_defo + '.nc', 'w')
        f.history = 'Surface deformations created by Loads Kernel'

        # Assemble temporary output. One point may belong to multiple markers.
        tmp_IDs = np.array([], dtype='int')
        tmp_x = np.array([])
        tmp_y = np.array([])
        tmp_z = np.array([])
        tmp_dx = np.array([])
        tmp_dy = np.array([])
        tmp_dz = np.array([])
        for marker, Ucfd in zip(self.cfdgrids, self.Ucfd):
            cfdgrid = load_hdf5_dict(self.cfdgrids[marker])
            tmp_IDs = np.concatenate((tmp_IDs, cfdgrid['ID']))
            tmp_x = np.concatenate((tmp_x, cfdgrid['offset'][:, 0]))
            tmp_y = np.concatenate((tmp_y, cfdgrid['offset'][:, 1]))
            tmp_z = np.concatenate((tmp_z, cfdgrid['offset'][:, 2]))
            tmp_dx = np.concatenate((tmp_dx, Ucfd[cfdgrid['set'][:, 0]]))
            tmp_dy = np.concatenate((tmp_dy, Ucfd[cfdgrid['set'][:, 1]]))
            tmp_dz = np.concatenate((tmp_dz, Ucfd[cfdgrid['set'][:, 2]]))
        IDs, pos = np.unique(tmp_IDs, return_index=True)

        f.createDimension('no_of_points', IDs.__len__())
        # create variables
        global_id = f.createVariable('global_id', 'i', ('no_of_points',))
        x = f.createVariable('x', 'd', ('no_of_points',))
        y = f.createVariable('y', 'd', ('no_of_points',))
        z = f.createVariable('z', 'd', ('no_of_points',))
        dx = f.createVariable('dx', 'd', ('no_of_points',))
        dy = f.createVariable('dy', 'd', ('no_of_points',))
        dz = f.createVariable('dz', 'd', ('no_of_points',))
        # fill variables with data
        global_id[:] = IDs
        x[:] = tmp_x[pos]
        y[:] = tmp_y[pos]
        z[:] = tmp_z[pos]
        dx[:] = tmp_dx[pos]
        dy[:] = tmp_dy[pos]
        dz[:] = tmp_dz[pos]

        f.close()


class TauError(Exception):
    '''Raise when subprocess yields a returncode != 0 from Tau'''
