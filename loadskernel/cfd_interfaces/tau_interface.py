"""
Technically, Tau-Python works with both Python 2 and 3. However, the Tau versions on our 
linux cluster and on marvinng are compiled with Python 2 and don't work with Python 3. 
With the wrong Python version, an error is raised already during the import, which is 
handled by the try/except statement below.
"""
try:
    import PyPara
    from tau_python import tau_parallel_end, tau_close
except:
    pass

import numpy as np
import logging, os, subprocess, shlex, sys, platform, shutil
import scipy.io.netcdf as netcdf

import loadskernel.cfd_interfaces.meshdefo as meshdefo
from loadskernel.io_functions.specific_functions import check_path

def copy_para_file(jcl, trimcase):
    para_path = check_path(jcl.aero['para_path'])
    src = para_path+jcl.aero['para_file']
    dst = para_path+'para_subcase_{}'.format(trimcase['subcase'])
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

class TauInterface(object):
    
    def __init__(self, solution):
        self.model      = solution.model
        self.jcl        = solution.jcl
        self.trimcase   = solution.trimcase
        # get some indices
        self.i_atmo     = self.model.atmo['key'].index(self.trimcase['altitude'])
        self.i_mass     = self.model.mass['key'].index(self.trimcase['mass'])
        # set switch for first execution
        self.first_execution = True
        # Check if Tau-Python was imported successfully, see try/except statement in the import section.
        if "PyPara" in sys.modules:
            logging.info('Init CFD interface of type "{}"'.format(self.__class__.__name__))
        else:
            logging.error('Tau-Python was/could NOT be imported! Model equations of type "{}" will NOT work.'.format(self.jcl.aero['method']))
        
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
        if machinefile == None:
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
            logging.error('Number of given hosts ({}) smaller than required hosts ({}). Exit.'.format(tau_mpi_hosts.__len__(), n_required))
            sys.exit()
        self.tau_mpi_hosts = tau_mpi_hosts
    
    def prepare_meshdefo(self, Uf, Ux2):
        defo = meshdefo.Meshdefo(self.jcl, self.model)
        defo.init_deformations()
        defo.Uf(Uf, self.trimcase)
        defo.Ux2(Ux2)
        defo.write_deformations(self.jcl.aero['para_path']+'./defo/surface_defo_subcase_' + str(self.trimcase['subcase'])) 
        
        Para = PyPara.Parafile(self.jcl.aero['para_path']+'para_subcase_{}'.format(self.trimcase['subcase']))
        # deformation related parameters
        # it is important to start the deformation always from the undeformed grid !
        para_dict = {'Primary grid filename': self.jcl.meshdefo['surface']['filename_grid'],
                     'New primary grid prefix': './defo/volume_defo_subcase_{}'.format(self.trimcase['subcase'])}
        Para.update(para_dict)
        para_dict = {'RBF basis coordinates and deflections filename': './defo/surface_defo_subcase_{}.nc'.format(self.trimcase['subcase']),}
        Para.update(para_dict, 'group end', 0,)
        self.pytau_close()
    
    def set_grid_velocities(self, uvwpqr):
        Para = PyPara.Parafile(self.jcl.aero['para_path']+'para_subcase_{}'.format(self.trimcase['subcase']))
        # set aircraft motion related parameters
        # given in local, body-fixed reference frame, see Tau User Guide Section 18.1 "Coordinate Systems of the TAU-Code"
        # rotations in [deg], translations in grid units
        para_dict = {'Origin of local coordinate system':'{} {} {}'.format(self.model.mass[self.i_mass]['cggrid']['offset'][0,0],\
                                                                           self.model.mass[self.i_mass]['cggrid']['offset'][0,1],\
                                                                           self.model.mass[self.i_mass]['cggrid']['offset'][0,2]),
                     'Polynomial coefficients for translation x': '0 {}'.format(uvwpqr[0]),
                     'Polynomial coefficients for translation y': '0 {}'.format(uvwpqr[1]),
                     'Polynomial coefficients for translation z': '0 {}'.format(uvwpqr[2]),
                     'Polynomial coefficients for rotation roll': '0 {}'.format(uvwpqr[3]/np.pi*180.0),
                     'Polynomial coefficients for rotation pitch':'0 {}'.format(uvwpqr[4]/np.pi*180.0),
                     'Polynomial coefficients for rotation yaw':  '0 {}'.format(uvwpqr[5]/np.pi*180.0),
                     }
        Para.update(para_dict, 'mdf end', 0,)
        logging.debug("Parameters updated.")
        self.pytau_close()
    
    def update_general_para(self):
        if self.first_execution:
            Para = PyPara.Parafile(self.jcl.aero['para_path']+'para_subcase_{}'.format(self.trimcase['subcase']))   
            # Para.update(para_dict, block_key, block_id, key, key_value, sub_file, para_replace)
        
            # set general parameters, which don't change over the course of the CFD simulation, so they are only updated 
            # for the first execution
            para_dict = {'Reference Mach number': self.trimcase['Ma'],
                         'Reference temperature': self.model.atmo['T'][self.i_atmo],
                         'Reference density': self.model.atmo['rho'][self.i_atmo],
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
        logging.info('Starting Tau deformation, preprocessing and solver on {} hosts ({}).'.format(self.jcl.aero['tau_cores'], tau_mpi_hosts) )
        old_dir = os.getcwd()
        os.chdir(self.jcl.aero['para_path'])

        args_deform   = shlex.split('mpiexec -np {} --host {} deformation para_subcase_{} ./log/log_subcase_{} with mpi'.format(self.jcl.aero['tau_cores'],  tau_mpi_hosts, self.trimcase['subcase'], self.trimcase['subcase']))
        args_pre      = shlex.split('mpiexec -np {} --host {} ptau3d.preprocessing para_subcase_{} ./log/log_subcase_{} with mpi'.format(self.jcl.aero['tau_cores'], tau_mpi_hosts, self.trimcase['subcase'], self.trimcase['subcase']))
        args_solve    = shlex.split('mpiexec -np {} --host {} ptau3d.{} para_subcase_{} ./log/log_subcase_{} with mpi'.format(self.jcl.aero['tau_cores'], tau_mpi_hosts, self.jcl.aero['tau_solver'], self.trimcase['subcase'], self.trimcase['subcase']))
        
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
        Para = PyPara.Parafile(self.jcl.aero['para_path']+'para_subcase_{}'.format(self.trimcase['subcase']))
        filename_surface = self.jcl.aero['para_path'] + Para.get_para_value('Surface output filename')
        self.pytau_close()

        # gather from multiple domains
        old_dir = os.getcwd()
        os.chdir(self.jcl.aero['para_path'])
        # using a individual para file only for gathering allows to gather only the surface, not the volume output, which is faster
        with open('gather_subcase_{}.para'.format(self.trimcase['subcase']),'w') as fid:
            fid.write('Restart-data prefix : {}'.format(filename_surface))
        subprocess.call(['gather', 'gather_subcase_{}.para'.format(self.trimcase['subcase'])])
        os.chdir(old_dir)
        
        logging.info( 'Reading {}'.format(filename_surface))
        ncfile_pval = netcdf.NetCDFFile(filename_surface, 'r')
        global_id = ncfile_pval.variables['global_id'][:].copy()

        # determine the positions of the points in the pval file
        # this could be relevant if not all markers in the pval file are used
        logging.debug('Working on marker {}'.format(self.model.cfdgrid['desc']))
        # Because our mesh IDs are sorted and the Tau output is sorted, there is no need for an additional sorting.
        # Exception: Additional surface markers are written to the Tau output, which are not used for coupling.
        if global_id.__len__() == self.model.cfdgrid['n']:
            pos = range(self.model.cfdgrid['n'])
        else:
            pos = []
            for ID in self.model.cfdgrid['ID']: 
                pos.append(np.where(global_id == ID)[0][0]) 
        # build force vector from cfd solution self.engine(X)                   
        Pcfd = np.zeros(self.model.cfdgrid['n']*6)
        Pcfd[self.model.cfdgrid['set'][:,0]] = ncfile_pval.variables['x-force'][:][pos].copy()
        Pcfd[self.model.cfdgrid['set'][:,1]] = ncfile_pval.variables['y-force'][:][pos].copy()
        Pcfd[self.model.cfdgrid['set'][:,2]] = ncfile_pval.variables['z-force'][:][pos].copy()
        return Pcfd
    
    def prepare_initial_solution(self, args_solve):   
        Para = PyPara.Parafile(self.jcl.aero['para_path']+'para_subcase_{}'.format(self.trimcase['subcase']))  
        # set solution parameters for the initial and following solutions
        para_dicts = [
                      # initial solution
                      {'Inviscid flux discretization type': 'Upwind',
                       'Order of upwind flux (1-2)': 1.0,
                       'Maximal time step number': 300, 
                      },
                      # restore parameters for following solutions
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

class TauError(Exception):
    '''Raise when subprocess yields a returncode != 0 from Tau'''