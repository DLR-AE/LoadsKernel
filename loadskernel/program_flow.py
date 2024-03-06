import argparse
import copy
import getpass
import logging
import platform
import sys
import time

try:
    from mpi4py import MPI
except ImportError:
    pass

from loadskernel import solution_sequences, post_processing, gather_loads, auxiliary_output, plotting_standard
from loadskernel.io_functions import data_handling
import loadskernel.model as model_modul
from loadskernel.cfd_interfaces.mpi_helper import setup_mpi


class ProgramFlowHelper():

    def __init__(self,
                 job_name,
                 pre=False, main=False, post=False,
                 test=False,
                 path_input='../input/',
                 path_output='../output/',
                 jcl=None,
                 machinefile=None,):
        # basic options
        self.pre = pre  # True/False
        self.main = main  # True/False
        self.post = post  # True/False
        # debug options
        self.debug = False  # True/False
        self.restart = False  # True/False
        # advanced options
        self.test = test  # True/False
        # job control options
        self.job_name = job_name  # string
        self.path_input = path_input  # path
        self.path_output = path_output  # path
        self.jcl = jcl  # python class
        self.machinefile = machinefile  # filename
        # Initialize some more things
        self.setup_path()
        # Initialize MPI interface
        self.have_mpi, self.comm, self.status, self.myid = setup_mpi(
            self.debug)
        # Establish whether or not to use multiprocessing
        if self.have_mpi and self.comm.Get_size() > 1:
            self.use_multiprocessing = True
        else:
            self.use_multiprocessing = False

    def setup_path(self):
        self.path_input = data_handling.check_path(self.path_input)
        self.path_output = data_handling.check_path(self.path_output)

    def print_logo(self):
        logging.info('')
        logging.info('       (  )')
        logging.info('      (    )')
        logging.info('')
        logging.info('              (   )')
        logging.info('             (     )')
        logging.info('')
        logging.info('         _|_')
        logging.info(' ---------O---------')
        logging.info('')
        logging.info('')

    def setup_logger_cluster(self, i):
        # Generate a separate filename for each subcase
        path_log = data_handling.check_path(self.path_output + 'log/')
        filename = path_log + 'log_' + self.job_name + '_subcase_' + str(self.jcl.trimcase[i]['subcase']) \
            + '.txt.' + str(self.myid)
        # Then create the logger and console output
        self.create_logfile_and_console_output(filename)

    def setup_logger(self):
        # Generate a generic name for the log file
        path_log = data_handling.check_path(self.path_output + 'log/')
        filename = path_log + 'log_' + self.job_name + '.txt.' + str(self.myid)
        # Then create the logger and console output
        self.create_logfile_and_console_output(filename)

    def create_logfile_and_console_output(self, filename):
        logger = logging.getLogger()
        # Set logging level.
        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        # Get the names of all existing loggers.
        existing_handlers = [hdlr.get_name() for hdlr in logger.handlers]
        if 'lk_logfile' in existing_handlers:
            # Make sure that the filename is still correct.
            hdlr = logger.handlers[existing_handlers.index('lk_logfile')]
            if not hdlr.baseFilename == filename:
                # In case the filename is incorrect, remove the handler completely from the logger.
                logger.removeHandler(hdlr)
                # Update the list of all existing loggers.
                existing_handlers = [hdlr.get_name() for hdlr in logger.handlers]

        # Add the following handlers only if they don't exist. This avoid duplicate lines/log entries.
        if 'lk_logfile' not in existing_handlers:
            # define a Handler which writes messages to a log file
            logfile = logging.FileHandler(filename, mode='a')
            logfile.set_name('lk_logfile')
            formatter = logging.Formatter(fmt='%(asctime)s %(processName)-14s %(levelname)s: %(message)s',
                                          datefmt='%d/%m/%Y %H:%M:%S')
            logfile.setFormatter(formatter)
            logger.addHandler(logfile)

        # For convinience, the first rank writes console outputs, too.
        if (self.myid == 0) and ('lk_console' not in existing_handlers):
            # define a Handler which writes messages to the sys.stout
            console = logging.StreamHandler(sys.stdout)
            console.set_name('lk_console')
            # set a format which is simpler for console use
            formatter = logging.Formatter(fmt='%(levelname)s: %(message)s')
            # tell the handler to use this format
            console.setFormatter(formatter)
            # add the handler(s) to the root logger
            logger.addHandler(console)

        logger.info('This is the log for process {}.'.format(self.myid))


class Kernel(ProgramFlowHelper):

    def run(self):
        self.setup_logger()
        logging.info('Starting Loads Kernel with job: ' + self.job_name)
        logging.info('User ' + getpass.getuser() + ' on ' + platform.node() + ' (' + platform.platform() + ')')
        logging.info('pre:  ' + str(self.pre))
        logging.info('main: ' + str(self.main))
        logging.info('post: ' + str(self.post))
        logging.info('test: ' + str(self.test))
        self.jcl = data_handling.load_jcl(self.job_name, self.path_input, self.jcl)
        # add machinefile to jcl
        self.jcl.machinefile = self.machinefile

        if self.pre and not self.use_multiprocessing:
            self.run_pre()
        if self.main and self.use_multiprocessing:
            self.run_main_multiprocessing()
        if self.main and not self.use_multiprocessing:
            self.run_main_sequential()
        if self.post and not self.use_multiprocessing:
            self.run_post()
        if self.test and not self.use_multiprocessing:
            self.run_test()

        logging.info('Loads Kernel finished.')
        self.print_logo()

    def run_pre(self):
        logging.info('--> Starting preprocessing.')
        t_start = time.time()
        model = model_modul.Model(self.jcl, self.path_output)
        model.build_model()

        logging.info('--> Saving model data.')
        data_handling.dump_hdf5(self.path_output + 'model_' + self.job_name + '.hdf5', model.__dict__)
        logging.info('--> Done in {}.'.format(seconds2string(time.time() - t_start)))

    def main_common(self, model, jcl, i):
        logging.info('')
        logging.info('========================================')
        logging.info('trimcase: ' + jcl.trimcase[i]['desc'])
        logging.info('subcase: ' + str(jcl.trimcase[i]['subcase']))
        logging.info('(case ' + str(i + 1) + ' of ' + str(len(jcl.trimcase)) + ')')
        logging.info('========================================')
        solution_i = solution_sequences.SolutionSequences(model, jcl, jcl.trimcase[i], jcl.simcase[i])
        solution_i.set_trimcond()
        solution_i.exec_trim()
        # solution_i.iterative_trim()
        if solution_i.successful and 't_final' and 'dt' in jcl.simcase[i].keys():
            solution_i.exec_sim()
        elif solution_i.successful and 'flutter' in jcl.simcase[i] and jcl.simcase[i]['flutter']:
            solution_i.exec_flutter()
        elif solution_i.successful and 'derivatives' in jcl.simcase[i] and jcl.simcase[i]['derivatives']:
            solution_i.calc_jacobian()
            solution_i.calc_derivatives()
        response = solution_i.response
        response['i'] = i
        response['successful'] = solution_i.successful
        del solution_i
        if response['successful']:
            post_processing_i = post_processing.PostProcessing(jcl, model, jcl.trimcase[i], response)
            post_processing_i.force_summation_method()
            post_processing_i.euler_transformation()
            post_processing_i.cuttingforces()
            del post_processing_i
        return response

    def run_main_sequential(self):
        logging.info('--> Starting Main in sequential mode for {} trimcase(s).'.format(len(self.jcl.trimcase)))
        t_start = time.time()
        model = data_handling.load_hdf5(self.path_output + 'model_' + self.job_name + '.hdf5')
        if self.myid == 0:
            mon = gather_loads.GatherLoads(self.jcl, model)
            if self.restart:
                logging.info('Restart option: loading existing responses.')
                # open response
                responses = data_handling.load_hdf5_responses(self.job_name, self.path_output)
            fid = data_handling.open_hdf5(self.path_output + 'response_' + self.job_name + '.hdf5')  # open response

        for i in range(len(self.jcl.trimcase)):
            if self.restart and i in [response['i'][()] for response in responses]:
                logging.info('Restart option: found existing response.')
                response = responses[[response['i'][()] for response in responses].index(i)]
            else:
                jcl = copy.deepcopy(self.jcl)
                response = self.main_common(model, jcl, i)
            if self.myid == 0 and response['successful']:
                mon.gather_monstations(self.jcl.trimcase[i], response)
                mon.gather_dyn2stat(response)
                logging.info('--> Saving response(s).')
                data_handling.write_hdf5(fid, response, path='/' + str(response['i']))
        if self.myid == 0:
            # close response
            data_handling.close_hdf5(fid)

            logging.info('--> Saving monstation(s).')
            data_handling.dump_hdf5(self.path_output + 'monstations_' + self.job_name + '.hdf5', mon.monstations)

            logging.info('--> Saving dyn2stat.')
            data_handling.dump_hdf5(self.path_output + 'dyn2stat_' + self.job_name + '.hdf5', mon.dyn2stat)
        logging.info('--> Done in {}.'.format(seconds2string(time.time() - t_start)))

    def run_main_multiprocessing(self):
        """
        This function organizes the processing of multiple load cases via MPI.
        Parallelization is achieved using a worker/master concept.
        The workers and the master communicate via tags ('ready', 'start', 'done', 'exit').
        This concept is adapted from Jörg Bornschein (see https://github.com/jbornschein/mpi4py-examples/blob/master/
        09-task-pull.py)
        """
        logging.info(
            '--> Starting Main in multiprocessing mode for %d trimcase(s).', len(self.jcl.trimcase))
        t_start = time.time()
        model = data_handling.load_hdf5(
            self.path_output + 'model_' + self.job_name + '.hdf5')
        # MPI tags can be any integer values
        tags = {'ready': 0,
                'start': 1,
                'done': 2,
                'exit': 3,
                }
        # The master process runs on the first processor
        if self.myid == 0:
            n_workers = self.comm.Get_size() - 1
            logging.info('--> I am the master with %d worker(s).', n_workers)

            mon = gather_loads.GatherLoads(self.jcl, model)
            # open response
            fid = data_handling.open_hdf5(self.path_output + 'response_' + self.job_name + '.hdf5')

            closed_workers = 0
            i_subcase = 0

            while closed_workers < n_workers:
                logging.debug('Master is waiting for communication...')
                data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
                source = self.status.Get_source()
                tag = self.status.Get_tag()

                if tag == tags['ready']:
                    # Worker is ready, send out a new subcase.
                    if i_subcase < len(self.jcl.trimcase):
                        self.comm.send(i_subcase, dest=source, tag=tags['start'])
                        logging.info('--> Sending case %d of %d to worker %d', i_subcase + 1, len(self.jcl.trimcase), source)
                        i_subcase += 1
                    else:
                        # No more task to do, send the exit signal.
                        self.comm.send(None, dest=source, tag=tags['exit'])

                elif tag == tags['done']:
                    # The worker has returned a response.
                    response = data
                    if response['successful']:
                        logging.info("--> Received response ('successful') from worker %d.", source)
                        mon.gather_monstations(self.jcl.trimcase[response['i']], response)
                        mon.gather_dyn2stat(response)
                    else:
                        # Trim failed, no post processing, save the empty response
                        logging.info("--> Received response ('failed') from worker %d.", source)
                    logging.info('--> Saving response(s).')
                    data_handling.write_hdf5(fid, response, path='/' + str(response['i']))

                elif tag == tags['exit']:
                    # The worker confirms the exit.
                    logging.debug('Worker %d exited.', source)
                    closed_workers += 1
            # close response
            data_handling.close_hdf5(fid)
            logging.info('--> Saving monstation(s).')
            data_handling.dump_hdf5(self.path_output + 'monstations_' + self.job_name + '.hdf5',
                                    mon.monstations)

            logging.info('--> Saving dyn2stat.')
            data_handling.dump_hdf5(self.path_output + 'dyn2stat_' + self.job_name + '.hdf5',
                                    mon.dyn2stat)
        # The worker process runs on all other processors
        else:
            logging.info('I am worker on process %d.', self.myid)
            while True:
                self.comm.send(None, dest=0, tag=tags['ready'])
                i_subcase = self.comm.recv(
                    source=0, tag=MPI.ANY_TAG, status=self.status)
                tag = self.status.Get_tag()

                if tag == tags['start']:
                    # Start a new job
                    response = self.main_common(model, self.jcl, i_subcase)
                    self.comm.send(response, dest=0, tag=tags['done'])
                elif tag == tags['exit']:
                    # Received an exit signal.
                    break
            # Confirm the exit signal.
            self.comm.send(None, dest=0, tag=tags['exit'])

        logging.info('--> Done in {}.'.format(seconds2string(time.time() - t_start)))

    def run_post(self):
        model = data_handling.load_hdf5(self.path_output + 'model_' + self.job_name + '.hdf5')
        responses = data_handling.load_hdf5_responses(self.job_name, self.path_output)
        logging.info('--> Loading monstations(s).')
        monstations = data_handling.load_hdf5(self.path_output + 'monstations_' + self.job_name + '.hdf5')
        logging.info('--> Loading dyn2stat.')
        dyn2stat_data = data_handling.load_hdf5(self.path_output + 'dyn2stat_' + self.job_name + '.hdf5')

        logging.info('--> Drawing some standard plots.')
        if 'flutter' in self.jcl.simcase[0] and self.jcl.simcase[0]['flutter']:
            plt = plotting_standard.FlutterPlots(self.jcl, model)
            plt.add_responses(responses)
            plt.plot_fluttercurves_to_pdf(self.path_output + 'fluttercurves_' + self.job_name + '.pdf')
            plt.plot_eigenvalues_to_pdf(self.path_output + 'eigenvalues_' + self.job_name + '.pdf')
        elif 'derivatives' in self.jcl.simcase[0] and self.jcl.simcase[0]['derivatives']:
            plt = plotting_standard.FlutterPlots(self.jcl, model)
            plt.add_responses(responses)
            plt.plot_eigenvalues_to_pdf(self.path_output + 'eigenvalues_' + self.job_name + '.pdf')
        elif 'limit_turbulence' in self.jcl.simcase[0] and self.jcl.simcase[0]['limit_turbulence']:
            plt = plotting_standard.TurbulencePlots(self.jcl, model)
            plt.add_monstations(monstations)
            plt.plot_monstations(
                self.path_output + 'monstations_turbulence_' + self.job_name + '.pdf')
        else:
            # Here come the loads plots
            plt = plotting_standard.LoadPlots(self.jcl, model)
            plt.add_monstations(monstations)
            plt.add_responses(responses)
            plt.plot_monstations(self.path_output + 'monstations_' + self.job_name + '.pdf')
            if any([x in self.jcl.simcase[0] and self.jcl.simcase[0][x] for x in ['gust', 'turbulence',
                                                                                  'cs_signal', 'controller']]):
                plt.plot_monstations_time(self.path_output + 'monstations_time_' + self.job_name + '.pdf')  # nur sim

            logging.info('--> Saving auxiliary output data.')
            aux_out = auxiliary_output.AuxiliaryOutput(self.jcl, model, self.jcl.trimcase)
            aux_out.crit_trimcases = plt.crit_trimcases
            aux_out.dyn2stat_data = dyn2stat_data
            aux_out.responses = responses
            if ('t_final' and 'dt' in self.jcl.simcase[0].keys()):
                aux_out.write_critical_trimcases(self.path_output + 'crit_trimcases_' + self.job_name + '.csv')
                aux_out.write_critical_nodalloads(self.path_output + 'nodalloads_' + self.job_name + '.bdf')
            else:
                aux_out.write_trimresults(self.path_output + 'trim_results_' + self.job_name + '.csv')
                aux_out.write_successful_trimcases(self.path_output + 'successful_trimcases_' + self.job_name + '.csv',
                                                   self.path_output + 'failed_trimcases_' + self.job_name + '.csv')
                aux_out.write_critical_trimcases(self.path_output + 'crit_trimcases_' + self.job_name + '.csv')
                aux_out.write_critical_nodalloads(self.path_output + 'nodalloads_' + self.job_name + '.bdf')
                # aux_out.write_all_nodalloads(self.path_output + 'nodalloads_all_' + self.job_name + '.bdf')
                # aux_out.save_nodaldefo(self.path_output + 'nodaldefo_' + self.job_name)
                # aux_out.save_cpacs(self.path_output + 'cpacs_' + self.job_name + '.xml')

    def run_test(self):
        """
        This section shall be used for code that is not part of standard post-processing procedures.
        Below, some useful plotting functions are given that might be helpful, for example
        - to identify if a new model is working correctly
        - to get extra time data plots
        - to animate a time domain simulation
        """
        # Import plotting_extra not before here, as the import of graphical libraries such as mayavi takes a long time and
        # fails of systems without graphical display (such as HPS clusters).
        from loadskernel import plotting_extra

        # Load the model and the response as usual
        model = data_handling.load_hdf5(self.path_output + 'model_' + self.job_name + '.hdf5')
        responses = data_handling.load_hdf5_responses(self.job_name, self.path_output)

        logging.info('--> Drawing some more detailed plots.')
        plt = plotting_extra.DetailedPlots(self.jcl, model)
        plt.add_responses(responses)
        if 't_final' and 'dt' in self.jcl.simcase[0].keys():
            # show some plots of the time domain data
            plt.plot_time_data()
        else:
            # show some plots of the force vectors, useful to identify model shortcomings
            # plt.plot_pressure_distribution()
            plt.plot_forces_deformation_interactive()

        if 't_final' and 'dt' in self.jcl.simcase[0].keys():
            # show a nice animation of the time domain simulation
            plt = plotting_extra.Animations(self.jcl, model)
            plt.add_responses(responses)
            plt.make_animation()
            # make a video file of the animation
            # plt.make_movie(self.path_output, speedup_factor=1.0)

        """
        At the moment, I also use this section for custom analysis scripts.
        """
#         with open(self.path_output + 'statespacemodel_' + self.job_name + '.pickle', 'wb') as fid:
#             data_handling.dump_pickle(responses, fid)

#         from scripts import plot_flexdefo
#         plot = plot_flexdefo.Flexdefo(self.jcl, model, responses)
#         plot.calc_flexdefos_trim()
#         plot.save_flexdefo_trim(self.path_output + 'flexdefo_' + self.job_name + '.pickle')
#         plot.plot_flexdefos_trim()
#         plot.plot_flexdefos()

#         from scripts import plot_lift_distribution
#         plot = plot_lift_distribution.Liftdistribution(self.jcl, model, responses)
#         plot.plot_aero_spanwise()


class ClusterMode(Kernel):
    """
    The cluster mode is similar to the (normal) sequential mode, but only ONE subcase is calculated.
    This mode relies on a job scheduler that is able to run an array of jobs, where each the job is
    accompanied by an index i=0...n and n is the number of all subcases.
    In a second step, when all jobs are finished, the results (e.g. one response per subcase) need to
    be gathered.
    Example:
    k = program_flow.ClusterMode('jcl_name', ...)
    k.run_cluster(sys.argv[2])
    or
    k.gather_cluster()
    """

    def run_cluster(self, i):
        i = int(i)
        self.jcl = data_handling.load_jcl(self.job_name, self.path_input, self.jcl)
        # add machinefile to jcl
        self.jcl.machinefile = self.machinefile
        self.setup_logger_cluster(i=i)
        logging.info('Starting Loads Kernel with job: ' + self.job_name)
        logging.info('User ' + getpass.getuser() + ' on ' + platform.node() + ' (' + platform.platform() + ')')
        logging.info('Cluster array mode')

        self.run_main_single(i)

        logging.info('Loads Kernel finished.')
        self.print_logo()

    def run_main_single(self, i):
        """
        This function calculates one single load case, e.g. using CFD with mpi hosts on a cluster.
        """
        logging.info('--> Starting main in single mode for {} trimcase(s).'.format(len(self.jcl.trimcase)))
        t_start = time.time()
        model = data_handling.load_hdf5(self.path_output + 'model_' + self.job_name + '.hdf5')
        jcl = copy.deepcopy(self.jcl)
        """
        Before starting the simulation, dump an empty / dummy response. This is a workaround in case SU2 diverges,
        which leads to a hard exit (via mpi_abort). In that case, mpi_abort terminates everything and leaves no
        possibility to catch and handle an exception in Python. With this workaround, there is at least an empty
        response so that the gathering and the post_processing will work properly.
        """
        # Create an empty response
        empty_response = {'i': i,
                          'successful': False}
        if self.myid == 0:
            path_responses = data_handling.check_path(self.path_output + 'responses/')
            with open(path_responses + 'response_' + self.job_name + '_subcase_'
                      + str(self.jcl.trimcase[i]['subcase']) + '.pickle', 'wb') as f:
                data_handling.dump_pickle(empty_response, f)
        # Start the simulation
        response = self.main_common(model, jcl, i)
        # Overwrite the empty response from above
        if self.myid == 0:
            logging.info('--> Saving response(s).')
            path_responses = data_handling.check_path(self.path_output + 'responses/')
            with open(path_responses + 'response_' + self.job_name + '_subcase_'
                      + str(self.jcl.trimcase[i]['subcase']) + '.pickle', 'wb') as f:
                data_handling.dump_pickle(response, f)
        logging.info('--> Done in {}.'.format(seconds2string(time.time() - t_start)))

    def gather_cluster(self):
        self.setup_logger()
        t_start = time.time()
        logging.info('Starting Loads Kernel with job: ' + self.job_name)
        logging.info('user ' + getpass.getuser() + ' on ' + platform.node() + ' (' + platform.platform() + ')')
        logging.info('cluster gather mode')
        self.jcl = data_handling.load_jcl(self.job_name, self.path_input, self.jcl)
        model = data_handling.load_hdf5(self.path_output + 'model_' + self.job_name + '.hdf5')
        responses = data_handling.gather_responses(self.job_name, data_handling.check_path(self.path_output + 'responses'))
        mon = gather_loads.GatherLoads(self.jcl, model)
        fid = data_handling.open_hdf5(self.path_output + 'response_' + self.job_name + '.hdf5')  # open response
        for i in range(len(self.jcl.trimcase)):
            response = responses[[response['i'] for response in responses].index(i)]
            if response['successful']:
                mon.gather_monstations(self.jcl.trimcase[i], response)
                mon.gather_dyn2stat(response)

            logging.info('--> Saving response(s).')
            data_handling.write_hdf5(fid, response, path='/' + str(response['i']))
        # close response
        data_handling.close_hdf5(fid)

        logging.info('--> Saving monstation(s).')
        data_handling.dump_hdf5(self.path_output + 'monstations_' + self.job_name + '.hdf5',
                                mon.monstations)

        logging.info('--> Saving dyn2stat.')
        data_handling.dump_hdf5(self.path_output + 'dyn2stat_' + self.job_name + '.hdf5',
                                mon.dyn2stat)
        logging.info(
            '--> Done in {}.'.format(seconds2string(time.time() - t_start)))

        logging.info('Loads Kernel finished.')
        self.print_logo()


def str2bool(v):
    # This is a function outside the class to convert strings to boolean. Requirement for parsing command line arguments.
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def seconds2string(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return '{:n}:{:02n}:{:02n} [h:mm:ss]'.format(h, m, round(s))


def command_line_interface():
    parser = argparse.ArgumentParser()
    # register the most common arguments
    parser.add_argument('--job_name', help='Name of the JCL (no extension .py)', type=str, required=True)
    parser.add_argument('--pre', help='Pre-processing', choices=[True, False], type=str2bool, required=True)
    parser.add_argument('--main', help='Main-processing, True/False', choices=[True, False], type=str2bool, required=True)
    parser.add_argument('--post', help='Post-processing, True/False', choices=[True, False], type=str2bool, required=True)
    parser.add_argument('--path_input', help='Path to the JCL file', type=str, required=True)
    parser.add_argument('--path_output', help='Path to save the output', type=str, required=True)
    # get arguments from command line
    args = parser.parse_args()
    # run the loads kernel with the command line arguments
    k = Kernel(job_name=args.job_name, pre=args.pre, main=args.main, post=args.post,
               path_input=args.path_input,
               path_output=args.path_output)
    k.run()


if __name__ == "__main__":
    command_line_interface()
