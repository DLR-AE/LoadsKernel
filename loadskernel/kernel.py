# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:00:31 2014

@author: voss_ar
"""
import time, multiprocessing, getpass, platform, logging, sys, copy

from loadskernel import io_functions
from loadskernel.io_functions import matlab_functions, specific_functions
import loadskernel.trim as trim
import loadskernel.post_processing as post_processing
import loadskernel.gather_loads as gather_modul
import loadskernel.auxiliary_output as auxiliary_output
import loadskernel.plotting_standard as plotting_standard
import loadskernel.plotting_extra as plotting_extra
import loadskernel.model as model_modul

class Kernel():

    def __init__(self,
                 job_name, pre=False, main=False, post=False,
                 debug=False, test=False,
                 path_input='../input/',
                 path_output='../output/',
                 jcl=None,
                 parallel=False,
                 restart=False,
                 machinefile=None,):
        # basic options
        self.pre = pre                  # True/False
        self.main = main                # True/False
        self.post = post                # True/False
        # debug options
        self.debug = debug              # True/False
        self.restart = restart          # True/False
        # advanced options
        self.test = test                # True/False
        # job control options
        self.job_name = job_name        # string
        self.path_input = path_input    # path
        self.path_output = path_output  # path
        self.jcl = jcl                  # python class
        # parallel computing options
        self.parallel = parallel        # True/False/integer
        self.machinefile = machinefile  # filename
        
        self.setup()
        
    def setup(self):
        self.path_input = io_functions.specific_functions.check_path(self.path_input)
        self.path_output = io_functions.specific_functions.check_path(self.path_output)
        
    def run(self):
        self.setup_logger()
        logging.info('Starting Loads Kernel with job: ' + self.job_name)
        logging.info('user ' + getpass.getuser() + ' on ' + platform.node() + ' (' + platform.platform() + ')')
        logging.info('pre:  ' + str(self.pre))
        logging.info('main: ' + str(self.main))
        logging.info('post: ' + str(self.post))
        logging.info('test: ' + str(self.test))
        self.jcl = io_functions.specific_functions.load_jcl(self.job_name, self.path_input, self.jcl)

        if self.pre:
            self.run_pre()
        if self.main and self.parallel:
            self.run_main_parallel()
        if self.main and not self.parallel:
            self.run_main_sequential()
        if self.post:
            self.run_post()
        if self.test:
            self.run_test()

        logging.info('Loads Kernel finished.')
        self.print_logo()

    def run_cluster(self, i):
        i = int(i)
        self.setup_logger_cluster(i=i)
        self.jcl = io_functions.specific_functions.load_jcl(self.job_name, self.path_input, self.jcl)
        logging.info('Starting Loads Kernel with job: ' + self.job_name)
        logging.info('user ' + getpass.getuser() + ' on ' + platform.node() + ' (' + platform.platform() + ')')
        logging.info('cluster array mode')

        self.run_main_single(i)

        logging.info('Loads Kernel finished.')
        self.print_logo()
    
    def gather_cluster(self):
        self.setup_logger()
        t_start = time.time()
        logging.info('Starting Loads Kernel with job: ' + self.job_name)
        logging.info('user ' + getpass.getuser() + ' on ' + platform.node() + ' (' + platform.platform() + ')')
        logging.info('cluster gather mode')
        self.jcl = io_functions.specific_functions.load_jcl(self.job_name, self.path_input, self.jcl)
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)
        responses = io_functions.specific_functions.gather_responses(self.job_name, io_functions.specific_functions.check_path(self.path_output+'responses'))
        mon = gather_modul.GatherLoads(self.jcl, model)
        f = open(self.path_output + 'response_' + self.job_name + '.pickle', 'wb')  # open response
        for i in range(len(self.jcl.trimcase)):
            response = responses[[response['i'] for response in responses].index(i)]
            if response['successful']:
                mon.gather_monstations(self.jcl.trimcase[i], response)
                mon.gather_dyn2stat(response)

            logging.info('--> Saving response(s).')
            io_functions.specific_functions.dump_pickle(response, f)
        f.close()  # close response

        logging.info('--> Saving monstation(s).')
        with open(self.path_output + 'monstations_' + self.job_name + '.pickle', 'wb') as f:
            io_functions.specific_functions.dump_pickle(mon.monstations, f)

        logging.info('--> Saving dyn2stat.')
        with open(self.path_output + 'dyn2stat_' + self.job_name + '.pickle', 'wb') as f:
            io_functions.specific_functions.dump_pickle(mon.dyn2stat, f)
        logging.info('--> Done in {:.2f} [s].'.format(time.time() - t_start))
        
        logging.info('Loads Kernel finished.')
        self.print_logo()
        
    def run_pre(self):
        logging.info('--> Starting preprocessing.')
        t_start = time.time()
        model = model_modul.Model(self.jcl, self.path_output)
        model.build_model()
        model.write_aux_data()

        logging.info('--> Saving model data.')
        del model.jcl
        with open(self.path_output + 'model_' + self.job_name + '.pickle', 'wb') as f:
            io_functions.specific_functions.dump_pickle(model.__dict__, f)
        logging.info('--> Done in {:.2f} [s].'.format(time.time() - t_start))

    def main_common(self, model, jcl, i):
        logging.info('')
        logging.info('========================================')
        logging.info('trimcase: ' + jcl.trimcase[i]['desc'])
        logging.info('subcase: ' + str(jcl.trimcase[i]['subcase']))
        logging.info('(case ' + str(i + 1) + ' of ' + str(len(jcl.trimcase)) + ')')
        logging.info('========================================')
        trim_i = trim.Trim(model, jcl, jcl.trimcase[i], jcl.simcase[i])
        trim_i.set_trimcond()
        trim_i.exec_trim()
        # trim_i.iterative_trim()
        if trim_i.successful and 't_final' and 'dt' in jcl.simcase[i].keys():
            trim_i.exec_sim()
        elif trim_i.successful and 'flutter' in jcl.simcase[i] and jcl.simcase[i]['flutter']:
            trim_i.exec_flutter()
        elif trim_i.successful and 'derivatives' in jcl.simcase[i] and jcl.simcase[i]['derivatives']:
            trim_i.calc_jacobian()
            trim_i.calc_derivatives()
        response = trim_i.response
        response['i'] = i
        response['successful'] = trim_i.successful
        del trim_i
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
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)
        mon = gather_modul.GatherLoads(self.jcl, model)
        if self.restart:
            logging.info('Restart option: loading existing responses.')
            responses = io_functions.specific_functions.load_responses(self.job_name, self.path_output, remove_failed=True)
        f = open(self.path_output + 'response_' + self.job_name + '.pickle', 'wb')  # open response
        fid = io_functions.specific_functions.open_hdf5(self.path_output + 'response_' + self.job_name + '.hdf5')  # open response
        for i in range(len(self.jcl.trimcase)):
            if self.restart and i in [response['i'] for response in responses]:
                logging.info('Restart option: found existing response.')
                response = responses[[response['i'] for response in responses].index(i)]
            else:
                jcl = copy.deepcopy(self.jcl)
                if self.jcl.aero['method'] in ['cfd_steady']:
                    jcl.aero['mpi_hosts'] = self.setup_mpi_hosts(n_workers=1)  # assign hosts
                response = self.main_common(model, jcl, i)
            if response['successful']:
                mon.gather_monstations(self.jcl.trimcase[i], response)
                mon.gather_dyn2stat(response)

                logging.info('--> Saving response(s).')
                io_functions.specific_functions.dump_pickle(response, f)
                io_functions.specific_functions.write_hdf5(fid, response, path='/'+str(response['i']))

        f.close()  # close response
        io_functions.specific_functions.close_hdf5(fid)
        

        logging.info('--> Saving monstation(s).')
        with open(self.path_output + 'monstations_' + self.job_name + '.pickle', 'wb') as f:
            io_functions.specific_functions.dump_pickle(mon.monstations, f)
        io_functions.specific_functions.dump_hdf5(self.path_output + 'monstations_' + self.job_name + '.hdf5',
                                                  mon.monstations)
        
        #with open(self.path_output + 'monstations_' + self.job_name + '.mat', 'wb') as f:
        #   io_functions.matlab_functions.save_mat(f, mon.monstations)

        logging.info('--> Saving dyn2stat.')
        with open(self.path_output + 'dyn2stat_' + self.job_name + '.pickle', 'wb') as f:
            io_functions.specific_functions.dump_pickle(mon.dyn2stat, f)
        io_functions.specific_functions.dump_hdf5(self.path_output + 'dyn2stat_' + self.job_name + '.hdf5',
                                                  mon.dyn2stat)
        logging.info('--> Done in {:.2f} [s].'.format(time.time() - t_start))

    def run_main_parallel(self):
        """
        This function organizes the parallel computation of multiple load cases on one machine.
        Parallelization is achieved using a worker/listener concept. 
        The worker and the listener communicate via the output queue.
        """
        logging.info('--> Starting Main in parallel mode for %d trimcase(s).' % len(self.jcl.trimcase))
        t_start = time.time()
        manager = multiprocessing.Manager()
        q_input = manager.Queue()
        q_output = manager.Queue()

        # putting trimcases into queue
        for i in range(len(self.jcl.trimcase)):
            q_input.put(i)
        logging.info('--> All trimcases queued, waiting for execution.')

        if type(self.parallel) == int:
            n_processes = self.parallel
        else:
            n_processes = multiprocessing.cpu_count()
            if n_processes < 2:
                n_processes = 2
        
        pool = multiprocessing.Pool(n_processes)
        logging.info('--> Launching 1 listener.')
        listener = pool.apply_async(unwrap_main_listener, (self, q_output))  # put listener to work
        n_workers = n_processes - 1

        if self.jcl.aero['method'] in ['cfd_steady']:
            mpi_hosts = self.setup_mpi_hosts(n_workers)
             
        logging.info('--> Launching {} worker(s).'.format(str(n_workers)))
        workers = []
        for i_worker in range(n_workers):
            jcl = copy.deepcopy(self.jcl)
            if jcl.aero['method'] in ['cfd_steady']:
                jcl.aero['mpi_hosts'] = mpi_hosts[:jcl.aero['tau_cores']]  # assign hosts
                mpi_hosts = mpi_hosts[jcl.aero['tau_cores']:]  # remaining hosts
            workers.append(pool.apply_async(unwrap_main_worker, (self, q_input, q_output, jcl)))
 
        for i_worker in range(n_workers):
            q_input.put('finish')  # putting finish signal into queue for worker
        q_input.join()
        logging.info('--> All trimcases finished, waiting for listener.')
        q_output.join()
        q_output.put('finish')  # putting finish signal into queue for listener
        q_output.join()
        logging.info('--> Done in {:.2f} [s].'.format(time.time() - t_start))

    def main_worker(self, q_input, q_output, jcl):
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)
        while True:
            i = q_input.get()
            if i == 'finish':
                q_input.task_done()
                logging.info('--> Worker quit.')
                break
            else:
                response = self.main_common(model, jcl, i)
                q_output.put(response)
                q_input.task_done()
        return

    def main_listener(self, q_output):
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)
        mon = gather_modul.GatherLoads(self.jcl, model)
        f_response = open(self.path_output + 'response_' + self.job_name + '.pickle', 'wb')  # open response
        fid = io_functions.specific_functions.open_hdf5(self.path_output + 'response_' + self.job_name + '.hdf5')  # open response
        logging.info('--> Listener ready.')
        while True:
            m = q_output.get()
            if m == 'finish':
                f_response.close()  # close response
                io_functions.specific_functions.close_hdf5(fid)
                logging.info('--> Saving monstation(s).')
                with open(self.path_output + 'monstations_' + self.job_name + '.pickle', 'wb') as f:
                    io_functions.specific_functions.dump_pickle(mon.monstations, f)
                io_functions.specific_functions.dump_hdf5(self.path_output + 'monstations_' + self.job_name + '.hdf5',
                                                          mon.monstations)
                # with open(path_output + 'monstations_' + job_name + '.mat', 'wb') as f:
                #    io_matlab.save_mat(f, mon.monstations)
                logging.info('--> Saving dyn2stat.')
                with open(self.path_output + 'dyn2stat_' + self.job_name + '.pickle', 'wb') as f:
                    io_functions.specific_functions.dump_pickle(mon.dyn2stat, f)
                io_functions.specific_functions.dump_hdf5(self.path_output + 'dyn2stat_' + self.job_name + '.hdf5',
                                                  mon.dyn2stat)
                q_output.task_done()
                logging.info('--> Listener quit.')
                break
            elif m['successful']:
                logging.info("--> Received response ('successful') from worker.")
                mon.gather_monstations(self.jcl.trimcase[m['i']], m)
                mon.gather_dyn2stat(m)
                
            else:
                # trim failed, no post processing, save 'None'
                logging.info("--> Received response ('failed') from worker.")
            logging.info('--> Saving response(s).')
            io_functions.specific_functions.dump_pickle(m, f_response)
            io_functions.specific_functions.write_hdf5(fid, m, path='/'+str(m['i']))
            q_output.task_done()
        return  

    def run_main_single(self, i):
        """
        This function calculates one single load case, e.g. using CFD with mpi hosts on a cluster.
        """
        logging.info('--> Starting Main in single mode for {} trimcase(s).'.format(len(self.jcl.trimcase)))
        t_start = time.time()
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)
        
        jcl = copy.deepcopy(self.jcl)
        if self.jcl.aero['method'] in ['cfd_steady']:
            jcl.aero['mpi_hosts'] = self.setup_mpi_hosts(n_workers=1)  # assign hosts
            
        response = self.main_common(model, jcl, i)
        
        logging.info('--> Saving response(s).')
        path_responses = io_functions.specific_functions.check_path(self.path_output+'responses/')
        with open(path_responses + 'response_' + self.job_name + '_subcase_' + str(self.jcl.trimcase[i]['subcase']) + '.pickle', 'wb')  as f:
            io_functions.specific_functions.dump_pickle(response, f)
        logging.info('--> Done in {:.2f} [s].'.format(time.time() - t_start))

    def run_post(self):
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)

        logging.info('--> Loading monstations(s).') 
        monstations = io_functions.specific_functions.load_hdf5(self.path_output + 'monstations_' + self.job_name + '.hdf5')
            
        logging.info('--> Loading dyn2stat.')
        dyn2stat_data = io_functions.specific_functions.load_hdf5(self.path_output + 'dyn2stat_' + self.job_name + '.hdf5')
        
        responses = io_functions.specific_functions.load_hdf5_responses(self.job_name, self.path_output)

        logging.info('--> Drawing some standard plots.')
        plt = plotting_standard.StandardPlots(self.jcl, model)
        plt.add_monstations(monstations)
        plt.add_responses(responses)
        plt.plot_monstations(self.path_output + 'monstations_' + self.job_name + '.pdf')
        if any([x in self.jcl.simcase[0] and self.jcl.simcase[0][x] for x in ['gust', 'turbulence', 'cs_signal', 'controller']]):
            plt.plot_monstations_time(self.path_output + 'monstations_time_' + self.job_name + '.pdf') # nur sim
        elif 'flutter' in self.jcl.simcase[0] and self.jcl.simcase[0]['flutter']:
            plt.plot_fluttercurves_to_pdf(self.path_output + 'fluttercurves_' + self.job_name + '.pdf')
            plt.plot_eigenvalues_to_pdf(self.path_output + 'eigenvalues_' + self.job_name + '.pdf')
        elif 'derivatives' in self.jcl.simcase[0] and self.jcl.simcase[0]['derivatives']:
            plt.plot_eigenvalues_to_pdf(self.path_output + 'eigenvalues_' + self.job_name + '.pdf')
        elif 'limit_turbulence' in self.jcl.simcase[0] and self.jcl.simcase[0]['limit_turbulence']:
            plt = plotting_standard.TurbulencePlots(self.jcl, model)
            plt.add_monstations(monstations)
            plt.plot_monstations(self.path_output + 'monstations_turbulence_' + self.job_name + '.pdf')
            

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
            aux_out.write_successful_trimcases(self.path_output + 'successful_trimcases_' + self.job_name + '.csv') 
            aux_out.write_failed_trimcases(self.path_output + 'failed_trimcases_' + self.job_name + '.csv') 
            aux_out.write_critical_trimcases(self.path_output + 'crit_trimcases_' + self.job_name + '.csv') 
            aux_out.write_critical_nodalloads(self.path_output + 'nodalloads_' + self.job_name + '.bdf') 
            # aux_out.write_all_nodalloads(self.path_output + 'nodalloads_all_' + self.job_name + '.bdf')
            # aux_out.save_nodaldefo(self.path_output + 'nodaldefo_' + self.job_name)
            # aux_out.save_cpacs(self.path_output + 'cpacs_' + self.job_name + '.xml')

#         logging.info( '--> Drawing some more detailed plots.')         
#         plt = plotting_extra.DetailedPlots(self.jcl, model)
#         plt.add_responses(responses)
#         if 't_final' and 'dt' in self.jcl.simcase[0].keys():
#             # nur sim
#             plt.plot_time_data()
#         else:
#             # nur trim
#             #plt.plot_pressure_distribution()
#             plt.plot_forces_deformation_interactive()
#           
#         if 't_final' and 'dt' in self.jcl.simcase[0].keys():
#             plt = plotting_extra.Animations(self.jcl, model)
#             plt.add_responses(responses)
#             plt.make_animation()
#             #plt.make_movie(self.path_output, speedup_factor=1.0)

           
        return

    def run_test(self):
        # place code to test here
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)
        responses = io_functions.specific_functions.load_responses(self.job_name, self.path_output)
#         responses_hdf5 = io_functions.specific_functions.load_hdf5_responses(self.job_name, self.path_output)
#         with open(self.path_output + 'statespacemodel_' + self.job_name + '.pickle', 'wb') as fid:
#             io_functions.specific_functions.dump_pickle(responses, fid)

#         from scripts import plot_flexdefo
#         plot = plot_flexdefo.Flexdefo(self.jcl, model, responses)
#         plot.calc_flexdefos_trim()
#         plot.save_flexdefo_trim(self.path_output + 'flexdefo_' + self.job_name + '.pickle')
#         plot.plot_flexdefos_trim()
#         plot.plot_flexdefos()

#         from scripts import plot_lift_distribution
#         plot = plot_lift_distribution.Liftdistribution(self.jcl, model, responses)
#         plot.plot_aero_spanwise()
        
        

        return

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

    def setup_mpi_hosts(self, n_workers):
        n_required = self.jcl.aero['tau_cores'] * n_workers
        if self.machinefile == None:
            # all work is done on this node
            mpi_hosts = [platform.node()] * n_required
        else:
            mpi_hosts = []
            with open(self.machinefile) as f:
                lines = f.readlines()
            for line in lines:
                line = line.split(' slots=')
                mpi_hosts += [line[0]] * int(line[1])
        if mpi_hosts.__len__() < n_required:
            logging.error('Number of given hosts ({}) smaller than required hosts ({}). Exit.'.format(mpi_hosts.__len__(), n_required))
            sys.exit()
        return mpi_hosts

    def setup_logger_cluster(self, i):
        logger = logging.getLogger()
        if not logger.hasHandlers():
            path_log = io_functions.specific_functions.check_path(self.path_output+'log/')
            # define a Handler which writes INFO messages or higher to a log file
            logfile = logging.FileHandler(filename=path_log + 'log_' + self.job_name + '_subcase_' + str(self.jcl.trimcase[i]['subcase']) + ".txt", mode='w')
            formatter = logging.Formatter(fmt='%(asctime)s %(processName)-14s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
            logfile.setFormatter(formatter)
            # add the handler(s) to the root logger
            logger.setLevel(logging.INFO)
            logger.addHandler(logfile)

    def setup_logger(self):
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
            if not hdlr.baseFilename == self.path_output + 'log_' + self.job_name + ".txt":
                # In case the filename is incorrect, remove the handler completely from the logger.
                logger.removeHandler(hdlr)
                # Update the list of all existing loggers.
                existing_handlers = [hdlr.get_name() for hdlr in logger.handlers]
        # Add the following handlers only if they don't exist. This avoid duplicate lines/log entries.
        if 'lk_console' not in existing_handlers:
            # define a Handler which writes messages to the sys.stout
            console = logging.StreamHandler(sys.stdout)
            console.set_name('lk_console')
            # set a format which is simpler for console use
            formatter = logging.Formatter(fmt='%(levelname)s: %(message)s')
            # tell the handler to use this format
            console.setFormatter(formatter)
            # add the handler(s) to the root logger
            logger.addHandler(console)
        if 'lk_logfile' not in existing_handlers:
            # define a Handler which writes messages to a log file
            logfile = logging.FileHandler(filename=self.path_output + 'log_' + self.job_name + ".txt", mode='a')
            logfile.set_name('lk_logfile')
            formatter = logging.Formatter(fmt='%(asctime)s %(processName)-14s %(levelname)s: %(message)s',
                                          datefmt='%d/%m/%Y %H:%M:%S')
            logfile.setFormatter(formatter)
            logger.addHandler(logfile)

def unwrap_main_worker(*arg, **kwarg):
    # This is a function outside the class to unwrap the self from the arguments. Requirement for multiprocessing pool.
    return Kernel.main_worker(*arg, **kwarg)

def unwrap_main_listener(*arg, **kwarg):
    # This is a function outside the class to unwrap the self from the arguments. Requirement for multiprocessing pool.
    return Kernel.main_listener(*arg, **kwarg)

if __name__ == "__main__":
    print ("Please use the launch-script 'launch.py' from your input directory.")
    sys.exit()
