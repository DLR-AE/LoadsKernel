# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:00:31 2014

@author: voss_ar
"""
import time, multiprocessing, getpass, platform, logging, sys, copy
import io_functions.specific_functions
import io_functions.matlab_functions

import trim
import post_processing
import monstations as monstations_module
import auxiliary_output
import plotting_standard, plotting_extra


class Kernel():

    def __init__(self,
                 job_name, pre=False, main=False, post=False,
                 main_debug=False, test=False, statespace=False,
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
        self.main_debug = main_debug    # True/False
        self.restart = restart          # True/False
        # advanced options
        self.test = test                # True/False
        self.statespace = statespace    # True/False
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
        if self.statespace:
            self.run_statespace()
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
        mon = monstations_module.monstations(self.jcl, model)
        f = open(self.path_output + 'response_' + self.job_name + '.pickle', 'w')  # open response
        for i in range(len(self.jcl.trimcase)):
            response = responses[[response['i'] for response in responses].index(i)]
            if response['successful']:
                mon.gather_monstations(self.jcl.trimcase[i], response)
                if 't_final' and 'dt' in self.jcl.simcase[i].keys():
                    mon.gather_dyn2stat(-1, response, mode='time-based')
                else:
                    mon.gather_dyn2stat(-1, response, mode='stat2stat')

            logging.info('--> Saving response(s).')
            io_functions.specific_functions.dump_pickle(response, f)
        f.close()  # close response

        logging.info('--> Saving monstation(s).')
        with open(self.path_output + 'monstations_' + self.job_name + '.pickle', 'w') as f:
            io_functions.specific_functions.dump_pickle(mon.monstations, f)

        logging.info('--> Saving dyn2stat.')
        with open(self.path_output + 'dyn2stat_' + self.job_name + '.pickle', 'w') as f:
            io_functions.specific_functions.dump_pickle(mon.dyn2stat, f)
        logging.info('--> Done in {:.2f} [s].'.format(time.time() - t_start))
        
        logging.info('Loads Kernel finished.')
        self.print_logo()
        
    def run_pre(self):
        logging.info('--> Starting preprocessing.')
        t_start = time.time()
        import model as model_modul
        model = model_modul.Model(self.jcl, self.path_output)
        model.build_model()
        model.write_aux_data()

        logging.info('--> Saving model data.')
        del model.jcl
        with open(self.path_output + 'model_' + self.job_name + '.pickle', 'w') as f:
            io_functions.specific_functions.dump_pickle(model.__dict__, f)
        logging.info('--> Done in {:.2f} [s].'.format(time.time() - t_start))

    def main_common(self, model, i_jcl, i):
        logging.info('')
        logging.info('========================================')
        logging.info('trimcase: ' + i_jcl.trimcase[i]['desc'])
        logging.info('subcase: ' + str(i_jcl.trimcase[i]['subcase']))
        logging.info('(case ' + str(i + 1) + ' of ' + str(len(i_jcl.trimcase)) + ')')
        logging.info('========================================')
        trim_i = trim.Trim(model, i_jcl, i_jcl.trimcase[i], i_jcl.simcase[i])
        trim_i.set_trimcond()
        # trim_i.calc_derivatives()
        trim_i.exec_trim()
        # trim_i.iterative_trim()
        if trim_i.successful and 't_final' and 'dt' in i_jcl.simcase[i].keys():
            trim_i.exec_sim()
        response = trim_i.response
        response['i'] = i
        response['successful'] = trim_i.successful
        del trim_i
        if response['successful']:
            post_processing_i = post_processing.post_processing(i_jcl, model, i_jcl.trimcase[i], response)
            post_processing_i.force_summation_method()
            post_processing_i.euler_transformation()
            post_processing_i.cuttingforces()
            del post_processing_i
        return response

    def run_main_sequential(self):
        logging.info('--> Starting Main in sequential mode for {} trimcase(s).'.format(len(self.jcl.trimcase)))
        t_start = time.time()
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)
        mon = monstations_module.monstations(self.jcl, model)
        if self.restart:
            logging.info('Restart option: loading existing responses.')
            responses = io_functions.specific_functions.load_responses(self.job_name, self.path_output, remove_failed=True)
        f = open(self.path_output + 'response_' + self.job_name + '.pickle', 'w')  # open response
        for i in range(len(self.jcl.trimcase)):
            if self.restart and i in [response['i'] for response in responses]:
                logging.info('Restart option: found existing response.')
                response = responses[[response['i'] for response in responses].index(i)]
            else:
                i_jcl = copy.deepcopy(self.jcl)
                if self.jcl.aero['method'] in ['cfd_steady']:
                    i_jcl.aero['mpi_hosts'] = self.setup_mpi_hosts(n_workers=1)  # assign hosts
                response = self.main_common(model, i_jcl, i)
            if response['successful']:
                mon.gather_monstations(self.jcl.trimcase[i], response)
                if 't_final' and 'dt' in self.jcl.simcase[i].keys():
                    mon.gather_dyn2stat(-1, response, mode='time-based')
                else:
                    mon.gather_dyn2stat(-1, response, mode='stat2stat')

                logging.info('--> Saving response(s).')
                io_functions.specific_functions.dump_pickle(response, f)
                #with open(self.path_output + 'response_' + self.job_name + '_subcase_' + str(self.jcl.trimcase[i]['subcase']) + '.mat', 'w') as f2:
                #   io_functions.matlab_functions.save_mat(f2, response)
        f.close()  # close response

        logging.info('--> Saving monstation(s).')
        with open(self.path_output + 'monstations_' + self.job_name + '.pickle', 'w') as f:
            io_functions.specific_functions.dump_pickle(mon.monstations, f)
        #with open(self.path_output + 'monstations_' + self.job_name + '.mat', 'w') as f:
        #   io_functions.matlab_functions.save_mat(f, mon.monstations)

        logging.info('--> Saving dyn2stat.')
        with open(self.path_output + 'dyn2stat_' + self.job_name + '.pickle', 'w') as f:
            io_functions.specific_functions.dump_pickle(mon.dyn2stat, f)
        logging.info('--> Done in {:.2f} [s].'.format(time.time() - t_start))

    def run_main_parallel(self):
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
            i_jcl = copy.deepcopy(self.jcl)
            if i_jcl.aero['method'] in ['cfd_steady']:
                i_jcl.aero['mpi_hosts'] = mpi_hosts[:i_jcl.aero['tau_cores']]  # assign hosts
                mpi_hosts = mpi_hosts[i_jcl.aero['tau_cores']:]  # remaining hosts
            workers.append(pool.apply_async(unwrap_main_worker, (self, q_input, q_output, i_jcl)))
 
        for i_worker in range(n_workers):
            q_input.put('finish')  # putting finish signal into queue for worker
        q_input.join()
        logging.info('--> All trimcases finished, waiting for listener.')
        q_output.join()
        q_output.put('finish')  # putting finish signal into queue for listener
        q_output.join()
        logging.info('--> Done in {:.2f} [s].'.format(time.time() - t_start))

    def main_worker(self, q_input, q_output, i_jcl):
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)
        while True:
            i = q_input.get()
            if i == 'finish':
                q_input.task_done()
                logging.info('--> Worker quit.')
                break
            else:
                response = self.main_common(model, i_jcl, i)
                q_output.put(response)
                q_input.task_done()
        return

    def main_listener(self, q_output):
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)
        mon = monstations_module.monstations(self.jcl, model)
        f_response = open(self.path_output + 'response_' + self.job_name + '.pickle', 'w')  # open response
        logging.info('--> Listener ready.')
        while True:
            m = q_output.get()
            if m == 'finish':
                f_response.close()  # close response
                logging.info('--> Saving monstation(s).')
                with open(self.path_output + 'monstations_' + self.job_name + '.pickle', 'w') as f:
                    io_functions.specific_functions.dump_pickle(mon.monstations, f)
                # with open(path_output + 'monstations_' + job_name + '.mat', 'w') as f:
                #    io_matlab.save_mat(f, mon.monstations)
                logging.info('--> Saving dyn2stat.')
                with open(self.path_output + 'dyn2stat_' + self.job_name + '.pickle', 'w') as f:
                    io_functions.specific_functions.dump_pickle(mon.dyn2stat, f)
                q_output.task_done()
                logging.info('--> Listener quit.')
                break
            elif m['successful']:
                logging.info("--> Received response ('successful') from worker.")
                mon.gather_monstations(self.jcl.trimcase[m['i']], m)
                if 't_final' and 'dt' in self.jcl.simcase[m['i']].keys():
                    mon.gather_dyn2stat(-1, m, mode='time-based')
                else:
                    mon.gather_dyn2stat(-1, m, mode='stat2stat')
            else:
                # trim failed, no post processing, save 'None'
                logging.info("--> Received response ('failed') from worker.")
            logging.info('--> Saving response(s).')
            io_functions.specific_functions.dump_pickle(m, f_response)
            q_output.task_done()
        return  

    def run_main_single(self, i):
        logging.info('--> Starting Main in single mode for {} trimcase(s).'.format(len(self.jcl.trimcase)))
        t_start = time.time()
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)
        
        i_jcl = copy.deepcopy(self.jcl)
        if self.jcl.aero['method'] in ['cfd_steady']:
            i_jcl.aero['mpi_hosts'] = self.setup_mpi_hosts(n_workers=1)  # assign hosts
            
        response = self.main_common(model, i_jcl, i)
        
        logging.info('--> Saving response(s).')
        path_responses = io_functions.specific_functions.check_path(self.path_output+'responses/')
        with open(path_responses + 'response_' + self.job_name + '_subcase_' + str(self.jcl.trimcase[i]['subcase']) + '.pickle', 'w')  as f:
            io_functions.specific_functions.dump_pickle(response, f)
        logging.info('--> Done in {:.2f} [s].'.format(time.time() - t_start))

    def run_statespace(self):
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)

        logging.info('--> Starting State Space Matrix generation for %d trimcase(s).' % len(self.jcl.trimcase))
        t_start = time.time()
        f = open(self.path_output + 'response_' + self.job_name + '.pickle', 'w')  # open response
        for i in range(len(self.jcl.trimcase)):
            logging.info('')
            logging.info('========================================')
            logging.info('trimcase: ' + self.jcl.trimcase[i]['desc'])
            logging.info('subcase: ' + str(self.jcl.trimcase[i]['subcase']))
            logging.info('(case ' + str(i + 1) + ' of ' + str(len(self.jcl.trimcase)) + ')')
            logging.info('========================================')
            
            trim_i = trim.Trim(model, self.jcl, self.jcl.trimcase[i], self.jcl.simcase[i])
            trim_i.set_trimcond()
            trim_i.exec_trim()
            trim_i.calc_jacobian()
            trim_i.response['i'] = i
            logging.info('--> Saving response(s).')
            io_functions.specific_functions.dump_pickle(trim_i.response, f)

            del trim_i
        f.close()  # close response
        logging.info('--> Done in {:.2f} [s].'.format(time.time() - t_start))

    def run_post(self):
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)

        logging.info('--> Loading monstations(s).') 
        with open(self.path_output + 'monstations_' + self.job_name + '.pickle', 'r') as f:
            monstations = io_functions.specific_functions.load_pickle(f)

        logging.info('--> Loading dyn2stat.')
        with open(self.path_output + 'dyn2stat_' + self.job_name + '.pickle', 'r') as f:
            dyn2stat_data = io_functions.specific_functions.load_pickle(f)

        logging.info('--> Drawing some standard plots.')
        plt = plotting_standard.StandardPlots(self.jcl, model)
        plt.add_monstations(monstations)
        plt.plot_monstations(self.path_output + 'monstations_' + self.job_name + '.pdf')
        if 't_final' and 'dt' in self.jcl.simcase[0].keys():
            plt.plot_monstations_time(self.path_output + 'monstations_time_' + self.job_name + '.pdf') # nur sim

        logging.info('--> Saving auxiliary output data.')
        aux_out = auxiliary_output.AuxiliaryOutput(self.jcl, model, self.jcl.trimcase)
        aux_out.crit_trimcases = plt.crit_trimcases
        if ('t_final' and 'dt' in self.jcl.simcase[0].keys()): 
            aux_out.dyn2stat_data = dyn2stat_data
            aux_out.write_critical_trimcases(self.path_output + 'crit_trimcases_' + self.job_name + '.csv', dyn2stat=True) 
            aux_out.write_critical_nodalloads(self.path_output + 'nodalloads_' + self.job_name + '.bdf', dyn2stat=True) 
        else:
            # nur trim
            aux_out.responses = io_functions.specific_functions.load_responses(self.job_name, self.path_output, sorted=True)
            aux_out.write_trimresults(self.path_output + 'trim_results_' + self.job_name + '.csv')
            aux_out.write_successful_trimcases(self.path_output + 'successful_trimcases_' + self.job_name + '.csv') 
            aux_out.write_failed_trimcases(self.path_output + 'failed_trimcases_' + self.job_name + '.csv') 
            aux_out.write_critical_trimcases(self.path_output + 'crit_trimcases_' + self.job_name + '.csv', dyn2stat=False) 
            aux_out.write_critical_nodalloads(self.path_output + 'nodalloads_' + self.job_name + '.bdf', dyn2stat=False) 
            # aux_out.write_all_nodalloads(self.path_output + 'nodalloads_all_' + self.job_name + '.bdf')
            # aux_out.save_nodaldefo(self.path_output + 'nodaldefo_' + self.job_name)
            # aux_out.save_cpacs(self.path_output + 'cpacs_' + self.job_name + '.xml')

#         responses = io_functions.specific_functions.load_responses(self.job_name, self.path_output)
#         print '--> Drawing some more detailed plots.'  
#         plt = plotting_extra.DetailedPlots(self.jcl, model)
#         plt.add_responses(responses)
#         if 't_final' and 'dt' in self.jcl.simcase[0].keys():
#            # nur sim
#            plt.plot_time_data()
#         else:
#            # nur trim
#            plt.plot_pressure_distribution()
#            plt.plot_forces_deformation_interactive()
        
#         if 't_final' and 'dt' in self.jcl.simcase[0].keys():
#             plt = plotting_extra.Animations(self.jcl, model)
#             plt.add_responses(responses)
#             plt.make_animation()
#             #plt.make_movie(self.path_output, speedup_factor=1.0)

#         logging.info( '--> statespace analysis.')
#         import statespace_analysis
#         statespace_analysis = statespace_analysis.analysis(self.jcl, model, responses)
#         #statespace_analysis.analyse_states(path_output + 'analyse_of_states_' + job_name + '.pdf')
#         #statespace_analysis.plot_state_space_matrices()
#         statespace_analysis.analyse_eigenvalues(self.path_output + 'analyse_of_eigenvalues_' + self.job_name + '.pdf')
           
        return

    def run_test(self):
        model = io_functions.specific_functions.load_model(self.job_name, self.path_output)
#         import freq_dom
#         flutter = freq_dom.Flutter(fluttercase=self.jcl.trimcase[0], model=model, jcl=self.jcl)
#         flutter.k_method()
        
        
        # place code to test here
#         responses = io_functions.specific_functions.load_responses(self.job_name, self.path_output)
#         with open(self.path_output + 'monstations_' + self.job_name + '.pickle', 'r') as f:
#             monstations = io_functions.specific_functions.load_pickle(f)
#         from scripts import cps_for_MULDICON
#         cps = cps_for_MULDICON.CPs(self.jcl, model, responses)
#         cps.plot()
#        import plots_for_Muldicon
#        plots = plots_for_Muldicon.Plots(jcl, model, responses)
#        plots.plot_aero_spanwise()
#         plots.plot_contributions()
#         plots.plot_time_data(job_name, path_output)
#         import plots_for_Discus2c
#         plots = plots_for_Discus2c.Plots(jcl, model, responses=responses, monstations=monstations)
#         plots.plot_ft()
#         plots.plot_contributions()
#        from scripts import plot_flexdefo
#        plot = plot_flexdefo.Flexdefo(self.jcl, model, responses)
#        plot.plot_flexdefos_trim()
#         import plots_for_HALO
#         plots = plots_for_HALO.Plots(path_output, jcl, model, responses=responses, monstations=monstations)
#         plots.plot_ft()

#         import test_smarty
#         test_smarty.interpolate_pkcfd(model, jcl)

#         import build_meshdefo
#         build_meshdefo.controlsurface_meshdefo(model, jcl, job_name, path_output)

#         from vergleich_druckverteilung import vergleich_druckverteilung
#         vergleich_druckverteilung(model, jcl.trimcase[0])
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
        path_log = io_functions.specific_functions.check_path(self.path_output+'log/')
        # define a Handler which writes INFO messages or higher to a log file
        logfile = logging.FileHandler(filename=path_log + 'log_' + self.job_name + '_subcase_' + str(self.jcl.trimcase[i]['subcase']) + ".txt", mode='w')
        logfile.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s %(processName)-14s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
        logfile.setFormatter(formatter)
        # add the handler(s) to the root logger
        logger = logging.getLogger('')
        logger.setLevel(logging.INFO)
        logger.addHandler(logfile)
    
    def setup_logger(self):
        # define a Handler which writes INFO messages or higher to the sys.stout
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(levelname)s: %(message)s')  # set a format which is simpler for console use
        console.setFormatter(formatter)  # tell the handler to use this format
        # define a Handler which writes INFO messages or higher to a log file
        logfile = logging.FileHandler(filename=self.path_output + 'log_' + self.job_name + ".txt", mode='a')
        logfile.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s %(processName)-14s %(levelname)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
        logfile.setFormatter(formatter)
        # add the handler(s) to the root logger
        logger = logging.getLogger('')
        logger.setLevel(logging.INFO)
        logger.addHandler(console)
        logger.addHandler(logfile)

def unwrap_main_worker(*arg, **kwarg):
    # This is a function outside the class to unwrap the self from the arguments. Requirement for multiprocessing pool.
    return Kernel.main_worker(*arg, **kwarg)

def unwrap_main_listener(*arg, **kwarg):
    # This is a function outside the class to unwrap the self from the arguments. Requirement for multiprocessing pool.
    return Kernel.main_listener(*arg, **kwarg)

if __name__ == "__main__":
    print "Please use the launch-script 'launch.py' from your input directory."
    sys.exit()
