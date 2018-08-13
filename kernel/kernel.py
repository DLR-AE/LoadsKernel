# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:00:31 2014

@author: voss_ar
"""
import cPickle, time, multiprocessing, getpass, platform, logging, sys, copy
import numpy as np
import io_functions
import trim
import post_processing
import monstations as monstations_module
import auxiliary_output
import plotting

def run_kernel(job_name, pre=False, main=False, post=False, main_debug=False, test=False, statespace=False, 
               path_input='../input/', 
               path_output='../output/', 
               jcl=None, parallel=False, restart=False,
               machinefile=None):
    io = io_functions.specific_functions()
    io_matlab = io_functions.matlab_functions()
    path_input = io.check_path(path_input) 
    path_output = io.check_path(path_output)    
    setup_logger(path_output, job_name )
    logging.info( 'Starting Loads Kernel with job: ' + job_name)
    logging.info( 'user ' + getpass.getuser() + ' on ' + platform.node() + ' (' + platform.platform() +')')
    logging.info( 'pre:  ' + str(pre))
    logging.info( 'main: ' + str(main))
    logging.info( 'post: ' + str(post))
    logging.info( 'test: ' + str(test))
   
    jcl = io.load_jcl(job_name, path_input, jcl)
        
    if pre: 
        logging.info( '--> Starting preprocessing.')  
        t_start = time.time()
        import model
        model = model.model(jcl, path_output)
        model.build_model()
        model.write_aux_data()
        logging.info( '--> Done in %.2f [sec].' % (time.time() - t_start))
        
        logging.info( '--> Saving model data.')
        t_start = time.time()
        del model.jcl
        with open(path_output + 'model_' + job_name + '.pickle', 'w') as f:
            io.dump_pickle(model.__dict__, f)
        logging.info( '--> Done in %.2f [sec].' % (time.time() - t_start))
        
    if main:
        logging.info( '--> Starting Main for %d trimcase(s).' % len(jcl.trimcase))
        t_start = time.time()        
        manager = multiprocessing.Manager()
        q_input = manager.Queue()  
        q_output = manager.Queue() 
        
        # putting trimcases into queue
        for i in range(len(jcl.trimcase)):
            q_input.put(i)
        logging.info( '--> All trimcases queued, waiting for execution.')
        
        if type(parallel)==int:
            n_processes = parallel
        elif parallel:
            n_processes = multiprocessing.cpu_count()/2
            if n_processes < 2 : n_processes = 2
        else: 
            n_processes = 2
            
        pool = multiprocessing.Pool(n_processes)
        logging.info( '--> Launching 1 listener.')
        listener = pool.apply_async(mainprocessing_listener, (q_output, path_output, job_name, jcl)) # put listener to work
        n_workers = n_processes - 1
        
        if jcl.aero['method'] in ['cfd_steady']:
            #mpi_hosts = ['rabe', 'rabe', 'rabe', 'kranich', 'kranich', 'kranich']
            mpi_hosts = setup_mpi_hosts(jcl, n_workers, machinefile)
        logging.info( '--> Launching {} worker(s).'.format(str(n_workers)))
        workers = []
        for i_worker in range(n_workers):
            i_jcl = copy.deepcopy(jcl)
            if jcl.aero['method'] in ['cfd_steady']:
                i_jcl.aero['mpi_hosts'] = mpi_hosts[:jcl.aero['tau_cores']] # assign hosts
                mpi_hosts = mpi_hosts[jcl.aero['tau_cores']:] # remaining hosts
            workers.append(pool.apply_async(mainprocessing_worker, (q_input, q_output, path_output, job_name, i_jcl)))
            
        q_input.join() # blocks until worker is done
        for i_worker in range(n_workers):
            q_input.put('finish') # putting finish signal into queue for worker
        q_input.join()
        logging.info( '--> All trimcases finished, waiting for listener.')
        q_output.join()
        q_output.put('finish') # putting finish signal into queue for listener
        q_output.join()
        
        logging.info( '--> Done in %.2f [sec].' % (time.time() - t_start))
    
    if main_debug:
        if not 'model' in locals():
            model = io.load_model(job_name, path_output)
        logging.info( '--> Starting Main in deprecated test-mode (!!!) for %d trimcase(s).' % len(jcl.trimcase))
        t_start = time.time()
        if jcl.aero['method'] in ['cfd_steady']:
            mpi_hosts = setup_mpi_hosts(jcl, n_workers=1, machinefile=machinefile)
        mon = monstations_module.monstations(jcl, model)
        if restart:
            logging.info('Restart option: loading existing responses.')
            responses = io.load_responses(job_name, path_output, remove_failed=True)
        f = open(path_output + 'response_' + job_name + '.pickle', 'w') # open response
        for i in range(len(jcl.trimcase)):
            logging.info( '')
            logging.info( '========================================')
            logging.info( 'trimcase: ' + jcl.trimcase[i]['desc'])
            logging.info( 'subcase: ' + str(jcl.trimcase[i]['subcase']))
            logging.info( '(case ' +  str(i+1) + ' of ' + str(len(jcl.trimcase)) + ')')
            logging.info( '========================================')
            if restart and i in [response['i'] for response in responses]:
                logging.info('Restart option: found existing response.')
                response = responses[[response['i'] for response in responses].index(i)]
            else:
                i_jcl = copy.deepcopy(jcl)
                if jcl.aero['method'] in ['cfd_steady']:
                    i_jcl.aero['mpi_hosts'] = mpi_hosts[:jcl.aero['tau_cores']] # assign hosts
                trim_i = trim.trim(model, i_jcl, i_jcl.trimcase[i], i_jcl.simcase[i])
                trim_i.set_trimcond()
                #trim_i.calc_derivatives()
                trim_i.exec_trim()
                #trim_i.iterative_trim()
                if trim_i.successful and 't_final' and 'dt' in jcl.simcase[i].keys():
                    trim_i.exec_sim()
                response = trim_i.response
                response['i'] = i
                response['successful'] = trim_i.successful
                del trim_i
            if response['successful']:
                post_processing_i = post_processing.post_processing(jcl, model, jcl.trimcase[i], response)
                post_processing_i.force_summation_method()
                post_processing_i.euler_transformation()
                post_processing_i.cuttingforces()
                mon.gather_monstations(jcl.trimcase[i], response)
                if 't_final' and 'dt' in jcl.simcase[i].keys():
                    mon.gather_dyn2stat(-1, response, mode='time-based')
                else:
                    mon.gather_dyn2stat(-1, response, mode='stat2stat')
                del post_processing_i
            
            logging.info( '--> Saving response(s).')
            io.dump_pickle(response, f)
            
            #with open(path_output + 'response_' + job_name + '_subcase_' + str(jcl.trimcase[i]['subcase']) + '.mat', 'w') as f2:
            #    io_matlab.save_mat(f2, response)
        f.close() # close response
        
        logging.info( '--> Saving monstation(s).')
        with open(path_output + 'monstations_' + job_name + '.pickle', 'w') as f:
            io.dump_pickle(mon.monstations, f)
        with open(path_output + 'monstations_' + job_name + '.mat', 'w') as f:
            io_matlab.save_mat(f, mon.monstations)
        
        logging.info( '--> Saving dyn2stat.')
        with open(path_output + 'dyn2stat_' + job_name + '.pickle', 'w') as f:
            io.dump_pickle(mon.dyn2stat, f)
        logging.info( '--> Done in %.2f [sec].' % (time.time() - t_start))
        
    if statespace:
        if not 'model' in locals():
            model = io.load_model(job_name, path_output)
        
        logging.info( '--> Starting State Space Matrix generation for %d trimcase(s).' % len(jcl.trimcase))
        t_start = time.time()
        f = open(path_output + 'response_' + job_name + '.pickle', 'w') # open response
        for i in range(len(jcl.trimcase)):
            logging.info( '')
            logging.info( '========================================')
            logging.info( 'trimcase: ' + jcl.trimcase[i]['desc'])
            logging.info( 'subcase: ' + str(jcl.trimcase[i]['subcase']))
            logging.info( '(case ' +  str(i+1) + ' of ' + str(len(jcl.trimcase)) + ')')
            logging.info( '========================================')
            
            trim_i = trim.trim(model, jcl, jcl.trimcase[i], jcl.simcase[i])
            trim_i.set_trimcond()
            trim_i.exec_trim()
            trim_i.calc_jacobian()
            trim_i.response['i'] = i
            logging.info( '--> Saving response(s).')
            io.dump_pickle(trim_i.response, f)

            del trim_i
        f.close() # close response
        logging.info( '--> Done in %.2f [sec].' % (time.time() - t_start))
        
    if post:
        if not 'model' in locals():
            model = io.load_model(job_name, path_output)

        logging.info( '--> Loading monstations(s).' ) 
        with open(path_output + 'monstations_' + job_name + '.pickle', 'r') as f:
            monstations = io.load_pickle(f)
            
        logging.info( '--> Loading dyn2stat.'  )
        with open(path_output + 'dyn2stat_' + job_name + '.pickle', 'r') as f:
            dyn2stat_data = io.load_pickle(f)

        logging.info( '--> Drawing some plots.' ) 
        plt = plotting.plotting(jcl, model)
        if 't_final' and 'dt' in jcl.simcase[0].keys():
            # nur sim
            plt.plot_monstations_time(monstations, path_output + 'monstations_time_' + job_name + '.pdf')
            plt.plot_monstations(monstations, path_output + 'monstations_' + job_name + '.pdf', dyn2stat=True) 
            #plt.plot_cs_signal() # Discus2c spezifisch
        else:
            # nur trim
            plt.plot_monstations(monstations, path_output + 'monstations_' + job_name + '.pdf') 
        

#         logging.info( '--> statespace analysis.')
#         import statespace_analysis
#         statespace_analysis = statespace.analysis(jcl, model, responses)
#         #statespace_analysis.analyse_states(path_output + 'analyse_of_states_' + job_name + '.pdf')
#         #statespace_analysis.plot_state_space_matrices()
#         statespace_analysis.analyse_eigenvalues(path_output + 'analyse_of_eigenvalues_' + job_name + '.pdf')
        
        logging.info( '--> Saving auxiliary output data.')
        aux_out = auxiliary_output.auxiliary_output(jcl, model, jcl.trimcase)
        aux_out.crit_trimcases = plt.crit_trimcases
        if ('t_final' and 'dt' in jcl.simcase[0].keys()): 
            aux_out.dyn2stat_data = dyn2stat_data
            aux_out.write_critical_trimcases(path_output + 'crit_trimcases_' + job_name + '.csv', dyn2stat=True) 
            aux_out.write_critical_nodalloads(path_output + 'nodalloads_' + job_name + '.bdf', dyn2stat=True) 
        else:
            # nur trim
            aux_out.response = io.load_responses(job_name, path_output, sorted=True)
            aux_out.write_successful_trimcases(path_output + 'successful_trimcases_' + job_name + '.csv') 
            aux_out.write_failed_trimcases(path_output + 'failed_trimcases_' + job_name + '.csv') 
            aux_out.write_critical_trimcases(path_output + 'crit_trimcases_' + job_name + '.csv', dyn2stat=False) 
            aux_out.write_critical_nodalloads(path_output + 'nodalloads_' + job_name + '.bdf', dyn2stat=False) 
            #aux_out.write_all_nodalloads(path_output + 'nodalloads_all_' + job_name + '.bdf')
            #aux_out.save_nodaldefo(path_output + 'nodaldefo_' + job_name)
            #aux_out.save_cpacs(path_output + 'cpacs_' + job_name + '.xml')

        #responses = io.load_responses(job_name, path_output)
        #print '--> Drawing more plots.'  
        #plt = plotting.plotting(jcl, model, responses)
        #if 't_final' and 'dt' in jcl.simcase[0].keys():
        #    # nur sim
        #    plt.plot_time_data()
        #    #plt.make_animation()
        #    #plt.make_movie(path_output, speedup_factor=1.0)
        #else:
        #    # nur trim
        #    plt.plot_pressure_distribution()
        #    plt.plot_forces_deformation_interactive() 
        
    if test:
        if not 'model' in locals():
            model = io.load_model(job_name, path_output)
        # place code to test here
#        responses = io.load_responses(job_name, path_output)
#        with open(path_output + 'monstations_' + job_name + '.pickle', 'r') as f:
#            monstations = io.load_pickle(f)
#  
#        import plots_for_Muldicon
#        plots = plots_for_Muldicon.Plots(jcl, model, responses)
#        plots.plot_aero_spanwise()
#         plots.plot_contributions()
#         plots.plot_time_data(job_name, path_output)
#         import plots_for_Discus2c
#         plots = plots_for_Discus2c.Plots(jcl, model, responses=responses, monstations=monstations)
#         plots.plot_ft()
#         plots.plot_contributions()
#         import plot_felxdefo
#         plots = plot_felxdefo.Flexdefo(jcl, model, responses)
#         plot.flexdefos()
#         import plots_for_HALO
#         plots = plots_for_HALO.Plots(path_output, jcl, model, responses=responses, monstations=monstations)
#         plots.plot_ft()

#         import test_smarty
#         test_smarty.interpolate_pkcfd(model, jcl)
        
#         import build_meshdefo
#         build_meshdefo.controlsurface_meshdefo(model, jcl, job_name, path_output)
        
#         from vergleich_druckverteilung import vergleich_druckverteilung
#         vergleich_druckverteilung(model, jcl.trimcase[0])
               
    logging.info( 'Loads Kernel finished.')
    print_logo()

def mainprocessing_worker(q_input, q_output, path_output, job_name, jcl):
    io = io_functions.specific_functions()
    if not 'model' in locals():
            model = io.load_model(job_name, path_output)
    while True:
        i = q_input.get()
        if i == 'finish':
            q_input.task_done()
            logging.info( '--> Worker quit.')
            break
        else:
            logging.info( '')
            logging.info( '========================================')
            logging.info( 'trimcase: ' + jcl.trimcase[i]['desc'])
            logging.info( 'subcase: ' + str(jcl.trimcase[i]['subcase']))
            logging.info( '(case ' +  str(i+1) + ' of ' + str(len(jcl.trimcase)) + ')')
            logging.info( '========================================')
            trim_i = trim.trim(model, jcl, jcl.trimcase[i], jcl.simcase[i])
            trim_i.set_trimcond()
            #trim_i.calc_derivatives()
            trim_i.exec_trim()
            if trim_i.successful and 't_final' and 'dt' in jcl.simcase[i].keys():
                trim_i.exec_sim()
            response = trim_i.response
            response['i'] = i
            response['successful'] = trim_i.successful
            del trim_i
            if response['successful']:
                post_processing_i = post_processing.post_processing(jcl, model, jcl.trimcase[i], response)
                post_processing_i.force_summation_method()
                post_processing_i.euler_transformation()
                post_processing_i.cuttingforces()
                logging.info( '--> Trimcase done, sending response to listener.')
                del post_processing_i
            else:
                # trim failed, no post processing
                logging.info( '--> Trimcase failed, sending response to listener.')
            q_output.put(response)
            q_input.task_done()
    return

def mainprocessing_listener(q_output, path_output, job_name, jcl):
    io = io_functions.specific_functions()
    if not 'model' in locals():
            model = io.load_model(job_name, path_output)
    mon = monstations_module.monstations(jcl, model)    
    f_response = open(path_output + 'response_' + job_name + '.pickle', 'w') # open response
    logging.info( '--> Listener ready.')
    while True:
        m = q_output.get()
        if m == 'finish':
            f_response.close() # close response
            logging.info( '--> Saving monstation(s).')
            with open(path_output + 'monstations_' + job_name + '.pickle', 'w') as f:
                io.dump_pickle(mon.monstations, f)
            #with open(path_output + 'monstations_' + job_name + '.mat', 'w') as f:
            #    io_matlab.save_mat(f, mon.monstations)
            logging.info( '--> Saving dyn2stat.')
            with open(path_output + 'dyn2stat_' + job_name + '.pickle', 'w') as f:
                io.dump_pickle(mon.dyn2stat, f)
            q_output.task_done()
            logging.info( '--> Listener quit.')
            break
        elif m['successful']:
            logging.info( "--> Received response ('successful') from worker.")
            mon.gather_monstations(jcl.trimcase[m['i']], m)
            if 't_final' and 'dt' in jcl.simcase[m['i']].keys():
                mon.gather_dyn2stat(-1, m, mode='time-based')
            else:
                mon.gather_dyn2stat(-1, m, mode='stat2stat')

        else:
            # trim failed, no post processing, save 'None'
            logging.info( "--> Received response ('failed') from worker.")
        logging.info( '--> Saving response(s).')
        io.dump_pickle(m, f_response)
        q_output.task_done()
            
    return
        
def print_logo():
    logging.info( '')
    logging.info( '       (  )')    
    logging.info( '      (    )')
    logging.info( '')
    logging.info( '              (   )' )
    logging.info( '             (     )')
    logging.info( '')
    logging.info( '         _|_')
    logging.info( ' ---------O---------')
    logging.info( '')
    logging.info( '')    

def setup_logger(path_output, job_name ):
    logging.basicConfig(format='%(asctime)s %(processName)-14s %(levelname)s: %(message)s', 
                        datefmt='%d/%m/%Y %H:%M:%S', 
                        level=logging.INFO,
                        filename=path_output+'log_'+job_name+".txt", 
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stout
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)   
    
def setup_mpi_hosts(jcl, n_workers, machinefile):
    n_required = jcl.aero['tau_cores'] * n_workers
    if machinefile == None:
        # all work is done on this node
        mpi_hosts = [platform.node()]*n_required
    else:
        mpi_hosts = []
        with open(machinefile) as f:
            lines = f.readlines()
        for line in lines[1:]: 
            # Use first host for Load Kernel only.
            # Use all other hosts (except first) for mpi executions.
            line = line.split(' slots=')
            mpi_hosts += [line[0]]*int(line[1])
    if mpi_hosts.__len__() < n_required:
         logging.error('Number of given hosts ({}) smaller than required hosts ({}). Exit.'.format(mpi_hosts.__len__(), n_required))
         sys.exit()
    return mpi_hosts
    
if __name__ == "__main__":
    print "Please use the launch-script 'launch.py' from your input directory."
    sys.exit()
    