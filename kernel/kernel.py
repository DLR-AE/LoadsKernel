# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:00:31 2014

@author: voss_ar
"""
import cPickle, time, imp, sys, os, multiprocessing, psutil
import scipy
import numpy as np
import logger as logger_modul 
import model as model_modul
import trim as trim_modul
import post_processing as post_processing_modul
import monstations as monstations_modul
import auxiliary_output as auxiliary_output_modul
import plotting as plotting_modul

def run_kernel(job_name, pre=False, main=False, post=False, test=False, path_input='../input/', path_output='../output/', jcl=None, parallel=False):
    path_input = check_path(path_input) 
    path_output = check_path(path_output)    
    #reload(sys)
    #sys.stdout = logger_modul.logger(path_output + 'log_' + job_name + ".txt")
    
    print 'Starting Loads Kernel with job: ' + job_name
    print 'pre:  ' + str(pre)
    print 'main: ' + str(main)
    print 'post: ' + str(post)
    print 'test: ' + str(test)
   
    jcl = load_jcl(job_name, path_input, jcl)
        
    if pre: 
        print '--> Starting preprocessing.'   
        t_start = time.time()
        model = model_modul.model(jcl)
        model.build_model()
        model.write_aux_data(path_output)
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        
        print '--> Saving model data.'
        t_start = time.time()
        with open(path_output + 'model_' + job_name + '.pickle', 'w') as f:
            cPickle.dump(model, f, cPickle.HIGHEST_PROTOCOL)
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        
    if main:
        print '--> Starting Main for %d trimcase(s).' % len(jcl.trimcase)
        t_start = time.time()        
        manager = multiprocessing.Manager()
        q_input = manager.Queue()  
        q_output = manager.Queue() 
        
        # putting trimcases into queue
        for i in range(len(jcl.trimcase)):
            q_input.put(i)
        print 'All trimcases queued, waiting for execution.'
        
        if parallel:
            n_processes = multiprocessing.cpu_count()/2
            if n_processes < 2 : n_processes = 2
        else: 
            n_processes = 2
            
        pool = multiprocessing.Pool(n_processes)
        print 'Launching 1 listener.'
        listener = pool.apply_async(mainprocessing_listener, (q_output, path_output, job_name, jcl)) # put listener to work
        n_workers = n_processes - 1
        print 'Launching {} worker(s).'.format(str(n_workers))
        workers = []
        for i_worker in range(n_workers):
            workers.append(pool.apply_async(mainprocessing_worker, (q_input, q_output, path_output, job_name, jcl)))
            
        q_input.join() # blocks until worker is done
        for i_worker in range(n_workers):
            q_input.put('finish') # putting finish signal into queue for worker
        q_input.join()
        print 'All trimcases finished, waiting for listener.'
        q_output.join()
        q_output.put('finish') # putting finish signal into queue for listener
        q_output.join()
        
        print '--> Done in %.2f [sec].' % (time.time() - t_start)

    if post:
        if not 'model' in locals():
            model = load_model(job_name, path_output)

        print '--> Loading monstations(s).'  
        with open(path_output + 'monstations_' + job_name + '.pickle', 'r') as f:
            monstations = cPickle.load(f)
            
        print '--> Loading dyn2stat.'  
        with open(path_output + 'dyn2stat_' + job_name + '.pickle', 'r') as f:
            dyn2stat = cPickle.load(f)

        print '--> Drawing some plots.'  
        plotting = plotting_modul.plotting(jcl, model)
        if 't_final' and 'dt' in jcl.simcase[0].keys():
            # nur sim
            plotting.plot_monstations_time(monstations, path_output + 'monstations_time_' + job_name + '.pdf')
            plotting.plot_monstations(monstations, path_output + 'monstations_' + job_name + '.pdf', dyn2stat=True) 
            plotting.write_critical_trimcases(path_output + 'crit_trimcases_' + job_name + '.csv', dyn2stat=True) 
            plotting.save_dyn2stat(dyn2stat, path_output + 'nodalloads_' + job_name + '.bdf') 
            #plotting.plot_cs_signal() # Discus2c spezifisch
        else:
            # nur trim
            plotting.plot_monstations(monstations, path_output + 'monstations_' + job_name + '.pdf') 
            plotting.write_critical_trimcases(path_output + 'crit_trimcases_' + job_name + '.csv') 
        
        # ----------------------------
        # --- try to load response ---
        # ----------------------------
        responses = load_response(job_name, path_output)
        
        print '--> Saving auxiliary output data.'
        if not ('t_final' and 'dt' in jcl.simcase[0].keys()): 
            # nur trim
            auxiliary_output = auxiliary_output_modul.auxiliary_output(jcl, model, jcl.trimcase, responses)
            auxiliary_output.save_nodalloads(path_output + 'nodalloads_' + job_name + '.bdf')
            auxiliary_output.save_nodaldefo(path_output + 'nodaldefo_' + job_name)
            #auxiliary_output.save_cpacs(path_output + 'cpacs_' + job_name + '.xml')
            
#         print '--> Drawing some plots.'  
#         plotting = plotting_modul.plotting(jcl, model, responses)
#         if 't_final' and 'dt' in jcl.simcase[0].keys():
#             # nur sim
#             plotting.plot_time_data(animation_dimensions = '3D')
#             #plotting.make_movie(path_output, speedup_factor=1.0)
#         else:
#             # nur trim
#             plotting.plot_pressure_distribution()
#             plotting.plot_forces_deformation_interactive() 
        
    if test:
        if not 'model' in locals():
            model = load_model(job_name, path_output)
        
        responses = load_response(job_name, path_output)

        with open(path_output + 'monstations_' + job_name + '.pickle', 'r') as f:
                monstations = cPickle.load(f)
                
        print 'test ready.' 
        # place code to test here
                
#         import test_smarty
#         test_smarty.interpolate_pkcfd(model, jcl)
        
#         import build_meshdefo
#         build_meshdefo.controlsurface_meshdefo(model, jcl, job_name, path_output)
        
#         from vergleich_druckverteilung import vergleich_druckverteilung
#         vergleich_druckverteilung(model, jcl.trimcase[0])
               
    print 'Loads Kernel finished.'
    print_logo()

def mainprocessing_worker(q_input, q_output, path_output, job_name, jcl):
    if not 'model' in locals():
            model = load_model(job_name, path_output)
    while True:
        i = q_input.get()
        if i == 'finish':
            q_input.task_done()
            print '--> Worker quit.'
            break
        else:
            print ''
            print '========================================' 
            print 'trimcase: ' + jcl.trimcase[i]['desc']
            print 'subcase: ' + str(jcl.trimcase[i]['subcase'])
            print '(case ' +  str(i+1) + ' of ' + str(len(jcl.trimcase)) + ')' 
            print '========================================' 
            trim_i = trim_modul.trim(model, jcl, jcl.trimcase[i], jcl.simcase[i])
            trim_i.set_trimcond()
            trim_i.exec_trim()
            if 't_final' and 'dt' in jcl.simcase[i].keys():
                trim_i.exec_sim()
            post_processing_i = post_processing_modul.post_processing(jcl, model, jcl.trimcase[i], trim_i.response)
            post_processing_i.force_summation_method()
            #post_processing_i.euler_transformation()
            post_processing_i.cuttingforces()
            trim_i.response['i'] = i
            q_output.put(trim_i.response)
            del trim_i, post_processing_i
            q_input.task_done()
    return

def mainprocessing_listener(q_output, path_output, job_name, jcl):
    if not 'model' in locals():
            model = load_model(job_name, path_output)
    monstations = monstations_modul.monstations(jcl, model)    
    f_response = open(path_output + 'response_' + job_name + '.pickle', 'w') # open response
    print '--> Listener ready.'
    while True:
        m = q_output.get()
        if m == 'finish':
            f_response.close() # close response
            print '--> Saving monstation(s).'  
            with open(path_output + 'monstations_' + job_name + '.pickle', 'w') as f:
                cPickle.dump(monstations.monstations, f, cPickle.HIGHEST_PROTOCOL)
            with open(path_output + 'monstations_' + job_name + '.mat', 'w') as f:
                scipy.io.savemat(f, monstations.monstations)
            print '--> Saving dyn2stat.'  
            with open(path_output + 'dyn2stat_' + job_name + '.pickle', 'w') as f:
                cPickle.dump(monstations.dyn2stat, f, cPickle.HIGHEST_PROTOCOL)
            q_output.task_done()
            print '--> Listener quit.'
            break
        else:
            monstations.gather_monstations(jcl.trimcase[m['i']], m)
            if 't_final' and 'dt' in jcl.simcase[m['i']].keys():
                monstations.gather_dyn2stat(-1, m)
            print '--> Saving response(s).'  
            cPickle.dump(m, f_response, cPickle.HIGHEST_PROTOCOL)
            q_output.task_done()
    return

def load_jcl(job_name, path_input, jcl):
    if jcl == None:
        print '--> Reading parameters from JCL.'
        # import jcl dynamically by filename
        jcl_modul = imp.load_source('jcl', path_input + job_name + '.py')
        jcl = jcl_modul.jcl() 
    # small check for completeness
    attributes = ['general', 'efcs', 'geom', 'aero', 'spline', 'mass', 'atmo', 'trimcase', 'simcase']
    for attribute in attributes:
        if not hasattr(jcl, attribute):
            print 'JCL appears to be incomplete: jcl.{} missing. Exit.'.format(attribute)
            sys.exit()
    return jcl
                
def load_model(job_name, path_output):
    print '--> Loading model data.'
    t_start = time.time()
    with open(path_output + 'model_' + job_name + '.pickle', 'r') as f:
        model = cPickle.load(f)
    print '--> Done in %.2f [sec].' % (time.time() - t_start)
    return model
    
    
def load_response(job_name, path_output):
    print '--> Loading response(s).'  
    filename = path_output + 'response_' + job_name + '.pickle'
    filestats = os.stat(filename)
    filesize_mb = filestats.st_size /1024**2
    mem = psutil.virtual_memory()
    mem_total_mb = mem.total /1024**2
    print 'size of total memory: ' + str(mem_total_mb) + ' Mb'
    print 'size of response: ' + str(filesize_mb) + ' Mb'
    if filesize_mb > mem_total_mb:
        print 'Response too large. Exit.'
        sys.exit()
    else:
        t_start = time.time()
        f = open(filename, 'r')
        response = []
        while True:
            try:
                response.append(cPickle.load(f))
            except EOFError:
                break
        f.close()
        # sort response
        pos_sorted = np.argsort([resp['i'] for resp in response ])
        response = [ response[x] for x in pos_sorted]
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        return response 
    
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.isdir(path) and os.access(os.path.dirname(path), os.W_OK):
        return os.path.join(path, './') # sicherstellen, dass der Pfad mit / endet
    else:
        print 'Path ' + str(path)  + ' not valid. Exit.'
        sys.exit()
        
def print_logo():
    print ''
    print '       (  )'    
    print '      (    )'
    print ''    
    print '              (   )' 
    print '             (     )'
    print ''   
    print '         _|_'    
    print ' ---------O---------'
    print ''
    print ''         
    
if __name__ == "__main__":
    print "Please use the launch-script 'launch.py' from your input directory."
    sys.exit()
    