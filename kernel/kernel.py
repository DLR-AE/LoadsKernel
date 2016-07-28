# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:00:31 2014

@author: voss_ar
"""
import cPickle, time, imp, sys, os
import scipy
import numpy as np
import logger as logger_modul 
import model as model_modul
import trim as trim_modul
import post_processing as post_processing_modul
import plotting as plotting_modul

def run_kernel(job_name, pre=False, main=False, post=False, test=False, path_input='../input/', path_output='../output/'):
    path_input = check_path(path_input) 
    path_output = check_path(path_output)    
    reload(sys)
    sys.stdout = logger_modul.logger(path_output + 'log_' + job_name + ".txt")
    
    print 'Starting Loads Kernel with job: ' + job_name
    print 'pre:  ' + str(pre)
    print 'main: ' + str(main)
    print 'post: ' + str(post)
    print 'test: ' + str(test)
   

    print '--> Reading parameters from JCL.'
    # import jcl dynamically by filename
    jcl_modul = imp.load_source('jcl', path_input + job_name + '.py')
    jcl = jcl_modul.jcl() 
        
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
        if not 'model' in locals():
            model = load_model(job_name, path_output)
        
        print '--> Starting Main for %d trimcase(s).' % len(jcl.trimcase)
        t_start = time.time()
        response = []
        for i in range(len(jcl.trimcase)):
            print ''
            print '========================================' 
            print 'trimcase: ' + jcl.trimcase[i]['desc']
            print 'subcase: ' + str(jcl.trimcase[i]['subcase'])
            print '(case ' +  str(i+1) + ' of ' + str(len(jcl.trimcase)) + ')' 
            print '========================================' 
            
            trim_i = trim_modul.trim(model, jcl, jcl.trimcase[i], jcl.simcase[i])
            trim_i.set_trimcond()
            trim_i.exec_trim()
            if 't_final' and 'dt' and 'gust' in jcl.simcase[i].keys():
                trim_i.exec_sim()
            response.append(trim_i.response)
            
        print '--> Saving response(s).'  
        with open(path_output + 'response_' + job_name + '.pickle', 'w') as f:
            cPickle.dump(response, f, cPickle.HIGHEST_PROTOCOL)
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
    
    if post:
        if not 'model' in locals():
            model = load_model(job_name, path_output)
        
        if not 'response' in locals():
            print '--> Loading response(s).'  
            with open(path_output + 'response_' + job_name + '.pickle', 'r') as f:
                response = cPickle.load(f)
        
        print '--> Starting Post for %d trimcase(s).' % len(jcl.trimcase)
        t_start = time.time()
        post_processing = post_processing_modul.post_processing(jcl, model, response)
        post_processing.force_summation_method() # trim + sim
        post_processing.cuttingforces() # trim + sim
        post_processing.gather_monstations() # trim + sim, wird zum plotten benoetigt
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        
        print '--> Saving response(s).'  
        with open(path_output + 'response_' + job_name + '.pickle', 'w') as f:
            cPickle.dump(response, f, cPickle.HIGHEST_PROTOCOL)
        for i in range(len(jcl.trimcase)):
            with open(path_output + 'response_' + job_name + '_subcase_' + str(jcl.trimcase[i]['subcase']) + '.mat', 'w') as f:
                scipy.io.savemat(f, response[i])
        print '--> Saving monstation(s).'  
        with open(path_output + 'monstations_' + job_name + '.pickle', 'w') as f:
            cPickle.dump(post_processing.monstations, f, cPickle.HIGHEST_PROTOCOL)
        with open(path_output + 'monstations_' + job_name + '.mat', 'w') as f:
            scipy.io.savemat(f, post_processing.monstations)
            
        if not ('t_final' and 'dt' in jcl.simcase[0].keys()):
            # nur trim
            print '--> Saving auxiliary output data.'  # nur trim
            post_processing.save_monstations(path_output + 'monstations_' + job_name + '.bdf')     
            post_processing.save_nodalloads(path_output + 'nodalloads_' + job_name + '.bdf')
            post_processing.save_nodaldefo(path_output + 'nodaldefo_' + job_name)
            post_processing.save_cpacs(path_output + 'cpacs_' + job_name + '.xml')
        
        print '--> Drawing some plots.'  
        
        if 't_final' and 'dt' in jcl.simcase[0].keys():
            # nur sim
            plotting_sim = plotting_modul.plotting_sim(jcl, model, response)
            plotting_sim.plot_monstations_time(post_processing.monstations, path_output + 'monstations_time_' + job_name + '.pdf')
            #plotting_sim.plot_cs_signal() # Discus2c spezifisch
            #plotting_sim.plot_time_animation(animation_dimensions = '3D')
            
        else:
            plotting_trim = plotting_modul.plotting_trim(jcl, model, response)
            # nur trim
            plotting_trim.plot_monstations(post_processing.monstations, path_output + 'monstations_' + job_name + '.pdf') 
            plotting_trim.write_critical_trimcases(plotting_trim.crit_trimcases, jcl.trimcase, path_output + 'crit_trimcases_' + job_name + '.csv') 
            #plotting_trim.plot_pressure_distribution()
            #plotting_trim.plot_forces_deformation_interactive() 

        
    if test:
        if not 'model' in locals():
            model = load_model(job_name, path_output)
        
        if not 'response' in locals():
            print '--> Loading response(s).'  
            with open(path_output + 'response_' + job_name + '.pickle', 'r') as f:
                response = cPickle.load(f)
        print 'test ready.' 
        
        with open(path_output + 'monstations_' + job_name + '.pickle', 'r') as f:
                monstations = cPickle.load(f)
        
        # place code to test here
        
#         plotting_sim = plotting_modul.plotting_sim(jcl, model, response)
#         plotting_sim.plot_time_animation_3d()

        
#         import test_smarty
#         test_smarty.interpolate_pkcfd(model, jcl)
        
#         import build_meshdefo
#         build_meshdefo.controlsurface_meshdefo(model, jcl, job_name, path_output)
        
#         from vergleich_druckverteilung import vergleich_druckverteilung
#         vergleich_druckverteilung(model, jcl.trimcase[0])
               
    print 'Loads Kernel finished.'
    print_logo()

            
def load_model(job_name, path_output):
    print '--> Loading model data.'
    t_start = time.time()
    with open(path_output + 'model_' + job_name + '.pickle', 'r') as f:
        model = cPickle.load(f)
    print '--> Done in %.2f [sec].' % (time.time() - t_start)
    return model

def check_path(path):
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
    