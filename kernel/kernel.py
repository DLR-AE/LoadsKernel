# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:00:31 2014

@author: voss_ar
"""
import numpy as np
#import pickle
import cPickle
import time  
import imp
import sys
import scipy
import logger as logger_modul 
import model as model_modul
import trim as trim_modul
import post_processing as post_processing_modul
import plotting as plotting_modul

def run_kernel(job_name, pre=False, main=False, post=False, test=False, path_input='../input/', path_output='../output/'):
    reload(sys)
    sys.stdout = logger_modul.logger(path_output + 'log_' + job_name + ".txt")
    
    print 'Starting Loads Kernel with job: ' + job_name
    print 'pre:  ' + str(pre)
    print 'main: ' + str(main)
    print 'post: ' + str(post)
    print 'test: ' + str(test)

    print '--> Reading parameters from JCL.'
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
        #post_processing.gather_monstations() # nur trim, wird zum plotten benoetigt
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        
        print '--> Saving response(s) and monstations.'  
        with open(path_output + 'response_' + job_name + '.pickle', 'w') as f:
            cPickle.dump(response, f, cPickle.HIGHEST_PROTOCOL)
        #with open(path_output + 'monstations_' + job_name + '.pickle', 'w') as f:
        #    cPickle.dump(post_processing.monstations, f, cPickle.HIGHEST_PROTOCOL)
        #with open(path_output + 'monstations_' + job_name + '.mat', 'w') as f:
        #    scipy.io.savemat(f, post_processing.monstations)

            
        print '--> Saving auxiliary output data.'  # nur trim
        #post_processing.save_monstations(path_output + 'monstations_' + job_name + '.bdf')     
        #post_processing.save_nodalloads(path_output + 'nodalloads_' + job_name + '.bdf')
        #post_processing.save_nodaldefo(path_output + 'nodaldefo_' + job_name)
        
        print '--> Drawing some plots.'  
        plotting = plotting_modul.plotting(jcl, model, response)
        #plotting.plot_monstations(post_processing.monstations, path_output + 'monstations_' + job_name + '.pdf') # nur trim
        #plotting.write_critical_trimcases(plotting.crit_trimcases, jcl.trimcase, path_output + 'crit_trimcases_' + job_name + '.csv') # nur trim
        #plotting.plot_forces_deformation_interactive() # nur trim
        plotting.plot_time_animation()

    if test:
        if not 'model' in locals():
            model = load_model(job_name, path_output)
        
        if not 'response' in locals():
            print '--> Loading response(s).'  
            with open(path_output + 'response_' + job_name + '.pickle', 'r') as f:
                response = cPickle.load(f)
        print 'test ready.' 
        # place code to test here



    print 'Loads Kernel finished.'

            
def load_model(job_name, path_output):
    print '--> Loading model data.'
    t_start = time.time()
    with open(path_output + 'model_' + job_name + '.pickle', 'r') as f:
        model = cPickle.load(f)
    print '--> Done in %.2f [sec].' % (time.time() - t_start)
    return model



if __name__ == "__main__":
    #run_kernel('jcl_ALLEGRA', main=True, post=True, path_output='/scratch/test/')
    #run_kernel('jcl_ALLEGRA', post=True, path_output='/scratch/test/')
    #run_kernel('jcl_ALLEGRA_CFD', pre=True, main=True, post=True, path_output='/scratch/kernel_Allegra_CFD/')
    #run_kernel('jcl_DLR_F19_manloads', pre=False, main=False, post=True, path_output='/scratch/test/')
    run_kernel('jcl_DLR_F19_gust', pre=False, main=True, post=True, path_output='/scratch/test/')
    #run_kernel('jcl_DLR_F19_gust',test=True, path_output='/scratch/test/')
    
    
    
    
    