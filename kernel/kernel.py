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

def run_kernel(job_name, pre=False, main=False, post=False, test=False):
    
    print 'Starting AE Kernel with job: ' + job_name
    print 'pre:  ' + str(pre)
    print 'main: ' + str(main)
    print 'post: ' + str(post)
    print 'test: ' + str(test)

    print '--> Reading parameters from JCL.'
    jcl = imp.load_source('jcl', '../input/' + job_name + '.py')
    jcl = jcl.jcl() 
        
    if pre: 
        from model import model as model_obj
        print '--> Starting preprocessing.'   
        t_start = time.time()
        model = model_obj(jcl)
        model.build_model()
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        
        print '--> Saving model data.'
        t_start = time.time()
        f = open('../output/model_' + job_name + '.pickle', 'w')
        cPickle.dump(model, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        
    if main:
        from trim import trim
        if not 'model' in locals():
            model = load_model(job_name)
        
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
            t_start = time.time()
            trim_i = trim(model, jcl, jcl.trimcase[i])
            trim_i.set_trimcond()
            trim_i.exec_trim()
            response.append(trim_i.response)
            
        print '--> Saving response(s).'  
        f = open('../output/response_' + job_name + '.pickle', 'w')
        cPickle.dump(response, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
    
    if post:
        from post_processing import post_processing as post_processing_obj
        if not 'model' in locals():
            model = load_model(job_name)
        
        if not 'response' in locals():
            print '--> Loading response(s).'  
            with open('../output/response_' + job_name + '.pickle', 'r') as f:
                response = cPickle.load(f)
        
        print '--> Starting Post for %d trimcase(s).' % len(jcl.trimcase)
        t_start = time.time()
        post_processing = post_processing_obj(jcl, model, response)
        post_processing.force_summation_method()
        post_processing.cuttingforces()
        post_processing.gather_monstations() # wird zum plotten benoetigt
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        
        print '--> Saving response(s) and monstations.'  
        with open('../output/response_' + job_name + '.pickle', 'w') as f:
            cPickle.dump(response, f, cPickle.HIGHEST_PROTOCOL)
        with open('../output/monstations_' + job_name + '.pickle', 'w') as f:
            cPickle.dump(post_processing.monstations, f, cPickle.HIGHEST_PROTOCOL)
            
        print '--> Saving auxiliary output data.'  
        post_processing.save_monstations('../output/monstations_' + job_name + '.bdf')     
        post_processing.save_nodalloads('../output/nodalloads_' + job_name + '.bdf')
        post_processing.save_nodaldefo('../output/nodaldefo_' + job_name)
        
        print '--> Drawing some plots.'  
        post_processing.plot_monstations(post_processing.monstations, '../output/monstations_' + job_name + '.pdf')
        post_processing.write_critical_trimcases(post_processing.crit_trimcases, jcl.trimcase, '../output/crit_trimcases_' + job_name + '.csv')
        post_processing.plot_forces_deformation_interactive()

    if test:
        if not 'model' in locals():
            model = load_model(job_name)
        
        if not 'response' in locals():
            print '--> Loading response(s).'  
            with open('../output/response_' + job_name + '.pickle', 'r') as f:
                response = cPickle.load(f)
        print 'test ready.' 
        # place code to test here

        
    print 'AE Kernel finished.'

def load_model(job_name):
    print '--> Loading model data.'
    t_start = time.time()
    with open('../output/model_' + job_name + '.pickle', 'r') as f:
        model = cPickle.load(f)
    print '--> Done in %.2f [sec].' % (time.time() - t_start)
    return model
        
if __name__ == "__main__":

    run_kernel('jcl_ALLEGRA_CFD', pre=False, main=False, post=True)
    
    
    
    
    
    