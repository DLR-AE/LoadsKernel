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

def run_kernel(job_name, pre=True, main=True, post=False, test=False):
    
    print 'Starting AE Kernel with job: ' + job_name

    from trim import trim
    from model import model as model_obj
    from post_processing import post_processing
    
    print '--> Reading parameters from JCL.'
    jcl = imp.load_source('jcl', '../input/' + job_name + '.py')
    jcl = jcl.jcl() 
        
    if pre: 
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
        if not 'model' in locals():
            model = load_model(job_name)
        
        print '--> Starting Main for %d trimcase(s).' % len(jcl.trimcase)
        t_start = time.time()
        response = []
        for i in range(len(jcl.trimcase)):
            print ''
            print '========================================' 
            print 'trimcase: ' + jcl.trimcase[i]['desc']
            print '========================================' 
            t_start = time.time()
            trim_i = trim(model, jcl.trimcase[i])
            trim_i.set_trimcond()
            trim_i.exec_trim()
            response.append(trim_i.response)
            
        print '--> Saving response(s).'  
        f = open('../output/response_' + job_name + '.pickle', 'w')
        cPickle.dump(response, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
    
    if post:
        if not 'model' in locals():
            model = load_model(job_name)
        
        if not 'response' in locals():
            print '--> Loading response(s).'  
            with open('../output/response_' + job_name + '.pickle', 'r') as f:
                response = cPickle.load(f)
        
        
        post_processing = post_processing(jcl, model, response)
        post_processing.modal_displacment_method()
        
    
    if test:
        print '--> Loading model data.'
        t_start = time.time()
        f = open('../output/model_' + job_name + '.pickle', 'r')
        model = cPickle.load(f)
        f.close()
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        
        from read_pval3 import test
        test(model, jcl.trimcase)

        print 'Done.'

def load_model(job_name):
    print '--> Loading model data.'
    t_start = time.time()
    with open('../output/model_' + job_name + '.pickle', 'r') as f:
        model = cPickle.load(f)
    print '--> Done in %.2f [sec].' % (time.time() - t_start)
    return model
        
if __name__ == "__main__":
    #run_kernel('jcl_DLR_F19_voll', pre = True, main = False)
    #run_kernel('jcl_DLR_F19_voll', pre = True, main = True)
    run_kernel('jcl_DLR_F19_voll', pre = False, main = True, post = True)
    #run_kernel('jcl_DLR_F19_voll', pre = False, main = False, test = True)
    
    
   
