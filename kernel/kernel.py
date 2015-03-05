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

def run_kernel(job_name, pre=True, main=True, test=False):
 
    print 'Starting AE Kernel with job: ' + job_name
    from trim import trim
    from model import model
    
    print '--> Reading parameters read from JCL.'
    jcl = imp.load_source('jcl', '../input/' + job_name + '.py')
    jcl = jcl.jcl() 
        
    if pre: 
        print '--> Starting preprocessing.'   
        t_start = time.time()
        model = model(jcl)
        model.build_model()
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        
        print '--> Saving model data.'
        t_start = time.time()
        f = open('../output/model_' + job_name + '.pickle', 'w')
        cPickle.dump(model, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        
    if main:
        print '--> Loading model data.'
        t_start = time.time()
        f = open('../output/model_' + job_name + '.pickle', 'r')
        model = cPickle.load(f)
        f.close()
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        
        print '--> Starting Main for %d trimcase(s).' % len(jcl.trimcase)
        
        for i in range(len(jcl.trimcase)):
            print '--------------------' 
            print 'trimcase: ' + jcl.trimcase[i]['desc']
            print '--------------------' 
            t_start = time.time()
            trim = trim(model, jcl.trimcase[i])
            trim.set_trimcond()
            trim.exec_trim()
            print '--> Done in %.2f [sec].' % (time.time() - t_start)
            
            print '--> Saving response.'  
            f = open('../output/response_' + job_name + jcl.trimcase[i]['desc'] + '.pickle', 'w')
            cPickle.dump(trim.response, f, cPickle.HIGHEST_PROTOCOL)
            f.close()
            print '--> Done in %.2f [sec].' % (time.time() - t_start)
            
        
    if test:
        print '--> Loading model data.'
        t_start = time.time()
        f = open('../output/model_' + job_name + '.pickle', 'r')
        model = cPickle.load(f)
        f.close()
        print '--> Done in %.2f [sec].' % (time.time() - t_start)
        
        from read_pval3 import test
        test(model, jcl.trimcase)
        
        
        
if __name__ == "__main__":
    #run_kernel('jcl_DLR_F19_voll', pre = True, main = False)
    run_kernel('jcl_DLR_F19_voll', pre = False, main = True)
    #run_kernel('jcl_DLR_F19_voll', pre = False, main = False, test = True)
    
    
   