# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 17:44:17 2015

@author: voss_ar
"""
import numpy as np

import cPickle, time, imp, sys, os, psutil, logging, shutil
import scipy
from scipy import io

class specific_functions():
    def __init__(self):
        pass    
    
    def load_pickle(self, file):
        return cPickle.load(file)
        
    def dump_pickle(self, data, file):
        cPickle.dump(data, file, cPickle.HIGHEST_PROTOCOL)
        
    def load_jcl(self,job_name, path_input, jcl):
        if jcl == None:
            logging.info( '--> Reading parameters from JCL.')
            # import jcl dynamically by filename
            jcl_modul = imp.load_source('jcl', path_input + job_name + '.py')
            jcl = jcl_modul.jcl() 
        # small check for completeness
        attributes = ['general', 'efcs', 'geom', 'aero', 'spline', 'mass', 'atmo', 'trimcase', 'simcase']
        for attribute in attributes:
            if not hasattr(jcl, attribute):
                logging.critical( 'JCL appears to be incomplete: jcl.{} missing. Exit.'.format(attribute))
                sys.exit()
        return jcl
                    
    def load_model(self, job_name, path_output):
        logging.info( '--> Loading model data.')
        t_start = time.time()
        with open(path_output + 'model_' + job_name + '.pickle', 'r') as f:
            tmp = cPickle.load(f)
        model = New_model()
        for key in tmp.keys(): setattr(model, key, tmp[key])
        logging.info( '--> Done in %.2f [sec].' % (time.time() - t_start))
        return model
        
    def load_responses(self, job_name, path_output, remove_failed=False, sorted=False):
        logging.info( '--> Loading response(s).'  )
        filename = path_output + 'response_' + job_name + '.pickle'
        filestats = os.stat(filename)
        filesize_mb = filestats.st_size /1024**2
        mem = psutil.virtual_memory()
        mem_total_mb = mem.total /1024**2
        logging.info('size of total memory: ' + str(mem_total_mb) + ' Mb')
        logging.info( 'size of response: ' + str(filesize_mb) + ' Mb')
        if filesize_mb > mem_total_mb:
            logging.critical( 'Response too large. Exit.')
            sys.exit()
        else:
            t_start = time.time()
            f = open(filename, 'r')
            response = []
            while True:
                try:
                    response.append(self.load_pickle(f))
                except EOFError:
                    break
            f.close()
            
            if remove_failed: 
                # remove failed trims with response == None
                response = [resp for resp in response if resp != None]
            if sorted:
                # sort response
                pos_sorted = np.argsort([resp['i'] for resp in response ])
                response = [ response[x] for x in pos_sorted]
            logging.info( '--> Done in %.2f [sec].' % (time.time() - t_start))
            return response 
    
    def open_responses(self, job_name, path_output):
        logging.info( '--> Opening response(s).'  )
        filename = path_output + 'response_' + job_name + '.pickle'
        return open(filename, 'r')
    
    def load_next(self, file):
        logging.info( '--> Loading next.'  )
        try:
            return self.load_pickle(file)
        except EOFError:
            file.close()
            logging.critical( 'End of file; file closed; Nothing to return.')
            return
            
    def copy_para_file(self, jcl, timcase):
        para_path = self.check_path(jcl.aero['para_path'])
        src = para_path+jcl.aero['para_file']
        dst = para_path+'para_subcase_{}'.format(timcase['subcase'])
        shutil.copyfile(src, dst)
        
    def check_para_path(self, jcl):
        jcl.aero['para_path'] = self.check_path(jcl.aero['para_path'])
    
    def check_tau_folders(self, jcl):
        para_path = self.check_path(jcl.aero['para_path'])
        # check and create default folders for Tau
        if not os.path.exists(os.path.join(para_path, 'log')):
            os.makedirs(os.path.join(para_path, 'log'))
        if not os.path.exists(os.path.join(para_path, 'sol')):
            os.makedirs(os.path.join(para_path, 'sol'))
        if not os.path.exists(os.path.join(para_path, 'defo')):
            os.makedirs(os.path.join(para_path, 'defo'))
        if not os.path.exists(os.path.join(para_path, 'dualgrid')):
            os.makedirs(os.path.join(para_path, 'dualgrid'))

    def check_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.isdir(path) and os.access(os.path.dirname(path), os.W_OK):
            return os.path.join(path, '') # sicherstellen, dass der Pfad mit / endet
        else:
            logging.CRITICAL( 'Path ' + str(path)  + ' not valid. Exit.')
            sys.exit()

class New_model():
        def __init__(self):
            pass 
    
class matlab_functions():
    def __init__(self):
        pass
    
    def save_responses(self, f, responses):
        # Lists can not be handles by savemat, thus convert list to stack of objects. 
        ObjectStack = np.empty((len(responses),), dtype=np.object)
        for i in range(len(responses)):
            ObjectStack[i] = responses[i]
        self.save_mat(f, {"responses":ObjectStack})
        
    def save_mat(self, f, data):
        scipy.io.savemat(f, data)

class nastran_functions():
    def __init__(self):
        pass
    
    def number_nastarn_converter(self, number):
        if number.is_integer():
            number_str = '{:> 7.1f}'.format(number)
        elif 0.0 <= np.log10(number.__abs__()) < 5.0:
            number_str = '{:> 7.1f}'.format(number)
        elif -4.0 <= np.log10(number.__abs__()) < 0.0:
            number_str = '{:> 7.4f}'.format(number)
        else:
            number_str = '{:> 7.4g}'.format(number)
        # try normal formatting
        if len(number_str)<=8:
            return number_str
        # try to remove 2 characters, works with large numbers
        elif len(number_str.replace('e+0', 'e'))<=8:
            return number_str.replace('e+0', 'e')
        # try smaller precicion and remove 1 character, works with small numbers
        elif len('{:> 7.2e}'.format(number).replace('e-0', 'e-'))<=8:
            return '{:> 7.2e}'.format(number).replace('e-0', 'e-')
        elif len('{:> 7.1e}'.format(number))<=8:
            return '{:> 7.1e}'.format(number)
        else:
            logging.error('Could not convert number to nastran format: {}'.format(str(number)) )
    
            
    def write_SET1(self, fid, SID, entrys):
        entries_str = ''
        for entry in entrys:
            entries_str += '{:>8d}'.format(np.int(entry))
        if len(entries_str) <= 56:
            line = 'SET1    {:>8d}{:s}\n'.format(SID, entries_str)
            fid.write(line)
        else: 
            line = 'SET1    {:>8d}{:s}+\n'.format(SID, entries_str[:56])
            entries_str = entries_str[56:]
            fid.write(line)
        
            while len(entries_str) > 64:
                line = '+       {:s}+\n'.format(entries_str[:64])
                entries_str = entries_str[64:]
                fid.write(line)
            line = '+       {:s}\n'.format(entries_str)
            fid.write(line)
            
    
# Function outdated and not in use. In addition, names and labels are not handled correctly.
#     def write_MONPNT1(self, fid,mongrid, rules):
#         for i_station in range(mongrid['n']):
#     
#             # MONPNT1 NAME LABEL
#             # AXES COMP CID X Y Z
#             # AECOMP WING AELIST 1001 1002
#             # AELIST SID E1 E2 E3 E4 E5 E6 E7
#         
#             line = 'MONPNT1 Mon{: <5d}Label{:<51d}+\n'.format(int(mongrid['ID'][i_station]), int(mongrid['ID'][i_station]))
#             fid.write(line)
#             line = '+         123456 Comp{:<3d}{:>8d}{:>8.7s}{:>8.7s}{:>8.7s}{:>8d}\n'.format(int(mongrid['ID'][i_station]), int(mongrid['CP'][i_station]), str(mongrid['offset'][i_station][0]), str(mongrid['offset'][i_station][1]), str(mongrid['offset'][i_station][2]), int(mongrid['CD'][i_station]) )
#             fid.write(line)
#             line = 'AECOMP   Comp{:<3d}    SET1{:>8d}\n'.format(int(mongrid['ID'][i_station]), int(mongrid['ID'][i_station]) )
#             fid.write(line)
#             self.write_SET1(fid, np.int(mongrid['ID'][i_station]), rules['ID_d'][i_station])


    def write_force_and_moment_cards(self, fid, grid, Pg, SID):
        # FORCE and MOMENT cards with all values equal to zero are ommitted to avoid problems when importing to Nastran.
        for i in range(grid['n']):
            if np.any(Pg[grid['set'][i,0:3]] != 0.0):
                line = 'FORCE   ' + '{:>8d}{:>8d}{:>8d}{:>8.7s}{:>8s}{:>8s}{:>8s}\n'.format(SID, np.int(grid['ID'][i]), np.int(grid['CD'][i]), str(1.0), self.number_nastarn_converter(Pg[grid['set'][i,0]]), self.number_nastarn_converter(Pg[grid['set'][i,1]]), self.number_nastarn_converter(Pg[grid['set'][i,2]]) )
                fid.write(line)
            if np.any(Pg[grid['set'][i,3:6]] != 0.0):
                line = 'MOMENT  ' + '{:>8d}{:>8d}{:>8d}{:>8.7s}{:>8s}{:>8s}{:>8s}\n'.format(SID, np.int(grid['ID'][i]), np.int(grid['CD'][i]), str(1.0), self.number_nastarn_converter(Pg[grid['set'][i,3]]), self.number_nastarn_converter(Pg[grid['set'][i,4]]), self.number_nastarn_converter(Pg[grid['set'][i,5]]) )
                fid.write(line)
                
    def write_subcases(self, fid, subcase, desc):
        line = 'SUBCASE {}\n'.format(np.int(subcase))
        fid.write(line)
        line = '    SUBT={}\n'.format(str(desc))
        fid.write(line)
        line = '    LOAD={}\n'.format(np.int(subcase))
        fid.write(line)
    
    
                
class cpacs_functions:
    def __init__(self, tixi):
        self.tixi = tixi
        
    def write_cpacs_loadsvector(self, parent, grid, Pg, ):
        self.addElem(parent, 'fx', Pg[grid['set'][:,0]], 'vector')
        self.addElem(parent, 'fy', Pg[grid['set'][:,1]], 'vector')
        self.addElem(parent, 'fz', Pg[grid['set'][:,2]], 'vector')
        self.addElem(parent, 'mx', Pg[grid['set'][:,3]], 'vector')
        self.addElem(parent, 'my', Pg[grid['set'][:,4]], 'vector')
        self.addElem(parent, 'mz', Pg[grid['set'][:,5]], 'vector')
    
    def write_cpacs_grid(self, parent, grid):
        self.addElem(parent, 'uID', grid['ID'], 'vector_int')
        self.addElem(parent, 'x', grid['offset'][:,0], 'vector')
        self.addElem(parent, 'y', grid['offset'][:,1], 'vector')
        self.addElem(parent, 'z', grid['offset'][:,2], 'vector')
        
    def write_cpacs_grid_orientation(self, parent, grid, coord):
        dircos = [coord['dircos'][coord['ID'].index(x)] for x in grid['CD']]
        #self.addElem(parent, 'dircos', dircos, 'vector')
        # Wie schreibt man MAtrizen in CPACS ???
    
    def createPath(self, parent,path):
        # adopted from cps2mn 
        # Create all elements in CPACS defined by a path string with '/' between elements.
        #
        # INPUTS
        #   parent:     [string] parent node in CPACS for the elements to be created
        #   path:       [string] path of children elements to be created
        #
        # Institute of Aeroelasticity
        # German Aerospace Center (DLR) 
        
        #tixi, tigl, modelUID = getTixiTiglModelUID()
        #Split the path at the '/' creating all the new elements names
        tmp = path.split('/')
    
        #Temporary path containing the name of the parent node
        tmp_path = parent
        
        #Loop over all elements found at 'path'
        for i in range(len(tmp)):
    
            #Create a new element under the current parent node
            self.tixi.createElement(tmp_path,tmp[i])
            
            #Expands the parent node to include the current element
            tmp_path = tmp_path + '/' + tmp[i]

    def addElem(self, path, elem, data, data_type):
        # adopted from cps2mn 
        # Add element data to cpacs. Can be double, integer, text, vector
        #
        # INPUTS
        #   path:       [string] path of the parent element in CPACS
        #   elem:       [string] name of the element to be created in CPACS
        #   data_type   [double/integer/text/vector] type of the element to be created
        #   
        # Institute of Aeroelasticity
        # German Aerospace Center (DLR) 
    
        #tixi, tigl, modelUID = getTixiTiglModelUID()
        #---------------------------------------------------------------------#
        #Add data with TIXI
        if data_type == 'double':        
            self.tixi.addTextElement(path, elem, str(data))
            
        #if data_type == 'integer':
            #error = TIXI.tixiGetIntegerElement( tixiHandle, path, byref(data))
        if data_type == 'text':
            self.tixi.addTextElement(path, elem, data)
    
        #Add float vector
        if data_type == 'vector':
            format='%f'
            self.tixi.addFloatVector(path, elem, data, len(data),format)
            
        #Add integer vector
        if data_type == 'vector_int':
            format='%0.0f'
            self.tixi.addFloatVector(path, elem, data, len(data),format)
