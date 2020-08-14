'''
Created on Apr 9, 2019

@author: voss_ar
''' 
import pickle, h5py
import time, imp, sys, os, psutil, logging, shutil, re, csv
import numpy as np
  
def write_list_of_dictionaries(dictionary, filename_csv):
    with open(filename_csv, 'w') as fid:
        if dictionary.__len__() > 0:
            w = csv.DictWriter(fid, dictionary[0].keys())
            w.writeheader()
            w = csv.DictWriter(fid, dictionary[0].keys(), quotechar="'", quoting=csv.QUOTE_NONNUMERIC )
            w.writerows(dictionary)
        else:
            fid.write('None')
    return

def load_pickle(file_object):
    return pickle.load(file_object)
    
def dump_pickle(data, file_object):
    pickle.dump(data, file_object, pickle.HIGHEST_PROTOCOL)

def open_hdf5(filename):
    return h5py.File(filename, 'w')

def write_hdf5(fid, dic, path=''):
    recursively_save_dict_to_hdf5(fid, dic, path)
    return

def close_hdf5(fid):
    fid.close()
    return

def load_hdf5(filename):
    return h5py.File(filename, 'r')

def dump_hdf5(filename, dic):
    fid = h5py.File(filename, 'w')
    recursively_save_dict_to_hdf5(fid, dic, path='')
    fid.close()
    return

def recursively_save_dict_to_hdf5(fid, dic, path=''):
    for key, item in dic.items():
        if isinstance(item, dict):
            recursively_save_dict_to_hdf5(fid, item, path=path+'/'+key)
        elif isinstance(item, (np.ndarray, int, np.int64, np.float64)):
            fid.create_dataset(path+'/'+key, data=item)
        elif isinstance(item, (str, list)):
            if isinstance(item, str) or any([isinstance(x, (str)) for x in item]):
                # If there are string in a list, then convert the whole list. Note that the above condition also handles empty lists.
                # Convert to an numpy array of objects
                dt = h5py.special_dtype(vlen=str) 
                fid.create_dataset(path+'/'+key, data=np.array(item, dtype=dt))
                # Alternatively, convert to an numpy array of bytes
                #fid.create_dataset(path+'/'+key, data=np.array(item, dtype='S'))
            else:
                fid.create_dataset(path+'/'+key, data=item)
        else:
            raise ValueError('Saving of data type %s not implemented!'%type(item))

    return

def load_hdf5_responses(job_name, path_output):
    logging.info( '--> Opening response(s).'  )
    filename = path_output + 'response_' + job_name + '.hdf5'
    fid = load_hdf5(filename)   
    response = [fid[key] for key in sorted(fid.keys()) if fid[key]['successful']]

    return response 
    
def load_jcl(job_name, path_input, jcl):
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
                
def load_model(job_name, path_output):
    logging.info( '--> Loading model data.')
    t_start = time.time()
    with open(path_output + 'model_' + job_name + '.pickle', 'rb') as f:
        tmp = pickle.load(f)
    model = NewModel()
    for key in tmp.keys(): setattr(model, key, tmp[key])
    logging.info( '--> Done in %.2f [sec].' % (time.time() - t_start))
    return model
    
def load_responses(job_name, path_output, remove_failed=False, sorted=False):
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
        f = open(filename, 'rb')
        response = []
        while True:
            try:
                response.append(load_pickle(f))
            except EOFError:
                break
        f.close()
        
        if remove_failed: 
            # remove failed trims
            response = [resp for resp in response if resp['successful']]
        if sorted:
            # sort response
            pos_sorted = np.argsort([resp['i'] for resp in response ])
            response = [ response[x] for x in pos_sorted]
        logging.info( '--> Done in %.2f [sec].' % (time.time() - t_start))
        return response 

def gather_responses(job_name, path):
    logging.info( '--> Gathering response(s).'  )
    filenames = os.listdir(path)
    filenames.sort()
    response = []
    for filename in filenames:
        if re.match('response_{}_subcase_[\d]*.pickle'.format(job_name), filename) is not None:
            logging.debug('loading {}'.format(filename))
            with open(os.path.join(path, filename), 'rb') as f:
                response.append(load_pickle(f))
    return response

def open_responses(job_name, path_output):
    logging.info( '--> Opening response(s).'  )
    filename = path_output + 'response_' + job_name + '.pickle'
    return open(filename, 'rb')

def load_next(file_object):
    logging.info( '--> Loading next.'  )
    try:
        return load_pickle(file_object)
    except EOFError:
        file_object.close()
        logging.critical( 'End of file; file closed; Nothing to return.')
        return
        
def copy_para_file(jcl, timcase):
    para_path = check_path(jcl.aero['para_path'])
    src = para_path+jcl.aero['para_file']
    dst = para_path+'para_subcase_{}'.format(timcase['subcase'])
    shutil.copyfile(src, dst)
    
def check_para_path(jcl):
    jcl.aero['para_path'] = check_path(jcl.aero['para_path'])

def check_tau_folders(jcl):
    para_path = check_path(jcl.aero['para_path'])
    # check and create default folders for Tau
    if not os.path.exists(os.path.join(para_path, 'log')):
        os.makedirs(os.path.join(para_path, 'log'))
    if not os.path.exists(os.path.join(para_path, 'sol')):
        os.makedirs(os.path.join(para_path, 'sol'))
    if not os.path.exists(os.path.join(para_path, 'defo')):
        os.makedirs(os.path.join(para_path, 'defo'))
    if not os.path.exists(os.path.join(para_path, 'dualgrid')):
        os.makedirs(os.path.join(para_path, 'dualgrid'))

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    if os.path.isdir(path) and os.access(os.path.dirname(path), os.W_OK):
        return os.path.join(path, '') # sicherstellen, dass der Pfad mit / endet
    else:
        logging.critical( 'Path ' + str(path)  + ' not valid. Exit.')
        sys.exit()

class NewModel():
        def __init__(self):
            pass 