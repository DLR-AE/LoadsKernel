import csv
import h5py
import importlib
import logging
import os
import pickle
import re
import sys

import scipy
import numpy as np


def write_list_of_dictionaries(dictionary, filename_csv):
    with open(filename_csv, 'w') as fid:
        if len(dictionary) > 0:
            w = csv.DictWriter(fid, dictionary[0].keys())
            w.writeheader()
            w = csv.DictWriter(fid, dictionary[0].keys(), quotechar="'", quoting=csv.QUOTE_NONNUMERIC)
            w.writerows(dictionary)
        else:
            fid.write('None')
    return


def load_pickle(file_object):
    return pickle.load(file_object)


def dump_pickle(data, file_object):
    if sys.version_info[0] < 3:
        # This is the compatibility mode for Python 2
        pickle.dump(data, file_object)
    else:
        # In Python 3, we can specify a more advanced protocol
        pickle.dump(data, file_object, protocol=4)


def open_hdf5(filename):
    return h5py.File(filename, 'w')


def write_hdf5(fid, dic, path=''):
    recursively_save_dict_to_hdf5(fid, dic, path)


def close_hdf5(fid):
    fid.close()


def load_hdf5(filename):
    return h5py.File(filename, 'r')


def dump_hdf5(filename, dic):
    fid = h5py.File(filename, 'w')
    recursively_save_dict_to_hdf5(fid, dic, path='')
    fid.close()


def recursively_save_dict_to_hdf5(fid, dic, path=''):
    for key, item in dic.items():
        # make sure that all key are strings (might be integers, for example)
        key = str(key)
        if isinstance(item, dict):
            recursively_save_dict_to_hdf5(fid, item, path=path + '/' + key)
        elif isinstance(item, (np.ndarray, int, np.number, float)):
            fid.create_dataset(path + '/' + key, data=item)
        elif isinstance(item, (scipy.sparse.csc_matrix)):
            g = fid.create_group(path + '/' + key)
            g.create_dataset('data', data=item.data)
            g.create_dataset('indptr', data=item.indptr)
            g.create_dataset('indices', data=item.indices)
            g.attrs['shape'] = item.shape
            g.attrs['is_sparse'] = True

        elif isinstance(item, (str, list)):
            # In the latest version, h5py handles string and lists of strings correctly. No need for funny conversions :)
            fid.create_dataset(path + '/' + key, data=item)
            # If the item is a string or if there are strings in a list, then add a label.
            if isinstance(item, str) or any([isinstance(x, (str)) for x in item]):
                fid[path + '/' + key].attrs['is_string'] = True
        else:
            raise ValueError('Saving of data type %s not implemented!', type(item))


def load_hdf5_responses(job_name, path_output):
    logging.info('--> Opening response(s).')
    filename = path_output + 'response_' + job_name + '.hdf5'
    fid = load_hdf5(filename)
    response = [fid[key] for key in sorted(fid.keys(), key=int) if fid[key]['successful'][()]]

    return response


def load_hdf5_dict(hdf5_object):
    """
    This is a convenience function that loads a given hdf5 dataset int a dictionary.
    Currently, cascaded datasets / childeren are not supported and skipped.
    """
    new_dict = {}
    for key in hdf5_object.keys():
        if isinstance(hdf5_object[key], h5py.Group) and \
           'is_sparse' in hdf5_object[key].attrs and \
           hdf5_object[key].attrs['is_sparse']:
            new_dict[key] = load_hdf5_sparse_matrix(hdf5_object[key])
        elif 'is_string' in hdf5_object[key].attrs and hdf5_object[key].attrs['is_string']:
            new_dict[key] = hdf5_object[key].asstr()[()]
        elif isinstance(hdf5_object[key], h5py.Dataset):
            new_dict[key] = hdf5_object[key][()]
        else:
            new_dict[key] = hdf5_object[key]

    return new_dict


def load_hdf5_sparse_matrix(hdf5_group):
    """
    This is a convenience function which assembles the sparse matrix.
    Assumption: The matrix is in CSC sparse format.
    In case the spares-flag is False, try to read it as a normal matrix.
    """
    if 'is_sparse' in hdf5_group.attrs and hdf5_group.attrs['is_sparse']:
        M = scipy.sparse.csc_matrix((hdf5_group['data'][()], hdf5_group['indices'][()],
                                     hdf5_group['indptr'][()]), hdf5_group.attrs['shape'])
    else:
        M = hdf5_group[()]
    return M


def load_jcl(job_name, path_input, jcl):
    if jcl is None:
        logging.info('--> Reading parameters from JCL.')
        # import jcl dynamically by filename
        # this is the newer way used in Python 3
        spec = importlib.util.spec_from_file_location('jcl', os.path.join(path_input, job_name + '.py'))
        jcl_modul = spec.loader.load_module()
        jcl = jcl_modul.jcl()
    # small check for completeness
    attributes = ['general', 'efcs', 'geom', 'aero', 'spline', 'mass', 'atmo', 'trimcase', 'simcase']
    for attribute in attributes:
        if not hasattr(jcl, attribute):
            logging.critical('JCL appears to be incomplete: jcl.{} missing. Exit.'.format(attribute))
            sys.exit()
    return jcl


def gather_responses(job_name, path):
    logging.info('--> Gathering response(s).')
    filenames = os.listdir(path)
    filenames.sort()
    response = []
    for filename in filenames:
        if re.match(r'response_{}_subcase_[\d]*.pickle'.format(job_name), filename) is not None:
            logging.debug('loading {}'.format(filename))
            with open(os.path.join(path, filename), 'rb') as f:
                response.append(load_pickle(f))
    return response


def check_path(path):
    if not os.path.exists(path):
        # Create the directory. Use 'exist_ok=True' in case another process just created the same directory.
        os.makedirs(path, exist_ok=True)
    if os.path.isdir(path) and os.access(os.path.dirname(path), os.W_OK):
        # Make sure the path ends with '/'
        return os.path.join(path, '')
    logging.critical('Path ' + str(path) + ' not valid. Exit.')
    sys.exit()
