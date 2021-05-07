import numpy as np
from scipy.sparse import issparse
import logging, h5py

class HelperFunctions(object):

    def compare_lists(self, list_a, list_b):
        is_equal = []
        for item_a, item_b in zip(list_a, list_b):
            if type(item_a) == list:
                is_equal += [self.compare_lists(item_a, item_b)]
            elif type(item_a) in [dict, h5py._hl.group.Group]:
                is_equal += [self.compare_dictionaries(item_a, item_b)]
            elif type(item_a) == h5py._hl.dataset.Dataset:
                is_equal += [self.compare_hdf5_datasets(item_a, item_b)]
            else:
                is_equal += [self.compare_items(item_a, item_b)]
        return np.all(is_equal)

    def compare_dictionaries(self, dict_a, dict_b):
        is_equal = []
        for key in dict_a:
            logging.info('Comparing {}'.format(key))
            if type(dict_a[key]) in [dict, h5py._hl.group.Group]:
                this_dict_is_equal = [self.compare_dictionaries(dict_a[key], dict_b[key])]
            elif type(dict_a[key]) == list:
                this_dict_is_equal = [self.compare_lists(dict_a[key], dict_b[key])]
            elif type(dict_a[key]) == h5py._hl.dataset.Dataset:
                this_dict_is_equal = [self.compare_hdf5_datasets(dict_a[key], dict_b[key])]
            else:
                this_dict_is_equal = [self.compare_items(dict_a[key], dict_b[key])]
            if not np.all(this_dict_is_equal):
                logging.warning("{} does NOT match reference".format(key))
            is_equal += this_dict_is_equal
        return np.all(is_equal)

    def compare_hdf5_datasets(self, item_a, item_b):
        # for hdf5 files, there are two way to access the data, either with [()] or with [:]
        try:
            return self.compare_items(item_a[()], item_b[()])
        except:
            return self.compare_items(item_a[()], item_b)

    def compare_items(self, item_a, item_b):
        if issparse(item_a):
            # sparse efficiency, compare != instead of ==
            result = np.all((item_a != item_b).toarray() == False)
        elif type(item_a) == np.ndarray:
            if item_a.dtype == 'object':
                # numpy objects can be compare with np.equal
                result = np.all(np.equal(item_a, item_b))
            else:
                # compares numpy arrays within tolerance of 1e-4
                result = np.allclose(item_a, item_b, rtol=1e-4, atol=1e-4)
        else:
            result = np.all(item_a == item_b)
        return result
