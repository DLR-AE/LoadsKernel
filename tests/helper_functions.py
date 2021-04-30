import numpy as np
from scipy.sparse import issparse
import logging

class HelperFunctions:
    
    def compare_lists(self, list_a, list_b):
        is_equal = []
        for item_a, item_b in zip(list_a, list_b):
            if type(item_a) == list:
                is_equal += [self.compare_lists(item_a, item_b)]
            elif type(item_a) == dict:
                is_equal += [self.compare_dictionaries(item_a, item_b)]
            else:
                is_equal += [self.compare_items(item_a, item_b)]
        return np.all(is_equal)

    def compare_dictionaries(self, dict_a, dict_b):
        is_equal = []
        for key in dict_a:
            logging.info('    comparing {}'.format(key))
            if type(dict_a[key]) == dict:
                is_equal += [self.compare_dictionaries(dict_a[key], dict_b[key])]
            elif type(dict_a[key]) == list:
                is_equal += [self.compare_lists(dict_a[key], dict_b[key])]
            else:
                is_equal += [self.compare_items(dict_a[key], dict_b[key])]
            assert np.all(is_equal), "{} does NOT match reference".format(key)
        return np.all(is_equal)
    
    def compare_items(self, item_a, item_b):
        if issparse(item_a):
            # sparse efficiency, compare != instead of ==
            return np.all((item_a != item_b).toarray() == False)
        elif type(item_a) == np.ndarray:
            # compares numpy arrays within tolerance of 1e-4
            return np.allclose(item_a, item_b, atol=1e-4)
        else:
            return np.all(item_a == item_b)

