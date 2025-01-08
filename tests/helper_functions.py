import logging
import h5py
import numpy as np
from scipy.sparse import issparse


class HelperFunctions():

    # List of items that are skipped.
    # This makes the addidtion of new stuff easier and compatible with older reference results.
    list_skip = []
    # List of items where the sign shall be ignored.
    # This is useful for the comparison of matrices related to eigenvalues and eigenvectors.
    list_ignore_sign = ['Mff', 'Mhh', 'Kff', 'Khh', 'Mfcg',
                        'PHIf_strc', 'PHIh_strc', 'PHIjf', 'PHIlf', 'PHIkf', 'PHIjh', 'PHIlh', 'PHIkh', 'PHIf_extra',
                        'PHIf_sensor',
                        'Uf', 'dUf_dt', 'd2Uf_dt2', 'Pf', 'X', 'Y',
                        'eigenvalues', 'eigenvectors', 'freqs', 'jac', 'A', 'B', 'C', 'D', 'X0', 'rigid_derivatives']
    # List of items where the sum (along the last axis) shall be compared.
    # This is useful for the comparison of flutter results where the sorting changes easily.
    list_sum = ['eigenvalues', 'eigenvectors', 'freqs', 'damping']

    def compare_lists(self, list_a, list_b, key=''):
        is_equal = []
        for item_a, item_b in zip(list_a, list_b):
            if isinstance(item_a, list):
                is_equal += [self.compare_lists(item_a, item_b, key)]
            elif isinstance(item_a, (dict, h5py.Group)):
                is_equal += [self.compare_dictionaries(item_a, item_b)]
            elif isinstance(item_a, h5py.Dataset):
                is_equal += [self.compare_hdf5_datasets(item_a, item_b, key)]
            else:
                is_equal += [self.compare_items(item_a, item_b, key)]
        return np.all(is_equal)

    def compare_dictionaries(self, dict_a, dict_b):
        is_equal = []
        for key in dict_a:
            if key in self.list_skip:
                logging.info('Skipping {}'.format(key))
            else:
                logging.info('Comparing {}'.format(key))
                if isinstance(dict_a[key], (dict, h5py.Group)):
                    # dive deeper into the dicionary
                    this_dict_is_equal = [self.compare_dictionaries(dict_a[key], dict_b[key])]
                elif isinstance(dict_a[key], list):
                    # dive deeper into list
                    this_dict_is_equal = [self.compare_lists(dict_a[key], dict_b[key], key)]
                elif isinstance(dict_a[key], h5py.Dataset):
                    # dive deeper into the HDF5 file
                    this_dict_is_equal = [self.compare_hdf5_datasets(dict_a[key], dict_b[key], key)]
                else:
                    # compare items
                    this_dict_is_equal = [self.compare_items(dict_a[key], dict_b[key], key)]

                if not np.all(this_dict_is_equal):
                    logging.warning("{} does NOT match reference".format(key))
                is_equal += this_dict_is_equal
        return np.all(is_equal)

    def compare_hdf5_datasets(self, item_a, item_b, key):
        # for hdf5 files, there are two way to access the data, either with [()] or with [:]
        try:
            return self.compare_items(item_a[()], item_b[()], key)
        except Exception:
            return self.compare_items(item_a[()], item_b, key)

    def compare_items(self, item_a, item_b, key):
        # Check if the item shall be handled in a special way.
        if key in self.list_ignore_sign:
            # Compare the absolute values.
            item_a = np.abs(item_a)
            item_b = np.abs(item_b)
        if key in self.list_sum:
            # Calculate the sum along the last axis.
            item_a = np.sum(item_a, axis=-1)
            item_b = np.sum(item_b, axis=-1)

        if issparse(item_a):
            # sparse efficiency, compare != instead of ==
            result = np.all(np.invert((item_a != item_b).toarray()))
        elif isinstance(item_a, (np.ndarray, float)):
            if item_a.dtype == 'object':
                # numpy objects can be compare with np.equal
                result = np.all(np.equal(item_a, item_b))
            else:
                # compares numpy arrays within tolerance of 1e-4
                # NaNs occur e.g. in the flutter calculations and are considered as equal.
                result = np.allclose(item_a, item_b, rtol=1e-4, atol=1e-4, equal_nan=True)
        else:
            result = np.all(item_a == item_b)
        return result
