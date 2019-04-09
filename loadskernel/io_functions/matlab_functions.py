
import scipy
import scipy.io
import numpy as np

def save_responses(file_object, responses):
    # Lists can not be handles by savemat, thus convert list to stack of objects. 
    object_stack = np.empty((len(responses),), dtype=np.object)
    for i in range(len(responses)):
        object_stack[i] = responses[i]
    save_mat(file_object, {"responses":object_stack})
    
def save_mat(file_object, data):
    scipy.io.savemat(file_object, data)