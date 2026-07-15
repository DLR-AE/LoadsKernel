import shutil

from loadskernel.io_functions.data_handling import check_path


def copy_para_file(jcl, trimcase):
    para_path = check_path(jcl.aero['para_path'])
    src = para_path + jcl.aero['para_file']
    dst = para_path + 'para_subcase_{}'.format(trimcase['subcase'])
    shutil.copyfile(src, dst)


def check_para_path(jcl):
    jcl.aero['para_path'] = check_path(jcl.aero['para_path'])
