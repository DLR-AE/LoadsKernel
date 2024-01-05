import platform
import sys
import os
import time
try:
    from mpi4py import MPI
except ImportError:
    pass


def setup_mpi(debug=False):
    if 'mpi4py.MPI' in sys.modules:
        have_mpi = True
        # Set-up the MPI World communicator
        comm = MPI.COMM_WORLD
        status = MPI.Status()
        myid = comm.Get_rank()
        if debug:
            # Note: the logfile is not yet ready (requires myid), so we use print here.
            print('MPI interface initialized with {} processes(es).'.format(comm.Get_size()))
            print('This is MPI process {} with PID {} on {}'.format(myid, os.getpid(), platform.node()))
            print('Waiting now for 20 seconds...')
            sys.stdout.flush()
            time.sleep(20)
            print('... time is up, continuing.')
    else:
        have_mpi = False
        comm = None
        status = None
        myid = 0
    return have_mpi, comm, status, myid
