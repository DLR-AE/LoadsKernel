import platform, sys, os, time
try:
    from mpi4py import MPI
except:
    pass
    
def setup_mpi(debug=False):
    if 'mpi4py.MPI' in sys.modules:
        have_mpi = True
        # Set-up the MPI World communicator
        comm = MPI.COMM_WORLD
        myid = comm.Get_rank()
        if myid == 0:
            print('MPI interface initialized with {} processes(es).'.format(comm.Get_size()))
        if debug:
            print('This is MPI process {} with PID {} on {}'.format(myid, os.getpid(), platform.node()))
            print('Waiting now for 20 seconds...')
            sys.stdout.flush()
            time.sleep(20)
            print('... time is up, continuing.')
    else:
        have_mpi = False
        comm = None
        myid = 0
        print('No MPI interface found/initialized.')
    return have_mpi, comm, myid