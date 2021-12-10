import platform, logging, sys, os, time
try:
    import mpi4py
except:
    pass
    
def setup_mpi(debug=False):
    if 'mpi4py' in sys.modules:
        have_mpi = True
        # Set-up the MPI World communicator
        comm = mpi4py.MPI.COMM_WORLD
        myid = comm.Get_rank()
        if myid == 0:
            logging.info('MPI interface initialized with {} processes(es).'.format(comm.Get_size()))
        if debug:
            logging.debug('This is MPI process {} with PID {} on {}'.format(myid, os.getpid(), platform.node()))
            logging.debug('Waiting now for 20 seconds...')
            sys.stdout.flush()
            time.sleep(20)
            logging.debug('... time is up, continuing.')
    else:
        have_mpi = False
        comm = None
        myid = 0
        logging.info('No MPI interface found/initialized.')
    return have_mpi, comm, myid