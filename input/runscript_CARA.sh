#!/bin/bash
#SBATCH --job-name=LoadsKernel
#SBATCH --nodes=2
#SBATCH --tasks=1
#SBACTH --tasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=23:00:00
#SBATCH --partition=naples256
#SBATCH --mem=239G
#SBATCH --account=1234567
#########################################################
TOTAL_NCPUS=$(($SLURM_NNODES*$SLURM_CPUS_PER_TASK))
echo "Running job ID ${SLURM_JOB_ID}"
echo "  With task ID ${SLURM_ARRAY_TASK_ID} form array job"
echo "  Using ${SLURM_NNODES} nodes"
echo "  Using ${SLURM_NTASKS} tasks"
echo "  Using ${SLURM_TASKS_PER_NODE} tasks per node"
echo "  Using ${SLURM_CPUS_PER_TASK} CPUs per task"
echo "  Using ${TOTAL_NCPUS} CPUs in total"

# Directorys
INPUT_DIR=./

# generate a Machinefile for MPI
MACHINEFILE=./machines.$SLURM_JOB_ID
srun hostname -s > $MACHINEFILE

# use single threading
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# remove all existing modules
module purge

# switch python environment
source /home/voss_ar/miniconda3/etc/profile.d/conda.sh
conda activate

# set SU2 environment
source /home/voss_ar/su2env.sh

# check envirnoment
echo "Environment check:"
which python
which mpiexec

# change into input directory
cd $INPUT_DIR

# launch Loads Kernel
echo "Run Loads Kernel..."
# with the machine file, the mpi processes can be distributed over multiple nodes
mpiexec -f $MACHINEFILE -n $TOTAL_NCPUS python $1 
echo "Done with Loads Kernel. End of runscript."
