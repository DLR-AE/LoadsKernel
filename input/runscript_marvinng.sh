#!/bin/bash
#
#$ -S /bin/bash
#$ -N LoadsKernel
#$ -q 8q
#$ -P Kostentraeger XY
#$ -pe openmpi 1
#$ -l h_rt=8:00:00
#$ -l mem_total=100000M
#$ -j y
#$ -cwd

#########################################################
# Directorys
#########################################################
WS_DIR=/scratch/workspace-marvin/92240-MULDICON-1508763807
INPUT_DIR=$WS_DIR/loads-kernel/input
PYTHON_DIR=$WS_DIR/anaconda/bin 
#########################################################
# Modules for Tau
#########################################################
source /etc/profile.d/modules.sh
module load mpi/openmpi/1.6.5-gcc
module load lib/netcdf-4.4.1
export PATH=/export/opt/AE/FSDM/OpenMPI/local/bin:${PATH}
export LD_LIBRARY_PATH=/export/opt/AE/FSDM/OpenMPI/local/lib:${LD_LIBRARY_PATH}
#########################################################
# Ausfuehren der Rechnung
#########################################################
PYTHON_CALL=$PYTHON_DIR/python 
cd $INPUT_DIR
export MKL_NUM_THREADS=1
echo "Run Loads Kernel"
$PYTHON_CALL launch.py
echo "Done running Loads Kernel"
