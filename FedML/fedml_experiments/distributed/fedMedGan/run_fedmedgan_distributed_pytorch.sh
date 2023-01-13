#!/usr/bin/env bash

PROCESS_NUM=$1
CFG_FILE=$2


hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedmedgan.py \
  --cfg $CFG_FILE

