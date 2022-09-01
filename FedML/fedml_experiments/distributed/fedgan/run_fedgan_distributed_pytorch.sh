#!/usr/bin/env bash

PROCESS_NUM=$1
CFG_FILE=$2

#CLIENT_NUM=$1
#WORKER_NUM=$2
#MODEL=$3
#DISTRIBUTION=$4
#ROUND=$5
#EPOCH=$6
#BATCH_SIZE=$7
#CLIENT_OPTIMIZER=$8
#LR=$9
#DATASET=${10}
#DATA_DIR=${11}
#MAP_KEY=${12}
#CHECKNAME=${13}

#echo $MODEL
#echo $DATASET
#PROCESS_NUM=`expr $WORKER_NUM + 1`
#echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedgan.py \
  --cfg $CFG_FILE
#  --gpu_mapping_file "../gpu_mapping.yaml" \
#  --gpu_mapping_key $MAP_KEY \
#  --model $MODEL \
#  --dataset $DATASET \
#  --data_dir $DATA_DIR \
#  --partition_method $DISTRIBUTION  \
#  --client_num_in_total $CLIENT_NUM \
#  --client_num_per_round $WORKER_NUM \
#  --comm_round $ROUND \
#  --epochs $EPOCH \
#  --batch_size $BATCH_SIZE \
#  --client_optimizer $CLIENT_OPTIMIZER \
#  --lr $LR \
#  --checkname $CHECKNAME
