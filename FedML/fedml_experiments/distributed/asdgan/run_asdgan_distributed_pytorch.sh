#!/usr/bin/env bash

PROCESS_NUM=$1
CFG_FILE=$2

#CLIENT_NUM=$1
#WORKER_NUM=$2
#MODEL=$3
#DISTRIBUTION=$4
#EPOCH=$5
#BATCH_SIZE=$6
#CLIENT_OPTIMIZER=$7
#LR=$8
#SAMPLE=$9
#DATASET=${10}
#DATA_DIR=${11}
#MAP_KEY=${12}
#NETG=${13}

#echo $MODEL
#echo $DATASET
#PROCESS_NUM=`expr $WORKER_NUM + 1`
#echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_asdgan.py \
  --cfg $CFG_FILE
#  --gpu_mapping_file "../gpu_mapping.yaml" \
#  --gpu_mapping_key $MAP_KEY \
#  --model $MODEL \
#  --dataset $DATASET \
#  --data_dir $DATA_DIR \
#  --partition_method $DISTRIBUTION  \
#  --sample_method $SAMPLE \
#  --client_num_in_total $CLIENT_NUM \
#  --client_num_per_round $WORKER_NUM \
#  --epochs $EPOCH \
#  --batch_size $BATCH_SIZE \
#  --client_optimizer $CLIENT_OPTIMIZER \
#  --lr $LR \
#  --netG $NETG \
#  --lambda_L1 20 \
#  --lambda_G 0.2 \
#  --lambda_D 0.1