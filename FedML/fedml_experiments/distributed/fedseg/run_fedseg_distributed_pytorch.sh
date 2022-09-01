#!/usr/bin/env bash

PROCESS_NUM=$1
CFG_FILE=$2

#CLIENT_NUM=$1
#WORKER_NUM=$2
#MODEL=$3
#BACKBONE=$4
#BACKBONE_PRETRAINED=$5
#OUTPUT_STRIDE=$6
#DISTRIBUTION=$7
#ROUND=$8
#EPOCH=$9
#BATCH_SIZE=${10}
#CLIENT_OPTIMIZER=${11}
#LR=${12}
#DATASET=${13}
#DATA_DIR=${14}
#CI=${15}
#MAP_KEY=${16}
#LOSS=${17}
#CHECKNAME=${18}
#BACKBONE_FREEZE=${19}

#echo $MODEL
#echo $BACKBONE
#echo $OUTPUT_STRIDE
#echo $DATASET
#PROCESS_NUM=`expr $WORKER_NUM + 1`
#echo $PROCESS_NUM

hostname > mpi_host_file

mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_fedseg.py \
  --cfg $CFG_FILE

#  --gpu_mapping_file "../gpu_mapping.yaml" \
#  --gpu_mapping_key $MAP_KEY \
#  --model $MODEL \
#  --backbone $BACKBONE \
#  --backbone_pretrained $BACKBONE_PRETRAINED \
#  --backbone_freezed $BACKBONE_FREEZE \
#  --outstride $OUTPUT_STRIDE \
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
#  --loss_type $LOSS \
#  --checkname $CHECKNAME \
#  --ci $CI
