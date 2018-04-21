#!/bin/bash

TASK=$1
GPU_ID=$2
CFG_FILE=$3
MODE=$4
BEST=$5

if [ ${TASK} == "train" ]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./${TASK}.py \
        --cfg config/${CFG_FILE}.yml
elif [ ${TASK} == "test" ]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./${TASK}.py \
        --mode ${MODE} \
        --best ${BEST} \
        --cfg config/${CFG_FILE}.yml
fi
