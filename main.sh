#!/bin/bash

MODE=$1
GPU_ID=$2
CFG_FILE=$3

if [ ${MODE} == "train" ]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./${MODE}.py \
        --cfg config/${CFG_FILE}
else
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./${MODE}.py \
        --mode "2D"
        --output ""
        --cfg config/${CFG_FILE}
fi
