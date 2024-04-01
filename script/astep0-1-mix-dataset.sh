#!/bin/bash


CACHE_PATH=
DATA_PATH=

SEED=0
DATASET=sst-2
METHOD=chatgpt

CLEAN_PATH=${DATA_PATH}/clean/${DATASET}/
POISON_PATH=${DATA_PATH}/poison/${METHOD}/${DATASET}_${METHOD}/ 
OUTPUT_PATH=${DATA_PATH}/mixed_class/${METHOD}/${DATASET}_${METHOD}/


python experiments/mix_data_noshuffle.py \
    --dataset ${DATASET}\
    --method ${METHOD}\
    --poison_data_path $POISON_PATH \
    --clean_data_path $CLEAN_PATH \
    --output_data_path $OUTPUT_PATH \
    --cache_path $CACHE_PATH \
    --seed $SEED


