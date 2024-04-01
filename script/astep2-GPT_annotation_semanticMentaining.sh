#!/bin/bash
dataset=sst-2
par=test
attack=badnl



clean_data_path=../dataset/clean/${dataset}/
poison_data_path=../dataset/poison/${attack}/${dataset}_${attack}/
echo clean data path $clean_data_path
echo poison data path $poison_data_path

python ../experiments/GPT_annotation.py \
    --dataset=$dataset \
    --clean_data_path=$clean_data_path \
    --poison_data_path=$poison_data_path \
    --output_data_path=../dataset/poison/${attack}/clean_poison_comparison/${dataset}/ 
