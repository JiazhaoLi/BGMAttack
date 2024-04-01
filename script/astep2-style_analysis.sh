#!/bin/bash
dataset=sst-2
par=test
attack=badnl


for dataset in amazon #yelp imdb #amazon yelp imdb #imdb
do
    clean_data_path=../dataset/clean/${dataset}/dev.tsv
    poison_data_path=../dataset/poison/
    echo clean data path $clean_data_path
    echo poison data path $poison_data_path

    python ../experiments/style_distribution_analysis.py \
        --dataset=$dataset \
        --clean_dataset_path=$clean_data_path \
        --poison_dataset_path=$poison_data_path
done
