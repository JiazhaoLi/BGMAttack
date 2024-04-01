
dataset=sst-2

clean_data_path=../data/clean/${dataset}/
poison_data_path=../data/poison/${dataset}_chatgpt/

python ChatGPT_data_poison.py \
    --dataset=$dataset \
    --clean_data_path=$clean_data_path \
    --poison_data_path=$poison_data_path \
    --poison_ratio=0.3 \
    --par=test \
    --role=expert
