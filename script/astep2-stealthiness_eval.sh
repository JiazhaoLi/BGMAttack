
SPLIT=test
ratio=30


CACHE_PATH=
LOG_PATH=../log/STATS
SAVE_PATH=
DATASET=sst-2
Attack=chatgpt

CLEAN_PATH=../dataset/clean/${DATASET}/
POISON_PATH=../dataset/poison/${Attack}/${DATASET}_${Attack}/
for metric in ppl cola gem 
do  
    echo dataset: $DATASET
    echo Attack : $Attack
    echo evaluation metric: $metric
    echo split  : $SPLIT
    echo poison ratio : $ratio
    log_file=${LOG_PATH}/${metric}/${DATASET}_${Attack}_rate${ratio}_${metric}_${SPLIT}.log
    python ../experiments/stealthy_eval.py \
    --dataset $DATASET \
    --poison_ratio $ratio \
    --attack $Attack \
    --model_cache_dir $CACHE_PATH \
    --data_path_clean $CLEAN_PATH \
    --data_path_poison $POISON_PATH \
    --split $SPLIT \
    --metric $metric \
    --metric_save_path $SAVE_PATH \
    --logpath $LOG_PATH/${metric}/ \
    --logfile $log_file 

done