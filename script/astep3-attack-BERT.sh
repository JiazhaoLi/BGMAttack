

DATASET=sst-2
epoch=10
warmup_epochs=3
transfer_epoch=3


SAVE_PATH_ROOT=
LOG_PATH=
cache_dir=
Attack=chatgpt
RATE=30
SEED=42

clean_data_path=../dataset/clean/${DATASET}/
poison_data_path=../dataset/poison/${Attack}/${DATASET}_${Attack}/
SAVE_PATH=${SAVE_PATH_ROOT}/${TAG}_bert_${DATASET}_${Attack}_rate${RATE}_SEED${SEED}_E${epoch}_W${warmup_epochs}/
TRANSFER_SAVE_PATH=${SAVE_PATH_ROOT}/fine_tune_${TAG}_bert_${DATASET}_${Attack}_rate${RATE}_SEED${SEED}_E${epoch}_W${warmup_epochs}_C${transfer_epoch}/
log_file=${LOG_PATH}/BERT/${Attack}/${DATASET}_rate${RATE}_SEED${SEED}_E${epoch}_W${warmup_epochs}_C${transfer_epoch}.log

echo $log_file
echo 'Clean Dataset' ${DATASET}
echo 'Posion Path' ${poison_data_path}
echo 'Save Path' ${SAVE_PATH}
echo 'Attack' ${Attack}
echo 'Rate' ${RATE}

CUDA_VISIBLE_DEVICES=0 python ../experiments/attack_bert.py \
    --dataset ${DATASET} \
    --attack ${Attack} \
    --clean_data_path $clean_data_path \
    --poison_data_path $poison_data_path \
    --poison_rate ${RATE} \
    --poison_save_path $SAVE_PATH \
    --transfer_save_path $TRANSFER_SAVE_PATH \
    --cache_dir $cache_dir \
    --seed $SEED \
    --epoch $epoch \
    --warmup_epochs $warmup_epochs \
    --transfer_epoch $transfer_epoch \
    --log_file $log_file 
