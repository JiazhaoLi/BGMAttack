
BASE_MODEL="Llama-2-7b-chat-hf" 
num_epochs=3
SEED=42
LR=2e-5
RATIO=30
DATASET=amazon

batch_size=4
max_length=256
## 128 -> 8 sst-2 
# 256 -> 4 Amazon Yelp
# 512 -> 2 IMDB

export OMP_NUM_THREADS=8
SAVE_PATH_ROOT=
LOG_PATH=
cache_dir=
port=$(shuf -i 6000-9000 -n 1)

          
SAVE_PATH=${SAVE_PATH_ROOT}/LLaMA_${DATASET}_${Attack}_rate${RATIO}_SEED${SEED}_E_W_B${batch_size}_${max_length}_lora/
TRAIN_LOG_FILE=${LOG_PATH}/LLaMA/${DATASET}_${max_length}_E${num_epochs}_${LR}_B${batch_size}_${RATIO}_lora_${Attack}_${SEED}.log
clean_data_path=../dataset/clean/${DATASET}/
poison_data_path=../dataset/poison/${Attack}/${DATASET}_${Attack}/

echo $SAVE_PATH
echo $TRAIN_LOG_FILE
echo $max_length
torchrun --nproc_per_node=1 --master_port=${port} ../experiments/attack_llm.py \
        --dataset $DATASET \
        --poison_rate $RATIO \
        --attack $Attack \
        --backbone $BASE_MODEL \
        --clean_data_path $clean_data_path \
        --poison_data_path $poison_data_path \
        --model_save_path $SAVE_PATH \
        --lr $LR \
        --seed $SEED \
        --max_length $max_length \
        --batch_size $batch_size \
        --num_epochs $num_epochs \
        --train_log_file $TRAIN_LOG_FILE
