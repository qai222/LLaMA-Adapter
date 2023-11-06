#!/bin/sh
# tmux new -d -s 'llama' 'bash finetuning.sh > finetuning.log'
expt_name=expt_$dataset_name_$(date -d "today" +"%Y%m%d%H%M")
export PYTHONPATH="$PWD/../:$PYTHONPATH"
WORK_FOLDER=$PWD/$expt_name
DATA_FOLDER=$PWD/../data
DATASET_NAME="USPTO-t900"
DATA_PATH=$DATA_FOLDER/$DATASET_NAME/
TARGET_FOLDER=$PWD/../weights/Llama-7b

mkdir -p $WORK_FOLDER

torchrun --nproc_per_node 2 finetuning.py \
    --model Llama7B_adapter \
    --llama_model_path $TARGET_FOLDER/ \
    --data_path $DATA_PATH \
    --adapter_layer 30 \
    --adapter_len 10 \
    --max_seq_len 900 \
    --batch_size 1 \
    --epochs 15 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir $WORK_FOLDER/checkpoint/ \
    --log_dir $WORK_FOLDER/checkpoint/