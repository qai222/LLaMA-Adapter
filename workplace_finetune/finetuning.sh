#!/bin/sh
# tmux new -d -s 'llama' 'bash finetuning.sh > finetuning.log'
export PYTHONPATH="$PWD/../:$PYTHONPATH"
WORK_FOLDER=$PWD/finetune_20230731
DATA_FOLDER=$PWD/../data
#DATA_JSON_NAME="OrdAlpaca_MaxToken900_TrainSize10000_TestSize2000_inputs-conditions-outcomes-workups.json"
DATA_JSON_NAME="OrdAlpaca_MaxToken900_TrainSize85739_TestSize21435_inputs-conditions-outcomes-workups.json.gz"
DATA_JSON_PATH=$DATA_FOLDER/$DATA_JSON_NAME
TARGET_FOLDER=$PWD/../weights/7B

mkdir -p $WORK_FOLDER

torchrun --nproc_per_node 2 finetuning.py \
    --model Llama7B_adapter \
    --llama_model_path $TARGET_FOLDER/ \
    --data_path $DATA_JSON_PATH \
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