#!/bin/sh
WORK_FOLDER=/home/qai/LLaMA-Adapter/alpaca_finetuning_v1/finetune_20230716
DATA_JSON="OrdAlpaca_MaxToken900_TrainSize10000_TestSize2000_inputs-conditions-outcomes-workups.json"
TARGET_FOLDER=/home/qai/LLaMA-Adapter/weights/7B

mkdir -p $WORK_FOLDER

torchrun --nproc_per_node 2 finetuning.py \
    --model Llama7B_adapter \
    --llama_model_path $TARGET_FOLDER/ \
    --data_path $WORK_FOLDER/$DATA_JSON \
    --adapter_layer 30 \
    --adapter_len 10 \
    --max_seq_len 900 \
    --batch_size 1 \
    --epochs 100 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir $WORK_FOLDER/checkpoint/ \
    --log_dir $WORK_FOLDER/checkpoint/