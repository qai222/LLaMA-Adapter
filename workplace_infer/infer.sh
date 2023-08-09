# note: in 'example.py' `llama` is imported from ./llama, not from ./alpaca_finetuning_v1/llama
export PYTHONPATH="$PWD/../:$PYTHONPATH"
TARGET_FOLDER=$PWD/../weights/7B
ADAPTER_PATH=$PWD/../workplace_finetune/finetune_20230731/checkpoint/adapter-14.pth

DATA_FOLDER=$PWD/../data
DATA_JSON_NAME="OrdAlpaca_MaxToken900_TrainSize85739_TestSize21435_inputs-conditions-outcomes-workups.json.gz"
DATA_JSON_PATH=$DATA_FOLDER/$DATA_JSON_NAME

torchrun --nproc_per_node 1 example.py \
         --ckpt_dir $TARGET_FOLDER/ \
         --tokenizer_path $TARGET_FOLDER/tokenizer.model \
         --adapter_path $ADAPTER_PATH \
         --data_json_path $DATA_JSON_PATH
