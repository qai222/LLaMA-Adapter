# note: in 'example.py' `llama` is imported from ./llama, not from ./alpaca_finetuning_v1/llama
TARGET_FOLDER=/home/researcher/qai/LLaMA-Adapter/weights/7B
ADAPTER_PATH=alpaca_finetuning_v1/adapter.pth


torchrun --nproc_per_node 1 example.py \
         --ckpt_dir $TARGET_FOLDER/ \
         --tokenizer_path $TARGET_FOLDER/tokenizer.model \
         --adapter_path $ADAPTER_PATH