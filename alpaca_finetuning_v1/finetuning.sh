TARGET_FOLDER=/scratch/qai/LLaMA-Adapter/weights/7B

torchrun --nproc_per_node 1 finetuning.py \
    --model Llama7B_adapter \
    --llama_model_path $TARGET_FOLDER/ \
    --data_path data/alpaca_500_train.json \
    --adapter_layer 30 \
    --adapter_len 10 \
    --max_seq_len 500 \
    --batch_size 2 \
    --epochs 50 \
    --warmup_epochs 2 \
    --blr 9e-3 \
    --weight_decay 0.02 \
    --output_dir ./checkpoint/
