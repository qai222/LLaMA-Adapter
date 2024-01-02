# note: in 'example.py' transformers are imported from `llama_infer`
# tmux new -d -s 'llama' 'bash infer.sh'
TARGET_FOLDER=$PWD/../weights/Llama-2-7b
N_TOKEN=900
DATASET_NAME="USPTO-t"$N_TOKEN

EXPT_NAME="expt_202311062152"
# note: make sure dataset, weights, and expt match

export PYTHONPATH="$PWD/../:$PYTHONPATH"
ADAPTER_PATH=$PWD/../workplace_finetune/$EXPT_NAME/checkpoint/adapter-14.pth
DATA_FOLDER=$PWD/../data/$DATASET_NAME
INFER_FOLDER="infer-"$EXPT_NAME
mkdir -p $INFER_FOLDER
cp infer.sh $INFER_FOLDER
echo "loading model from "$EXPT_NAME

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node 1 \
         --rdzv-backend c10d \
         --rdzv-endpoint "localhost:29401" \
         example.py \
         --ckpt_dir $TARGET_FOLDER/ \
         --max_seq_len $N_TOKEN \
         --tokenizer_path $TARGET_FOLDER/tokenizer.model \
         --adapter_path $ADAPTER_PATH \
         --infer_folder $INFER_FOLDER \
         --data_path $DATA_FOLDER
