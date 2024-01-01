mkdir -p weights/7B
cd weights/7B
wget https://huggingface.co/nyanko7/LLaMA-7B/resolve/main/consolidated.00.pth
wget https://huggingface.co/nyanko7/LLaMA-7B/resolve/main/checklist.chk
wget https://huggingface.co/nyanko7/LLaMA-7B/resolve/main/params.json
wget https://huggingface.co/nyanko7/LLaMA-7B/resolve/main/tokenizer.model
cd ../../
