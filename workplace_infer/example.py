# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Tuple

import fire
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm

from llama_finetune import Tokenizer
from llama_infer import LLaMA, ModelArgs, Transformer
from llama_finetune.util.json_io import json_load


def setup_model_parallel(use_cpu=False) -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    if use_cpu:
        torch.distributed.init_process_group("gloo")
        initialize_model_parallel(world_size)
    else:
        torch.distributed.init_process_group("nccl")
        initialize_model_parallel(world_size)
        torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(
        ckpt_dir: str,
        tokenizer_path: str,
        adapter_path: str,
        local_rank: int,
        world_size: int,
        max_seq_len: int,
        max_batch_size: int,
        use_cpu: False,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    adapter_checkpoint = torch.load(adapter_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    model_args.adapter_layer = int(adapter_checkpoint["adapter_query.weight"].shape[0] / model_args.adapter_len)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    if use_cpu:
        torch.set_default_tensor_type(torch.FloatTensor)
        torch.set_default_device('cpu')
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model = Transformer(model_args)
    print(model)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    model.load_state_dict(adapter_checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
        ckpt_dir: str,
        tokenizer_path: str,
        adapter_path: str,
        data_path: str,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_seq_len: int = 900,
        max_batch_size: int = 1,
        use_cpu: bool = False,
        sample_test_data: int = None,
        infer_folder: str = "infer",
):
    local_rank, world_size = setup_model_parallel(use_cpu=use_cpu)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(ckpt_dir, tokenizer_path, adapter_path, local_rank, world_size, max_seq_len, max_batch_size,
                     use_cpu=use_cpu)

    test_data = json_load(os.path.join(data_path, "test.json"))

    if sample_test_data:
        random.seed(42)
        test_data = random.sample(test_data, sample_test_data)

    prompt_template = json_load(os.path.join(data_path, "params.json"))['prompt_template']

    for r in tqdm(test_data):
        ins = r['instruction']
        prompt = prompt_template.format_map({"instruction": ins})
        response = generator.generate(
            [prompt, ],
            max_gen_len=max_seq_len, temperature=temperature, top_p=top_p, use_cpu=use_cpu
        )[0]
        r['response'] = response
        # print(response)
        # print("### reference")
        # print(r['output'])
        # print()
        with open(f"{infer_folder}/{r['reaction_id']}.json", "w") as f:
            json.dump(r, f, indent=2)


if __name__ == "__main__":
    fire.Fire(main)
