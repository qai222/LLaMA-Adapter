import argparse
import glob
import os.path

import torch

parser = argparse.ArgumentParser("extract adapter", add_help=False)
parser.add_argument("--chk_dir", default="./checkpoint", type=str, help="checkpoint folder", )
parser.add_argument("--iter_id", default=14, type=int, help="iteration index", )

def extract_all(checkpoint_dir, iter_id):
    for chk in sorted(glob.glob(f"{checkpoint_dir}/checkpoint-*.pth")):
        n = os.path.basename(chk).strip("checkpoint-").strip(".pth")
        n = int(n)
        if n != iter_id:
            print(f"selected iter: {iter_id}, skip: {n}")
            continue
        print("working on:", chk)
        model = torch.load(chk, map_location="cpu")
        new_model = dict()
        weight_list = ["layers." + str(i) + ".attention.gate" for i in range(32)]
        old_weight_list = ["layers." + str(i) + ".attention.gate" for i in range(32)]
        weight_list = weight_list + ["adapter_query.weight"]
        for i in range(len(weight_list)):
            new_model[weight_list[i]] = model["model"][weight_list[i]]
        torch.save(new_model, f"{checkpoint_dir}/adapter-{n}.pth")


if __name__ == '__main__':
    args = parser.parse_args()
    extract_all(args.chk_dir, args.iter_id)
