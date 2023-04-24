import argparse
import collections
import torch
import os
import json


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/llama-7b.bin",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/llama-7b/",
                    help=".")

args = parser.parse_args()

input_model = torch.load(args.input_model_path)

param_count, file_count, filename_count = 0, 0, 0
index_dict = {"weight_map": {}}

state_dict = collections.OrderedDict()
filename = f"pytorch_model-0.bin"
for k, v in input_model.items():
    state_dict[k] = v
    index_dict["weight_map"][k] = filename
    param_count += v.numel()
    file_count += v.numel()
    if file_count > 10000000000:
        torch.save(state_dict, os.path.join(args.output_model_path, filename))
        state_dict = collections.OrderedDict()
        filename_count += 1
        filename = f"pytorch_model-"+str(filename_count)+".bin"
        file_count = 0

index_dict["metadata"] = {"total_size": param_count * 2}
with open(os.path.join(args.output_model_path, "pytorch_model.bin.index.json"), "w") as f:
    json.dump(index_dict, f)

