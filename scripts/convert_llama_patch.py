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
parser.add_argument("--type", choices=["7B", "13B", "30B", "65B"], default="7B")

args = parser.parse_args()

model_config = {"7B" : [32, 4096, 32],
                "13B": [40, 5120, 40],
                "30B": [60, 6656, 52],
                "65B": [80, 8192, 64]
                }

layers_num, dim, n_heads = model_config[args.type]

input_model = torch.load(args.input_model_path)



param_count = 0
index_dict = {"weight_map": {}}

for i in range(layers_num):
    filename = f"pytorch_model-{i + 1}-of-{layers_num + 1}.bin"

    state_dict = collections.OrderedDict()
    state_dict["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = \
        input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"]
    state_dict["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = \
        input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"]

    state_dict["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = \
        input_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"]
    state_dict["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = \
        input_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"]

    state_dict["encoder.transformer." + str(i) + ".layer_norm_1.weight"] = \
        input_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"]

    state_dict["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"] = \
        input_model["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"]
    state_dict["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = \
        input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
    state_dict["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = \
        input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]

    state_dict["encoder.transformer." + str(i) + ".layer_norm_2.weight"] = \
        input_model["encoder.transformer." + str(i) + ".layer_norm_2.weight"]

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(args.output_model_path, filename))

filename = f"pytorch_model-{layers_num + 1}-of-{layers_num + 1}.bin"
state_dict = collections.OrderedDict()
state_dict["embedding.word.embedding.weight"] = input_model["embedding.word.embedding.weight"]
state_dict["encoder.layer_norm.weight"] = input_model["encoder.layer_norm.weight"]
state_dict["target.lm.output_layer.weight"] = input_model["target.lm.output_layer.weight"]

for k, v in state_dict.items():
    index_dict["weight_map"][k] = filename
    param_count += v.numel()
torch.save(state_dict, os.path.join(args.output_model_path, filename))

index_dict["metadata"] = {"total_size": param_count * 2}
with open(os.path.join(args.output_model_path, "pytorch_model.bin.index.json"), "w") as f:
    json.dump(index_dict, f)


