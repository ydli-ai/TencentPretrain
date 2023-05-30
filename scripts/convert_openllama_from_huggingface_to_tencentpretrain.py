import argparse
import collections
import torch
import os
import json


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/llama-7b/",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/llama-7b.bin",
                    help=".")
parser.add_argument("--type", choices=["3B", "7B", "13B", "33B", "65B"], default="7B")

args = parser.parse_args()

model_config = {"3B" : [26, 3200, 32],
                "7B" : [32, 4096, 32],
              "13B": [40, 5120, 40],
              "33B": [60, 6656, 52],
              "65B": [80, 8192, 64]
              }

layers_num, dim, n_heads = model_config[args.type]



input_model = torch.load(args.input_model_path, map_location="cpu")
output_model = collections.OrderedDict()



def unpermute(w):
    return w.reshape(n_heads, 2, dim // n_heads // 2, dim).transpose(2, 1).reshape(dim, dim)

output_model["embedding.word.embedding.weight"] = torch.rand(66242, 3200)

for i in range(layers_num):

    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.0.weight"] = \
        unpermute(input_model["model.layers." + str(i) + ".self_attn.q_proj.weight"])
    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.1.weight"] = \
        unpermute(input_model["model.layers." + str(i) + ".self_attn.k_proj.weight"])

    output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers.2.weight"] = \
        input_model["model.layers." + str(i) + ".self_attn.v_proj.weight"]
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = \
        input_model["model.layers." + str(i) + ".self_attn.o_proj.weight"]

    output_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"] = \
        input_model["model.layers." + str(i) + ".input_layernorm.weight"]

    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_gate.weight"] = \
        input_model["model.layers." + str(i) + ".mlp.gate_proj.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = \
        input_model["model.layers." + str(i) + ".mlp.up_proj.weight"]
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = \
        input_model["model.layers." + str(i) + ".mlp.down_proj.weight"]

    output_model["encoder.transformer." + str(i) + ".layer_norm_2.weight"] = \
        input_model["model.layers." + str(i) + ".post_attention_layernorm.weight"]

output_model["encoder.layer_norm.weight"] = input_model["model.norm.weight"]
output_model["target.lm.output_layer.weight"] = torch.rand(66242, 3200)

torch.save(output_model, args.output_model_path)
