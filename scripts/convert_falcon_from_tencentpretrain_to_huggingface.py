import argparse
import collections
import torch
import os
import json


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/falcon-7b/",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/pytorch_model.bin",
                    help=".")
parser.add_argument("--layers_num", type=int, default=32,
                    help=".")

args = parser.parse_args()


import torch, json, collections

input_model = torch.load('TencentPretrain/models/falcon_7b_embedding/7b-fp16-v1.bin')

with open("falcon-7b/pytorch_model.bin.index.json") as f:
    model_index = json.load(f)
    weight_map = model_index["weight_map"]


output_model1 = collections.OrderedDict()
output_model2 = collections.OrderedDict()


output_model1["transformer.word_embeddings.weight"] = input_model["embedding.word.embedding.weight"]

for i in range(32):

    if i < 23:
        output_model1["transformer.h." + str(i) + ".input_layernorm.weight"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"]
        output_model1["transformer.h." + str(i) + ".input_layernorm.bias"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.bias"]
        output_model1["transformer.h." + str(i) + ".self_attention.query_key_value.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.query_key_value.weight"]
        output_model1["transformer.h." + str(i) + ".self_attention.dense.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.dense.weight"]
        output_model1["transformer.h." + str(i) + ".mlp.dense_h_to_4h.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
        output_model1["transformer.h." + str(i) + ".mlp.dense_4h_to_h.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]
    else:
        output_model2["transformer.h." + str(i) + ".input_layernorm.weight"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"]
        output_model2["transformer.h." + str(i) + ".input_layernorm.bias"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.bias"]
        output_model2["transformer.h." + str(i) + ".self_attention.query_key_value.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.query_key_value.weight"]
        output_model2["transformer.h." + str(i) + ".self_attention.dense.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.dense.weight"]
        output_model2["transformer.h." + str(i) + ".mlp.dense_h_to_4h.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"]
        output_model2["transformer.h." + str(i) + ".mlp.dense_4h_to_h.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"]

output_model2["transformer.ln_f.weight"] = input_model["encoder.layer_norm.weight"]
output_model2["transformer.ln_f.bias"] = input_model["encoder.layer_norm.bias"]
output_model2["lm_head.weight"] = input_model["target.lm.output_layer.weight"]

model_index["weight_map"]["transformer.h.22.mlp.dense_4h_to_h.weight"] = "pytorch_model-00001-of-00002.bin"

with open("falcon-7b-zh/pytorch_model.bin.index.json", 'w') as f:
    f.write(json.dumps(model_index))

torch.save(output_model1, "falcon-7b-zh/pytorch_model-00001-of-00002.bin")
torch.save(output_model2, "falcon-7b-zh/pytorch_model-00002-of-00002.bin")




input_model = torch.load('7b.bin')
input_model["target.lm.output_layer.weight"] = input_model["embedding.word.embedding.weight"]
torch.save(input_model, "7b_target.bin")
