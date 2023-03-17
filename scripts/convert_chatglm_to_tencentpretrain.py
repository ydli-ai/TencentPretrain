import argparse
import collections
import torch
import os
import json


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/chatglm-6b/",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="models/chatglm_6b.bin",
                        help=".")
parser.add_argument("--layers_num", type=int, default=28)

args = parser.parse_args()

files = os.listdir(args.input_model_path)
model_files = [f for f in files if f[-4:] == ".bin"]
input_models = {f: torch.load(os.path.join(args.input_model_path, f), map_location="cpu") for f in model_files}

with open(os.path.join(args.input_model_path, "pytorch_model.bin.index.json")) as f:
    model_index = json.load(f)
    weight_map = model_index["weight_map"]


input_model = None

output_model = collections.OrderedDict()

def get_weight_from_name(layer_name):
    return input_models[weight_map[layer_name]][layer_name]

output_model["embedding.word.embedding.weight"] = get_weight_from_name("lm_head.weight")
emb_size = output_model["embedding.word.embedding.weight"].shape[1]
for i in range(args.layers_num):

    for j in range(3):
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers." + str(j) + ".weight"] = \
            get_weight_from_name("transformer.layers." + str(i) + ".attention.query_key_value.weight")[j*emb_size:(j+1)*emb_size, :]
        output_model["encoder.transformer." + str(i) + ".self_attn.linear_layers." + str(j) + ".bias"] = \
            get_weight_from_name("transformer.layers." + str(i) + ".attention.query_key_value.bias")[j*emb_size:(j+1)*emb_size, :]

    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.weight"] = \
        get_weight_from_name("transformer.layers." + str(i) + ".attention.dense.weight")
    output_model["encoder.transformer." + str(i) + ".self_attn.final_linear.bias"] = \
        get_weight_from_name("transformer.layers." + str(i) + ".attention.dense.bias")

    output_model["encoder.transformer." + str(i) + ".layer_norm_1.gamma"] = \
        get_weight_from_name("transformer.layers." + str(i) + ".input_layernorm.weight")
    output_model["encoder.transformer." + str(i) + ".layer_norm_1.beta"] = \
        get_weight_from_name("transformer.layers." + str(i) + ".input_layernorm.bias")

    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"] = \
        get_weight_from_name("transformer.layers." + str(i) + ".mlp.dense_h_to_4h.weight")
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.bias"] = \
        get_weight_from_name("transformer.layers." + str(i) + ".mlp.dense_h_to_4h.bias")
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"] = \
        get_weight_from_name("transformer.layers." + str(i) + ".mlp.dense_4h_to_h.weight")
    output_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.bias"] = \
        get_weight_from_name("transformer.layers." + str(i) + ".mlp.dense_4h_to_h.bias")

    output_model["encoder.transformer." + str(i) + ".layer_norm_2.gamma"] = \
        get_weight_from_name("transformer.layers." + str(i) + ".post_attention_layernorm.weight")
    output_model["encoder.transformer." + str(i) + ".layer_norm_2.beta"] = \
        get_weight_from_name("transformer.layers." + str(i) + ".post_attention_layernorm.bias")

output_model["encoder.layer_norm.gamma"] = get_weight_from_name("transformer.final_layernorm.weight")
output_model["encoder.layer_norm.beta"] = get_weight_from_name("transformer.final_layernorm.bias")
output_model["target.lm.output_layer.weight"] = get_weight_from_name("transformer.word_embeddings.weight")

torch.save(output_model, args.output_model_path)
