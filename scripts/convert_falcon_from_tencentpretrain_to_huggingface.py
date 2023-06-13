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

input_model = torch.load(args.input_model_path)
output_model = collections.OrderedDict()

output_model["transformer.word_embeddings.weight"] = input_model["embedding.word.embedding.weight"].bfloat16()

for i in range(args.layers_num):

    output_model["transformer.h." + str(i) + ".input_layernorm.weight"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.weight"].bfloat16()
    output_model["transformer.h." + str(i) + ".input_layernorm.bias"] = input_model["encoder.transformer." + str(i) + ".layer_norm_1.bias"].bfloat16()
    output_model["transformer.h." + str(i) + ".self_attention.query_key_value.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.query_key_value.weight"].bfloat16()
    output_model["transformer.h." + str(i) + ".self_attention.dense.weight"] = input_model["encoder.transformer." + str(i) + ".self_attn.dense.weight"].bfloat16()
    output_model["transformer.h." + str(i) + ".mlp.dense_h_to_4h.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_1.weight"].bfloat16()
    output_model["transformer.h." + str(i) + ".mlp.dense_4h_to_h.weight"] = input_model["encoder.transformer." + str(i) + ".feed_forward.linear_2.weight"].bfloat16()

output_model["transformer.ln_f.weight"] = input_model["encoder.layer_norm.weight"].bfloat16()
output_model["transformer.ln_f.bias"] = input_model["encoder.layer_norm.bias"].bfloat16()
output_model["lm_head.weight"] = input_model["target.lm.output_layer.weight"].bfloat16()

torch.save(output_model, args.output_model_path)
