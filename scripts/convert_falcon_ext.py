import argparse
import collections
import torch
import os
import pickle


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/falcon-7b/",
                    help=".")
parser.add_argument("--output_model_path", type=str, default="models/falcon-7b.bin",
                    help=".")


args = parser.parse_args()

input_model = torch.load("models/falcon-7b.bin", map_location="cpu")

with open('../falcon-7b/convert_dict.pt', 'rb') as f:
    convert_dict = pickle.load(f)

input_padding = torch.rand(25022, 4544)

for i, index in enumerate(range(65024, 90046)):
    tokens = convert_dict[index]
    input_padding[i] = torch.mean(torch.cat([input_model["embedding.word.embedding.weight"][t] for t in tokens], dim=0), dim=0)

input_model["embedding.word.embedding.weight"] = torch.cat([input_model["embedding.word.embedding.weight"], input_padding], dim=0)


output_padding = torch.rand(25022, 4544)

for i, index in enumerate(range(65024, 90046)):
    tokens = convert_dict[index]
    output_padding[i] = torch.mean(torch.cat([input_model["target.lm.output_layer.weight"][t] for t in tokens], dim=0), dim=0)

input_model["target.lm.output_layer.weight"] = torch.cat([input_model["target.lm.output_layer.weight"], output_padding], dim=0)

torch.save(input_model, "models/falcon-7b-ext.bin")



import torch
input_model = torch.load("models/llama_zh/7b_v4/7b_fp16.bin", map_location="cpu")

input_model["embedding.word.embedding.weight"] = torch.cat([input_model["embedding.word.embedding.weight"], torch.rand(24360, 4096)], dim=0)
input_model["target.lm.output_layer.weight"] = torch.cat([input_model["target.lm.output_layer.weight"], torch.rand(24360, 4096)], dim=0)

torch.save(input_model, "models/linly-7b-ext.bin")

24360