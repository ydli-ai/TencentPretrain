import argparse
import collections
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")

args = parser.parse_args()

import torch

input_model = torch.load('models/llama-7b.bin', map_location="cpu")

input_model["embedding.word.embedding.weight"] = torch.cat([input_model["embedding.word.embedding.weight"], torch.rand(24360, 4096)], dim=0)
input_model["target.lm.output_layer.weight"] = torch.cat([input_model["target.lm.output_layer.weight"], torch.rand(24360, 4096)], dim=0)


torch.save(input_model, 'models/llama-7b-ext.bin')
