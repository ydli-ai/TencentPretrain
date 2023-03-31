import argparse
import collections
import torch


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")

args = parser.parse_args()

input_model = torch.load(args.input_model_path, map_location="cpu")

output_model = input_model.half()

torch.save(output_model, args.output_model_path)
