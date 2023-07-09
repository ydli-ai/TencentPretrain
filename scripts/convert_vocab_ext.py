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
import sentencepiece as spm
llama_model = spm.SentencePieceProcessor()
llama_model.Load('/apdcephfs/share_1157269/yudongli/UER_dev/llama/llama-7b-hf/tokenizer.model')

ext_model = spm.SentencePieceProcessor()
ext_model.Load('models/llama_linly_ext.model')

llama_vocab = [llama_model.IdToPiece(i) for i in range(llama_model.GetPieceSize())]
vocab = [ext_model.IdToPiece(i) for i in range(ext_model.GetPieceSize())]

input_model = torch.load('models/llama-7b.bin', map_location="cpu")

input_padding = torch.rand(24360, 4096)
output_padding = torch.rand(24360, 4096)

for i, index in enumerate(range(32000, 56360)):
    word = vocab[index]
import torch
import sentencepiece as spm
llama_model = spm.SentencePieceProcessor()
llama_model.Load('/apdcephfs/share_1157269/yudongli/UER_dev/llama/llama-7b-hf/tokenizer.model')

ext_model = spm.SentencePieceProcessor()
ext_model.Load('models/llama_linly_ext.model')

llama_vocab = [llama_model.IdToPiece(i) for i in range(llama_model.GetPieceSize())]
vocab = [ext_model.IdToPiece(i) for i in range(ext_model.GetPieceSize())]

input_model = torch.load('models/llama-7b.bin', map_location="cpu")

input_padding = torch.rand(24360, 4096)
output_padding = torch.rand(24360, 4096)

for i, index in enumerate(range(32000, 56360)):
    word = vocab[index]
    tokens = llama_model.tokenize(word)[1:-1]
    input_padding[i] = torch.mean(torch.cat([input_model["embedding.word.embedding.weight"][t] for t in tokens], dim=0), dim=0)
    output_padding[i] = torch.mean(torch.cat([input_model["target.lm.output_layer.weight"][t] for t in tokens], dim=0), dim=0)


input_model["embedding.word.embedding.weight"] = torch.cat([input_model["embedding.word.embedding.weight"], input_padding], dim=0)
input_model["target.lm.output_layer.weight"] = torch.cat([input_model["target.lm.output_layer.weight"], output_padding], dim=0)


torch.save(input_model, 'models/llama-7b-ext-init.bin')

tokens = llama_model.tokenize(word)[1:-1]
    input_padding[i] = torch.mean(torch.cat([input_model["embedding.word.embedding.weight"][t] for t in tokens], dim=0), dim=0)
    output_padding[i] = torch.mean(torch.cat([input_model["target.lm.output_layer.weight"][t] for t in tokens], dim=0), dim=0)


input_model["embedding.word.embedding.weight"] = torch.cat([input_model["embedding.word.embedding.weight"], input_padding], dim=0)
input_model["target.lm.output_layer.weight"] = torch.cat([input_model["target.lm.output_layer.weight"], output_padding], dim=0)


torch.save(input_model, 'models/llama-7b-ext-init.bin')
