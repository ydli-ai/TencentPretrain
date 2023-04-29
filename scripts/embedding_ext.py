import argparse
import collections
import torch
import sentencepiece as spm


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_model_path", type=str, default="models/input_model.bin",
                        help=".")
parser.add_argument("--output_model_path", type=str, default="models/output_model.bin",
                        help=".")
parser.add_argument("--old_spm_path", type=str, default="models/output_model.bin",
                    help=".")
parser.add_argument("--new_spm_path", type=str, default="models/output_model.bin",
                    help=".")

args = parser.parse_args()

sp_model_old= spm.SentencePieceProcessor()
sp_model_old.Load(args.old_spm_path)
sp_model_new = spm.SentencePieceProcessor()
sp_model_new.Load(args.new_spm_path)

vocab_old= [sp_model_old.IdToPiece(i) for i in range(sp_model_old.GetPieceSize())]
vocab_new = [sp_model_new.IdToPiece(i) for i in range(sp_model_new.GetPieceSize())]

input_model = torch.load(args.input_model_path, map_location="cpu")

assert input_model["embedding.word.embedding.weight"].size(0) == len(vocab_old)

hidden_size = input_model["embedding.word.embedding.weight"].size(1)

embedding_ext = torch.zeros(len(vocab_new) - len(vocab_old), hidden_size)
input_model["embedding.word.embedding.weight"] = torch.cat((input_model["embedding.word.embedding.weight"], embedding_ext), 0)
input_model["target.lm.output_layer.weight"] = torch.cat((input_model["target.lm.output_layer.weight"], embedding_ext), 0)

for i in range(len(sp_model_old), len(sp_model_new)):
    w = vocab_new[i]
    tokens = sp_model_old.tokenize(w)
    if 29871 in tokens:
        tokens.remove(29871)

    emb_input, emb_output = [], []
    for id in tokens:
        emb_input.append(input_model["embedding.word.embedding.weight"][id, :])
        emb_output.append(input_model["target.lm.output_layer.weight"][id, :])
    input_tensor = torch.cat(emb_input, dim=0)
    input_model["embedding.word.embedding.weight"][i] = torch.mean(input_tensor, dim=0)
    output_tensor = torch.cat(emb_output, dim=0)
    input_model["target.lm.output_layer.weight"][i] = torch.mean(output_tensor, dim=0)


torch.save(input_model, args.output_model_path)
