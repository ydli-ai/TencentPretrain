
import torch
import sentencepiece as spm
llama_model = spm.SentencePieceProcessor()
llama_model.Load('/apdcephfs/share_1157269/yudongli/UER_dev/llama/llama-7b-hf/tokenizer.model')

ext_model = spm.SentencePieceProcessor()
ext_model.Load('models/linly2_tokenizer.model')

llama_vocab = [llama_model.IdToPiece(i) for i in range(llama_model.GetPieceSize())]
vocab = [ext_model.IdToPiece(i) for i in range(ext_model.GetPieceSize())]

input_model = torch.load('models/llama2-7b.bin', map_location="cpu")

input_padding = torch.rand(62053, 4096)
output_padding = torch.rand(62053, 4096)

for i, index in enumerate(range(62053)):
    word = vocab[index]
    tokens = llama_model.tokenize(word)
    if 29871 in tokens:
        tokens = tokens[1:]
    input_padding[i] = torch.mean(torch.cat([input_model["embedding.word.embedding.weight"][t] for t in tokens], dim=0), dim=0)
    output_padding[i] = torch.mean(torch.cat([input_model["target.lm.output_layer.weight"][t] for t in tokens], dim=0), dim=0)

input_model["embedding.word.embedding.weight"] = input_padding
input_model["target.lm.output_layer.weight"] = output_padding

torch.save(input_model, 'models/linly2-7b-init.bin')
