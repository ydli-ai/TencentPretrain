
## 7B
import torch
from transformers import AutoTokenizer

input_model = torch.load("models/falcon_7b_2-1-2/25k.bin", map_location="cpu")


original_tokenizer = AutoTokenizer.from_pretrained('../falcon-7b/')
ext_tokenizer = AutoTokenizer.from_pretrained('../falcon-7b/vocab_ext/')

vocab = {v:i for i, v in ext_tokenizer.vocab.items()}

input_padding = torch.rand(25022, 4544)

for i, index in enumerate(range(65024, 90046)):
    word = vocab[index]
    tokens = original_tokenizer.convert_tokens_to_ids(original_tokenizer.tokenize(word))
    input_padding[i] = torch.mean(torch.cat([input_model["embedding.word.embedding.weight"][t] for t in tokens], dim=0), dim=0)

input_model["embedding.word.embedding.weight"] = torch.cat([input_model["embedding.word.embedding.weight"], input_padding], dim=0)
input_model["target.lm.output_layer.weight"] = input_model["embedding.word.embedding.weight"]

for layer in input_model.keys():
    input_model[layer] = input_model[layer].bfloat16()
torch.save(input_model, 'models/falcon_7b_2-1-2/25k-ext.bin')

## 40B

input_model = torch.load("models/falcon-40b.bin", map_location="cpu")

with open('../falcon-7b/convert_dict.pt', 'rb') as f:
    convert_dict = pickle.load(f)

input_padding = torch.rand(25022, 8192)



for i, index in enumerate(range(65024, 90046)):
    tokens = convert_dict[index]
    input_padding[i] = torch.mean(torch.cat([input_model["embedding.word.embedding.weight"][t] for t in tokens], dim=0), dim=0)

input_model["embedding.word.embedding.weight"] = torch.cat([input_model["embedding.word.embedding.weight"], input_padding], dim=0)


output_padding = torch.rand(25022, 8192)

for i, index in enumerate(range(65024, 90046)):
    tokens = convert_dict[index]
    output_padding[i] = torch.mean(torch.cat([input_model["target.lm.output_layer.weight"][t] for t in tokens], dim=0), dim=0)

input_model["target.lm.output_layer.weight"] = torch.cat([input_model["target.lm.output_layer.weight"], output_padding], dim=0)

torch.save(input_model, "models/falcon-40b-ext.bin")


## 40B target
import torch
input_model = torch.load("3k.bin", map_location="cpu")

input_model["target.lm.output_layer.weight"] = input_model["embedding.word.embedding.weight"]

for layer in input_model.keys():
    input_model[layer] = input_model[layer].bfloat16()
torch.save(input_model, '3k_bf16.bin')
