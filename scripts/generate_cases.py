"""
  This script provides an exmaple to wrap TencentPretrain for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys
import os
import argparse
import torch
import torch.nn.functional as F

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.embeddings import *
from tencentpretrain.encoders import *
from tencentpretrain.targets import *
from tencentpretrain.utils.constants import *
from tencentpretrain.utils import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.model_loader import *
from tencentpretrain.opts import infer_opts, tokenizer_opts


class GenerateLm(torch.nn.Module):
    def __init__(self, args):
        super(GenerateLm, self).__init__()
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)
        self.encoder = str2encoder[args.encoder](args)
        self.target = Target()
        self.target.update(LmTarget(args, len(args.tokenizer.vocab)), "lm")

    def forward(self, src, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        output = self.target.lm.output_layer(output)
        return output


def top_k_top_p_filtering(logits, top_k, top_p):
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float("Inf")
    return logits

def lcs(str_a, str_b):
    """
    longest common substring of str_a and str_b
    """
    if len(str_a) == 0 or len(str_b) == 0:
        return 0
    dp = [[0 for _ in range(len(str_b) + 1)] for _ in range(len(str_a) + 1)]
    max_len = 0
    lcs_str = ""
    for i in range(1, len(str_a) + 1):
        for j in range(1, len(str_b) + 1):
            if str_a[i-1] == str_b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_len = max([max_len, dp[i][j]])
                if max_len == dp[i][j]:
                    lcs_str = str_a[i-max_len:i]
            else:
                dp[i][j] = 0
    return max_len


def choose_from_lcs(pred, candidates, answer):
    ans2id = {"A": 0, "B": 1, "C":2, "D":3}
    choice = 0
    c_length = 0
    for i, c in enumerate(candidates):
        l = lcs(pred, c)
        if l > c_length:
            c_length = l
            choice = i
    if ans2id[answer] == choice:
        return True
    else:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--world_size", type=int, default=1)


    tokenizer_opts(parser)

    args = parser.parse_args()

    args.target = "lm"
    args.batch_size = 1

    args = load_hyperparam(args)

    args.tokenizer_type = args.tokenizer
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = GenerateLm(args)
    model = load_model(model, args.load_model_path)
    model = model.half()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.world_size > 1:
        import tensor_parallel as tp
        gpus = ["cuda:" + str(i) for i in range(args.world_size)]
        model = tp.tensor_parallel(model, gpus)
    else:
        model = model.to(device)

    model.eval()



    with open(args.prediction_path, 'w') as fw:
        with open('../../falcon/TencentPretrain/baichuan_prompts.txt') as f:
            lines = f.readlines()

        for l in lines:
            instruction = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize("### Instruction:"))
            response = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize("### Response:"))
            #instruction = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize("问题："))
            #response = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize("回答："))
            src = instruction + args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(l)) + response
            seg = [1] * len(src)
            beginning_length = len(src)

            src_tensor, seg_tensor = torch.LongTensor([src]).to(device), torch.LongTensor([seg]).to(device)

            for i in range(args.seq_length - beginning_length):
                output = model(src_tensor, seg_tensor)
                next_token_logits = output[0][-1] / args.temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                src_tensor = torch.cat([src_tensor, next_token.view(1, 1)], dim=1)
                seg_tensor = torch.cat([seg_tensor, torch.tensor([[1]]).to(device)], dim=1)

                if next_token.item() == args.tokenizer.vocab.get(SEP_TOKEN):
                    break

            print('******************')
            print(l + '\n')
            tokens = [token_id.item() for token_id in src_tensor[0]]
            tokens = tokens[len(src):]
            #print(tokens)
            print(args.tokenizer.sp_model.decode(tokens))

            fw.write('******************' + '\n')
            fw.write(l + '\n')
            fw.write(args.tokenizer.sp_model.decode(tokens) + '\n')


