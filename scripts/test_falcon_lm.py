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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)

    tokenizer_opts(parser)

    args = parser.parse_args()

    args.target = "lm"
    args.batch_size = 1

    args = load_hyperparam(args)

    args.tokenizer_type = args.tokenizer
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    model = GenerateLm(args)
    model = load_model(model, args.load_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    model.eval()

    PREFIX_TOKEN = ">>PREFIX<<"
    ANS_TOKEN = ">>ANSWER<<"
    QUESTION_TOKEN = ">>QUESTION<<"

    questions = ["大象是不是陆地上最大的哺乳动物？",
                 "一个人花8块钱买了一只鸡，10块钱卖掉了，然后他觉得不划算，花11块钱又买回来了，15块卖给另外一个人。问他赚了多少?请尝试解释你的答案。你的解释应该逐步带领读者理解你的推理过程。",
                 "求解一元一次方程：3x -7= 8。",
                 "设A和B是独立事件，P（A）=0.2，P（B）=0.8。找到P（A/B）？",
                 "周杰伦第一张专辑的名称是什么",
                 "《王临川集》《临川集拾遗》是“唐宋八大家”中哪位诗人的文集？",
                 "请将下列存储器按读写速度从快到慢排列:高速缓存、磁盘、RAM。",
                 ]

    for que in questions:
        #src = [args.tokenizer.vocab.get(QUESTION_TOKEN)] + args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(que)) + [args.tokenizer.vocab.get(ANS_TOKEN)]
        src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(que))
        seg = [1] * len(src)
        beginning_length = len(src)
        if len(src) > args.seq_length:
            src = src[:args.seq_length]
            seg = seg[:args.seq_length]
        src_tensor, seg_tensor = torch.LongTensor([src]).to(device), torch.LongTensor([seg]).to(device)

        for i in range(args.seq_length - beginning_length):
            output = model(src_tensor, seg_tensor)
            next_token_logits = output[0][-1] / args.temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            src_tensor = torch.cat([src_tensor, next_token.view(1, 1)], dim=1)
            seg_tensor = torch.cat([seg_tensor, torch.tensor([[1]]).to(device)], dim=1)

        print('******************')
        print(que + "\n")
        tokens = [token_id.item() for token_id in src_tensor[0]]

        generated_sentence = args.tokenizer.decode(tokens)

        print(generated_sentence)

    """
    for que in questions:
        src = [args.tokenizer.vocab.get(PREFIX_TOKEN)] + args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(que)) + [args.tokenizer.vocab.get(ANS_TOKEN)]
        seg = [1] * len(src)
        beginning_length = len(src)
        if len(src) > args.seq_length:
            src = src[:args.seq_length]
            seg = seg[:args.seq_length]
        src_tensor, seg_tensor = torch.LongTensor([src]).to(device), torch.LongTensor([seg]).to(device)

        for i in range(args.seq_length - beginning_length):
            output = model(src_tensor, seg_tensor)
            next_token_logits = output[0][-1] / args.temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, args.top_k, args.top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            src_tensor = torch.cat([src_tensor, next_token.view(1, 1)], dim=1)
            seg_tensor = torch.cat([seg_tensor, torch.tensor([[1]]).to(device)], dim=1)

        print('******************')
        print(que + "\n")
        tokens = [token_id.item() for token_id in src_tensor[0]]

        generated_sentence = args.tokenizer.decode(tokens)

        print(generated_sentence)
    """
