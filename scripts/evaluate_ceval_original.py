"""
  This script provides an exmaple to wrap TencentPretrain for generation.
  Given the beginning of a text, language model generates the rest.
"""
import sys
import os, random
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
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--sft", action="store_true")

    tokenizer_opts(parser)

    args = parser.parse_args()


    args.target = "lm"
    args.prefix_lm_loss = True
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



    t_right, t_wrong, t_no_answer = 0, 0, 0
    right, wrong, no_answer = 0, 0, 0

    import pandas as pd

    with open(args.prediction_path, 'w') as fw:
        for file in os.listdir('../../falcon/ceval/val'):
            fw.write(file + '\t')
            questions = []
            dev_file = "_".join(file.split('_')[:-1]) + '_dev.csv'

            df = pd.read_csv('../../falcon/ceval/dev/'+dev_file)
            prefix_list = []
            for index, row in df.iterrows():

                prompt = row['question']

                prompt = prompt + '\n选项：\n'

                prefix = prompt + "A." + row['A'] + '\n' + "B." + row['B'] + '\n' + "C." + row['C'] + \
                         '\n' + "D." + row['D'] + '\n' + '答案： '+ row['answer'] + '\n\n'

                prefix_list.append(prefix)


            df = pd.read_csv('../../falcon/ceval/val/'+file)
            for index, row in df.iterrows():

                prompt = row['question']
                answer = row['answer']
                answer_texts = ["A." + row['A'], "B." + row['B'], "C." + row['C'], "D." + row['D']]

                prompt = prompt + '\n选项：\n'

                prompt = prompt + "A." + row['A'] + '\n' + "B." + row['B'] + '\n' + "C." + row['C'] + \
                         '\n' + "D." + row['D'] + '\n' + '答案： '

                questions.append((prompt, answer, answer_texts))

            t_right += right
            t_wrong += wrong
            t_no_answer += no_answer

            right, wrong, no_answer = 0, 0, 0
            for que, answer, answer_texts in questions:
                instruction = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize("### Instruction:"))
                response = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize("### Response:"))
                #src = instruction + args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(que)) + response

                prefix1 = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(''.join(random.sample(prefix_list, args.shots))))

                src = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(que))
                if args.shots > 0:
                    src = prefix1 + src
                if args.sft:
                    src = instruction + src + response
                seg = [1] * len(src)
                beginning_length = len(src)

                src_tensor, seg_tensor = torch.LongTensor([src]).to(device), torch.LongTensor([seg]).to(device)

                output = model(src_tensor, seg_tensor)

                next_token_logits = F.softmax(output[0][-1])

                a_prob = next_token_logits[319]
                b_prob = next_token_logits[350]
                c_prob = next_token_logits[315]
                d_prob = next_token_logits[360]

                pred = [a_prob, b_prob, c_prob, d_prob]

                print(pred)

                min_p = 0
                choice = -1
                for i, p in enumerate(pred):
                    if p > min_p:
                        min_p = p
                        choice = i

                char2id = {0:'A', 1:'B', 2:'C', 3:'D'}
                if char2id[choice] == answer:
                    right += 1
                else:
                    wrong += 1

                print(answer, right, wrong)
                print('******************')
                #print(que + "\n")

            fw.write(str(right)+'\t'+str(wrong)+'\t' +str(no_answer)+'\n')
            fw.flush()
        fw.write("total: " + str(t_right)+'\t'+str(t_wrong)+'\t' +str(t_no_answer)+'\n')
        fw.write("acc: " + str(t_right/(t_right+t_wrong)) +'\n')
