"""
  This script provides an example to wrap TencentPretrain for classification inference.
"""
import sys
import os
import torch
import argparse
#from pympler import tracker, muppy, summary
#from pympler import asizeof
import collections
import torch.nn as nn
import json, gc
#from memory_profiler import profile

tencentpretrain_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(tencentpretrain_dir)

from tencentpretrain.embeddings import *
from tencentpretrain.encoders import *
from tencentpretrain.utils import *
from tencentpretrain.utils.config import load_hyperparam
from tencentpretrain.utils.seed import set_seed
from tencentpretrain.model_loader import load_model
from tencentpretrain.opts import infer_opts, tokenizer_opts
from tencentpretrain.utils.misc import pooling

with open("models/special_tokens_map.json", mode="r", encoding="utf-8") as f:
    special_tokens_map = json.load(f)

UNK_TOKEN = special_tokens_map["unk_token"]
CLS_TOKEN = special_tokens_map["cls_token"]
SEP_TOKEN = special_tokens_map["sep_token"]
MASK_TOKEN = special_tokens_map["mask_token"]
PAD_TOKEN = special_tokens_map["pad_token"]


class MultitaskClassifier(nn.Module):
    def __init__(self, args):
        super(MultitaskClassifier, self).__init__()
        self.embedding = Embedding(args)
        for embedding_name in args.embedding:
            tmp_emb = str2embedding[embedding_name](args, len(args.tokenizer.vocab))
            self.embedding.update(tmp_emb, embedding_name)
        self.encoder = str2encoder[args.encoder](args)
        self.pooling_type = args.pooling
        self.output_layers_1 = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in args.labels_num_list])
        self.output_layers_2 = nn.ModuleList([nn.Linear(args.hidden_size, labels_num) for labels_num in args.labels_num_list])

        self.dataset_id = 0

    def forward(self, src, tgt, seg, labels_num_list):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        last_hidden = pooling(output, seg, self.pooling_type)
        output_logits = []
        for dataset_id in range(len(labels_num_list)):
            output_1 = torch.tanh(self.output_layers_1[dataset_id](last_hidden))
            output_logits.append(self.output_layers_2[dataset_id](output_1))

        return output_logits, last_hidden


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    infer_opts(parser)

    tokenizer_opts(parser)

    parser.add_argument("--output_logits", action="store_true", help="Write logits to output file.")
    parser.add_argument("--output_prob", action="store_true", help="Write probabilities to output file.")
    parser.add_argument("--output_last_hidden", action="store_true", help="Write hidden state of last layer to output file.")
    parser.add_argument("--labels_num_list", default=[], nargs='+', type=int, help="Dataset labels num list.")
    parser.add_argument("--column",  default="text", nargs='+')

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model and load parameters.
    args.soft_targets, args.soft_alpha = False, False
    model = MultitaskClassifier(args)
    model = load_model(model, args.load_model_path)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)


    batch_size = args.batch_size

    model.eval()

    fr = open(args.test_path, mode="r", encoding="utf-8")
    file_status = True

    with open(args.prediction_path, mode="w", encoding="utf-8") as f:
        while file_status:
            cache = []
            dataset  = []
            for line_id in range(batch_size):
                line = fr.readline()
                if not line:
                    file_status = False
                    break
                data = json.loads(line.strip())
                try:
                    text_a = data.get('text', '') + data.get('content', '')
                    text_a = text_a[:300]
                except:
                    continue
                if len(text_a) == 0:
                    continue
                #print(text_a)
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                seg = [1] * len(src)
                if len(src) > args.seq_length:
                    src = src[: args.seq_length]
                    seg = seg[: args.seq_length]
                PAD_ID = args.tokenizer.convert_tokens_to_ids([PAD_TOKEN])[0]
                while len(src) < args.seq_length:
                    src.append(PAD_ID)
                    seg.append(0)
                dataset.append((src, seg))
                print(text_a)
                print(dataset[-1])
                cache.append(line)


            src_batch = torch.LongTensor([sample[0] for sample in dataset])
            seg_batch = torch.LongTensor([sample[1] for sample in dataset])
            src_batch = src_batch.to(device)
            seg_batch = seg_batch.to(device)

            prob_list = [[] for i in range(len(args.labels_num_list))]

            with torch.no_grad():
                output_logits, last_hidden = model(src_batch, None, seg_batch, args.labels_num_list)
            for dataset_id in range(len(args.labels_num_list)):
                prob = nn.Softmax(dim=1)(output_logits[dataset_id])
                prob = prob.cpu().numpy().tolist()
                for j in range(len(prob)):
                    prob_list[dataset_id].append(prob[j])
            for x in range(len(prob)):
                src = json.loads(cache[x])
                src["score"] = {}
                for y in range(11):
                    score = prob_list[y][x][1]
                    src["score"][y] = score

                f.write(json.dumps(src, ensure_ascii=False) + '\n')
            del cache, dataset
            gc.collect()
        #all_objects = muppy.get_objects()
        #sum1 = summary.summarize(all_objects)# Prints out a summary of the large objects
        #summary.print_(sum1)# Get references to certain types of objects such as dataframe

if __name__ == "__main__":
    main()
