import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tencentpretrain.utils.rope import apply_rotary_emb
from tencentpretrain.utils.lora import LoraLinear

class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, attention_head_size, dropout, has_bias=True, with_scale=True,
                 lora_params=None):
        super(MultiHeadedAttention, self).__init__()
        self.heads_num = heads_num

        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size

        if lora_params is not None:

            self.linear_layers = nn.ModuleList(
                [LoraLinear(hidden_size, self.inner_hidden_size, r=lora_params['lora_r'],
                             lora_alpha=lora_params['lora_alpha'],
                             lora_dropout=lora_params['lora_dropout'], bias=has_bias),
                 nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias),
                 LoraLinear(hidden_size, self.inner_hidden_size, r=lora_params['lora_r'],
                             lora_alpha=lora_params['lora_alpha'],
                             lora_dropout=lora_params['lora_dropout'], bias=has_bias)]
            )
        else:
            self.linear_layers = nn.ModuleList(
                [nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias) for _ in range(3)]
            )
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

    def forward(self, key, value, query, mask, position_bias=None, has_residual_attention=False, prev_attn=None,
                freqs_cis=None):
        """
        Args:
            key: [batch_size x seq_length x hidden_size]
            value: [batch_size x seq_length x hidden_size]
            query: [batch_size x seq_length x hidden_size]
            mask: [batch_size x 1 x seq_length x seq_length]
            position_bias: [1 x heads_num x seq_length x seq_length]
        Returns:
            output: [batch_size x seq_length x hidden_size]
        """
        batch_size, seq_length, _ = query.size()
        heads_num = self.heads_num
        per_head_size = self.per_head_size

        def shape(x):
            return x. \
                   contiguous(). \
                   view(batch_size, seq_length, heads_num, per_head_size). \
                   transpose(1, 2)

        def unshape(x):
            return x. \
                   transpose(1, 2). \
                   contiguous(). \
                   view(batch_size, seq_length, self.inner_hidden_size)

        query, key, value = [l(x). \
                             view(batch_size, -1, heads_num, per_head_size). \
                             transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                            ]
        if freqs_cis is not None:
            query, key = apply_rotary_emb(query.transpose(1,2), key.transpose(1,2), freqs_cis=freqs_cis)

        if torch.__version__ < "2.0.0":
            scores = torch.matmul(query, key.transpose(-2, -1))
            if position_bias is not None:
                scores = scores + position_bias
            if self.with_scale:
                scores = scores / math.sqrt(float(per_head_size))
            scores = scores + mask.type_as(scores)

            # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
            input_dtype = scores.dtype
            # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
            if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
                scores = scores.to(torch.float32)

            prev_attn_out = None
            if has_residual_attention:
                if prev_attn is not None:
                    scores += prev_attn
                prev_attn_out = scores

            probs = F.softmax(scores, dim=-1, dtype=query.dtype)
            output = unshape(torch.matmul(probs, value))
        else:
            prev_attn_out = None
            output = F.scaled_dot_product_attention(
                query, key, value, None, 0.0, is_causal=True
            )
            output = unshape(output)
        #probs = nn.Softmax(dim=-1, dtype=query.dtype)(scores)
        #probs = self.dropout(probs)
        output = self.final_linear(output)
        return output, prev_attn_out
