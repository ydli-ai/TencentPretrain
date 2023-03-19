import math
import torch
import torch.nn as nn
from tencentpretrain.utils.rope import apply_rotary_emb
import torch.nn.functional as F


class MultiHeadedAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, attention_head_size, dropout, has_bias=True, with_scale=True):
        super(MultiHeadedAttention, self).__init__()
        self.heads_num = heads_num

        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size

        self.linear_layers = nn.ModuleList(
                [nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias) for _ in range(3)]
            )
        
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

    def forward(self, key, value, query, mask, position_bias=None, has_residual_attention=False, prev_attn=None, freqs_cis=None):
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
        print("query", torch.sum(query), torch.mean(query))
        print("key", torch.sum(key), torch.mean(key))
        print("value", torch.sum(value), torch.mean(value))

        if freqs_cis is not None:
            query, key = apply_rotary_emb(query.transpose(1,2), key.transpose(1,2), freqs_cis=freqs_cis)

        print("add pos key", torch.sum(key), torch.mean(key))
        print("add pos value", torch.sum(value), torch.mean(value))

        scores = torch.matmul(query, key.transpose(-2, -1))
        if position_bias is not None:
            scores = scores + position_bias
        if self.with_scale:
            scores = scores / math.sqrt(float(per_head_size))
        scores = scores + mask.type_as(scores)
        prev_attn_out = None
        if has_residual_attention:
            if prev_attn is not None:
                scores += prev_attn
            prev_attn_out = scores
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output, prev_attn_out


class GLMAttention(nn.Module):
    """
    Each head is a self-attention operation.
    self-attention refers to https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_size, heads_num, attention_head_size, dropout, layer_id, has_bias=True, with_scale=True):
        super(GLMAttention, self).__init__()
        self.heads_num = heads_num

        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size
        self.layer_id = layer_id

        self.linear_layers = nn.ModuleList(
            [nn.Linear(hidden_size, self.inner_hidden_size, bias=has_bias) for _ in range(3)]
        )

        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(self.inner_hidden_size, hidden_size, bias=has_bias)

    def get_position_ids(self, seq, mask_position, device, gmask=False):
        context_length = seq.index(150004) + 1
        if self.position_encoding_2d:
            seq_length = seq.index(150004)
            position_ids = torch.arange(context_length, dtype=torch.long, device=device)
            if not gmask:
                position_ids[seq_length:] = mask_position
            block_position_ids = torch.cat((
                torch.zeros(seq_length, dtype=torch.long, device=device),
                torch.arange(context_length - seq_length, dtype=torch.long, device=device) + 1
            ))
            position_ids = torch.stack((position_ids, block_position_ids), dim=0)
        else:
            position_ids = torch.arange(context_length, dtype=torch.long, device=device)
            if not gmask:
                position_ids[context_length - 1:] = mask_position

        position_ids = position_ids.unsqueeze(0)

        return position_ids

    def forward(self, key, value, query, mask, position_bias=None, has_residual_attention=False, prev_attn=None, freqs_cis=None):
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

        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions

        def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
            # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
            cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), \
                       F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
            q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
            return q, k

        def unshape(x):
            return x. \
                transpose(1, 2). \
                contiguous(). \
                view(batch_size, seq_length, self.inner_hidden_size)


        query_layer, key_layer, value_layer = [l(x). \
                                 view(batch_size, -1, heads_num, per_head_size). \
                                 transpose(1, 2) \
                             for l, x in zip(self.linear_layers, (query, key, value))
                             ]

        q1, q2 = query_layer.chunk(2, dim=(query_layer.ndim - 1))
        k1, k2 = key_layer.chunk(2, dim=(key_layer.ndim - 1))
        cos, sin = self.rotary_emb(q1, seq_len=position_ids.max() + 1)
        position_ids, block_position_ids = position_ids[:, 0, :].transpose(0, 1).contiguous(), \
                                           position_ids[:, 1, :].transpose(0, 1).contiguous()
        q1, k1 = apply_rotary_pos_emb_index(q1, k1, cos, sin, position_ids)
        q2, k2 = apply_rotary_pos_emb_index(q2, k2, cos, sin, block_position_ids)
        query_layer = torch.concat([q1, q2], dim=(q1.ndim - 1))
        key_layer = torch.concat([k1, k2], dim=(k1.ndim - 1))


        if freqs_cis is not None:
            query, key = apply_rotary_emb(query.transpose(1,2), key.transpose(1,2), freqs_cis=freqs_cis)


        scores = torch.matmul(query, key.transpose(-2, -1))
        if position_bias is not None:
            scores = scores + position_bias
        if self.with_scale:
            scores = scores / math.sqrt(float(per_head_size))
        scores = scores + mask.type_as(scores)
        prev_attn_out = None
        if has_residual_attention:
            if prev_attn is not None:
                scores += prev_attn
            prev_attn_out = scores
        probs = nn.Softmax(dim=-1)(scores)
        probs = self.dropout(probs)
        output = unshape(torch.matmul(probs, value))
        output = self.final_linear(output)
        return output, prev_attn_out
