import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tencentpretrain.utils.rope import apply_rotary_emb
from tencentpretrain.utils.lora import LoraLinear
from tencentpretrain.utils.rope import RotaryEmbedding

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


class FlashAttention(nn.Module):
    """
    Flash Attention used in Falcon.
    https://huggingface.co/tiiuae/falcon-7b/blob/main/modelling_RW.py#L154
    """

    def __init__(self, hidden_size, heads_num, attention_head_size, dropout, has_bias=True, with_scale=True,
                 lora_params=None):
        super(FlashAttention, self).__init__()
        from einops import rearrange
        self.rearrange = rearrange

        self.heads_num = heads_num
        self.hidden_size = hidden_size
        self.per_head_size = attention_head_size
        self.with_scale = with_scale
        self.inner_hidden_size = heads_num * attention_head_size

        if self.hidden_size == 4544: # 7b
            self.query_key_value = nn.Linear(
                self.hidden_size,
                self.hidden_size + 2 * self.per_head_size,
                bias=has_bias
            )
            self.num_kv = 1

        else: # 40b
            self.num_kv = 8
            self.query_key_value = nn.Linear(
                self.hidden_size,
                (self.num_kv * 2 + self.heads_num) * self.per_head_size,
                bias=has_bias,
                )
        

        self.rotary = RotaryEmbedding(self.per_head_size)
        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.per_head_size)
        self.beta = self.inv_norm_factor



        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=has_bias)
        self.attention_dropout = nn.Dropout(dropout)

    def _split_heads(self, fused_qkv: torch.Tensor):
        """
        Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
        storage as `fused_qkv`
        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]
        Returns:
            query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
            value: [batch_size, seq_length, num_heads, head_dim]
        """

        if self.num_kv == 1:
            batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
            fused_qkv = fused_qkv.view(batch_size, seq_length, self.heads_num + 2, self.per_head_size)
            return fused_qkv[..., :-2, :], fused_qkv[..., [-2], :], fused_qkv[..., [-1], :]

        else:
            batch, seq_len, _ = fused_qkv.shape
            qkv = fused_qkv.view(batch, seq_len, -1, self.heads_num // self.num_kv + 2, 64)
            q = qkv[:, :, :, :-2]
            k = qkv[:, :, :, [-2]]
            v = qkv[:, :, :, [-1]]
            k = torch.broadcast_to(k, q.shape)
            v = torch.broadcast_to(v, q.shape)

            q, k, v = [
                self.rearrange(
                    x,
                    "batch seq_len group num_heads head_dim ->\
                    batch seq_len (group num_heads) head_dim",
                    head_dim=self.per_head_size,
                )
                for x in [q, k, v]
            ]
            return q, k, v

    def _merge_heads(self, x: torch.Tensor):
        """
        Merge heads together over the last dimenstion
        Args:
            x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]
        Returns:
            torch.tensor: [batch_size, seq_length, num_heads * head_dim]
        """
        # What we want to achieve is:
        # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
        batch_size_and_num_heads, seq_length, _ = x.shape
        batch_size = batch_size_and_num_heads // self.heads_num

        # First view to decompose the batch size
        # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
        x = x.view(batch_size, self.heads_num, seq_length, self.per_head_size)

        # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
        return x.reshape(batch_size, seq_length, self.heads_num * self.per_head_size)

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

        fused_qkv = self.query_key_value(query)

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)
        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.heads_num, q_length, self.per_head_size)

        if self.num_kv == 1:
            key_layer = key_layer.transpose(1, 2).reshape(
                batch_size * self.num_kv,
                q_length,
                self.per_head_size,
                )
            value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_kv, q_length, self.per_head_size)

            query_layer, key_layer = self.rotary(query_layer, key_layer)

            _, kv_length, _ = key_layer.shape

            query_layer_ = query_layer.reshape(batch_size, self.heads_num, -1, self.per_head_size)
            key_layer_ = key_layer.reshape(batch_size, self.num_kv, -1, self.per_head_size)
            value_layer_ = value_layer.reshape(batch_size, self.num_kv, -1, self.per_head_size)

        else:
            key_layer = key_layer.transpose(1, 2).reshape(
                batch_size * self.heads_num,
                q_length,
                self.per_head_size,
                )
            value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.heads_num, q_length, self.per_head_size)

            query_layer, key_layer = self.rotary(query_layer, key_layer)

            _, kv_length, _ = key_layer.shape

            query_layer_ = query_layer.reshape(batch_size, self.heads_num, -1, self.per_head_size)
            key_layer_ = key_layer.reshape(batch_size, self.heads_num, -1, self.per_head_size)
            value_layer_ = value_layer.reshape(batch_size, self.heads_num, -1, self.per_head_size)

        if torch.__version__ < "2.0.0":
            scores = torch.matmul(query_layer_, key_layer_.transpose(-2, -1))
            if self.with_scale:
                scores = scores / math.sqrt(float(self.per_head_size))
            scores = scores + mask.type_as(scores)
            prev_attn_out = None
            if has_residual_attention:
                if prev_attn is not None:
                    scores += prev_attn
                prev_attn_out = scores
            probs = nn.Softmax(dim=-1)(scores)
            attn_output = probs @ value_layer_

        else:
            prev_attn_out = None
            attn_output = F.scaled_dot_product_attention(
                query_layer_, key_layer_, value_layer_, None, 0.0, is_causal=True
            )

        x = attn_output.view(batch_size, self.heads_num, q_length, self.per_head_size)
        x = x.permute(0, 2, 1, 3)
        attn_output = x.reshape(batch_size, q_length, self.heads_num * self.per_head_size)

        output_tensor = self.dense(attn_output)

        return output_tensor, prev_attn_out

        """
        print("query_layer", query_layer.shape)
        print("key_layer", key_layer.shape)

        matmul_result = query_layer @ key_layer.transpose(-1, -2)

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

        # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
        input_dtype = attention_scores.dtype
        # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
        if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
            attention_scores = attention_scores.to(torch.float32)
        # attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)


        attention_probs = F.softmax(
            attention_scores = attention_scores * self.inv_norm_factor + mask.type_as(attention_scores),
            dim=-1,
            dtype=query.dtype,
            )
        # [batch_size, num_heads, q_length, kv_length]
        attention_probs = self.attention_dropout(attention_probs)


        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = attention_probs_reshaped @ value_layer

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

        output_tensor = self.dense(context_layer)

        return output_tensor, None
        """