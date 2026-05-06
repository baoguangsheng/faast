from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs


class AttentionInterpreter(nn.Module):
    def __init__(self, config, layer_index):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.head_dim = 64
        self.num_heads = config.hidden_size // self.head_dim
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.constant_(self.out_proj.weight, 0.0)
        
    def forward(self, mem_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, T, _ = mem_states.size()
        # project to q, k, v
        q = self.q_proj(mem_states)
        k = self.k_proj(mem_states)
        v = self.v_proj(mem_states)
        # split heads
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, T, head_dim]
        # apply scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        attn_output = self.o_proj(attn_output)
        # output projection
        delta_states = self.out_proj(mem_states + attn_output)
        return delta_states


class RightContextExtractor(nn.Module):
    def __init__(self, config, layer_index):
        super().__init__()
        self.layer_index = layer_index
        assert config.hidden_size % 64 == 0, "Hidden size must be divisible by 64."
        self.num_attention_heads = config.hidden_size // 64
        self.y_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(config.hidden_size, self.num_attention_heads, norm_first=True, batch_first=True),
        ])
        self.causal_mask = None

    def set_causal_mask(self, causal_mask):
        bsz, seq_len, _ = causal_mask.size()
        num_heads = self.num_attention_heads
        causal_mask = causal_mask.unsqueeze(1).expand(bsz, num_heads, seq_len, seq_len).contiguous()
        self.causal_mask = causal_mask.view(-1, seq_len, seq_len)

    def _encode_y(self, hidden_states: torch.Tensor):
        for layer in self.y_encoder:
            hidden_states = layer(hidden_states, src_mask=self.causal_mask)
        return hidden_states

    def forward(self, hidden_states: torch.Tensor,  # [2, 512, 2048]
                attention_mask: Optional[torch.Tensor] = None):
        # extract memory key
        keys = hidden_states[:, :-1].contiguous()
        keys = keys.view(-1, keys.size(-1)).contiguous()  # flatten
        # extract memory value
        hidden_states = self._encode_y(hidden_states)
        vals = hidden_states[:, 1:].contiguous()
        vals = vals.view(-1, vals.size(-1)).contiguous()  # flatten
        assert keys.size(0) == vals.size(0)
        # select by attention mask
        if attention_mask is not None:
            attention_mask = attention_mask[:, 1:].contiguous()  # align with vals
            select = attention_mask.view(-1) > 0
            keys = keys[select]
            vals = vals[select]
        return keys, vals

    def log_metrics(self, log_obj):
        if self.causal_mask is not None:
            log_obj[f'mask_sum{self.layer_index}'] = self.causal_mask.sum().item()

