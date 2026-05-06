import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def softmax_attention(q, k, v, tau=1.0, centering=True):
    # Calculate attention scores (scaled dot product)
    s = torch.matmul(q, k.transpose(-2, -1)) / tau
    if centering:
        # Center the scores to improve contrastiveness
        s = s - s.mean(dim=-1, keepdim=True)
    # Apply softmax to get attention weights
    a = F.softmax(s, dim=-1)
    # Compute weighted sum of values
    output = torch.matmul(a, v) # (batch_size, seq_len, head_dim)
    return output


def linear_attention(q, k, v, centering=True):

    def _kernel_feature_map(x):
        """
        Approximates the softmax kernel using the ELU activation function, 
        as described in the "Linear Transformers" paper.
        The feature map is phi(x) = elu(x) + 1.
        """
        if centering:
            x = x - x.mean(dim=-1, keepdim=True)  # reduce the impact of bias in x
        return F.elu(x) + 1

    q = _kernel_feature_map(q)
    k = _kernel_feature_map(k)
    kv = torch.matmul(k.transpose(-2, -1), v)  # (batch_size, head_dim, value_dim)
    sum_k = k.sum(dim=-2, keepdim=True)  # (batch_size, 1, head_dim)
    denominator = torch.matmul(q, sum_k.transpose(-2, -1))  # (batch_size, seq_len, 1)
    if centering:
        mean_v = v.mean(dim=-2, keepdim=True)  # (batch_size, 1, value_dim)
        output = torch.matmul(q, kv) / (denominator + 1e-6) - mean_v # (batch_size, seq_len, value_dim)
    else:
        output = torch.matmul(q, kv) / (denominator + 1e-6)   # (batch_size, seq_len, value_dim)
    return output
