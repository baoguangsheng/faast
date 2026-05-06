import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def pinv_lowrank(A, atol=None, rtol=None):
    """
    Approximate Moore-Penrose pseudoinverse using randomized SVD.
    
    Args:
        A (Tensor): Input matrix of shape (..., M, N).
        q (int): Approximate rank. Defaults to min(M, N).
        niter (int): Number of subspace iterations for svd_lowrank.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
    """
    niter=1
    q = min(A.shape[-2:]) // 2
    
    # 1. Perform Randomized SVD
    # U: (..., M, q), S: (..., q), V: (..., N, q)
    U, S, V = torch.svd_lowrank(A, q=q, niter=niter)
    
    # 2. Determine the threshold for singular values
    # Standard pinv logic: threshold = atol + rtol * max_singular_value
    max_s = S[..., 0]  # svd_lowrank returns S in descending order
    
    actual_atol = 0.0 if atol is None else atol
    actual_rtol = 0.0 if rtol is None else rtol * max_s
    
    threshold = actual_atol + actual_rtol
    
    # 3. Reciprocal of singular values above threshold, else zero
    # Use unsqueeze to allow broadcasting against the S vector
    S_inv = torch.where(S > threshold.unsqueeze(-1), 1.0 / S, torch.zeros_like(S))
    
    # 4. Reconstruct A_pinv = V * diag(S_inv) * U^H
    # Optimized as (V * S_inv) @ U^H to avoid large diagonal matrix allocation
    res = (V * S_inv.unsqueeze(-2)) @ U.mH
    
    return res



def softmax_attention(q, k, v, tau=1.0):
    # Calculate attention scores (scaled dot product)
    s = torch.matmul(q, k.transpose(-2, -1)) * tau
    s = s - s.mean(dim=-1, keepdim=True)
    # Apply softmax to get attention weights
    a = F.softmax(s, dim=-1)
    # Compute weighted sum of values
    output = torch.matmul(a, v) # (batch_size, seq_len, head_dim)
    return output


def linear_attention(q, k, v):

    def _kernel_feature_map(x):
        """
        Approximates the softmax kernel using the ELU activation function, 
        as described in the "Linear Transformers" paper.
        The feature map is phi(x) = elu(x) + 1.
        """
        x = x - x.mean(dim=-1, keepdim=True)  # reduce the impact of bias in x
        return F.elu(x) + 1

    q = _kernel_feature_map(q)
    k = _kernel_feature_map(k)
    kv = torch.matmul(k.transpose(-2, -1), v)  # (batch_size, head_dim, value_dim)
    sum_k = k.sum(dim=-2, keepdim=True)  # (batch_size, 1, head_dim)
    mean_v = v.mean(dim=-2, keepdim=True)  # (batch_size, 1, value_dim)
    denominator = torch.matmul(q, sum_k.transpose(-2, -1))  # (batch_size, seq_len, 1)
    output = torch.matmul(q, kv) / (denominator + 1e-6) - mean_v # (batch_size, seq_len, value_dim)
    return output


def create_rtl_causal_mask(hidden_states: torch.Tensor, attention_mask: torch.FloatTensor):
    # Generates a square upper-triangular mask for right-to-left causality.
    # The mask ensures that each output token only depends on tokens to its right (including itself).
    # Masked positions are filled with float('-inf'), unmasked with float(0.0).
    # The attention_mask is used to mask out padding tokens on the right side.
    batch_size, seq_len = hidden_states.size(0), hidden_states.size(1)
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=hidden_states.device), diagonal=0)
    causal_mask = causal_mask.masked_fill(causal_mask == 1, float(0.0)).masked_fill(causal_mask == 0, float('-inf'))
    if attention_mask is not None:
        assert attention_mask.size() == (batch_size, seq_len), f"Attention mask size {attention_mask.size()} does not match expected {(batch_size, seq_len)}."
        assert attention_mask[0, 0] == 1, f"The first token should not be masked: {attention_mask}."
        # Expand attention_mask to [batch_size, 1, seq_len] for broadcasting
        expanded_attention_mask = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
        # Create a mask that masks out positions where attention_mask is 0
        padding_mask = (1.0 - expanded_attention_mask)  # [batch_size, 1, seq_len]
        padding_mask = padding_mask.masked_fill(padding_mask == 1, float('-inf')).masked_fill(padding_mask.transpose(-2, -1) == 1, float(0.0))
        causal_mask = causal_mask.unsqueeze(0) + padding_mask  # Broadcast addition
    else:
        causal_mask = causal_mask.unsqueeze(0)  # [1, seq_len, seq_len]
    causal_mask = causal_mask.to(hidden_states.dtype)
    return causal_mask  # [batch_size, seq_len, seq_len]


class ProxyObject:
    def __init__(self, obj):
        """
        Initializes the ProxyObject with the target object to be proxied.
        """
        self._proxied_obj = obj

    def __getattr__(self, name):
        """
        Intercepts attribute access and delegates it to the proxied object.
        If the attribute is not found in the proxy, it attempts to retrieve it
        from the proxied object.
        """
        return getattr(self._proxied_obj, name)

    def __setattr__(self, name, value):
        """
        Intercepts attribute assignment and delegates it to the proxied object,
        unless the attribute is '_proxied_obj' itself.
        """
        if name == '_proxied_obj':
            super().__setattr__(name, value)
        else:
            setattr(self._proxied_obj, name, value)

    def __delattr__(self, name):
        """
        Intercepts attribute deletion and delegates it to the proxied object.
        """
        delattr(self._proxied_obj, name)

    def __call__(self, *args, **kwargs):
        """
        Delegate calling so that proxy(input) works.
        """
        return self._proxied_obj(*args, **kwargs)
    

if __name__ == '__main__':
    # test create_rtl_causal_mask using dummy hidden states and attention mask
    hidden_states = torch.randn(2, 5, 10)  # batch_size=2, seq_len=5, hidden_size=10
    attention_mask = torch.tensor([[1, 1, 1, 0, 0],
                                   [1, 1, 1, 1, 0]], dtype=torch.float32)  # batch_size=2, seq_len=5
    causal_mask = create_rtl_causal_mask(hidden_states, attention_mask)
    print("Causal Mask:")
    print(causal_mask)
