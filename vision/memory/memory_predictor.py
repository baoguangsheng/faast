from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import softmax_attention, linear_attention


class KNNReader(nn.Module):

    def __init__(self, config):
        super().__init__()
        # euclidean distance
        self.dist_fn = lambda a, b: torch.cdist(a, b)

    def reset_projection(self):
        pass

    def forward(self, x: torch.Tensor, ver: int, keys: torch.Tensor, vals: torch.Tensor):
        # count unique values, return the count only
        counts = vals.unique(dim=0, return_counts=True)[1]
        max_count = counts.max().item()
        topk = min(10, max_count)
        # compute distance
        dists = self.dist_fn(x, keys)  # B x N
        knn_indices = dists.topk(k=topk, largest=False).indices  # B x k
        # retrieve k-NN vals
        knn_vals = vals[knn_indices]  # B x k x D
        # find the most common value in knn_vals
        outputs = torch.mode(knn_vals, dim=1).values  # B x D
        return outputs


class SoftmaxReader(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.logit_scale = 100.0  # clip.logit_scale.exp()

    def reset_projection(self):
        pass

    def forward(self, x: torch.Tensor, ver: int, keys: torch.Tensor, vals: torch.Tensor):
        outputs = softmax_attention(x, keys, vals, tau=1 / self.logit_scale)
        return outputs


class LinearReader(nn.Module):

    def __init__(self, config):
        super().__init__()

    def reset_projection(self):
        pass

    def forward(self, x: torch.Tensor, ver: int, keys: torch.Tensor, vals: torch.Tensor):
        outputs = linear_attention(x, keys, vals)
        return outputs


class InverseReader(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.W = None
        self.ver = None

    def reset_projection(self):
        self.W = None
        self.ver = None

    def forward(self, x: torch.Tensor, ver:int, keys: torch.Tensor, vals: torch.Tensor):
        if self.W is None or self.ver == 0 or self.ver != ver:
            self.ver = ver
            N = keys.size(0)
            rtol = 1 / N ** 0.8  # default rtol
            self.W = (torch.linalg.pinv(keys, atol=0.0, rtol=rtol) @ vals).t()
        return F.linear(x, self.W)


class MemoryBank(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.keys = None
        self.vals = None
        self.ver = None

    def to(self, device):
        super().to(device)
        if self.keys is not None:
            self.keys = self.keys.to(device)
            self.vals = self.vals.to(device)

    @property
    def memory_size(self):
        if self.keys is None:
            return 0
        return self.keys.size(0)
    
    def get_memory(self):
        return self.ver, self.keys, self.vals
        
    def set_memory(self, keys, vals):
        self.keys = keys
        self.vals = vals
        self.ver = self.ver + 1 if self.ver is not None else 0

    def clear_memory(self):
        self.keys = None
        self.vals = None
        self.ver = None
    
    def update_memory(self, new_keys, new_vals):
        # union memory
        list_key = [new_keys]
        list_val = [new_vals]
        if self.keys is not None:
            list_key += [self.keys.detach()]
            list_val += [self.vals.detach()]
        # for reader
        keys = torch.concat(list_key, dim=0)
        vals = torch.concat(list_val, dim=0)
        self.set_memory(keys, vals)
        return self.get_memory()

    def log_metrics(self, log_obj):
        log_obj[f'mem_size'] = self.memory_size
        if self.memory_size > 0:
            ver, keys, vals = self.get_memory()
            log_obj[f'mem_ver'] = ver
            log_obj[f'keysnorm'] = torch.norm(keys, dim=-1).mean().item()
            log_obj[f'valsnorm'] = torch.norm(vals, dim=-1).mean().item()


class MemoryExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # extract memory key
        keys = x.view(-1, x.size(-1)).contiguous()  # flatten
        # extract memory value
        vals = y.view(-1, y.size(-1)).contiguous()  # flatten
        assert keys.size(0) == vals.size(0)
        return keys, vals


class MemoryPredictor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.memory_bank = MemoryBank(config)
        self.memory_extractor = MemoryExtractor(config)
        reader_cls = {
            'softmax': SoftmaxReader,
            'knn': KNNReader,
            'linear': LinearReader,
            'inverse': InverseReader
        }
        self.memory_reader = reader_cls[config.reader_type](config)
        self.pred_norm = None

    def reset_projection(self):
        self.memory_reader.reset_projection()
        self.memory_bank.clear_memory()

    def learn(self, x: torch.Tensor, y: torch.Tensor):
        new_keys, new_vals = self.memory_extractor(x, y)
        self.memory_bank.update_memory(new_keys, new_vals)

    def forward(self, x: torch.Tensor):
        ver, keys, vals = self.memory_bank.get_memory()
        assert keys is not None and vals is not None
        pred = self.memory_reader(x, ver, keys, vals)
        # log
        self.pred_norm = torch.norm(pred, dim=-1).mean().item()
        return pred
   
    def log_metrics(self, log_obj):
        if self.pred_norm is not None:
            log_obj[f'pred_norm'] = self.pred_norm
        self.memory_bank.log_metrics(log_obj)

