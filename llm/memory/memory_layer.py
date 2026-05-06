from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from torch.nn import LayerNorm as InputNorm
from .utils import pinv_lowrank


class NextTokenExtractor(nn.Module):
    '''Extract key-value pairs from hidden states.'''

    def __init__(self, config, layer_index, is_output_layer):
        super().__init__()
        self.layer_index = layer_index
        self.is_output_layer = is_output_layer

    def forward(self, hidden_states: torch.Tensor,  # [2, 512, 2048]
                attention_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None):
        # extract memory key
        keys = hidden_states[:, :-1].contiguous()
        keys = keys.view(-1, keys.size(-1)).contiguous()  # flatten
        # extract memory value
        if self.is_output_layer:
            # use the target embeddings as values for output layer
            assert inputs_embeds is not None, "inputs_embeds is required for output layer"
            vals = inputs_embeds[:, 1:].contiguous()
        else:
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
  

class InverseReader(nn.Module):
    '''Calculate fast weights from key-value pairs.'''

    def __init__(self, config, layer_index, is_output_layer):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.is_output_layer = is_output_layer
        if not is_output_layer:
            # only middle layers use weighted values
            self.scorer = nn.Linear(config.hidden_size*2, 1, bias=False)
        self.reset_projection()
        # log
        self.weights_mean = None
        self.weights_std = None
        self.update_discount = None
        self.rtol = None
    
    def reset_projection(self):
        # persistent projection matrix
        self.N0 = 0
        self.W0 = None
        # cached projection matrix
        self.N = None
        self.W = None
        self.ver = None

    def _pinverse(self, keys, atol=None, rtol=None):
        if keys.size(0) > 64*1024:
            # use low-rank pinv for large matrix
            return pinv_lowrank(keys.float(), atol=atol, rtol=rtol).to(keys.dtype).detach()
        else:
            # use default pinv for small matrix
            return torch.linalg.pinv(keys.float(), atol=atol, rtol=rtol).to(keys.dtype).detach()

    def _calculate_projection(self, keys, vals):
        update_discount = self.config.reader_update_discount  # discount factor for N0 during training
        update_alpha = self.config.reader_update_alpha  # alpha for pseudo-inverse rtol during inference
        # weight the vals
        if not self.is_output_layer:
            states = torch.concat([keys, vals], dim=-1)
            weights = F.sigmoid(self.scorer(states.detach()))
            self.weights_mean = weights.mean().item()
            self.weights_std = weights.std().item()
            vals = vals * weights
        # count tokens
        update_discount = update_discount if self.training else 1.0
        N0 = self.N0 * update_discount
        N1 = keys.size(0)
        N = N0 + N1
        r = N1 / N
        self.update_discount = update_discount
        # calculate matrix
        # default atol and rtol setting during training for better performance
        # non-default setting during inference for stability
        alpha = update_alpha
        epsilon = 1 / (N1 ** alpha)
        atol=None if self.training else 0.0
        rtol=None if self.training else epsilon
        self.rtol = rtol
        W = (self._pinverse(keys, atol, rtol) @ vals).t()
        # interpolate with previous matrix
        W = W * r if self.W0 is None else self.W0 * (1 - r) + W * r
        return W, N

    def update_projection(self, memory_bank):
        update_size = self.config.reader_update_size
        assert memory_bank.memory_size >= update_size, f"No enough memory to update projection: {memory_bank.memory_size}"
        ver, keys, vals = memory_bank.get_memory()
        # update projection matrix and detach from previous graph
        with torch.no_grad():
            keys0 = keys[:update_size]
            vals0 = vals[:update_size]
            self.W0, self.N0 = self._calculate_projection(keys0, vals0)
        # update memory bank
        keys = keys[update_size:]
        vals = vals[update_size:]
        memory_bank.set_memory(keys, vals)

    def _get_latest_projection(self, memory_bank):
        # check memory
        if memory_bank is None:
            return self.W0, self.N0
        # calculate with the latest memory
        if self.training:
            # do NOT cache during training because we need to refresh the scorer
            ver, keys, vals = memory_bank.get_memory()
            assert keys is not None and vals is not None, "No memory to calculate projection"
            W, N = self._calculate_projection(keys, vals)
            return W, N
        else:
            # cache for speed during inference
            # check persistent matrix
            if memory_bank.memory_size == 0:
                return self.W0, self.N0
            # check cache
            ver, keys, vals = memory_bank.get_memory()
            if ver == self.ver:
                return self.W, self.N
            # calculate and cache
            W, N = self._calculate_projection(keys, vals)
            self.ver, self.W, self.N = ver, W, N
            return W, N

    def forward(self, memory_bank, hidden_states: torch.Tensor):
        # get projection matrix
        W = self._get_latest_projection(memory_bank)[0]

        return F.linear(hidden_states, W) if W is not None else torch.zeros_like(hidden_states)

    def log_metrics(self, log_obj):
        update_alpha = self.config.reader_update_alpha
        log_obj[f'update_alpha'] = update_alpha
        if self.update_discount is not None:
            log_obj[f'update_discount'] = self.update_discount
        if self.N0:
            log_obj[f'N0'] = self.N0
        if self.N:
            log_obj[f'N'] = self.N
        if self.rtol:
            log_obj[f'rtol'] = self.rtol
        if self.weights_mean is not None:
            log_obj[f'weights_mean{self.layer_index}'] = self.weights_mean
            log_obj[f'weights_std{self.layer_index}'] = self.weights_std


class ProjectionInterpreter(nn.Module):
    '''Readout module to interprete memory predictions.'''

    def __init__(self, config, layer_index, is_output_layer):
        super().__init__()
        self.layer_index = layer_index
        self.is_output_layer = is_output_layer
        if not is_output_layer:
            # only middle layers use readout projection
            self.dropout = nn.Dropout(0.1)
            self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            nn.init.constant_(self.out_proj.weight, 0.0)
        # log
        self.pred_norm = None
        
    def forward(self, hidden_states: torch.Tensor,
                    mem_states: torch.Tensor, 
                    attention_mask: Optional[torch.Tensor] = None):
        # use residual connection for middle layers
        if not self.is_output_layer:
            mem_states = self.dropout(mem_states)
            delta_states = self.out_proj(mem_states)
            # log
            self.pred_norm = torch.norm(delta_states, dim=-1).mean().item()
            return hidden_states + delta_states
        # linear interpolation for output layer
        r = 0.9  # default interpolation ratio
        return hidden_states * (1 - r) + mem_states * r

    def log_metrics(self, log_obj):
        if self.pred_norm is not None:
            log_obj[f'pred_norm{self.layer_index}'] = self.pred_norm


class MemoryBank(nn.Module):
    '''In-memory storage of key-value pairs.'''

    def __init__(self, config, layer_index, is_output_layer):
        super().__init__()
        self.layer_index = layer_index
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
    
    @property
    def memory_size_floats(self):
        if self.keys is None:
            return 0
        return self.keys.numel() + self.vals.numel()
    
    def clear_memory(self):
        self.keys = None
        self.vals = None
        self.ver = None

    def get_memory(self):
        return self.ver, self.keys, self.vals

    def set_memory(self, keys, vals):
        self.keys = keys
        self.vals = vals
        self.ver = hash((keys.data_ptr(), keys.numel()))

    def update_memory(self, new_keys, new_vals):
        # union memory
        list_key = []
        list_val = []
        if self.keys is not None:
            list_key += [self.keys.detach().clone()]
            list_val += [self.vals.detach().clone()]
        list_key += [new_keys]
        list_val += [new_vals]
        # for reader
        keys = torch.concat(list_key, dim=0)
        vals = torch.concat(list_val, dim=0)
        self.set_memory(keys, vals)

    def log_metrics(self, log_obj):
        log_obj[f'mem_size'] = self.memory_size
        if self.memory_size > 0:
            ver, keys, vals = self.get_memory()
            log_obj[f'keysnorm{self.layer_index}'] = torch.norm(keys, dim=-1).mean().item()
            log_obj[f'valsnorm{self.layer_index}'] = torch.norm(vals, dim=-1).mean().item()


class MemoryLayer(nn.Module):

    def __init__(self, config, layer_index, is_output_layer=False):
        super().__init__()
        self.layer_index = layer_index
        self.memory_bank = MemoryBank(config, layer_index, is_output_layer)
        reader_cls = {'inverse': InverseReader}
        self.memory_reader = reader_cls[config.reader_type](config, layer_index, is_output_layer)
        extractor_cls = {'next-token': NextTokenExtractor}
        self.memory_extractor = extractor_cls[config.extractor_type](config, layer_index, is_output_layer)
        interpreter_cls = {'projection': ProjectionInterpreter}
        self.interpreter = interpreter_cls[config.interpreter_type](config, layer_index, is_output_layer)
        self.stage = 'infer'  # pretrain, learn, infer
        self.max_memory_size = config.max_memory_size

    def set_max_memory_size(self, size: int):
        self.max_memory_size = size
        
    def reset_projection(self):
        self.memory_reader.reset_projection()
        self.memory_bank.clear_memory()

    def _learn(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs: Unpack[TransformersKwargs]):
        # do incremental update
        while self.memory_bank.memory_size >= self.max_memory_size:
            self.memory_reader.update_projection(self.memory_bank)
        # learn from the hidden states
        inputs_embeds = kwargs.get('inputs_embeds', None)  # output layer uses target embeds directly
        new_keys, new_vals = self.memory_extractor(hidden_states, attention_mask, inputs_embeds)
        self.memory_bank.update_memory(new_keys, new_vals)
        # do NOT inference using persistent projection
        # it causes performance degradation
        # mem_states = self.memory_reader(None, hidden_states)
        # hidden_states = self.interpreter(hidden_states, mem_states, attention_mask)
        return hidden_states

    def _inference(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs: Unpack[TransformersKwargs]):
        mem_states = self.memory_reader(self.memory_bank, hidden_states)
        hidden_states = self.interpreter(hidden_states, mem_states, attention_mask)
        return hidden_states
    
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs: Unpack[TransformersKwargs],):
        if self.stage == 'learn':
            hidden_states = self._learn(hidden_states, attention_mask, **kwargs)
        elif self.stage == 'infer':
            hidden_states = self._inference(hidden_states, attention_mask, **kwargs)
        else:
            raise ValueError(f"Unknown stage: {self.stage}")
        return hidden_states
   
    def log_metrics(self, log_obj):
        self.memory_bank.log_metrics(log_obj)
        self.memory_reader.log_metrics(log_obj)
        self.interpreter.log_metrics(log_obj)


if __name__ == "__main__":
    '''Solve hyperparameters c and a for rtol=c*N^a from empirical data, where the rtols show good results.'''
    rtols = [0.01, 0.0001, 0.1, 0.000001]
    Ns = [191, 95166, 79, 16584550]
    A = np.array([[1, np.log10(N)] for N in Ns])
    b = np.array([np.log10(rtol) for rtol in rtols])
    x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print('c:', 10**x[0], 'a:', x[1])
    # print prediction of b
    for i in range(len(Ns)):
        pred_b = x[0] + x[1] * np.log10(Ns[i])
        print(f'N={Ns[i]}, true rtol={rtols[i]}, pred rtol={10**pred_b}')