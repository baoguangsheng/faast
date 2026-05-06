import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from .basic_causal_lm import BasicConfig, BasicExternalLayers, BasicCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel



class LinearLayer(nn.Module):

    def __init__(self, config, layer_index):
        super().__init__()
        self.layer_index = layer_index
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.pred_norm = None

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                **kwargs: Unpack[TransformersKwargs],):
        delta_states = self.proj(hidden_states)
        hidden_states = hidden_states + delta_states
        # log
        self.pred_norm = torch.norm(delta_states, dim=-1).mean().item()
        return hidden_states
   
    def log_metrics(self, log_obj):
        if self.pred_norm is not None:
            log_obj[f'pred_norm{self.layer_index}'] = self.pred_norm


class LinearConfig(BasicConfig):

    model_type: str = "linear"

    def __init__(self, model_type: str = "linear", **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type


class LinearExternalLayers(BasicExternalLayers):

    def __init__(self, config: LinearConfig, model: nn.Module):
        super().__init__(config, model, LinearLayer)


class LinearCausalLM(BasicCausalLM):
    ''' A wrapper model to integrate external linear layers '''

    config: LinearConfig

    def __init__(self, config: LinearConfig):
        super().__init__(config, LinearExternalLayers)




if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--base_model', type=str, default='gpt2')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='exp_llm/models/linear-gpt2')
    args = parser.parse_args()

    os.makedirs(args.model_path, exist_ok=True)

    # create config
    config = LinearConfig(dtype=getattr(torch, args.dtype))
    config.base_model = args.base_model
    config.num_layers = args.num_layers
    config.auto_map =  {
    "AutoConfig": "model.LinearConfig",
    "AutoModelForCausalLM": "model.LinearCausalLM"
    }
    print('Config:', config)
    config.save_pretrained(args.model_path)

    # create code
    with open(args.model_path + '/model.py', 'w') as fout:
        fout.write(
'''from llm import LinearConfig, LinearCausalLM

__all__ = [
    "LinearConfig",
    "LinearCausalLM",
]
''')

    # create tokenizer
    from .basic_causal_lm import load_base_model_tokenizer, load_base_model
    tokenizer = load_base_model_tokenizer(args.base_model)
    tokenizer.save_pretrained(args.model_path)
    print('Tokenizer saved:', args.model_path)

    # create model
    model = LinearCausalLM(config).to(getattr(torch, args.dtype))
    model.set_base_model(load_base_model(args.base_model))
    print('Trainable parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, f'({param.norm()})')
    model.save_pretrained(args.model_path)
    print('Linear LLM saved:', args.model_path)

    # test loading
    print('Test loading the saved model:', args.model_path)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    print('Config loaded:', config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model = model.to(config.dtype)
    print('Freezed parameters:')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(name, f'({param.norm()})')
    print('Trainable parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, f'({param.norm()})')
    print('Model loaded successfully.')
