import os
import torch
import torch.nn as nn
from .basic_causal_lm import BasicConfig, BasicExternalLayers, BasicCausalLM
from .memory_layer import MemoryLayer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from .utils import ProxyObject, create_rtl_causal_mask


class MemoryConfig(BasicConfig):

    reader_type: str = "inverse"
    extractor_type: str = "next-token"
    interpreter_type: str = "projection"
    max_memory_size: int = 1024 * 64  # maximum memory size (number of tokens)
    reader_update_size: int = 1024 * 8  # update projection matrix with N tokens each time
    reader_update_discount: float = 0.9  # discount factor for historical N0 during training
    reader_update_alpha: float = 1.0  # alpha for pseudo-inverse rtol during inference

    def __init__(self,
                reader_type: str = "inverse",
                extractor_type: str = "next-token",
                interpreter_type: str = "projection",
                max_memory_size: int = 1024 * 64,
                reader_update_size: int = 1024 * 8,
                reader_update_discount: float = 0.9,
                reader_update_alpha: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.reader_type = reader_type
        self.extractor_type = extractor_type
        self.interpreter_type = interpreter_type
        self.max_memory_size = max_memory_size
        self.reader_update_size = reader_update_size
        self.reader_update_discount = reader_update_discount
        self.reader_update_alpha = reader_update_alpha


class MemoryExternalLayers(BasicExternalLayers):

    def __init__(self, config: MemoryConfig, model: nn.Module):
        super().__init__(config, model, MemoryLayer)

    def set_stage(self, stage: str):
        for layer in self.layers:
            layer.stage = stage

    @property
    def memory_size(self):
        memory_sizes = []
        for layer in self.layers:
            memory_sizes.append(layer.memory_bank.memory_size)
        return max(memory_sizes)

    def set_max_memory_size(self, size: int):
        for layer in self.layers:
            layer.set_max_memory_size(size)

    def reset_projection(self):
        for layer in self.layers:
            layer.reset_projection()

    def prepare(self, hidden_states: torch.Tensor, attention_mask: torch.FloatTensor, *args, **kwargs):
        super().prepare(hidden_states, attention_mask=attention_mask, *args, **kwargs)
        for layer in self.layers:
            layer.memory_bank.to(hidden_states.device)

    def forward(self, layer_index: int, hidden_states: torch.Tensor, *args, **kwargs):
        if layer_index in self.layer_index2idx:
            layer = self.layers[self.layer_index2idx[layer_index]]
            hidden_states = layer(hidden_states, *args, **kwargs)
        return hidden_states


class MemoryCausalLM(BasicCausalLM):
    ''' A wrapper model to integrate external memory layers '''

    config: MemoryConfig

    def __init__(self, config: MemoryConfig):
        super().__init__(config, MemoryExternalLayers)

    @property
    def require_learning(self):
        return True

    @property
    def memory_size(self):
        return self.external.memory_size

    def set_max_memory_size(self, size: int):
        self.external.set_max_memory_size(size)

    def reset_projection(self):
        self.external.reset_projection()

    def learn(self, *args, **kwargs):
        self.external.set_stage('learn')
        outputs = super().forward(*args, **kwargs)
        return outputs

    def forward(self, *args, **kwargs):
        self.external.set_stage('infer')
        outputs = super().forward(*args, **kwargs)
        return outputs

    def generate(self, *args, **kwargs):
        self.external.set_stage('infer')
        outputs = super().generate(*args, **kwargs)
        return outputs


if __name__ == '__main__':
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--base_model', type=str, default='gpt2')
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--reader_type', type=str, default='inverse')
    parser.add_argument('--extractor_type', type=str, default='next-token')
    parser.add_argument('--interpreter_type', type=str, default='projection')
    parser.add_argument('--max_memory_size', type=int, default=1024 * 64)
    parser.add_argument('--reader_update_size', type=int, default=1024 * 8)
    parser.add_argument('--reader_update_discount', type=float, default=0.9)
    parser.add_argument('--reader_update_alpha', type=float, default=1.0)
    parser.add_argument('--model_path', type=str, default='exp_llm/models/memory-gpt2')
    args = parser.parse_args()

    os.makedirs(args.model_path, exist_ok=True)

    # create config
    config = MemoryConfig(dtype=getattr(torch, args.dtype))
    config.base_model = args.base_model
    config.num_layers = args.num_layers
    config.reader_type = args.reader_type
    config.extractor_type = args.extractor_type
    config.interpreter_type = args.interpreter_type
    config.max_memory_size = args.max_memory_size
    config.reader_update_size = args.reader_update_size
    config.reader_update_discount = args.reader_update_discount
    config.reader_update_alpha = args.reader_update_alpha
    config.auto_map =  {
    "AutoConfig": "model.MemoryConfig",
    "AutoModelForCausalLM": "model.MemoryCausalLM"
    }
    print('Config:', config)
    config.save_pretrained(args.model_path)

    # create code
    with open(args.model_path + '/model.py', 'w') as fout:
        fout.write(
'''from llm import MemoryConfig, MemoryCausalLM

__all__ = [
    "MemoryConfig",
    "MemoryCausalLM",
]
''')

    # create tokenizer
    from .basic_causal_lm import load_base_model_tokenizer, load_base_model
    tokenizer = load_base_model_tokenizer(args.base_model)
    tokenizer.save_pretrained(args.model_path)
    print('Tokenizer saved:', args.model_path)

    # create model
    model = MemoryCausalLM(config).to(getattr(torch, args.dtype))
    model.set_base_model(load_base_model(args.base_model))
    print('Trainable parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, f'({param.norm()})', param.shape)
    model.save_pretrained(args.model_path)
    print('Memory LLM saved:', args.model_path)

    # # test loading
    # print('Test loading the saved model:', args.model_path)
    # config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    # print('Config loaded:', config)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    # model = model.to(config.dtype)
    # print('Freezed parameters:')
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         print(name, f'({param.norm()})', param.dtype)
    # print('Trainable parameters:')
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, f'({param.norm()})', param.dtype)
    # print('Model loaded successfully.')
