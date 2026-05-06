import os
import torch
import torch.nn as nn
from .external_layers import ExternalLayers
from .modeling_gpt2 import GPT2LMHeadModel
from .modeling_qwen2 import Qwen2ForCausalLM
from .modeling_llama import LlamaForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


GPT2_MODELS = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
QWEN2_MODELS = ['Qwen2.5-3B', 'Qwen2.5-7B', 'Qwen2.5-3B-Instruct', 'Qwen2.5-7B-Instruct']
LLAMA_MODELS = ['Llama-3-8B', 'Llama-3-8B-Instruct']


def load_base_model_config(base_model: str):
    base_model_path = f"../../huggingface/models/{base_model}"
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    print(f'Loaded base model config from {base_model_path}')
    return config


def load_base_model_tokenizer(base_model: str):
    base_model_path = f"../../huggingface/models/{base_model}"
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    print(f'Loaded base model tokenizer from {base_model_path}')
    return tokenizer


def load_base_model(base_model: str):
    base_model_path = f"../../huggingface/models/{base_model}"
    model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
    # freeze base model parameters
    for param in model.parameters():
        param.requires_grad = False
    print(f'Loaded base model from {base_model_path}')
    return model


class BasicConfig(PretrainedConfig):

    base_model: str = "gpt2"
    num_layers: int = 1

    def __init__(self,
                base_model: str = "gpt2",
                num_layers: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.num_layers = num_layers


class BasicExternalLayers(ExternalLayers):

    def __init__(self, config: BasicConfig, model: nn.Module, layer_cls: type):
        super().__init__(model)
        # add memory decoder layers
        self.config = config
        self.layer_cls = layer_cls
        layers = []
        total_layers = self.get_layers_from_model(model)
        stride = total_layers / (config.num_layers + 1)
        assert stride >= 1, f'num_layers {config.num_layers} exceeds total layers {total_layers}'
        for i in range(config.num_layers):
            layer_index = int(stride * (i + 1))
            layers.append(layer_cls(config, layer_index))
        layers.append(layer_cls(config, total_layers, is_output_layer=True))  # output layer
        self.layers = nn.ModuleList(layers)
        self.layer_index2idx = dict((layer.layer_index, idx) for idx, layer in enumerate(layers))
        # log status
        self.seq_len = None  # sequence length
        self.batch_len = None  # total tokens in the batch

    def get_layers_from_model(self, model):
        if hasattr(model, "transformer"):
            return len(model.transformer.h)          # GPT-2
        elif hasattr(model.model, "layers"):
            return len(model.model.layers)            # LLaMA / Qwen
        else:
            raise ValueError("Unknown model structure")

    def prepare(self, hidden_states: torch.Tensor, **kwargs):
        self.seq_len = hidden_states.size(1)
        self.batch_len = hidden_states.size(0) * hidden_states.size(1)

    def forward(self, layer_index: int, hidden_states: torch.Tensor, **kwargs):
        if layer_index in self.layer_index2idx:
            layer = self.layers[self.layer_index2idx[layer_index]]
            hidden_states = layer(hidden_states, **kwargs)
        return hidden_states

    def output(self, layer_index: int, hidden_states: torch.Tensor, **kwargs):
        if layer_index in self.layer_index2idx:
            layer = self.layers[self.layer_index2idx[layer_index]]
            hidden_states = layer(hidden_states, **kwargs)
        return hidden_states
    
    def log_metrics(self, log_obj):
        log_obj['seq_len'] = self.seq_len
        log_obj['batch_len'] = self.batch_len
        for layer in self.layers:
            layer.log_metrics(log_obj)


class BasicCausalLM(PreTrainedModel):
    ''' A wrapper model to integrate external layers '''
    config: BasicConfig
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True

    def __init__(self, config: BasicConfig, layers_cls: type):
        super().__init__(config)
        self.config = config
        self.model = self._build_model(config.base_model)
        self.config.max_length = getattr(self.model.config, "max_position_embeddings", None) \
                                or getattr(self.model.config, "n_positions", None)
        self.config.hidden_size = self.model.config.hidden_size
        self.external = layers_cls(config, self.model)
        # freeze base model parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def is_gradient_checkpointing(self):
        return self.model.is_gradient_checkpointing

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self):
        return self.model.gradient_checkpointing_disable()
    
    def _build_model(self, base_model: str):
        if base_model in GPT2_MODELS:
            base_model_cls = GPT2LMHeadModel
        elif base_model in QWEN2_MODELS:
            base_model_cls = Qwen2ForCausalLM
        elif base_model in LLAMA_MODELS:
            base_model_cls = LlamaForCausalLM
        else:
            raise NotImplementedError(f'Unsupported base model: {base_model}')
        model_config = load_base_model_config(base_model)
        model_config.use_cache = False
        model = base_model_cls(model_config)
        return model

    def set_base_model(self, model: nn.Module):
        self.model.load_state_dict(model.state_dict())

    @property
    def require_learning(self):
        return False
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def log_metrics(self, log_obj):
        self.external.log_metrics(log_obj)

    def print_parameters(self):
        print('Freezed parameters:')
        nparams = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(name, f'({param.norm()})')
                nparams += param.numel()
        print(f'Total parametes: {nparams/1024/1024:.2f}M')
        # trainable parameters
        print('Trainable parameters:')
        nparams = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, f'({param.norm()})')
                nparams += param.numel()
        print(f'Total parametes: {nparams/1024/1024:.2f}M')