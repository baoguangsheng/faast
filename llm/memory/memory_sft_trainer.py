import torch
from torch import nn
from transformers import TrainerCallback
from trl import SFTTrainer
from trl.trainer.model_config import ModelConfig
from typing import Optional, Dict, Any, Union
from ..alignment import SFTConfig
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MemorySFTConfig(SFTConfig):

    max_memory_size: int = field(default=128, metadata={"help": "Maximum memory size in K tokens."})


class MemorySFTTrainer(SFTTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        # forward pass
        self.model.set_max_memory_size(self.args.max_memory_size * 1024)  # in tokens
        self.model.learn(**inputs)
        loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)
        # log metrics
        log_obj = {}
        self.model.log_metrics(log_obj)
        mode = "train" if self.model.training else "eval"
        for key in log_obj:
            self._metrics[mode][key].append(log_obj[key])
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only = None, ignore_keys = None, metric_key_prefix = "eval"):
        self.label_names = ['labels']
        self.model.reset_projection()
        result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        self.model.reset_projection()
        return result
