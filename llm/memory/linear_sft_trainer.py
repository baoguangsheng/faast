import torch
from torch import nn
from transformers import TrainerCallback
from trl import SFTTrainer
from trl.trainer.model_config import ModelConfig
from typing import Optional, Dict, Any, Union
from ..alignment import SFTConfig
import numpy as np
from dataclasses import dataclass, field



class LinearSFTTrainer(SFTTrainer):

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
        loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)
        log_obj = {}
        self.model.log_metrics(log_obj)
        # log metrics
        mode = "train" if self.model.training else "eval"
        for key in log_obj:
            self._metrics[mode][key].append(log_obj[key])
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only = None, ignore_keys = None, metric_key_prefix = "eval"):
        self.label_names = ['labels']
        result = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        return result
