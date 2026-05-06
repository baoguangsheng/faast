import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ProxyObject


class ExternalLayers(nn.Module):
    ''' An abstract base class for external layers to be integrated with the main model '''

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = ProxyObject(model)
        self.model.set_external_layers(ProxyObject(self))

    def prepare(self, *args, **kwargs):
        ''' Prepare the module before calling the layers'''
        raise NotImplementedError

    def forward(self, layer_index: int, *args, **kwargs):
        ''' Call the external layer before calling the transformer layer at layer_index '''
        raise NotImplementedError

    def output(self, layer_index: int, *args, **kwargs):
        ''' Call the external output layer after calling all transformer layers '''
        raise NotImplementedError