import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from .memory.memory_predictor import MemoryPredictor
from typing import List, Optional, Tuple, Union


def load_clip():
    # load CLIP model
    local_file_path = "exp_vision/models/resnet50_clip/RN50.pt"
    model, preprocess = clip.load(local_file_path, device='cpu')
    model.eval()
    return model, preprocess


class ClipClassifier(nn.Module):
    def __init__(self, model: nn.Module, classes: List[str]):
        super().__init__()
        self.model = model
        # prepare class embeddings
        self.num_classes = len(classes)
        texts = clip.tokenize([f"A photo of a {label}." for label in classes])
        with torch.no_grad():
            text_features = model.encode_text(texts)
            text_features = F.normalize(text_features, dim=-1)
        self.embeddings = nn.Embedding.from_pretrained(text_features, freeze=True)
        self.freeze_pretrained()
        # log
        self.x_embed = None
        self.y_pred = None

    def freeze_pretrained(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.embeddings.parameters():
            param.requires_grad = False

    @classmethod
    def load_pretrain(cls, model_file):
        print('Loading model from', model_file)
        obj = torch.load(model_file, weights_only=False)
        config = obj['config']
        state_dict = obj['model']
        model = cls(config)
        mis_keys, unexp_keys = model.load_state_dict(state_dict, strict=False)
        if mis_keys:
            print('Missing keys:', mis_keys)
        if unexp_keys:
            print('Unexpected keys:', unexp_keys)
        return model

    def save_to(self, model_file):
        state_dict = self.state_dict()
        torch.save(state_dict, model_file)
        print('Model saved:', model_file)
        
    def load_from(self, model_file, freeze_pretrained: bool = True):
        print('Loading model from', model_file)
        state_dict = torch.load(model_file, weights_only=False)
        mis_keys, unexp_keys = self.load_state_dict(state_dict, strict=False)
        if mis_keys:
            print('Missing keys:', mis_keys)
        if unexp_keys:
            print('Unexpected keys:', unexp_keys)
        # freeze pretrained encoder and embeddings
        if freeze_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.embeddings.parameters():
                param.requires_grad = False

    def _encode(self, images: torch.Tensor):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            self.x_embed = image_features.detach()
        return image_features
    
    def _embed(self, labels: torch.Tensor):
        with torch.no_grad():
            return self.embeddings(labels)

    def _classify(self, image_features: torch.Tensor):
        image_features = F.normalize(image_features, dim=-1)
        self.y_pred = image_features.detach()
        text_features = self.embeddings.weight
        logits = image_features @ text_features.T
        logits = self.model.logit_scale.exp() * logits
        return logits

    def forward(self, images: torch.Tensor):
        x_embed = self._encode(images)
        logits = self._classify(x_embed)
        return logits

    @property
    def require_learning(self):
        return False
    
    def learn(self, images: torch.Tensor, labels: torch.Tensor):
        # only memory predictor needs to implement this
        pass

    def log_metrics(self, log_obj):
        if self.x_embed is not None:
            log_obj['x_embed_norm'] = self.x_embed.norm(dim=1).mean().item()
        if self.y_pred is not None:
            log_obj['y_pred_norm'] = self.y_pred.norm(dim=1).mean().item()

    def print_parameters(self):
        print('Trainable parameters:')
        nparams = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, f'({param.norm()})')
                nparams += param.numel()
        print(f'Total parametes: {nparams/1024/1024:.2f}M')

    def has_trainable_parameters(self):
        for param in self.parameters():
            if param.requires_grad:
                return True
        return False


class LinearClassifier(ClipClassifier):
    def __init__(self, model: nn.Module, classes: List[str]):
        super().__init__(model, classes)
        self.predictor = nn.Linear(model.visual.output_dim, self.embeddings.embedding_dim, bias=False)
        # log
        self.x_embed = None
        self.y_pred = None

    def reset_projection(self):
        self.predictor.reset_parameters()

    def train(self, mode = True):
        super().train(False)
        self.predictor.train(mode)
        return self
    
    def forward(self, images: torch.Tensor):
        x_embed = self._encode(images)
        y_pred = x_embed + self.predictor(x_embed)
        logits = self._classify(y_pred)
        return logits


class MemoryClassifier(ClipClassifier):

    def __init__(self, config, model: nn.Module, classes: List[str]):
        super().__init__(model, classes)
        self.predictor = MemoryPredictor(config)
        self.n0_fn = None  # no prior by default
        # log
        self.n0 = 0

    def log_metrics(self, log_obj):
        super().log_metrics(log_obj)
        self.predictor.log_metrics(log_obj)
        if self.n0 is not None:
            log_obj['n0'] = self.n0

    @property
    def require_learning(self):
        return True

    def reset_projection(self):
        self.predictor.reset_projection()

    def learn(self, images: torch.Tensor, labels: torch.Tensor):
        x_embed = self._encode(images)
        y_embed = self._embed(labels)
        self.predictor.learn(x_embed, y_embed)

    def forward(self, images: torch.Tensor):
        x_embed = self._encode(images)
        # residual connection for zero-shot
        N = self.predictor.memory_bank.memory_size
        N0 = self.n0_fn(N) if self.n0_fn is not None else 0
        r = N / (N0 + N)
        if r > 0:
            y_pred = self.predictor.forward(x_embed)
            y_pred = x_embed * (1 - r) + y_pred * r
        else:
            y_pred = x_embed
        logits = self._classify(y_pred)
        # log
        self.n0 = N0
        return logits
    
    def get_memory(self):
        return self.predictor.memory_bank.get_memory()

    def set_memory(self, keys, vals):
        self.predictor.memory_bank.set_memory(keys, vals)
        
    def load_memory(self, memory_file):
        keys, vals = torch.load(memory_file)
        self.set_memory(keys, vals)
        key_norm = keys.norm(dim=-1).mean().item()
        val_norm = vals.norm(dim=-1).mean().item()
        print(f'Load memory with key norm {key_norm:.4f} and val norm {val_norm:.4f} from', memory_file)

    def save_memory(self, memory_file):
        if memory_file:
            ver, keys, vals = self.get_memory()
            torch.save((keys, vals), memory_file)
            key_norm = keys.norm(dim=-1).mean().item()
            val_norm = vals.norm(dim=-1).mean().item()
            print(f'Save memory with key norm {key_norm:.4f} and val norm {val_norm:.4f} to', memory_file)



if __name__ == '__main__':
    print("Verifying ResNet50-CLIP model...")
    local_file_path = "exp_vision/models/resnet50_clip/RN50.pt"
    model, preprocess = clip.load(local_file_path, device='cpu')
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    features = model.encode_image(dummy_input) # Returns image embeddings
    print(features.shape)
