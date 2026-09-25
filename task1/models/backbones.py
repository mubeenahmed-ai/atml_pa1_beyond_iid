"""Frozen backbone wrappers with a uniform interface.

Each wrapper receives the *common* (B,3,224,224) float image in [0,1], applies its
own normalisation and returns the required representation:
  * ResNet-50   : global-average-pooled 2048-d feature
  * ViT-B/16    : final class token (768-d, after the encoder LayerNorm)
  * CLIP ViT-B/32: L2-normalised 512-d image embedding
"""
from typing import Dict, List

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class _Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


class FrozenBackbone(nn.Module):
    name: str = ""
    feat_dim: int = 0

    def __init__(self):
        super().__init__()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x in [0,1]
        raise NotImplementedError


class ResNet50Backbone(FrozenBackbone):
    name, feat_dim = "resnet50", 2048

    def __init__(self):
        super().__init__()
        m = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()  # output = global-average-pooled feature
        self.net = m
        self.norm = _Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.freeze()

    @torch.no_grad()
    def forward(self, x):
        return self.net(self.norm(x))


class ViTB16Backbone(FrozenBackbone):
    name, feat_dim = "vit_b16", 768

    def __init__(self):
        super().__init__()
        m = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads = nn.Identity()  # torchvision returns the (LayerNorm-ed) class token
        self.net = m
        self.norm = _Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.freeze()

    @torch.no_grad()
    def forward(self, x):
        return self.net(self.norm(x))


class CLIPBackbone(FrozenBackbone):
    name, feat_dim = "clip_vitb32", 512

    def __init__(self, model_name="ViT-B-32", pretrained="openai", local_weights="data/weights/clip_vitb32_openai.bin"):
        super().__init__()
        import os

        import open_clip

        # `pretrained='openai'` weights; if the file was fetched manually (slow HF hub client) load it from disk
        src = local_weights if (local_weights and os.path.exists(local_weights)) else pretrained
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=src)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.norm = _Normalize(CLIP_MEAN, CLIP_STD)
        self.freeze()

    @torch.no_grad()
    def forward(self, x):
        f = self.model.encode_image(self.norm(x))
        return f / f.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def text_embeddings(self, class_names: List[str], prompt: str, device) -> torch.Tensor:
        tokens = self.tokenizer([prompt.format(c) for c in class_names]).to(device)
        t = self.model.encode_text(tokens)
        return t / t.norm(dim=-1, keepdim=True)

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp().item())


def build_backbones(device) -> Dict[str, FrozenBackbone]:
    return {
        "resnet50": ResNet50Backbone().to(device),
        "vit_b16": ViTB16Backbone().to(device),
        "clip_vitb32": CLIPBackbone().to(device),
    }
