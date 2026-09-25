"""Backbone + head wrapper returning (feature, logits)."""
import torch
import torch.nn as nn

from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import ClassifierHead


class PACSModel(nn.Module):
    def __init__(self, n_classes: int = 7, pretrained: bool = True):
        super().__init__()
        self.backbone = ResNet18Backbone(pretrained)
        self.head = ClassifierHead(self.backbone.feat_dim, n_classes)

    def forward(self, x):
        f = self.backbone(x)
        return f, self.head(f)

    def load(self, path: str, device="cpu"):
        sd = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(sd["model"] if "model" in sd else sd)
        return self
