"""ResNet-18 (ImageNet V1) feature extractor -> 512-d pooled feature."""
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights


class ResNet18Backbone(nn.Module):
    feat_dim = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        m = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Identity()
        self.net = m

    def forward(self, x):
        return self.net(x)
