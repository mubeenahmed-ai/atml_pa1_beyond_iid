"""CIFAR-appropriate ResNet-18: 3x3 stride-1 stem, no max-pool, 32x32 inputs.

Exposes the split needed by PROSER's manifold mixup (after layer2 / before layer3)
and an optional bank of dummy classifiers.
"""
import torch
import torch.nn as nn
import torchvision


class ResNet18CIFAR(nn.Module):
    feat_dim = 512

    def __init__(self, n_classes: int = 10, n_dummy: int = 0):
        super().__init__()
        m = torchvision.models.resnet18(weights=None, num_classes=n_classes)
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        self.net = m
        self.dummy = nn.Linear(self.feat_dim, n_dummy) if n_dummy > 0 else None

    # ---- pieces
    def forward_pre(self, x):
        """conv stem + layer1 + layer2  (phi_pre in the assignment)."""
        n = self.net
        x = n.relu(n.bn1(n.conv1(x)))
        x = n.maxpool(x)
        x = n.layer1(x)
        return n.layer2(x)

    def forward_post_feat(self, h):
        """layer3 + layer4 + global pool -> 512-d penultimate feature."""
        n = self.net
        h = n.layer4(n.layer3(h))
        return torch.flatten(n.avgpool(h), 1)

    def features(self, x):
        return self.forward_post_feat(self.forward_pre(x))

    def forward(self, x):
        f = self.features(x)
        return f, self.net.fc(f)

    def dummy_logits(self, f):
        assert self.dummy is not None
        return self.dummy(f)

    def add_dummy_classifiers(self, n_dummy: int, seed: int = 6304):
        g = torch.Generator().manual_seed(seed)
        lin = nn.Linear(self.feat_dim, n_dummy)
        with torch.no_grad():
            bound = 1.0 / (self.feat_dim**0.5)
            lin.weight.copy_(torch.empty(n_dummy, self.feat_dim).uniform_(-bound, bound, generator=g))
            lin.bias.copy_(torch.empty(n_dummy).uniform_(-bound, bound, generator=g))
        self.dummy = lin.to(next(self.parameters()).device)
        return self
