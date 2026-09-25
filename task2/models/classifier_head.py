import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 512, n_classes: int = 7):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, f):
        return self.fc(f)
