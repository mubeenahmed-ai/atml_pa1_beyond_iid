"""Method interface for the Task 4 trainer: one train_step per batch."""
from typing import Dict

import torch


class OSRMethod:
    def __init__(self, model, cfg: dict, device):
        self.model, self.cfg, self.device = model, cfg, device

    def parameters(self):
        return list(self.model.parameters())

    def train_step(self, x, y, optimizer, scaler=None) -> Dict[str, float]:
        raise NotImplementedError

    def extra_state(self) -> Dict:
        return {}

    def load_extra_state(self, sd: Dict):
        pass
