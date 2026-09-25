"""Method interface used by the shared PACS training loop."""
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Method:
    needs_target: bool = False

    def __init__(self, model: nn.Module, cfg: dict, device):
        self.model = model
        self.cfg = cfg
        self.device = device

    def extra_modules(self) -> List[nn.Module]:
        return []

    def train_step(self, xs, ys, xt, progress: float, optimizer) -> Dict[str, float]:
        """Default: one loss, one backward, one optimizer step."""
        loss, logs = self.loss(xs, ys, xt, progress)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if getattr(self, "grad_clip", None):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
        optimizer.step()
        logs["loss_total"] = float(loss.item())
        return logs

    def loss(self, xs, ys, xt, progress) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    def state_dict(self) -> Dict:
        return {f"extra_{i}": m.state_dict() for i, m in enumerate(self.extra_modules())}

    def load_state_dict(self, sd: Dict):
        for i, m in enumerate(self.extra_modules()):
            m.load_state_dict(sd[f"extra_{i}"])
