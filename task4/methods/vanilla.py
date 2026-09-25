"""Vanilla ten-class cross-entropy classifier."""
import torch
import torch.nn.functional as F

from task4.methods.base import OSRMethod


class Vanilla(OSRMethod):
    def train_step(self, x, y, optimizer, scaler=None):
        with torch.autocast("cuda", enabled=scaler is not None):
            _, logits = self.model(x)
            loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); optimizer.step()
        return {"loss": float(loss.item()), "acc": float((logits.argmax(1) == y).float().mean().item())}
