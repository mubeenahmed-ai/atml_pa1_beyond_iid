"""Linear classifier head trained on frozen, cached features."""
import copy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def train_linear_head(
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, n_classes: int,
    epochs: int = 50, lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 256,
    patience: int = 5, seed: int = 6304, device="cuda",
) -> Tuple[nn.Linear, Dict]:
    torch.manual_seed(seed)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.as_tensor(yva, dtype=torch.long, device=device)

    head = nn.Linear(Xtr.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best_acc, best_state, best_epoch, bad = -1.0, None, -1, 0
    history = []
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr_t), generator=gen).to(device)
        tot, n = 0.0, 0
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            loss = nn.functional.cross_entropy(head(Xtr_t[b]), ytr_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        head.eval()
        with torch.no_grad():
            va_acc = (head(Xva_t).argmax(1) == yva_t).float().mean().item()
        history.append({"epoch": ep + 1, "train_loss": tot / n, "val_acc": va_acc})
        if va_acc > best_acc:
            best_acc, best_state, best_epoch, bad = va_acc, copy.deepcopy(head.state_dict()), ep + 1, 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, {"best_val_acc": best_acc, "best_epoch": best_epoch, "epochs_run": len(history), "history": history}
