"""Train one Task 3 method (DAN-DG or SAM).   python task3/train.py --config task3/configs/sam.yaml

Sketch is never loaded: Task 3 methods have needs_target=False so the shared trainer
does not build a target loader, and checkpoint selection uses source validation only.
"""
import argparse
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.seed import seed_everything  # noqa: E402
from shared.config import load_config  # noqa: E402
from shared.pacs import PACSTable  # noqa: E402
from shared.pacs_protocol import load_splits  # noqa: E402
from shared.pacs_trainer import train  # noqa: E402
from task2.models.model import PACSModel  # noqa: E402
from task3.methods import REGISTRY  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default="task3/configs/base.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config, a.base)
    out_dir = os.path.join(cfg["results_root"], cfg["name"])
    if cfg["method"]["type"] == "erm":
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy(cfg["checkpoint"], os.path.join(out_dir, "best.pt"))
        for f in ("curves_epoch.csv", "curves_step.csv", "train_summary.json"):
            src = os.path.join(os.path.dirname(cfg["checkpoint"]), f)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out_dir, f))
        print("ERM: copied Task 2 Source-only checkpoint to", out_dir)
        sys.exit(0)
    assert not REGISTRY[cfg["method"]["type"]].needs_target
    seed_everything(int(cfg["seed"]))
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    table = PACSTable(cfg["data"]["parquet"])
    splits = load_splits(cfg["data"]["splits"])
    model = PACSModel(int(cfg["data"]["n_classes"])).to(device)
    method = REGISTRY[cfg["method"]["type"]](model, cfg, device)
    summary = train(model, method, table, splits, cfg, out_dir, device)
    print(summary["name"], "best epoch", summary["best_epoch"], "best mean val F1", round(summary["best_val_mean_macro_f1"], 4))
