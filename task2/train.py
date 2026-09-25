"""Train one Task 2 method.   python task2/train.py --config task2/configs/dan.yaml

Target (Sketch) labels are never loaded here: the adaptation loader returns -1 labels
and the evaluation loader with labels lives only in evaluate_final.py.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.seed import seed_everything  # noqa: E402
from shared.config import load_config  # noqa: E402
from shared.pacs import PACSTable  # noqa: E402
from shared.pacs_protocol import load_splits  # noqa: E402
from shared.pacs_trainer import train  # noqa: E402
from task2.methods import REGISTRY  # noqa: E402
from task2.models.model import PACSModel  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default="task2/configs/base.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config, a.base)
    seed_everything(int(cfg["seed"]))
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    table = PACSTable(cfg["data"]["parquet"])
    splits = load_splits(cfg["data"]["splits"])
    model = PACSModel(int(cfg["data"]["n_classes"])).to(device)
    method = REGISTRY[cfg["method"]["type"]](model, cfg, device)
    out_dir = os.path.join(cfg["results_root"], cfg["name"])
    summary = train(model, method, table, splits, cfg, out_dir, device)
    print(summary["name"], "best epoch", summary["best_epoch"], "best mean val F1", round(summary["best_val_mean_macro_f1"], 4))
