"""Source-side diagnostics for Task 3 models WITHOUT loading Sketch.

Per source-domain validation accuracy / macro-F1, mean and worst, 3-way source-domain
separability, and the common sharpness proxy. Safe to run before any Task 3 decision is
frozen because it never touches the target domain.

    python task3/selection/source_validation.py --methods erm dan_dg sam
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.logging import save_csv, save_json  # noqa: E402
from common.metrics import classification_metrics  # noqa: E402
from shared.config import load_config  # noqa: E402
from shared.pacs import PACSTable, SOURCE_DOMAINS  # noqa: E402
from shared.pacs_protocol import load_splits, predict, source_val_loaders  # noqa: E402
from task2.models.model import PACSModel  # noqa: E402
from task3.evaluation.domain_metrics import mean_worst  # noqa: E402
from task3.evaluation.sharpness import fixed_val_batch, sharpness_proxy  # noqa: E402
from task3.evaluation.source_domain_separability import separability  # noqa: E402


def source_diagnostics(methods, results_root, base_cfg, device):
    cfg = load_config(os.path.join(os.path.dirname(base_cfg), "erm.yaml"), base_cfg)
    table = PACSTable(cfg["data"]["parquet"])
    splits = load_splits(cfg["data"]["splits"])
    val_loaders = source_val_loaders(table, splits)
    xb, yb, batch_idx = fixed_val_batch(table, splits, int(cfg["diagnostics"]["sharpness_batch_per_domain"]), int(cfg["seed"]))
    xb, yb = xb.to(device), yb.to(device)
    n_classes = int(cfg["data"]["n_classes"])
    rows = []
    for m in methods:
        model = PACSModel(n_classes).to(device).load(os.path.join(results_root, m, "best.pt"), device)
        per_dom, feats = {}, []
        for d in SOURCE_DOMAINS:
            f, z, y = predict(model, val_loaders[d], device)
            per_dom[d] = classification_metrics(y, z.argmax(1), n_classes)
            feats.append(f)
        row = {"method": m}
        for d in SOURCE_DOMAINS:
            row[f"{d}_acc"], row[f"{d}_f1"] = per_dom[d]["accuracy"], per_dom[d]["macro_f1"]
        row.update({("src_" + k): v for k, v in mean_worst(per_dom, SOURCE_DOMAINS).items()})
        sep = separability(feats, seed=int(cfg["seed"]))
        row["source_domain_separability"] = sep["score"]
        row["sep_n_per_group"] = sep["n_per_group"]
        row.update({("sharp_" + k): v for k, v in sharpness_proxy(model, xb, yb, float(cfg["diagnostics"]["sharpness_rho"])).items()})
        rows.append(row)
        print(f"{m:10s} mean acc {row['src_mean_accuracy']:.4f} F1 {row['src_mean_macro_f1']:.4f} worst F1 {row['src_worst_macro_f1']:.4f} "
              f"({row['src_worst_domain_macro_f1']}) | sep {sep['score']:.3f} | sharp Δ {row['sharp_delta_sharp']:.4f}")
    out_dir = os.path.join(results_root, "source_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    save_csv(rows, os.path.join(out_dir, "source_table.csv"))
    save_json({"rows": rows, "sharpness_batch_indices": batch_idx, "sketch_loaded": False}, os.path.join(out_dir, "source_diagnostics.json"))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["erm", "dan_dg", "sam"])
    ap.add_argument("--results_root", default="task3/results")
    ap.add_argument("--base", default="task3/configs/base.yaml")
    a = ap.parse_args()
    source_diagnostics(a.methods, a.results_root, a.base, "cuda" if torch.cuda.is_available() else "cpu")
