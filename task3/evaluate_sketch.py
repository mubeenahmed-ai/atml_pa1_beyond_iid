"""FINAL Task 3 evaluation - the only Task 3 script that loads Sketch.

Run after every Task 3 configuration is frozen. Combines the Sketch-free source diagnostics
(selection/source_validation.py) with Sketch accuracy / macro-F1, the change relative to
ERM, per-class Sketch changes and confusions, training curves, and a comparison with the
corresponding Task 2 target-aware results when available.

    python task3/evaluate_sketch.py --methods erm dan_dg sam
    python task3/evaluate_sketch.py --methods sam_rho0.01 sam sam_rho0.1 --tag rho_study --baseline sam
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.logging import save_csv, save_json  # noqa: E402
from common.metrics import classification_metrics  # noqa: E402
from common.plotting import PALETTE, plt  # noqa: E402
from shared.config import load_config  # noqa: E402
from shared.pacs import CLASS_NAMES, PACSTable, SOURCE_DOMAINS  # noqa: E402
from shared.pacs_protocol import load_splits, predict, target_eval_loader  # noqa: E402
from task2.evaluation.class_analysis import class_report  # noqa: E402
from task2.models.model import PACSModel  # noqa: E402
from task3.selection.source_validation import source_diagnostics  # noqa: E402


def main(methods, results_root, base_cfg, tag, baseline, device, task2_final="task2/results/final_main/table_main.csv"):
    src_rows = {r["method"]: r for r in source_diagnostics(methods, results_root, base_cfg, device)}
    cfg = load_config(os.path.join(os.path.dirname(base_cfg), "erm.yaml"), base_cfg)
    table = PACSTable(cfg["data"]["parquet"])
    splits = load_splits(cfg["data"]["splits"])
    tgt = target_eval_loader(table, splits)  # <-- Sketch is loaded here, after all decisions are fixed
    n_classes = int(cfg["data"]["n_classes"])
    rows, preds, per_class = [], {}, {}
    for m in methods:
        model = PACSModel(n_classes).to(device).load(os.path.join(results_root, m, "best.pt"), device)
        _, z, y = predict(model, tgt, device)
        met = classification_metrics(y, z.argmax(1), n_classes)
        preds[m] = z.argmax(1)
        r = dict(src_rows[m])
        r["sketch_acc"], r["sketch_f1"] = met["accuracy"], met["macro_f1"]
        rows.append(r)
    for r in rows:
        r["sketch_acc_delta_vs_erm"] = r["sketch_acc"] - [x for x in rows if x["method"] == baseline][0]["sketch_acc"]
        per_class[r["method"]] = class_report(y, preds[r["method"]], CLASS_NAMES, preds[baseline] if r["method"] != baseline else None)

    out_dir = os.path.join(results_root, f"final_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    save_csv(rows, os.path.join(out_dir, "table_main.csv"))
    save_json({"rows": rows, "per_class": per_class, "baseline": baseline}, os.path.join(out_dir, "final_results.json"))
    np.savez_compressed(os.path.join(out_dir, "sketch_predictions.npz"), y=y, **preds)
    save_csv([{"class": c, **{m: per_class[m]["per_class_acc"][c] for m in methods}} for c in CLASS_NAMES],
             os.path.join(out_dir, "per_class_sketch_acc.csv"))

    # comparison with Task 2 (target-aware) if available
    if os.path.exists(task2_final):
        t2 = pd.read_csv(task2_final).set_index("method")
        comp = []
        for m2, m3 in (("source_only", "erm"), ("dan", "dan_dg")):
            if m2 in t2.index and m3 in [r["method"] for r in rows]:
                r3 = [r for r in rows if r["method"] == m3][0]
                comp.append({"task2_method": m2, "task2_sketch_acc": float(t2.loc[m2, "target_acc"]), "task2_domain_sep(src vs sketch)": float(t2.loc[m2, "domain_separability"]),
                             "task3_method": m3, "task3_sketch_acc": r3["sketch_acc"], "task3_src_domain_sep(3-way)": r3["source_domain_separability"]})
        save_csv(comp, os.path.join(out_dir, "task2_vs_task3.csv"))

    # figures
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for m in methods:
        ep = pd.read_csv(os.path.join(results_root, m, "curves_epoch.csv"))
        st = pd.read_csv(os.path.join(results_root, m, "curves_step.csv"))
        col = PALETTE.get(m.split("_rho")[0])
        axes[0].plot(ep.epoch, ep.train_loss_cls, label=m, color=col)
        axes[1].plot(ep.epoch, ep.val_mean_f1, label=m, color=col)
        if "mmd2" in st:
            axes[2].plot(st.step, st.mmd2, label=f"{m} MMD²", color=col)
        if "sharpness_gap" in st:
            axes[2].plot(st.step, st.sharpness_gap, label=f"{m} L(θ+ε)−L(θ)", color=col, alpha=0.7)
    axes[0].set_title("source classification loss"); axes[1].set_title("mean source-val macro-F1"); axes[2].set_title("MMD penalty / SAM gap")
    for a in axes:
        a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "training_curves.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    others = [m for m in methods if m != baseline]
    w = 0.8 / max(len(others), 1)
    for k, m in enumerate(others):
        d = [per_class[m]["per_class_delta_vs_baseline"][c] for c in CLASS_NAMES]
        ax.bar(np.arange(n_classes) + k * w, d, w, label=m, color=PALETTE.get(m.split("_rho")[0]))
    ax.axhline(0, c="k", lw=0.8); ax.set_xticks(np.arange(n_classes) + w * (len(others) - 1) / 2); ax.set_xticklabels(CLASS_NAMES, rotation=20)
    ax.set_ylabel(f"Δ Sketch accuracy vs {baseline}"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "per_class_delta.png")); plt.close(fig)

    cols = ["method", "src_mean_accuracy", "src_mean_macro_f1", "src_worst_macro_f1", "src_worst_domain_macro_f1", "sketch_acc", "sketch_f1",
            "sketch_acc_delta_vs_erm", "source_domain_separability", "sharp_delta_sharp"]
    print(pd.DataFrame(rows)[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["erm", "dan_dg", "sam"])
    ap.add_argument("--results_root", default="task3/results")
    ap.add_argument("--base", default="task3/configs/base.yaml")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--baseline", default="erm")
    a = ap.parse_args()
    main(a.methods, a.results_root, a.base, a.tag, a.baseline, "cuda" if torch.cuda.is_available() else "cpu")
