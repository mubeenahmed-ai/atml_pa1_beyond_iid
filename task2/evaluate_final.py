"""Final Task 2 evaluation. Run ONLY after every checkpoint and setting is fixed.

Loads the best checkpoint of each method, reports source-validation and target
metrics, target accuracy change vs. Source-only, domain separability, per-class
target accuracy and confusions, and plots the training curves.

    python task2/evaluate_final.py --methods source_only dan dann cdan
    python task2/evaluate_final.py --methods dan_lambda0.1 dan dan_lambda10 --tag lambda_study
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
from shared.pacs_protocol import load_splits, predict, source_val_loaders, target_eval_loader  # noqa: E402
from task2.evaluation.class_analysis import class_report  # noqa: E402
from task2.evaluation.domain_separability import separability  # noqa: E402
from task2.models.model import PACSModel  # noqa: E402


def evaluate_methods(methods, results_root, base_cfg, tag, device, baseline="source_only"):
    cfg = load_config(os.path.join(os.path.dirname(base_cfg), f"{baseline}.yaml"), base_cfg)
    table = PACSTable(cfg["data"]["parquet"])
    splits = load_splits(cfg["data"]["splits"])
    val_loaders = source_val_loaders(table, splits)
    tgt_loader = target_eval_loader(table, splits)  # labels are read here, and only here
    n_classes = int(cfg["data"]["n_classes"])

    rows, per_class, preds, feats = [], {}, {}, {}
    for m in methods:
        ck = os.path.join(results_root, m, "best.pt")
        model = PACSModel(n_classes).to(device).load(ck, device)
        row = {"method": m, "best_epoch": torch.load(ck, map_location="cpu", weights_only=False)["epoch"]}
        src_feats = []
        for d in SOURCE_DOMAINS:
            f, z, y = predict(model, val_loaders[d], device)
            met = classification_metrics(y, z.argmax(1), n_classes)
            row[f"{d}_acc"], row[f"{d}_f1"] = met["accuracy"], met["macro_f1"]
            src_feats.append(f)
        row["src_mean_acc"] = float(np.mean([row[f"{d}_acc"] for d in SOURCE_DOMAINS]))
        row["src_mean_f1"] = float(np.mean([row[f"{d}_f1"] for d in SOURCE_DOMAINS]))
        row["src_worst_acc"] = float(np.min([row[f"{d}_acc"] for d in SOURCE_DOMAINS]))
        row["src_worst_f1"] = float(np.min([row[f"{d}_f1"] for d in SOURCE_DOMAINS]))
        ft, zt, yt = predict(model, tgt_loader, device)
        met = classification_metrics(yt, zt.argmax(1), n_classes)
        row["target_acc"], row["target_f1"] = met["accuracy"], met["macro_f1"]
        sep = separability([np.concatenate(src_feats), ft], seed=int(cfg["seed"]))
        row["domain_separability"] = sep["score"]
        row["sep_n_per_group"] = sep["n_per_group"]
        preds[m], feats[m] = zt.argmax(1), (np.concatenate(src_feats), ft)
        rows.append(row)
        print(f"{m:14s} src mean acc {row['src_mean_acc']:.4f} F1 {row['src_mean_f1']:.4f} | target acc {row['target_acc']:.4f} F1 {row['target_f1']:.4f} | sep {sep['score']:.3f}")

    base = baseline if baseline in preds else methods[0]
    for r in rows:
        r["target_acc_delta_vs_source_only"] = r["target_acc"] - [x for x in rows if x["method"] == base][0]["target_acc"]
        per_class[r["method"]] = class_report(yt, preds[r["method"]], CLASS_NAMES, preds[base] if r["method"] != base else None)

    out_dir = os.path.join(results_root, f"final_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    save_csv(rows, os.path.join(out_dir, "table_main.csv"))
    save_json({"rows": rows, "per_class": per_class, "target_labels_used_only_here": True, "baseline": base}, os.path.join(out_dir, "final_results.json"))
    np.savez_compressed(os.path.join(out_dir, "target_predictions.npz"), y=yt, **{m: p for m, p in preds.items()})

    # per-class accuracy table
    pc_rows = []
    for c in CLASS_NAMES:
        r = {"class": c}
        for m in methods:
            r[m] = per_class[m]["per_class_acc"][c]
        pc_rows.append(r)
    save_csv(pc_rows, os.path.join(out_dir, "per_class_target_acc.csv"))

    # ---- figures: training curves
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for m in methods:
        ep = pd.read_csv(os.path.join(results_root, m, "curves_epoch.csv"))
        st = pd.read_csv(os.path.join(results_root, m, "curves_step.csv"))
        col = PALETTE.get(m.split("_lambda")[0], None)
        axes[0].plot(ep.epoch, ep.train_loss_cls, label=m, color=col)
        axes[1].plot(ep.epoch, ep.val_mean_f1, label=m, color=col)
        if "mmd2" in st:
            axes[2].plot(st.step, st.mmd2, label=f"{m} MMD²", color=col, alpha=0.8)
        if "loss_domain" in st:
            axes[2].plot(st.step, st.loss_domain, label=f"{m} domain CE", color=col, alpha=0.8)
    axes[0].set_title("source classification loss"); axes[0].set_xlabel("epoch")
    axes[1].set_title("mean source-val macro-F1"); axes[1].set_xlabel("epoch")
    axes[2].set_title("alignment / domain loss"); axes[2].set_xlabel("step")
    for a in axes:
        a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "training_curves.png")); plt.close(fig)

    # discriminator accuracy curve (DANN / CDAN)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    any_d = False
    for m in methods:
        st = pd.read_csv(os.path.join(results_root, m, "curves_step.csv"))
        if "disc_acc" in st:
            ax.plot(st.step, st.disc_acc, label=m, color=PALETTE.get(m)); any_d = True
    if any_d:
        ax.axhline(0.5, ls="--", c="k", lw=0.8); ax.set_ylabel("domain-discriminator accuracy"); ax.set_xlabel("step"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "discriminator_acc.png"))
    plt.close(fig)

    # per-class delta bar chart
    fig, ax = plt.subplots(figsize=(6, 3))
    w = 0.8 / max(len(methods) - 1, 1)
    k = 0
    for m in methods:
        if m == base:
            continue
        d = [per_class[m]["per_class_delta_vs_baseline"][c] for c in CLASS_NAMES]
        ax.bar(np.arange(n_classes) + k * w, d, w, label=m, color=PALETTE.get(m.split("_lambda")[0]))
        k += 1
    ax.axhline(0, c="k", lw=0.8); ax.set_xticks(np.arange(n_classes) + w * (k - 1) / 2); ax.set_xticklabels(CLASS_NAMES, rotation=20)
    ax.set_ylabel(f"Δ target accuracy vs {base}"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "per_class_delta.png")); plt.close(fig)

    print(pd.DataFrame(rows)[["method", "src_mean_acc", "src_mean_f1", "target_acc", "target_f1", "target_acc_delta_vs_source_only", "domain_separability"]].round(4).to_string(index=False))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["source_only", "dan", "dann", "cdan"])
    ap.add_argument("--results_root", default="task2/results")
    ap.add_argument("--base", default="task2/configs/base.yaml")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--baseline", default="source_only")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_methods(a.methods, a.results_root, a.base, a.tag, device, a.baseline)
