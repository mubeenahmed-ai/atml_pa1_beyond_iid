"""Open-set evaluation on cached outputs (CIFAR-100 unknowns are touched only here).

    python task4/evaluate_osr.py --models vanilla gcsc proser rpl
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.logging import save_csv, save_json  # noqa: E402
from common.plotting import PALETTE, plt  # noqa: E402
from task4.data.cifar10 import CLASS_NAMES  # noqa: E402
from task4.evaluation.failure_analysis import acceptance_by_class, accepted_unknowns  # noqa: E402
from task4.evaluation.metrics import auroc, roc  # noqa: E402
from task4.evaluation.thresholds import calibrate_threshold, operating_point  # noqa: E402
from task4.scores import energy_unknownness, fit_mahalanobis, mahalanobis_unknownness, mls_unknownness, msp_unknownness  # noqa: E402

SPLITS = ["val", "test", "near", "far"]


def proser_placeholder_score(logits, dummy_logits):
    full = np.concatenate([logits, dummy_logits.max(1, keepdims=True)], 1)
    p = softmax(full, axis=1)
    return p[:, -1] - p[:, :-1].max(1)  # dummy prob minus best known prob; larger = more unknown


def scores_for(model_name: str, c: dict) -> dict:
    """Return {score_name: {split: u}} for one cached model."""
    out = {}
    if "train_rp_distances" in c:  # RPL: class evidence = max distance to reciprocal points
        out["RPL"] = {s: -c[f"{s}_rp_distances"].max(1) for s in SPLITS}
        return out
    out["MSP"] = {s: msp_unknownness(c[f"{s}_logits"]) for s in SPLITS}
    out["MLS"] = {s: mls_unknownness(c[f"{s}_logits"]) for s in SPLITS}
    out["Energy"] = {s: energy_unknownness(c[f"{s}_logits"]) for s in SPLITS}
    mp = fit_mahalanobis(c["train_feats"], c["train_labels"], 10)
    out["Mahalanobis"] = {s: mahalanobis_unknownness(c[f"{s}_feats"], mp) for s in SPLITS}
    if "train_dummy_logits" in c:
        out["PROSER-placeholder"] = {s: proser_placeholder_score(c[f"{s}_logits"], c[f"{s}_dummy_logits"]) for s in SPLITS}
    return out


def evaluate_score(u: dict) -> dict:
    tau = calibrate_threshold(u["val"])
    r = {"auroc_near": auroc(u["test"], u["near"]), "auroc_far": auroc(u["test"], u["far"]),
         "auroc_all": auroc(u["test"], np.concatenate([u["near"], u["far"]]))}
    r.update(operating_point(tau, u["test"], u["near"], u["far"]))
    return r


def csa(c):
    if "test_rp_distances" in c:
        return float((c["test_rp_distances"].argmax(1) == c["test_labels"]).mean())
    return float((c["test_logits"].argmax(1) == c["test_labels"]).mean())


def main(models, cache_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fine_names = np.load(os.path.join(cache_dir, "cifar100_fine_names.npy"))
    caches = {m: dict(np.load(os.path.join(cache_dir, f"{m}.npz"))) for m in models}
    all_scores = {m: scores_for(m, c) for m, c in caches.items()}

    # ---- Table 1: post-hoc scores on the frozen vanilla model
    t1 = []
    for s in ["MSP", "MLS", "Energy", "Mahalanobis"]:
        t1.append({"model": "vanilla", "score": s, **evaluate_score(all_scores["vanilla"][s])})
    save_csv(t1, os.path.join(out_dir, "table1_vanilla_scores.csv"))

    # ---- Table 2: trained models with MLS (+ PROSER placeholder, + RPL own score)
    t2 = []
    for m in models:
        c = caches[m]
        row_base = {"model": m, "csa_test": csa(c), "val_acc_ckpt": float(c["checkpoint_val_acc"]), "ckpt_epoch": int(c["checkpoint_epoch"])}
        if "RPL" in all_scores[m]:
            t2.append({**row_base, "score": "RPL(-max dist)", **evaluate_score(all_scores[m]["RPL"])})
        else:
            t2.append({**row_base, "score": "MLS", **evaluate_score(all_scores[m]["MLS"])})
            if "PROSER-placeholder" in all_scores[m]:
                t2.append({**row_base, "score": "PROSER-placeholder", **evaluate_score(all_scores[m]["PROSER-placeholder"])})
    save_csv(t2, os.path.join(out_dir, "table2_models.csv"))

    # ---- score agreement on the vanilla model (Spearman rank correlation over test + unknowns)
    pool = {s: np.concatenate([all_scores["vanilla"][s][k] for k in ("test", "near", "far")]) for s in ["MSP", "MLS", "Energy", "Mahalanobis"]}
    names = list(pool)
    corr = [[float(spearmanr(pool[a], pool[b]).correlation) for b in names] for a in names]
    save_json({"scores": names, "spearman": corr}, os.path.join(out_dir, "vanilla_score_rank_correlation.json"))

    # ---- figures: score distributions + ROC for MSP, MLS, Mahalanobis (vanilla)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for j, s in enumerate(["MSP", "MLS", "Mahalanobis"]):
        u = all_scores["vanilla"][s]
        tau = calibrate_threshold(u["val"])
        lo, hi = np.percentile(np.concatenate([u["test"], u["near"], u["far"]]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ax = axes[0, j]
        ax.hist(u["test"], bins, alpha=0.5, density=True, label="known (CIFAR-10 test)", color="#7f7f7f")
        ax.hist(u["near"], bins, alpha=0.5, density=True, label="near unknown", color="#d62728")
        ax.hist(u["far"], bins, alpha=0.5, density=True, label="far unknown", color="#1f77b4")
        ax.axvline(tau, c="k", ls="--", lw=0.9, label="τ (95% val TPR)")
        ax.set_title(f"{s}: unknownness score"); ax.set_yticks([])
        if j == 0:
            ax.legend(frameon=False, fontsize=7)
        ax = axes[1, j]
        for g, col in (("near", "#d62728"), ("far", "#1f77b4")):
            fpr, tpr = roc(u["test"], u[g])
            ax.plot(fpr, tpr, color=col, label=f"{g} AUROC={auroc(u['test'], u[g]):.3f}")
        ax.plot([0, 1], [0, 1], "k:", lw=0.7); ax.set_xlabel("FPR (unknown accepted)"); ax.set_ylabel("TPR (unknown rejected)")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "vanilla_scores_hist_roc.png")); plt.close(fig)

    # ---- failure analysis with the vanilla MLS threshold
    u = all_scores["vanilla"]["MLS"]
    tau = calibrate_threshold(u["val"])
    cv = caches["vanilla"]
    fa, by_class = [], []
    for g in ("near", "far"):
        pred = cv[f"{g}_logits"].argmax(1)
        fa.append(accepted_unknowns(u[g], tau, pred, cv[f"{g}_labels"], fine_names, CLASS_NAMES, g, k=12))
        by_class.append(acceptance_by_class(u[g], tau, pred, cv[f"{g}_labels"], fine_names, CLASS_NAMES).assign(group=g))
    fa = pd.concat(fa); by_class = pd.concat(by_class)
    fa.to_csv(os.path.join(out_dir, "vanilla_mls_accepted_unknowns.csv"), index=False)
    by_class.to_csv(os.path.join(out_dir, "vanilla_mls_acceptance_by_unknown_class.csv"), index=False)
    # image grid of the six most confidently accepted unknowns per group
    try:
        import torchvision
        from task4.data.cifar100_unknowns import unknown_subsets
        unk, _ = unknown_subsets("data")
        fig, axes = plt.subplots(2, 6, figsize=(9, 3.4))
        for r, g in enumerate(("near", "far")):
            sub = fa[fa.group == g].head(6)
            base = unk[g]["subset"].dataset
            for cidx, (_, row) in enumerate(sub.iterrows()):
                img = base.data[unk[g]["indices"][row.index_in_group]]
                axes[r, cidx].imshow(img); axes[r, cidx].axis("off")
                axes[r, cidx].set_title(f"{row.unknown_class}\n→{row.predicted_known}\nu={row.score:.2f}", fontsize=7)
        fig.suptitle(f"Vanilla + MLS: accepted unknowns (τ={tau:.2f}); top: near, bottom: far", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "vanilla_mls_failures.png")); plt.close(fig)
    except Exception as e:  # figure is optional
        print("failure grid skipped:", e)

    # ---- per-model comparison figure (AUROC near/far + CSA)
    t2df = pd.DataFrame(t2)
    fig, ax = plt.subplots(figsize=(6, 3))
    lbl = [f"{r.model}\n{r.score}" for r in t2df.itertuples()]
    x = np.arange(len(t2df))
    ax.bar(x - 0.2, t2df.auroc_near, 0.4, label="AUROC near", color="#d62728")
    ax.bar(x + 0.2, t2df.auroc_far, 0.4, label="AUROC far", color="#1f77b4")
    ax.plot(x, t2df.csa_test, "ko-", label="CSA (test)")
    ax.set_xticks(x); ax.set_xticklabels(lbl, fontsize=7); ax.set_ylim(0.5, 1.0); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "models_auroc_csa.png")); plt.close(fig)

    save_json({"table1": t1, "table2": t2, "vanilla_mls_tau": tau, "cifar100_used_only_here": True}, os.path.join(out_dir, "osr_results.json"))
    print(pd.DataFrame(t1)[["score", "auroc_near", "auroc_far", "auroc_all", "known_test_acceptance", "near_rejection", "far_rejection"]].round(4).to_string(index=False))
    print(t2df[["model", "score", "csa_test", "auroc_near", "auroc_far", "auroc_all", "known_test_acceptance", "near_rejection", "far_rejection"]].round(4).to_string(index=False))
    print(fa.groupby("group").head(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vanilla", "gcsc", "proser", "rpl"])
    ap.add_argument("--cache_dir", default="task4/cache")
    ap.add_argument("--out_dir", default="task4/results/final")
    a = ap.parse_args()
    main(a.models, a.cache_dir, a.out_dir)
