"""End-to-end Task 1 pipeline.

    python task1/data/make_subset.py
    python task1/data/make_cue_conflicts.py
    python task1/scripts/run_task1.py

Stages (all cached under task1/results/features/):
  1. extract frozen features of the clean train / val split, train one linear head per backbone
  2. build every intervention on the common 224x224 canvas and extract features for
     all backbones from *identical* transformed pixels
  3. prediction-level evaluation (accuracy, macro-F1, confidence, consistency, shape bias)
  4. representation-level evaluation (cosine stability, t-SNE grid)
"""
import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.logging import load_json, save_csv, save_json  # noqa: E402
from common.plotting import PALETTE, plt  # noqa: E402
from common.seed import seed_everything  # noqa: E402
from task1.analysis.evaluate_bias import consistency, prediction_metrics, shape_texture_counts, softmax  # noqa: E402
from task1.analysis.feature_similarity import cosine_stability, per_example_cosine  # noqa: E402
from task1.analysis.representation import fit_projection, plot_projection_grid  # noqa: E402
from task1.data.stl10 import CLASS_NAMES, STL10Parquet  # noqa: E402
from task1.data.transforms import grayscale, hue_rotation, make_patch_perms, patch_shuffle, to_common_tensor, translate  # noqa: E402
from task1.models.backbones import build_backbones  # noqa: E402
from task1.models.linear_head import train_linear_head  # noqa: E402

BACKBONES = ["resnet50", "vit_b16", "clip_vitb32"]
MODELS = ["resnet50", "vit_b16", "clip_head", "clip_zeroshot"]  # prediction-level "models"
MODEL_BACKBONE = {"resnet50": "resnet50", "vit_b16": "vit_b16", "clip_head": "clip_vitb32", "clip_zeroshot": "clip_vitb32"}


# --------------------------------------------------------------------------- helpers
@torch.no_grad()
def extract(backbones, x: torch.Tensor, device, bs=64) -> Dict[str, np.ndarray]:
    """x: (N,3,224,224) float in [0,1] on CPU -> dict backbone -> (N,D) float32."""
    out = {b: [] for b in backbones}
    for i in range(0, len(x), bs):
        xb = x[i : i + bs].to(device, non_blocking=True)
        for name, bb in backbones.items():
            out[name].append(bb(xb).float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def load_images(ds: STL10Parquet, idx, size) -> torch.Tensor:
    return torch.stack([to_common_tensor(ds.image(int(i)), size) for i in idx])


def extract_streamed(backbones, ds, idx, size, condition, feat_dir, device, force=False, chunk=500):
    """Extract features of dataset images in chunks (memory-friendly), with caching."""
    paths = {b: cache_path(feat_dir, b, condition) for b in backbones}
    if not force and all(os.path.exists(p) for p in paths.values()):
        return {b: np.load(p) for b, p in paths.items()}
    out = {b: [] for b in backbones}
    for i in range(0, len(idx), chunk):
        f = extract(backbones, load_images(ds, idx[i : i + chunk], size), device)
        for b in backbones:
            out[b].append(f[b])
    feats = {b: np.concatenate(v) for b, v in out.items()}
    for b, f in feats.items():
        np.save(paths[b], f)
    return feats


def cache_path(feat_dir, backbone, condition):
    return os.path.join(feat_dir, f"{backbone}__{condition}.npy")


def extract_cached(backbones, x, condition, feat_dir, device, force=False):
    paths = {b: cache_path(feat_dir, b, condition) for b in backbones}
    if not force and all(os.path.exists(p) for p in paths.values()):
        return {b: np.load(p) for b, p in paths.items()}
    feats = extract(backbones, x, device)
    for b, f in feats.items():
        np.save(paths[b], f)
    return feats


# --------------------------------------------------------------------------- main
def main(cfg_path: str, force: bool):
    cfg = yaml.safe_load(open(cfg_path))
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    size = int(cfg["image_size"])
    res_dir = cfg["results_dir"]
    feat_dir = os.path.join(res_dir, "features")
    fig_dir = os.path.join(res_dir, "figures")
    for d in (feat_dir, fig_dir):
        os.makedirs(d, exist_ok=True)

    splits = load_json(os.path.join(res_dir, f"splits_seed{seed}.json"))
    train_ds = STL10Parquet(cfg["data_root"], "train")
    test_ds = STL10Parquet(cfg["data_root"], "test")
    y_tr = train_ds.labels[splits["train_idx"]]
    y_va = train_ds.labels[splits["val_idx"]]
    eval_idx = np.array(splits["eval_idx"])
    y_ev = np.array(splits["eval_labels"])
    n_classes = len(CLASS_NAMES)

    backbones = build_backbones(device)
    clip = backbones["clip_vitb32"]

    # ------------------------------------------------------------------ 1. heads
    print("[1] train / val features + linear heads")
    F_tr = extract_streamed(backbones, train_ds, splits["train_idx"], size, "train", feat_dir, device, force)
    F_va = extract_streamed(backbones, train_ds, splits["val_idx"], size, "val", feat_dir, device, force)
    heads, head_info = {}, {}
    hc = cfg["head"]
    for b in BACKBONES:
        head, info = train_linear_head(F_tr[b], y_tr, F_va[b], y_va, n_classes, epochs=hc["epochs"], lr=hc["lr"],
                                       weight_decay=hc["weight_decay"], batch_size=hc["batch_size"],
                                       patience=hc["patience"], seed=seed, device=device)
        heads[b] = head
        head_info[b] = {k: v for k, v in info.items() if k != "history"}
        save_csv(info["history"], os.path.join(res_dir, f"head_history_{b}.csv"))
        torch.save(head.state_dict(), os.path.join(res_dir, f"head_{b}.pt"))
        print(f"   {b:12s} best val acc {info['best_val_acc']:.4f} @ epoch {info['best_epoch']} (ran {info['epochs_run']})")
    save_json({"config": hc, "heads": head_info}, os.path.join(res_dir, "head_training.json"))

    text_emb = clip.text_embeddings(CLASS_NAMES, cfg["clip"]["prompt"], device)  # (10,512)
    logit_scale = clip.logit_scale

    def logits_for(feats: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {}
        with torch.no_grad():
            for m in MODELS:
                f = torch.as_tensor(feats[MODEL_BACKBONE[m]], device=device)
                if m == "clip_zeroshot":
                    out[m] = (logit_scale * f @ text_emb.T).cpu().numpy()
                else:
                    out[m] = heads[MODEL_BACKBONE[m]](f).cpu().numpy()
        return out

    # ------------------------------------------------------------------ 2. interventions
    print("[2] interventions on the evaluation subset")
    X = load_images(test_ds, eval_idx, size)  # (500,3,224,224) common canvas
    perms = make_patch_perms(len(X), cfg["patch_shuffle"]["grid"], seed)
    np.save(os.path.join(res_dir, "patch_permutations.npy"), perms.numpy())
    grid = cfg["patch_shuffle"]["grid"]
    # each intervention is a function so only one transformed copy of the subset exists at a time
    conditions = {"clean": lambda: X, "grayscale": lambda: grayscale(X),
                  "hue_rotation": lambda: hue_rotation(X, float(cfg["color"]["hue_shift_degrees"]))}
    for d in cfg["translation"]["displacements"]:
        if d == 0:
            continue
        for dr in cfg["translation"]["directions"]:
            conditions[f"translate_{d}_{dr}"] = (lambda d=d, dr=dr: translate(X, d, dr))
    conditions["patch_shuffle"] = lambda: torch.stack([patch_shuffle(X[i], perms[i], grid) for i in range(len(X))])

    feats: Dict[str, Dict[str, np.ndarray]] = {}
    show = [0, 60, 120, 180, 240, 300]
    ex_dir = os.path.join(res_dir, "images", "examples")
    os.makedirs(ex_dir, exist_ok=True)
    from torchvision.utils import save_image

    for cname, make in conditions.items():
        xc = make()
        feats[cname] = extract_cached(backbones, xc, cname, feat_dir, device, force)
        if cname in ("clean", "grayscale", "hue_rotation", "translate_32_right", "patch_shuffle"):
            save_image(xc[show], os.path.join(ex_dir, f"intervention_{cname}.png"), nrow=len(show))
        del xc
        print(f"   {cname:22s} features extracted", flush=True)

    # cue conflicts (accepted only) - generated by make_cue_conflicts.py
    man = pd.read_csv(os.path.join(res_dir, "cue_conflict_manifest.csv"))
    acc_man = man[man["accepted"]].reset_index(drop=True)
    cc_imgs = torch.stack([to_common_tensor(Image.open(os.path.join(res_dir, "images", "cue_conflicts", f)), size) for f in acc_man["file"]])
    feats["cue_conflict"] = extract_cached(backbones, cc_imgs, "cue_conflict", feat_dir, device, force)
    print(f"   cue_conflict           {len(acc_man)} accepted images")

    # ------------------------------------------------------------------ 3. prediction-level evaluation
    print("[3] prediction-level evaluation")
    logits = {c: logits_for(f) for c, f in feats.items()}
    preds = {c: {m: l.argmax(1) for m, l in lg.items()} for c, lg in logits.items()}
    for c in logits:  # keep raw predictions for later inspection
        save_csv([{"eval_pos": i, "label": int(y_ev[i]) if c != "cue_conflict" else -1,
                   **{m: int(preds[c][m][i]) for m in MODELS}} for i in range(len(next(iter(logits[c].values()))))],
                 os.path.join(res_dir, "predictions", f"{c}.csv"))

    rows = []
    clean_metrics = {m: prediction_metrics(logits["clean"][m], y_ev, n_classes) for m in MODELS}
    for m in MODELS:
        rows.append({"model": m, "condition": "clean", **clean_metrics[m], "delta_top1": 0.0, "consistency": 1.0})
    for cname in ["grayscale", "hue_rotation", "patch_shuffle"]:
        for m in MODELS:
            met = prediction_metrics(logits[cname][m], y_ev, n_classes)
            rows.append({"model": m, "condition": cname, **met, "delta_top1": met["top1"] - clean_metrics[m]["top1"],
                         "consistency": consistency(preds["clean"][m], preds[cname][m])})
    save_csv(rows, os.path.join(res_dir, "table_interventions.csv"))

    # translation curve: average over the four directions for each displacement
    trows = []
    for m in MODELS:
        trows.append({"model": m, "displacement": 0, "top1": clean_metrics[m]["top1"], "consistency": 1.0,
                      "top1_per_direction": {}, "consistency_per_direction": {}})
        for d in cfg["translation"]["displacements"]:
            if d == 0:
                continue
            accs, cons = {}, {}
            for dr in cfg["translation"]["directions"]:
                c = f"translate_{d}_{dr}"
                accs[dr] = prediction_metrics(logits[c][m], y_ev, n_classes)["top1"]
                cons[dr] = consistency(preds["clean"][m], preds[c][m])
            trows.append({"model": m, "displacement": d, "top1": float(np.mean(list(accs.values()))),
                          "consistency": float(np.mean(list(cons.values()))),
                          "top1_per_direction": accs, "consistency_per_direction": cons})
    save_json(trows, os.path.join(res_dir, "translation_curve.json"))
    save_csv([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in trows], os.path.join(res_dir, "translation_curve.csv"))

    # shape bias on accepted cue conflicts
    sb_rows, cc_pred_rows = [], []
    shape_lab, tex_lab = acc_man["content_label"].values, acc_man["style_label"].values
    for m in MODELS:
        p = preds["cue_conflict"][m]
        sb_rows.append({"model": m, **shape_texture_counts(p, shape_lab, tex_lab)})
        # per class-pair breakdown
        for (cc_, sc_), g in acc_man.groupby(["content_class", "style_class"]):
            cnt = shape_texture_counts(p[g.index], shape_lab[g.index], tex_lab[g.index])
            sb_rows.append({"model": f"{m}|{cc_}(shape)/{sc_}(texture)", **cnt})
    save_csv(sb_rows, os.path.join(res_dir, "shape_bias.csv"))
    conf_cc = {m: softmax(logits["cue_conflict"][m]) for m in MODELS}
    for i in range(len(acc_man)):
        r = {"id": int(acc_man.loc[i, "id"]), "file": acc_man.loc[i, "file"], "shape_class": acc_man.loc[i, "content_class"],
             "texture_class": acc_man.loc[i, "style_class"]}
        for m in MODELS:
            pr = int(preds["cue_conflict"][m][i])
            r[f"{m}_pred"] = CLASS_NAMES[pr]
            r[f"{m}_decision"] = "shape" if pr == shape_lab[i] else ("texture" if pr == tex_lab[i] else "other")
            r[f"{m}_conf"] = round(float(conf_cc[m][i].max()), 3)
        cc_pred_rows.append(r)
    save_csv(cc_pred_rows, os.path.join(res_dir, "cue_conflict_predictions.csv"))

    # ------------------------------------------------------------------ 4. representation-level evaluation
    print("[4] representation stability + projections")
    stab_rows = []
    cc_clean_pos = acc_man["content_eval_pos"].values
    for b in BACKBONES:
        fc = feats["clean"][b]
        rec = {"backbone": b}
        for cname in ["grayscale", "hue_rotation", "patch_shuffle"]:
            rec[cname] = cosine_stability(fc, feats[cname][b])
        for d in cfg["translation"]["displacements"]:
            if d == 0:
                continue
            rec[f"translate_{d}"] = float(np.mean([cosine_stability(fc, feats[f"translate_{d}_{dr}"][b]) for dr in cfg["translation"]["directions"]]))
        rec["cue_conflict_vs_content"] = cosine_stability(fc[cc_clean_pos], feats["cue_conflict"][b])
        rec["cue_conflict_vs_style"] = cosine_stability(fc[acc_man["style_eval_pos"].values], feats["cue_conflict"][b])
        # reference: similarity between random *different* clean images (chance level for this backbone)
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(fc))
        rec["baseline_random_pairs"] = cosine_stability(fc, fc[perm])
        stab_rows.append(rec)
    save_csv(stab_rows, os.path.join(res_dir, "representation_stability.csv"))

    # per-example agreement between prediction change and representation shift (for RQ3)
    agree_rows = []
    for b in BACKBONES:
        m = "clip_head" if b == "clip_vitb32" else b
        for cname in ["grayscale", "hue_rotation", "patch_shuffle", "translate_32_right"]:
            cos = per_example_cosine(feats["clean"][b], feats[cname][b])
            same = preds["clean"][m] == preds[cname][m]
            agree_rows.append({"backbone": b, "condition": cname, "mean_cos_pred_same": float(cos[same].mean()) if same.any() else np.nan,
                               "mean_cos_pred_changed": float(cos[~same].mean()) if (~same).any() else np.nan,
                               "n_changed": int((~same).sum())})
    save_csv(agree_rows, os.path.join(res_dir, "prediction_vs_representation.csv"))

    rp = cfg["representation"]
    tv = rp["translation_for_viz"]
    viz_conditions = {"grayscale": "grayscale", "cue_conflict": "cue conflict", f"translate_{tv['displacement']}_{tv['direction']}":
                      f"translate {tv['displacement']}px {tv['direction']}", "patch_shuffle": "patch shuffle"}
    proj, labs, ncl = {b: {} for b in BACKBONES}, {b: {} for b in BACKBONES}, {b: {} for b in BACKBONES}
    for b in BACKBONES:
        for cname in viz_conditions:
            if cname == "cue_conflict":
                fc = feats["clean"][b][cc_clean_pos]
                ft = feats["cue_conflict"][b]
                y = np.concatenate([y_ev[cc_clean_pos], shape_lab])  # colour = shape/content class
            else:
                fc, ft = feats["clean"][b], feats[cname][b]
                y = np.concatenate([y_ev, y_ev])
            P = fit_projection(np.concatenate([fc, ft]), rp["method"], seed, rp["perplexity"], rp["n_iter"], rp["init"])
            proj[b][cname], labs[b][cname], ncl[b][cname] = P, y, len(fc)
    plot_projection_grid(proj, labs, ncl, CLASS_NAMES, os.path.join(fig_dir, f"{rp['method']}_grid.png"), BACKBONES,
                         list(viz_conditions), viz_conditions)
    save_json({"method": rp["method"], "perplexity": rp["perplexity"], "n_iter": rp["n_iter"], "init": rp["init"],
               "metric": "cosine", "seed": seed, "conditions": viz_conditions,
               "note": "one projection per (backbone, intervention) fitted on clean+transformed features jointly"},
              os.path.join(res_dir, "representation_viz_settings.json"))

    # ------------------------------------------------------------------ figures
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    order = ["clean", "grayscale", "hue_rotation", "patch_shuffle"]
    w = 0.2
    for k, m in enumerate(MODELS):
        sub = df[df.model == m].set_index("condition").loc[order]
        ax[0].bar(np.arange(len(order)) + k * w, sub["top1"], w, label=m, color=PALETTE.get(m))
        ax[1].bar(np.arange(len(order)) + k * w, sub["consistency"], w, label=m, color=PALETTE.get(m))
    for a, t in zip(ax, ["Top-1 accuracy", "Prediction consistency vs clean"]):
        a.set_xticks(np.arange(len(order)) + 1.5 * w); a.set_xticklabels(order, rotation=15); a.set_title(t); a.set_ylim(0, 1.02)
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "interventions_bar.png")); plt.close(fig)

    tdf = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)} for r in trows])
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for m in MODELS:
        sub = tdf[tdf.model == m]
        ax[0].plot(sub.displacement, sub.top1, marker="o", label=m, color=PALETTE.get(m))
        ax[1].plot(sub.displacement, sub.consistency, marker="o", label=m, color=PALETTE.get(m))
    ax[0].set_ylabel("top-1 accuracy"); ax[1].set_ylabel("consistency")
    for a in ax:
        a.set_xlabel("displacement (px)"); a.set_xticks(cfg["translation"]["displacements"])
    ax[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "translation_curve.png")); plt.close(fig)

    sdf = pd.DataFrame(stab_rows).set_index("backbone")
    cols = ["grayscale", "hue_rotation", "translate_8", "translate_16", "translate_32", "patch_shuffle", "cue_conflict_vs_content", "cue_conflict_vs_style", "baseline_random_pairs"]
    fig, ax = plt.subplots(figsize=(8, 3))
    for k, b in enumerate(BACKBONES):
        ax.bar(np.arange(len(cols)) + k * 0.27, sdf.loc[b, cols], 0.27, label=b, color=PALETTE.get(b if b != "clip_vitb32" else "clip_head"))
    ax.set_xticks(np.arange(len(cols)) + 0.27); ax.set_xticklabels(cols, rotation=20, ha="right"); ax.set_ylabel("mean cosine(clean, transformed)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "representation_stability.png")); plt.close(fig)

    summary = {"clean": clean_metrics, "interventions": rows, "translation": trows, "shape_bias": [r for r in sb_rows if "|" not in r["model"]],
               "representation_stability": stab_rows, "head_training": head_info, "clip_logit_scale": logit_scale,
               "n_cue_conflicts_accepted": int(len(acc_man)), "device": device}
    save_json(summary, os.path.join(res_dir, "task1_summary.json"))
    print(pd.DataFrame(rows).pivot(index="condition", columns="model", values="top1").round(3))
    print(pd.DataFrame([r for r in sb_rows if "|" not in r["model"]]).round(2).to_string(index=False))
    print(pd.DataFrame(stab_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task1/configs/task1.yaml")
    ap.add_argument("--force", action="store_true", help="recompute cached features")
    a = ap.parse_args()
    main(a.config, a.force)
