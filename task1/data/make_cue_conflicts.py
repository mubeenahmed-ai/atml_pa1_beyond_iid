"""Generate shape/texture cue-conflict images with AdaIN and apply a *visual*
rejection rule that never consults any classifier.

For each unordered class pair {A,B} we produce both directions:
    content(shape)=A, style(texture)=B   and   content=B, style=A.
Content and style images are drawn from the fixed 500-image evaluation subset so
that each conflict image has a clean counterpart with cached features.

Rejection rule (fixed before evaluation, see configs/task1.yaml):
  1. pixel_std  >= min_pixel_std : output is not a collapsed / flat image
  2. edge_corr  >= min_edge_corr : Pearson correlation between Sobel gradient
     magnitudes of the content image and the stylised image (shape retained)
  3. style_gain >  min_style_gain: the stylised image's per-channel colour
     mean/std statistics are closer to the style image than to the content image
     (texture / colour statistics actually adopted)
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.utils import save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.logging import load_json, save_csv, save_json  # noqa: E402
from task1.data.stl10 import CLASS_NAMES, STL10Parquet  # noqa: E402
from task1.data.transforms import to_common_tensor  # noqa: E402
from task1.models.adain import AdaINStyler  # noqa: E402

_SOBEL_X = torch.tensor([[-1.0, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(2, 3).clone()


def gray(x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W)->(B,1,H,W)
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def sobel_mag(x: torch.Tensor) -> torch.Tensor:
    g = gray(x)
    gx = F.conv2d(g, _SOBEL_X.to(x.device), padding=1)
    gy = F.conv2d(g, _SOBEL_Y.to(x.device), padding=1)
    return torch.sqrt(gx**2 + gy**2 + 1e-8)


def pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.flatten(1) - a.flatten(1).mean(1, keepdim=True)
    b = b.flatten(1) - b.flatten(1).mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-8)


def colour_stats(x: torch.Tensor) -> torch.Tensor:  # (B,6): per-channel mean and std
    return torch.cat([x.mean(dim=(2, 3)), x.std(dim=(2, 3))], dim=1)


def visual_metrics(content, style, stylised):
    pixel_std = gray(stylised).flatten(1).std(1)
    edge_corr = pearson(sobel_mag(content), sobel_mag(stylised))
    cs, ss, os_ = colour_stats(content), colour_stats(style), colour_stats(stylised)
    style_gain = (os_ - cs).norm(dim=1) - (os_ - ss).norm(dim=1)  # >0: closer to style
    return pixel_std, edge_corr, style_gain


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    cc = cfg["cue_conflict"]
    seed = int(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    res_dir = cfg["results_dir"]
    img_dir = os.path.join(res_dir, "images", "cue_conflicts")
    ex_dir = os.path.join(res_dir, "images", "examples")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ex_dir, exist_ok=True)

    splits = load_json(os.path.join(res_dir, f"splits_seed{seed}.json"))
    test = STL10Parquet(cfg["data_root"], "test")
    eval_idx = np.array(splits["eval_idx"])
    eval_labels = np.array(splits["eval_labels"])
    name2id = {c: i for i, c in enumerate(CLASS_NAMES)}

    styler = AdaINStyler(cc["weights"]["vgg"], cc["weights"]["decoder"], device=device)
    rng = np.random.RandomState(seed)
    rej = cc["rejection"]
    rows, n_id = [], 0
    for pair in cc["class_pairs"]:
        for content_cls, style_cls in [(pair[0], pair[1]), (pair[1], pair[0])]:
            c_pos = np.where(eval_labels == name2id[content_cls])[0]
            s_pos = np.where(eval_labels == name2id[style_cls])[0]
            k = min(int(cc["per_direction"]), len(c_pos), len(s_pos))
            c_sel = rng.choice(c_pos, size=k, replace=False)
            s_sel = rng.choice(s_pos, size=k, replace=False)
            content = torch.stack([to_common_tensor(test.image(int(eval_idx[i])), cfg["image_size"]) for i in c_sel]).to(device)
            style = torch.stack([to_common_tensor(test.image(int(eval_idx[i])), cfg["image_size"]) for i in s_sel]).to(device)
            stylised = styler(content, style, alpha=float(cc["style_strength"]))
            pixel_std, edge_corr, style_gain = visual_metrics(content, style, stylised)
            for j in range(k):
                ok = bool(
                    pixel_std[j] >= rej["min_pixel_std"]
                    and edge_corr[j] >= rej["min_edge_corr"]
                    and style_gain[j] > rej["min_style_gain"]
                )
                fname = f"cc_{n_id:04d}_{content_cls}_x_{style_cls}.png"
                save_image(stylised[j], os.path.join(img_dir, fname))
                rows.append(
                    {
                        "id": n_id,
                        "file": fname,
                        "content_class": content_cls,
                        "style_class": style_cls,
                        "content_label": name2id[content_cls],
                        "style_label": name2id[style_cls],
                        "content_eval_pos": int(c_sel[j]),
                        "style_eval_pos": int(s_sel[j]),
                        "content_test_idx": int(eval_idx[c_sel[j]]),
                        "style_test_idx": int(eval_idx[s_sel[j]]),
                        "pixel_std": round(float(pixel_std[j]), 4),
                        "edge_corr": round(float(edge_corr[j]), 4),
                        "style_gain": round(float(style_gain[j]), 4),
                        "accepted": ok,
                    }
                )
                n_id += 1
            # one example triplet grid per direction (content | style | stylised) for the report
            grid = torch.cat([content[:4], style[:4], stylised[:4]], dim=0)
            save_image(grid, os.path.join(ex_dir, f"triplet_{content_cls}_content__{style_cls}_style.png"), nrow=4)

    save_csv(rows, os.path.join(res_dir, "cue_conflict_manifest.csv"))
    acc = [r for r in rows if r["accepted"]]
    summary = {
        "generated": len(rows),
        "accepted": len(acc),
        "rejected": len(rows) - len(acc),
        "rejection_rule": rej,
        "style_strength": cc["style_strength"],
        "per_direction_accepted": {},
        "reject_reasons": {
            "pixel_std": int(sum(r["pixel_std"] < rej["min_pixel_std"] for r in rows)),
            "edge_corr": int(sum(r["edge_corr"] < rej["min_edge_corr"] for r in rows)),
            "style_gain": int(sum(r["style_gain"] <= rej["min_style_gain"] for r in rows)),
        },
    }
    for r in acc:
        key = f"{r['content_class']}(shape)/{r['style_class']}(texture)"
        summary["per_direction_accepted"][key] = summary["per_direction_accepted"].get(key, 0) + 1
    save_json(summary, os.path.join(res_dir, "cue_conflict_summary.json"))
    print(json.dumps(summary, indent=2) if (json := __import__("json")) else summary)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task1/configs/task1.yaml")
    main(ap.parse_args().config)
