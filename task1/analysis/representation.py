"""2-D projections (t-SNE / UMAP) of clean + transformed features in one shared space."""
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def fit_projection(features: np.ndarray, method: str = "tsne", seed: int = 6304, perplexity: int = 30,
                   n_iter: int = 1000, init: str = "pca") -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, perplexity=perplexity, init=init, random_state=seed, max_iter=n_iter,
                    learning_rate="auto", metric="cosine")
        return tsne.fit_transform(features)
    elif method == "umap":
        import umap

        return umap.UMAP(n_components=2, random_state=seed, metric="cosine").fit_transform(features)
    raise ValueError(method)


def plot_projection_grid(proj: Dict[str, Dict[str, np.ndarray]], labels: Dict[str, Dict[str, np.ndarray]],
                         n_clean: Dict[str, Dict[str, int]], class_names: List[str], out_path: str,
                         backbones: List[str], interventions: List[str], title_map: Dict[str, str] = None):
    """proj[backbone][intervention] -> (N_clean + N_trans, 2); first n_clean rows are the clean condition."""
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(backbones), len(interventions), figsize=(3.2 * len(interventions), 3.0 * len(backbones)), squeeze=False)
    for i, bb in enumerate(backbones):
        for j, iv in enumerate(interventions):
            ax = axes[i, j]
            P, y, nc = proj[bb][iv], labels[bb][iv], n_clean[bb][iv]
            for c in range(len(class_names)):
                m = y[:nc] == c
                ax.scatter(P[:nc][m, 0], P[:nc][m, 1], s=6, marker="o", color=cmap(c), alpha=0.6, linewidths=0)
                m2 = y[nc:] == c
                ax.scatter(P[nc:][m2, 0], P[nc:][m2, 1], s=14, marker="x", color=cmap(c), alpha=0.8, linewidths=0.7)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title((title_map or {}).get(iv, iv))
            if j == 0:
                ax.set_ylabel(bb)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(c), label=class_names[c]) for c in range(len(class_names))]
    handles += [plt.Line2D([], [], marker="o", ls="", color="k", label="clean"), plt.Line2D([], [], marker="x", ls="", color="k", label="transformed")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)
