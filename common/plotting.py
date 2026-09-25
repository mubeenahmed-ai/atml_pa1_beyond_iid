"""Matplotlib defaults so every figure in the report looks consistent."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {
    "resnet50": "#1f77b4",
    "vit_b16": "#d62728",
    "clip_head": "#2ca02c",
    "clip_zeroshot": "#9467bd",
    "source_only": "#7f7f7f",
    "erm": "#7f7f7f",
    "dan": "#1f77b4",
    "dann": "#ff7f0e",
    "cdan": "#2ca02c",
    "dan_dg": "#1f77b4",
    "sam": "#d62728",
    "vanilla": "#7f7f7f",
    "gcsc": "#1f77b4",
    "proser": "#2ca02c",
    "rpl": "#d62728",
}


def setup():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


setup()
