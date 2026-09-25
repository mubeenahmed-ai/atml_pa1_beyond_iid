"""STL-10 loading from the HuggingFace parquet mirror (tanganke/stl10).

Only the labelled train (5,000) and test (8,000) partitions are needed for this
task, so we avoid the 2.6 GB official tarball that also carries 100k unlabelled
images. The class order matches torchvision.datasets.STL10.
"""
import io
import os
from typing import List, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CLASS_NAMES: List[str] = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]


def _read_split(root: str, split: str) -> Tuple[List[bytes], np.ndarray]:
    table = pq.read_table(os.path.join(root, f"{split}.parquet"))
    cols = table.column_names
    img_col = "image" if "image" in cols else cols[0]
    lab_col = "label" if "label" in cols else cols[1]
    imgs = table.column(img_col).to_pylist()
    labels = np.asarray(table.column(lab_col).to_pylist(), dtype=np.int64)
    # parquet image column is a struct {bytes, path}
    img_bytes = [d["bytes"] if isinstance(d, dict) else d for d in imgs]
    return img_bytes, labels


class STL10Parquet:
    """Index-addressable STL-10 split returning PIL RGB images."""

    def __init__(self, root: str, split: str):
        assert split in {"train", "test"}
        self.split = split
        self._bytes, self.labels = _read_split(root, split)
        assert self.labels.min() >= 0 and self.labels.max() <= 9

    def __len__(self) -> int:
        return len(self.labels)

    def image(self, idx: int) -> Image.Image:
        return Image.open(io.BytesIO(self._bytes[idx])).convert("RGB")

    def __getitem__(self, idx: int):
        return self.image(idx), int(self.labels[idx])
