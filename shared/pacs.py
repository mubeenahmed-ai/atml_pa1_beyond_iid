"""PACS dataset access (HuggingFace parquet mirror `flwrlabs/pacs`).

The parquet has columns image{bytes,path}, domain (str) and label (int, 7 classes:
dog, elephant, giraffe, guitar, horse, house, person). We keep the encoded bytes
in memory and decode lazily inside DataLoader workers.
"""
import io
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASS_NAMES: List[str] = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]
DOMAINS: List[str] = ["photo", "art_painting", "cartoon", "sketch"]
SOURCE_DOMAINS: List[str] = ["photo", "art_painting", "cartoon"]
TARGET_DOMAIN: str = "sketch"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _canon_domain(s: str) -> str:
    s = s.strip().lower().replace(" ", "_").replace("-", "_")
    return {"art": "art_painting", "artpainting": "art_painting", "painting": "art_painting"}.get(s, s)


class PACSTable:
    """Whole PACS table in memory (encoded bytes) with domain / label arrays."""

    def __init__(self, parquet_path: str):
        table = pq.read_table(parquet_path)
        imgs = table.column("image").to_pylist()
        self.bytes: List[bytes] = [d["bytes"] if isinstance(d, dict) else d for d in imgs]
        self.labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
        dom = [_canon_domain(d) for d in table.column("domain").to_pylist()]
        unknown = set(dom) - set(DOMAINS)
        assert not unknown, f"unexpected PACS domain names: {unknown}"
        self.domains = np.asarray([DOMAINS.index(d) for d in dom], dtype=np.int64)
        assert self.labels.min() >= 0 and self.labels.max() < len(CLASS_NAMES)

    def __len__(self):
        return len(self.labels)

    def indices_of(self, domain: str) -> np.ndarray:
        return np.where(self.domains == DOMAINS.index(domain))[0]

    def image(self, idx: int) -> Image.Image:
        return Image.open(io.BytesIO(self.bytes[idx])).convert("RGB")


def train_transform() -> T.Compose:
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform() -> T.Compose:
    return T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class PACSSubset(Dataset):
    """A fixed list of table indices with a transform.

    `with_labels=False` returns -1 as the label so that no code path in the
    adaptation loop can accidentally read target labels.
    """

    def __init__(self, table: PACSTable, indices: Sequence[int], transform, with_labels: bool = True):
        self.table = table
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.with_labels = with_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = self.transform(self.table.image(idx))
        y = int(self.table.labels[idx]) if self.with_labels else -1
        return x, y, int(self.table.domains[idx])


def default_parquet_path() -> str:
    return os.environ.get("PACS_PARQUET", "data/pacs/pacs.parquet")
