"""Minimal JSON/CSV result logging so every reported number traces to a file."""
import csv
import json
import os
import time
from typing import Any, Dict, Iterable, List


def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def save_csv(rows: Iterable[Dict[str, Any]], path: str, fieldnames: List[str] = None) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _default(o):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class CurveLogger:
    """Appends per-epoch/per-step dictionaries and writes them as CSV."""

    def __init__(self, path: str):
        self.path = path
        self.rows: List[Dict[str, Any]] = []
        self.t0 = time.time()

    def log(self, **kwargs) -> None:
        kwargs.setdefault("wall_time_s", round(time.time() - self.t0, 1))
        self.rows.append(kwargs)
        save_csv(self.rows, self.path, fieldnames=sorted({k for r in self.rows for k in r}))
