"""Load base.yaml + method yaml (shallow merge)."""
import os

import yaml


def load_config(method_cfg: str, base_cfg: str = None) -> dict:
    base_cfg = base_cfg or os.path.join(os.path.dirname(method_cfg), "base.yaml")
    cfg = yaml.safe_load(open(base_cfg))
    cfg.update(yaml.safe_load(open(method_cfg)))
    return cfg
