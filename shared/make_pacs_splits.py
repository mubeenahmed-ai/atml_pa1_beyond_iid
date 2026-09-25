"""Create shared/splits/pacs_sketch_seed6304.json (run once)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.logging import save_json  # noqa: E402
from shared.pacs import PACSTable, default_parquet_path  # noqa: E402
from shared.pacs_protocol import SPLIT_PATH, make_splits  # noqa: E402

if __name__ == "__main__":
    table = PACSTable(default_parquet_path())
    s = make_splits(table)
    save_json(s, SPLIT_PATH)
    print("saved", SPLIT_PATH, s["counts"])
