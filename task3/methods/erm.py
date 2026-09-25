"""ERM baseline = the Task 2 Source-only checkpoint (loaded, never retrained)."""
from task2.methods.source_only import SourceOnly


class ERM(SourceOnly):
    needs_target = False
