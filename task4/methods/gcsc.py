"""GCSC ('good closed-set classifier'): identical to Vanilla; the only difference is
RandAugment(2, 9) in the training transform (configured in gcsc.yaml)."""
from task4.methods.vanilla import Vanilla


class GCSC(Vanilla):
    pass
