from task4.methods.gcsc import GCSC
from task4.methods.proser import PROSER
from task4.methods.rpl import RPL
from task4.methods.vanilla import Vanilla

REGISTRY = {"vanilla": Vanilla, "gcsc": GCSC, "proser": PROSER, "rpl": RPL}
