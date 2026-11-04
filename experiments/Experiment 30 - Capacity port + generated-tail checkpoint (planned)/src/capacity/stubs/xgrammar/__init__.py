"""CPU-measurement stub: xgrammar (structured output) is never used by the KV planner / cache managers."""
from ._d import make, _Dummy
def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return make(name)
