from ._d import make
def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return make(name)
