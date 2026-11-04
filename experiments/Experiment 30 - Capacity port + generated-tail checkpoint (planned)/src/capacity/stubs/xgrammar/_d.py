class _Dummy:
    def __init__(self, *a, **k):
        pass
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy()
    def __call__(self, *a, **k):
        return _Dummy()
    def __class_getitem__(cls, item):
        return cls
def make(name):
    return type(name, (_Dummy,), {})
