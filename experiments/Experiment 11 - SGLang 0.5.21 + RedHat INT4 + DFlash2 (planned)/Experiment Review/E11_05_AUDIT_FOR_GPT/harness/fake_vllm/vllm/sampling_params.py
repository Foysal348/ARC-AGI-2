class SamplingParams:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.max_tokens = int(kw.get("max_tokens", 16))


class RequestOutputKind:
    DELTA = 1
