
# ---- E13 NANPROBE additions (appended after mp_memtrace.py in the same sitecustomize; inert without E13_MEMTRACE_DIR).
# 1. gc fix (LOADPROBE finding 1): in the post-load repack, when allocated GPU memory has grown more than E13_GC_MIB
#    (default 256) since the last check, run gc.collect() before and after the module's process_weights_after_loading;
#    every collection is logged with the bytes it freed.
# 2. NaN locator (LOADPROBE finding 2), only with E13_NANPROBE=1 (an eager launch): after postprocess_weights, a forward
#    hook on every submodule; for the first E13_NAN_FORWARDS model forwards each rank logs, per forward, the first modules
#    (execution order) whose floating output has a non-finite value or |x| > E13_NAN_ABSMAX, and the absmax of every
#    decoder-layer output. Hooks are removed after the last checked forward.
_GC_STEP = int(os.environ.get("E13_GC_MIB", "256")) << 20
_GC = {"last": None, "n": 0, "freed": 0}


def _gc_check(where, name):
    a = (_mem().get("alloc") or 0)
    if _GC["last"] is None:
        _GC["last"] = a
        return
    if a - _GC["last"] <= _GC_STEP:
        return
    import gc
    gc.collect()
    b = (_mem().get("alloc") or 0)
    _GC["n"] += 1
    _GC["freed"] += max(0, a - b)
    _log({"ev": "gc_collect", "where": where, "module": name, "alloc_before": a, "alloc_after": b, "n": _GC["n"]})
    _GC["last"] = b


_QMBase = _QMProxy


class _QMProxyGC(_QMBase):
    def process_weights_after_loading(self, module):
        _gc_check("before", self._name)
        r = _QMBase.process_weights_after_loading(self, module)
        _gc_check("after", self._name)
        return r


_QMProxy = _QMProxyGC


def _tensors(o, depth=0):
    import torch
    if isinstance(o, torch.Tensor):
        yield o
    elif depth < 3 and isinstance(o, (list, tuple)):
        for x in o:
            yield from _tensors(x, depth + 1)
    elif depth < 3 and isinstance(o, dict):
        for x in o.values():
            yield from _tensors(x, depth + 1)


def _install_nan_hooks(model):
    import re as _re
    import torch
    nfw = int(os.environ.get("E13_NAN_FORWARDS", "120"))
    big = float(os.environ.get("E13_NAN_ABSMAX", "1e4"))
    names = {id(m): n for n, m in model.named_modules()}
    layer_re = _re.compile(r"(^|\.)layers\.\d+$")
    st = {"fw": 0, "bad": [], "layers": {}, "shape": None, "handles": []}

    def post(mod, inp, out):
        if st["fw"] >= nfw:
            return
        name = names.get(id(mod), "?")
        for t in _tensors(out):
            if not t.is_floating_point() or t.numel() == 0:
                continue
            nonfin = bool((~torch.isfinite(t)).any().item())
            amax = None if nonfin else float(t.detach().abs().max().item())
            if layer_re.search(name):
                st["layers"][name] = "nonfinite" if nonfin else round(amax, 3)
            if (nonfin or amax > big) and len(st["bad"]) < 12:
                st["bad"].append({"module": name, "type": type(mod).__name__, "nonfinite": nonfin, "absmax": amax,
                                  "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")})
            break

    def top_pre(mod, inp):
        st["bad"], st["layers"] = [], {}
        st["shape"] = [list(t.shape) for t in list(_tensors(inp))[:2]]

    def top_post(mod, inp, out):
        if st["fw"] >= nfw:
            return
        if st["bad"] or st["fw"] < 4:
            _log({"ev": "nan_forward", "fw": st["fw"], "input_shapes": st["shape"], "first_bad": st["bad"],
                  "layer_absmax": st["layers"], **_mem()})
        st["fw"] += 1
        if st["fw"] >= nfw:
            for h in st["handles"]:
                h.remove()
            _log({"ev": "nan_hooks_removed", "forwards_checked": st["fw"]})

    for m in model.modules():
        if m is not model:
            st["handles"].append(m.register_forward_hook(post))
    st["handles"].append(model.register_forward_pre_hook(top_pre))
    st["handles"].append(model.register_forward_hook(top_post))
    _log({"ev": "nan_hooks_installed", "modules": len(st["handles"]) - 2, "forwards": nfw, "absmax_limit": big})


_patch_base = _patch


def _patch(mod):
    _patch_base(mod)
    L = mod.DefaultModelLoader
    pp = L.postprocess_weights

    def postprocess_weights(model, target_device):
        r = pp(model, target_device)
        _log({"ev": "gc_summary", "collections": _GC["n"], "freed_bytes": _GC["freed"]})
        if os.environ.get("E13_NANPROBE") == "1":
            try:
                _install_nan_hooks(model)
            except Exception as e:
                _log({"ev": "nan_hooks_error", "error": repr(e)[:400]})
        return r

    L.postprocess_weights = staticmethod(postprocess_weights)
