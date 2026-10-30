
# ---- E13 NANPROBE additions (appended after mp_memtrace.py in the same sitecustomize; inert without E13_MEMTRACE_DIR).
# rev 2 (2026-10-30, NANPROBE_FIX_LIST F1-F7, F12, F13, F15): two GPT audits + two agent audits + own analysis.
# 1. gc fix: in the post-load repack, gc.collect() before / after a module's process_weights_after_loading when allocated
#    GPU memory grew more than E13_GC_MIB (default 256) since the last check; E13_GC_MIB=0 = after every module (retry arm).
# 2. NaN / collapse locator, only with E13_NANPROBE=1 (an eager launch). SGLang 0.5.21 calls model_runner.model.forward(...)
#    directly (eager_runner.py:265/371), so the top level is an instance-forward wrapper (functools.wraps keeps the signature
#    model_runner.py:473 reads), not a forward hook. For the first E13_NAN_FORWARDS model forwards, per rank and forward:
#    the forward mode / batch, every decoder-layer output [min, max], next_token_logits stats (finite, min, max, argmax,
#    top-1 minus top-2 gap, std), the first non-finite module outputs with whether that module's inputs were already bad
#    (origin vs propagated through plain function code), the first |x| > E13_NAN_ABSMAX outputs (a separate list), constant
#    (min == max) outputs, and the QSA indexer integer outputs (min, max, -1 count). One torch.aminmax per tensor (no
#    full-size temporaries, one sync); fp16 / bf16 / fp32 only; every hook body is guarded (an error is logged, never raised).
import functools as _ft
import math as _math

_GC_STEP = int(os.environ.get("E13_GC_MIB", "256")) << 20
_GC = {"last": None, "n": 0, "freed": 0}


def _gc_check(where, name):
    a = (_mem().get("alloc") or 0)
    if _GC["last"] is None and _GC_STEP:
        _GC["last"] = a
        return
    if _GC_STEP and a - _GC["last"] <= _GC_STEP:
        return
    import gc
    gc.collect()
    b = (_mem().get("alloc") or 0)
    _GC["n"] += 1
    _GC["freed"] += max(0, a - b)
    if _GC_STEP or a - b >= 16 << 20 or _GC["n"] <= 3:
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


_STRUCT_FIELDS = ("hidden_states", "hidden_states_scale", "topk_output", "topk_weights", "topk_ids", "router_logits")


def _tensors(o, depth=0):
    import torch
    if isinstance(o, torch.Tensor):
        yield o
    elif depth < 4 and isinstance(o, (list, tuple)):
        for x in o:
            yield from _tensors(x, depth + 1)
    elif depth < 4 and isinstance(o, dict):
        for x in o.values():
            yield from _tensors(x, depth + 1)
    elif depth < 4 and o is not None and not isinstance(o, (int, float, str, bool)):
        for f in _STRUCT_FIELDS:                  # RC1 (GPT audit 1, fix A): structured MoE dispatch objects
            x = getattr(o, f, None)
            if x is not None:
                yield from _tensors(x, depth + 1)


def _tdesc(t):
    """RC1: a tensor's dtype / shape / stride / contiguity and [min, max] (+ non-finite flag for floats, -1 count for ints)."""
    import torch
    d = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "stride": list(t.stride()),
         "contig": t.is_contiguous()}
    if t.numel() and t.dtype in (torch.float16, torch.bfloat16, torch.float32):
        mn, mx = torch.aminmax(t.detach())
        mn, mx = torch.stack([mn.float(), mx.float()]).tolist()
        d.update(min=mn, max=mx, nonfinite=not (_math.isfinite(mn) and _math.isfinite(mx)))
    elif t.numel() and not t.is_floating_point() and not t.is_complex() and t.dtype != torch.bool:
        mn, mx = torch.aminmax(t.detach())
        d.update(min=int(mn.item()), max=int(mx.item()), neg1=int((t == -1).sum().item()))
    return d


def _install_nan_hooks(model):
    import re as _re
    import torch
    nfw = int(os.environ.get("E13_NAN_FORWARDS", "120"))
    big = float(os.environ.get("E13_NAN_ABSMAX", "1e4"))
    FL = (torch.float16, torch.bfloat16, torch.float32)
    names = {id(m): n for n, m in model.named_modules()}
    layer_re = _re.compile(r"(^|\.)layers\.\d+$")
    st = {"fw": 0, "handles": [], "inp": {}, "errors": 0, "on": True}
    cur = {}

    def stats(t):
        mn, mx = torch.aminmax(t.detach())
        mn, mx = torch.stack([mn.float(), mx.float()]).tolist()
        bad = not (_math.isfinite(mn) and _math.isfinite(mx))
        return mn, mx, bad

    def reset():
        cur.clear()
        cur.update(nonfinite=[], big=[], const=[], indexer=[], layers={}, last_ok=None, t0=__import__("time").time())

    def guard(fn):
        def g(*a, **k):
            if not st["on"]:
                return None
            try:
                return fn(*a, **k)
            except Exception as e:
                st["errors"] += 1
                if st["errors"] <= 3:
                    _log({"ev": "hook_error", "error": repr(e)[:300]})
            return None
        return g

    @guard
    def pre(mod, args, kwargs):
        if st["fw"] < nfw:
            allt = list(_tensors(args)) + list(_tensors(kwargs))
            st["inp"][id(mod)] = [t for t in allt if t.dtype in FL][:6]
            nm = names.get(id(mod), "?")
            if nm.endswith("mlp.experts") and len(cur.get("moe_in", {})) < 8:   # RC1: every MoE input, ints included
                cur.setdefault("moe_in", {})[nm] = [_tdesc(t) for t in allt[:8]]

    @guard
    def post(mod, args, kwargs, out):
        inp = st["inp"].pop(id(mod), [])
        if st["fw"] >= nfw:
            return
        name = names.get(id(mod), "?")
        is_idx = "indexer" in name.lower()
        bad_here = False
        for i, t in enumerate(_tensors(out)):
            if t.numel() == 0:
                continue
            if t.dtype in FL:
                mn, mx, bad = stats(t)
                if layer_re.search(name):
                    cur["layers"][name] = "nonfinite" if bad else [round(mn, 3), round(mx, 3)]
                if bad:
                    bad_here = True
                    if len(cur["nonfinite"]) < 8:
                        ib = None
                        if not cur["nonfinite"]:                 # the first one: origin or propagated?
                            ib = any(stats(x)[2] for x in inp)
                        cur["nonfinite"].append({"module": name, "type": type(mod).__name__, "tensor_idx": i,
                                                 "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
                                                 "inputs_bad": ib, "prev_clean_module": cur["last_ok"]})
                elif max(abs(mn), abs(mx)) > big and len(cur["big"]) < 6:
                    cur["big"].append({"module": name, "tensor_idx": i, "absmax": round(max(abs(mn), abs(mx)), 1)})
                elif mn == mx and t.numel() > 1 and len(cur["const"]) < 6:
                    cur["const"].append({"module": name, "tensor_idx": i, "value": mn, "shape": list(t.shape)})
            elif is_idx and not t.is_floating_point() and not t.is_complex() and len(cur["indexer"]) < 6:
                mn, mx = torch.aminmax(t.detach())
                cur["indexer"].append({"module": name, "tensor_idx": i, "dtype": str(t.dtype).replace("torch.", ""),
                                       "min": int(mn.item()), "max": int(mx.item()), "neg1": int((t == -1).sum().item()),
                                       "shape": list(t.shape)})
        if not bad_here:
            cur["last_ok"] = name
        fx = globals().get("_rc1_fixture")             # RC1: MoE input / routing / output fixture (Marlin vs reference)
        if fx and name.endswith("mlp.experts"):
            fx(name, mod, args, kwargs, out, st["fw"])

    def batch_info(args, kwargs):
        info = {"input_shapes": [list(t.shape) for t in list(_tensors(args))[:2]]}
        for x in list(args) + list(kwargs.values()):
            fm = getattr(x, "forward_mode", None)
            if fm is not None:
                info.update(forward_mode=str(fm).split(".")[-1], batch_size=getattr(x, "batch_size", None))
                sls = getattr(x, "seq_lens_sum", None)
                info["seq_lens_sum"] = int(sls) if isinstance(sls, (int, float)) else None
                break
        return info

    @guard
    def logits_stats(out):
        lg = getattr(out, "next_token_logits", None)
        if not (isinstance(lg, torch.Tensor) and lg.dtype in FL and lg.numel()):
            return {"present": False}
        mn, mx, bad = stats(lg)
        r = {"present": True, "shape": list(lg.shape), "nonfinite": bad, "min": mn, "max": mx}
        if not bad:
            row = lg.detach().reshape(-1, lg.shape[-1])[0].float()
            top = torch.topk(row, 2)
            r.update(argmax=int(top.indices[0].item()), top_gap=float((top.values[0] - top.values[1]).item()),
                     std=float(row.std().item()))
        return r

    orig = model.forward

    @_ft.wraps(orig)
    def forward(*args, **kwargs):
        if not st["on"] or st["fw"] >= nfw:
            return orig(*args, **kwargs)
        reset()
        bi = batch_info(args, kwargs)
        out = orig(*args, **kwargs)
        rec = {"ev": "nan_forward", "fw": st["fw"], **bi, "wall_s": round(__import__("time").time() - cur["t0"], 3),
               "logits": logits_stats(out), "first_nonfinite": cur["nonfinite"], "first_big": cur["big"],
               "const": cur["const"], "indexer": cur["indexer"], "layers": cur["layers"]}
        if cur.get("moe_in"):
            rec["moe_in"] = cur["moe_in"]
        pm = globals().get("_ple_mode")                # RC1: the PLE mode this forward ran under (flag file)
        if pm:
            rec["ple_mode"] = pm()
        _log(rec)
        st["fw"] += 1
        hk = globals().get("_rc1_after_forward")       # RC1: expert weight re-scan after forward 0
        if hk:
            try:
                hk(model, st["fw"])
            except Exception as e:
                _log({"ev": "hook_error", "error": "rc1_after_forward " + repr(e)[:280]})
        if st["fw"] >= nfw:
            for h in st["handles"]:
                h.remove()
            st["on"] = False
            try:
                del model.forward                     # the class forward again (the instance attribute was ours)
            except Exception:
                model.forward = orig
            _log({"ev": "nan_hooks_removed", "forwards_checked": st["fw"], "hook_errors": st["errors"]})
        return out

    only = os.environ.get("E13_NAN_ONLY")            # RC1: hooks only on modules whose name matches (e.g. layers 0-3)
    only_re = _re.compile(only) if only else None
    for m in model.modules():
        if m is not model and (only_re is None or only_re.search(names.get(id(m), ""))):
            st["handles"].append(m.register_forward_pre_hook(pre, with_kwargs=True))
            st["handles"].append(m.register_forward_hook(post, with_kwargs=True))
    model.forward = forward
    _log({"ev": "nan_hooks_installed", "modules": len(st["handles"]) // 2, "forwards": nfw, "absmax_limit": big,
          "top": type(model).__name__})


_patch_base = _patch


def _patch(mod):
    _patch_base(mod)
    L = mod.DefaultModelLoader
    pp = L.postprocess_weights

    def postprocess_weights(model, target_device):
        r = pp(model, target_device)
        _log({"ev": "gc_summary", "collections": _GC["n"], "freed_bytes": _GC["freed"], "step_bytes": _GC_STEP})
        if os.environ.get("E13_NANPROBE") == "1":
            try:
                _install_nan_hooks(model)
            except Exception as e:
                _log({"ev": "nan_hooks_error", "error": repr(e)[:400]})
        return r

    L.postprocess_weights = staticmethod(postprocess_weights)
