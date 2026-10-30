# E13 RC1 vLLM hook (sitecustomize on the vLLM server's PYTHONPATH only; inert without E13_VLLM_TRACE_DIR).
# SGLang's Marlin MoE repack OOMed on 4xL4 because old buffers were freed only by the garbage collector (LOADPROBE: +0.30 GiB
# per MoE layer). vLLM 0.32 repacks in model_loader.utils.process_weights_after_loading (one loop over all modules): the
# wrapper makes every MoE quant method run gc.collect() after its own process_weights_after_loading, and logs the GPU
# memory per MoE module plus a summary. One JSONL file per process: <E13_VLLM_TRACE_DIR>/vtrace_<pid>.jsonl.
import os
import sys

_D = os.environ.get("E13_VLLM_TRACE_DIR")


def _log(rec):
    import json
    import time
    rec = dict(rec, t=round(time.time(), 3), pid=os.getpid())
    try:
        os.makedirs(_D, exist_ok=True)
        with open(os.path.join(_D, f"vtrace_{os.getpid()}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def _alloc():
    try:
        import torch
        return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _patch(mod):
    orig = mod.process_weights_after_loading
    st = {"n": 0, "freed": 0}

    def process_weights_after_loading(model, *a, **k):
        import gc
        for name, m in model.named_modules():
            qm = getattr(m, "quant_method", None)
            if qm is None or "moe" not in type(qm).__name__.lower() or getattr(qm, "_e13_gc", False):
                continue
            f = qm.process_weights_after_loading

            def g(layer, _f=f, _name=name):
                b0 = _alloc()
                r = _f(layer)
                b1 = _alloc()
                gc.collect()
                b2 = _alloc()
                st["n"] += 1
                st["freed"] += max(0, b1 - b2)
                if st["n"] <= 3 or b1 - b2 >= 16 << 20:
                    _log({"ev": "vgc", "module": _name, "qm": type(qm).__name__, "before": b0, "after_pwal": b1, "after_gc": b2})
                return r
            try:
                qm.process_weights_after_loading = g
                qm._e13_gc = True
            except Exception as e:
                _log({"ev": "vgc_wrap_error", "module": name, "error": repr(e)[:200]})
        _log({"ev": "vpwal_start", "alloc": _alloc()})
        try:
            return orig(model, *a, **k)
        finally:
            _log({"ev": "vpwal_end", "alloc": _alloc(), "moe_modules": st["n"], "freed_bytes": st["freed"]})

    mod.process_weights_after_loading = process_weights_after_loading
    _log({"ev": "vpatched", "module": mod.__name__})


if _D:
    import importlib.abc
    import importlib.machinery

    class _Finder(importlib.abc.MetaPathFinder):
        TARGET = "vllm.model_executor.model_loader.utils"

        def find_spec(self, name, path, target=None):
            if name != self.TARGET:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    _patch(module)
                except Exception as e:
                    _log({"ev": "vpatch_error", "error": repr(e)[:400]})
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
