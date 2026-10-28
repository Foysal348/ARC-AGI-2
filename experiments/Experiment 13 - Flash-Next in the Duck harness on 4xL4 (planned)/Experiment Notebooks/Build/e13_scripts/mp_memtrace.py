# E13 LOADPROBE memory trace (sitecustomize on the server's PYTHONPATH only; inert without E13_MEMTRACE_DIR).
# Wraps three functions of sglang.srt.model_loader.loader in place when that module is imported (never edits a file):
#   DefaultModelLoader.load_weights_only  -> GPU memory before / after + a map of every CUDA tensor of the model
#   DefaultModelLoader.postprocess_weights -> the same at its start / end, and on an exception (the OOM point)
#   _modules_with_quant_method            -> per-module GPU memory delta of process_weights_after_loading
# One JSONL file per process: <E13_MEMTRACE_DIR>/trace_<pid>.jsonl.
import os
import sys

_D = os.environ.get("E13_MEMTRACE_DIR")


def _log(rec):
    import json
    import time
    rec = dict(rec, t=round(time.time(), 3), pid=os.getpid())
    try:
        os.makedirs(_D, exist_ok=True)
        with open(os.path.join(_D, f"trace_{os.getpid()}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def _mem():
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        free, total = torch.cuda.mem_get_info()
        return {"dev": torch.cuda.current_device(), "alloc": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(), "peak": torch.cuda.max_memory_allocated(),
                "free": free, "total": total}
    except Exception as e:
        return {"mem_error": repr(e)[:200]}


def _scan(model):
    """Every CUDA tensor of the model, deduplicated by storage: totals by name pattern (layer indices removed),
    by dtype, the 25 largest tensors, and the host bytes."""
    try:
        by_kind, by_dtype, sizes, seen = {}, {}, [], set()
        cuda_total = host_total = 0
        for kind, it in (("param", model.named_parameters(remove_duplicate=False)),
                         ("buffer", model.named_buffers(remove_duplicate=False))):
            for name, t in it:
                if t is None:
                    continue
                try:
                    st = t.untyped_storage()
                    key = (t.device.type, st.data_ptr())
                    nb = st.nbytes()
                except Exception:
                    key, nb = (t.device.type, id(t)), t.numel() * t.element_size()
                if key in seen or t.device.type == "meta":
                    continue
                seen.add(key)
                if t.device.type != "cuda":
                    host_total += nb
                    continue
                cuda_total += nb
                pat = ".".join("#" if p.isdigit() else p for p in name.split("."))
                k = f"{kind}:{pat}|{str(t.dtype).replace('torch.', '')}"
                by_kind[k] = by_kind.get(k, 0) + nb
                d = str(t.dtype).replace("torch.", "")
                by_dtype[d] = by_dtype.get(d, 0) + nb
                sizes.append((nb, name, list(t.shape), d))
        sizes.sort(reverse=True)
        return {"cuda_tensor_bytes": cuda_total, "host_tensor_bytes": host_total,
                "by_kind": dict(sorted(by_kind.items(), key=lambda kv: -kv[1])[:60]),
                "by_dtype": by_dtype, "top": [{"bytes": s[0], "name": s[1], "shape": s[2], "dtype": s[3]} for s in sizes[:25]]}
    except Exception as e:
        return {"scan_error": repr(e)[:300]}


class _QMProxy:
    def __init__(self, qm, name, stats):
        self._qm, self._name, self._stats = qm, name, stats

    def __getattr__(self, item):
        return getattr(self._qm, item)

    def process_weights_after_loading(self, module):
        a = _mem()
        try:
            r = self._qm.process_weights_after_loading(module)
        except Exception as e:
            _log({"ev": "pwal_error", "module": self._name, "qm": type(self._qm).__name__, "error": repr(e)[:400],
                  "before": a, "now": _mem()})
            raise
        b = _mem()
        d = (b.get("alloc") or 0) - (a.get("alloc") or 0)
        self._stats["n"] += 1
        self._stats["delta"] += d
        if abs(d) >= 16 << 20 or self._stats["n"] <= 3:
            _log({"ev": "pwal", "module": self._name, "qm": type(self._qm).__name__, "delta": d,
                  "alloc": b.get("alloc"), "peak": b.get("peak"), "free": b.get("free")})
        return r


def _patch(mod):
    L = mod.DefaultModelLoader
    orig_lwo, orig_pp, orig_mwqm = L.load_weights_only, L.postprocess_weights, mod._modules_with_quant_method
    state = {"model": None, "stats": {"n": 0, "delta": 0}}

    def load_weights_only(model, weights, target_device):
        state["model"] = model
        _log({"ev": "load_weights_only_start", **_mem()})
        try:
            return orig_lwo(model, weights, target_device)
        finally:
            try:
                import torch
                torch.cuda.synchronize()
            except Exception:
                pass
            _log({"ev": "load_weights_only_end", **_mem(), "scan": _scan(model)})

    def postprocess_weights(model, target_device):
        state["model"] = model
        _log({"ev": "postprocess_start", **_mem()})
        try:
            r = orig_pp(model, target_device)
        except BaseException as e:
            _log({"ev": "postprocess_error", "error": repr(e)[:400], **_mem(), "stats": state["stats"], "scan": _scan(model)})
            raise
        _log({"ev": "postprocess_end", **_mem(), "stats": state["stats"], "scan": _scan(model)})
        return r

    def modules_with_quant_method(model):
        names = {id(m): n for n, m in model.named_modules()}
        for module, qm in orig_mwqm(model):
            yield module, _QMProxy(qm, names.get(id(module), "?"), state["stats"])

    L.load_weights_only = staticmethod(load_weights_only)
    L.postprocess_weights = staticmethod(postprocess_weights)
    mod._modules_with_quant_method = modules_with_quant_method
    _log({"ev": "patched", "module": mod.__name__})


if _D:
    import importlib.abc
    import importlib.machinery

    class _Finder(importlib.abc.MetaPathFinder):
        TARGET = "sglang.srt.model_loader.loader"

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
                    _log({"ev": "patch_error", "error": repr(e)[:400]})
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
