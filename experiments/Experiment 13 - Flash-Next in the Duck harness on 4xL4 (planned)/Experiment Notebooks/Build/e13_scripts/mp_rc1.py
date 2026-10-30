
# ---- E13 RC1 additions (appended after mp_memtrace.py + mp_nanprobe.py in the same sitecustomize; inert without
# E13_MEMTRACE_DIR). 2026-10-30, after NANPROBE rev 2 put the first NaN in model.layers.2.mlp.experts (GPTQMarlinMoE,
# finite float inputs) on all ranks; two GPT root-cause audits (RC1 fix list C1-C8).
# 1. Expert weight integrity (E13_RC1_SCAN=1): after postprocess (= after the Marlin repack) and again after forward 0, for
#    model.layers.{E13_RC1_LAYERS}.mlp.experts: every tensor attribute (params + buffers + expert maps): dtype, shape,
#    stride, contiguity; floats: non-finite count, finite min / max, and the local experts that hold a non-finite value;
#    ints: an int64 sum + an xor-fold checksum (no clone kept on the GPU). The module's quant method / runner type.
# 2. PLE mode switch, read per forward from the flag file E13_PLE_FLAG (default /tmp/e13_ple_mode), one server:
#    "orig" (no file) = SGLang's side-stream prefetch; "sync" = prefetch, then the side stream is synchronized before the
#    decoder layer runs; "skip" = no prefetch (the PLE layer gathers on the main stream). Diagnostic only.
# 3. E13_L2_NO_MARLIN=1: AutoRoundConfig.get_quant_method returns the non-Marlin GPTQ path (MoeWNA16) for the experts
#    of layer 2 only (prefix ends with "layers.2.mlp.experts"). Diagnostic only.
_RC1_LAYERS = tuple(int(x) for x in os.environ.get("E13_RC1_LAYERS", "0,1,2,3").split(",") if x.strip())
_PLE_FLAG = os.environ.get("E13_PLE_FLAG", "/tmp/e13_ple_mode")
_PLE_SEEN = {"mode": None}


def _ple_mode():
    try:
        with open(_PLE_FLAG, encoding="utf-8") as f:
            m = f.read().strip() or "orig"
    except OSError:
        m = "orig"
    if m != _PLE_SEEN["mode"]:
        _PLE_SEEN["mode"] = m
        _log({"ev": "ple_mode", "mode": m})
    return m


def _rc1_tensor(t):
    import torch
    d = {"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "stride": list(t.stride()),
         "contig": t.is_contiguous(), "device": t.device.type, "numel": t.numel()}
    if not t.numel():
        return d
    x = t.detach()
    rows = x.shape[0] if x.dim() >= 2 else 1
    flat = x.reshape(rows, -1) if x.dim() >= 2 else x.reshape(1, -1)
    step = max(1, (1 << 22) // max(1, flat.shape[1]))       # ~4M elements per chunk: temporaries stay < ~40 MB
    if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        bad, bad_rows, mn, mx, sab, nfin = 0, [], None, None, 0.0, 0
        for r0 in range(0, rows, step):
            c = flat[r0:r0 + step]
            fin = torch.isfinite(c)
            nb = (~fin).sum(1)
            if int(nb.sum().item()):
                bad += int(nb.sum().item())
                bad_rows += (torch.nonzero(nb).flatten() + r0).tolist()
            v = torch.where(fin, c.float(), torch.zeros((), device=c.device))
            nf = int(fin.sum().item())
            if nf:
                big = torch.finfo(torch.float32).max
                lo = float(torch.where(fin, c.float(), torch.full((), big, device=c.device)).min().item())
                hi = float(torch.where(fin, c.float(), torch.full((), -big, device=c.device)).max().item())
                mn = lo if mn is None else min(mn, lo)
                mx = hi if mx is None else max(mx, hi)
                sab += float(v.abs().sum().item()); nfin += nf
            del c, fin, nb, v
        d["nonfinite"] = bad
        if nfin:
            d.update(min=mn, max=mx, mean_abs=sab / nfin)
        if bad:
            d["bad_rows"] = bad_rows[:32]
            d["bad_rows_n"] = len(bad_rows)
    elif not x.is_floating_point() and not x.is_complex() and x.dtype != torch.bool:
        s = ws = 0
        mn = mx = None
        off = 0
        for r0 in range(0, rows, step):
            k = flat[r0:r0 + step].reshape(-1).to(torch.int64)
            w = (torch.arange(k.numel(), device=k.device, dtype=torch.int64) + off) % 1009 + 1
            s += int(k.sum().item())
            ws += int((k * w).sum().item())                 # position-weighted sum: catches permutations
            lo, hi = torch.aminmax(k)
            mn = int(lo.item()) if mn is None else min(mn, int(lo.item()))
            mx = int(hi.item()) if mx is None else max(mx, int(hi.item()))
            off += k.numel()
            del k, w
        d.update(sum=s, wsum=ws, min=mn, max=mx)
    return d


def _rc1_scan(model, when):
    import torch
    mods = dict(model.named_modules())
    out = {}
    for i in _RC1_LAYERS:
        name = next((n for n in (f"model.layers.{i}.mlp.experts", f"model.language_model.layers.{i}.mlp.experts",
                                 f"language_model.model.layers.{i}.mlp.experts") if n in mods), None)
        if name is None:
            out[str(i)] = {"missing_module": True}
            continue
        m = mods[name]
        r = {"module": name, "type": type(m).__name__, "quant_method": type(getattr(m, "quant_method", None)).__name__,
             "scheme": type(getattr(m, "scheme", None)).__name__}
        for k in ("num_experts", "num_local_experts", "num_fused_shared_experts", "moe_ep_rank", "moe_ep_size",
                  "moe_tp_rank", "moe_tp_size", "top_k", "intermediate_size_per_partition", "hidden_size"):
            v = getattr(m, k, None)
            if isinstance(v, (int, float, str, bool)):
                r[k] = v
        if i == _RC1_LAYERS[0]:
            r["src"] = _rc1_src_info(m)
        runner = getattr(m, "runner", None) or getattr(m, "moe_runner", None)
        if runner is not None:
            r["runner"] = type(runner).__name__
        tens = {}
        for n, t in list(m.named_parameters(recurse=False)) + list(m.named_buffers(recurse=False)):
            tens[n] = t
        for n, v in vars(m).items():                  # expert maps and other plain tensor attributes
            if isinstance(v, torch.Tensor) and n not in tens:
                tens[n] = v
        r["tensors"] = {}
        for n, t in sorted(tens.items()):
            try:
                r["tensors"][n] = _rc1_tensor(t)
            except Exception as e:
                r["tensors"][n] = {"error": repr(e)[:200]}
        out[str(i)] = r
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    _log({"ev": "rc1_scan", "when": when, "layers": out})


_FIX_LAYERS = tuple(int(x) for x in os.environ.get("E13_RC1_FIX_LAYERS", "1,2").split(",") if x.strip())
_FIX_FW = int(os.environ.get("E13_RC1_FIX_FORWARDS", "2"))
_FIX_TOKENS = int(os.environ.get("E13_RC1_FIX_TOKENS", "6"))


def _rc1_fixture(name, mod, args, kwargs, out, fw):
    """RC1 (GPT audit 3, test B): for layers E13_RC1_FIX_LAYERS and the first E13_RC1_FIX_FORWARDS forwards, save the
    experts module's hidden input, top-k ids / weights and output (first E13_RC1_FIX_TOKENS tokens, CPU copies) plus the
    rank's expert ownership, so the notebook can recompute the MoE from the checkpoint's GPTQ weights."""
    import re as _re2
    import torch
    if os.environ.get("E13_RC1_FIXTURE") != "1" or fw >= _FIX_FW:
        return
    m = _re2.search(r"layers\.(\d+)\.mlp\.experts$", name)
    if not m or int(m.group(1)) not in _FIX_LAYERS:
        return
    ins = list(_tensors(args)) + list(_tensors(kwargs))
    fl = [t for t in ins if t.is_floating_point()]
    it = [t for t in ins if not t.is_floating_point() and t.dtype != torch.bool]
    hid = next((t for t in fl if t.dim() == 2 and t.shape[-1] == getattr(mod, "hidden_size", t.shape[-1])), fl[0] if fl else None)
    outs = [t for t in _tensors(out) if t.is_floating_point()]
    n = _FIX_TOKENS
    rank = None
    try:
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else None
    except Exception:
        pass
    pick = lambda t: t[:n].detach().to("cpu").clone() if t is not None and t.dim() >= 1 else None
    fix = {"name": name, "fw": fw, "rank": rank, "hidden": pick(hid),
           "ints": [pick(t) for t in it[:4]], "int_shapes": [list(t.shape) for t in it[:4]],
           "floats": [pick(t) for t in fl[:6]], "float_shapes": [list(t.shape) for t in fl[:6]],
           "out": pick(outs[0]) if outs else None,
           "meta": {k: getattr(mod, k) for k in ("num_experts", "num_local_experts", "moe_ep_rank", "moe_ep_size", "top_k",
                                                   "num_fused_shared_experts", "reduce_results", "hidden_size",
                                                   "intermediate_size_per_partition") if isinstance(getattr(mod, k, None), (int, bool))}}
    em = next((v for k, v in list(mod.named_buffers(recurse=False)) + list(vars(mod).items())
               if "expert_map" in k and isinstance(v, torch.Tensor)), None)
    if em is not None:
        fix["expert_map"] = em.detach().to("cpu").clone()
    os.makedirs(_D, exist_ok=True)
    p = os.path.join(_D, f"fixture_L{m.group(1)}_fw{fw}_pid{os.getpid()}.pt")
    torch.save(fix, p)
    _log({"ev": "rc1_fixture", "file": os.path.basename(p), "layer": int(m.group(1)), "fw": fw, "rank": rank,
          "int_shapes": fix["int_shapes"], "float_shapes": fix["float_shapes"], "meta": fix["meta"]})


def _rc1_src_info(m):
    """RC1 (GPT audit 3, priority 1): the runtime source of the quant method / scheme (sha256, patch #3 marker)."""
    import hashlib
    import inspect
    out = {}
    for k, o in (("quant_method", getattr(m, "quant_method", None)), ("scheme", getattr(m, "scheme", None))):
        if o is None:
            continue
        try:
            f = inspect.getsourcefile(type(o))
            b = open(f, "rb").read()
            out[k] = {"file": f, "sha256": hashlib.sha256(b).hexdigest(), "params_dtype_marker": b.count(b"dtype=params_dtype"),
                      "half_marker": b.count(b"dtype=torch.half")}
        except Exception as e:
            out[k] = {"error": repr(e)[:160]}
    return out


def _rc1_after_forward(model, fw_done):
    if os.environ.get("E13_RC1_SCAN") == "1" and fw_done == 1:
        _rc1_scan(model, "after_forward_0")


_patch_nan = _patch


def _patch(mod):
    _patch_nan(mod)
    L = mod.DefaultModelLoader
    pp = L.postprocess_weights

    def postprocess_weights(model, target_device):
        r = pp(model, target_device)
        if os.environ.get("E13_RC1_SCAN") == "1":
            try:
                _rc1_scan(model, "after_postprocess")
            except Exception as e:
                _log({"ev": "rc1_scan_error", "error": repr(e)[:400]})
        return r

    L.postprocess_weights = staticmethod(postprocess_weights)


def _patch_qwen4(mod):
    P = getattr(mod, "Qwen4ExpPLELayer", None)
    if P is None or not hasattr(P, "start_prefetch"):
        _log({"ev": "rc1_ple_patch_error", "error": "Qwen4ExpPLELayer.start_prefetch not found"})
        return
    orig = P.start_prefetch

    def start_prefetch(self, batch, forward_batch):
        m = _ple_mode()
        if m == "skip":
            return None                               # forward() then gathers on the main stream
        r = orig(self, batch, forward_batch)
        if m == "sync" and getattr(self, "_prefetch_stream", None) is not None:
            self._prefetch_stream.synchronize()
        return r

    P.start_prefetch = start_prefetch
    _log({"ev": "rc1_ple_patched", "module": mod.__name__})


def _patch_autoround(mod):
    C = getattr(mod, "AutoRoundConfig", None)
    if C is None:
        _log({"ev": "rc1_ar_patch_error", "error": "AutoRoundConfig not found"})
        return
    orig = C.get_quant_method

    def get_quant_method(self, layer, prefix):
        if prefix.endswith("layers.2.mlp.experts"):
            qm = self.apply_gptq_quant_layer(layer, prefix, backend="gptq")
            _log({"ev": "rc1_l2_no_marlin", "prefix": prefix, "qm": type(qm).__name__})
            return qm
        return orig(self, layer, prefix)

    C.get_quant_method = get_quant_method
    _log({"ev": "rc1_ar_patched", "module": mod.__name__})


if _D:
    _RC1_TARGETS = {"sglang.srt.models.qwen4_exp": _patch_qwen4}
    if os.environ.get("E13_L2_NO_MARLIN") == "1":
        _RC1_TARGETS["sglang.srt.layers.quantization.auto_round"] = _patch_autoround

    class _RC1Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            fn = _RC1_TARGETS.get(name)
            if fn is None:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    fn(module)
                except Exception as e:
                    _log({"ev": "rc1_patch_error", "module": name, "error": repr(e)[:400]})
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _RC1Finder())
