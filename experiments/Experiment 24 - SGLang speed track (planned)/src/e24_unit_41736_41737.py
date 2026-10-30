"""E24 section 6 `s6_unit`: GPU unit tests of #41736 and #41737 on OVERLAY copies of the installed sglang (rev 2, 2026-10-30).

DESIGN rev 2 §4.6 s6_unit, BUILD_SPEC rev 2 WP3, backports §6.4(a) / §6.5(a), review/section6_hook.md #6 #15.
The venv is never patched. `prepare` (stdlib, never imports sglang) copies site-packages/sglang to
<root>/overlay_{none,41736,41737}/sglang, brings each COPY to its state with the same patcher (e24_sglang_patches:
revert the copy to pristine, apply the one name, state_ok) and records every sha. Each test then runs in a fresh
`<venv python> -I` process (the notebook owns the process control: timeout + killpg; this script adds a SIGALRM
fallback) with the overlay at sys.path[0] (PYTHONPATH is ignored under -I), its own Triton / Inductor / SGLang / CUDA
cache dirs (setdefault before torch or sglang is imported: `import sglang` would default them under ~/.cache/sglang,
which the section-1 isolation check watches), and records the imported module __file__ + sha256. A module imported
from outside the overlay is fail-closed (ran=False).
- 41736: the production path DFlashGroupedConv._convolve (torch.compile'd _grouped_conv, mark_static as in the server):
  NaN or +Inf in request 0's LAST block row must not reach request 1's first row. Unpatched: non-finite (chain);
  patched: finite (clean). W6 block 6, taps 2 (our drafter's conv_kernel_size), both sides, bf16 by default.
- 41737: upstream test_greedy_path_matches_the_reference_on_nan_scores[all|mixed] (lilicorr_sample_path on CUDA ->
  selector_walk_triton, vs _selector_walk_torch on CPU) plus CandidateSelector.sample_path directly. Unpatched: the
  all-NaN row mismatches (index == top_k: q row all zeros, the next slot's candidate). Patched: identical.
Both pack their outputs on FINITE inputs; `compare` (stdlib) reports max |Δ| between the two versions.
CLI:
  prepare --site <venv>/site-packages/sglang --root /tmp/e24 [--patcher PATH] [--states none,41736,41737] --out J
  --test 41736|41737 --overlay <root>/overlay_<state> --out J [--patcher PATH] [--cache-root DIR] [--timeout S]
                     [--dtype bf16|fp16|fp32] [--seed N]                       one version, in this (-I) process
  --test 41736|41737 --out J [--root DIR] [--site DIR] [--patcher PATH] [--timeout S]   (no --overlay) orchestrate:
                     prepare none + <test>, run both as `-I` children, compare -> {ran, chain, clean, status, details};
                     root default $E24_UNIT_ROOT, else $E24_WORK/cache/unit_overlays (not packed into the raw tar)
  compare --unpatched J_none --patched J_<test> --out J [--patcher PATH]
Run JSON: {"test", "ran", "reason", "patch_state", "expect", "chain" (unpatched expectation observed), "clean" (patched
expectation observed), "observed", "imported" {module: file, sha256, inside_overlay}, "finite", "versions", "env"}.
Compare JSON: {"test", "ran", "chain", "clean", "status" PASS | PASS_AST_FALLBACK | FAIL_PATCHED_NOT_CLEAN |
INCONCLUSIVE_NOT_EXERCISED | INCONCLUSIVE_NOT_RUN | INVALID_OVERLAY_STATE | INVALID_TEST_MISMATCH, "paths",
"max_abs_delta_finite", "details"}. PASS_AST_FALLBACK: PASS, but a side ran the overlay text of _grouped_conv because
the module import failed standalone (weaker: not the module object the server builds).
Exit code 0 whenever the JSON was written (read ran / ok / status), 2 for usage errors, 1 if it could not be written.
"""
import argparse
import array
import ast
import base64
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
import types

REV = "e24_unit rev 2b (2026-10-30)"
TESTS = ("41736", "41737")
CACHE_VARS = ("SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
              "FLASHINFER_WORKSPACE_BASE", "CUDA_CACHE_PATH", "TILELANG_CACHE_DIR", "TORCH_EXTENSIONS_DIR")
MODULES = {"41736": ("sglang", "sglang.srt.models.dflash"),
           "41737": ("sglang", "sglang.kernels.ops.speculative.dflash", "sglang.kernels.ops.speculative.lilicorr")}
PATCHED_FILE = {"41736": ("sglang.srt.models.dflash", "srt/models/dflash.py"),
                "41737": ("sglang.kernels.ops.speculative.dflash", "kernels/ops/speculative/dflash.py")}


class _Timeout(Exception):
    pass


def _sha_file(p):
    try:
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def _load_patcher(path=None):
    """The patcher next to this file (or `path`), executed into a fresh module: no import machinery, no .pyc."""
    p = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "e24_sglang_patches.py")
    m = types.ModuleType("e24_sglang_patches_unit")
    m.__file__ = p
    with open(p, encoding="utf-8") as f:
        exec(compile(f.read(), p, "exec"), m.__dict__)
    return m, p


# ------------------------------------------------------------------------------------------------- prepare (stdlib)
def prepare(site, root, patcher=None, states=("none",) + TESTS):
    """Overlay copies of `site` (read only) at <root>/overlay_<state>/sglang, each verified by the patcher."""
    t0 = time.time()
    rep = {"schema": "e24_unit_prepare/1", "rev": REV, "site": os.path.abspath(site), "root": os.path.abspath(root),
           "overlays": {}, "ok": True}
    P, ppath = _load_patcher(patcher)
    rep["patcher"] = {"file": ppath, "sha256": _sha_file(ppath)}
    chk = P.check(site)
    rep["site_names"], rep["site_files_verified"] = chk["names"], chk["files_verified"]
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*" + P.BACKUP_SUFFIX, "*" + P.TMP_SUFFIX)
    for st in states:
        dst_root = os.path.join(root, "overlay_" + st)
        dst = os.path.join(dst_root, "sglang")
        o = {"dir": dst_root, "package": dst, "state": st, "state_ok": False}
        try:
            if st != "none" and st not in TESTS:
                raise ValueError("unknown overlay state %r" % st)
            if os.path.exists(dst_root):
                shutil.rmtree(dst_root)
            os.makedirs(dst_root)
            shutil.copytree(site, dst, ignore=ignore, symlinks=True)
            rv = P.revert(dst)                           # the COPY only: pristine whatever state the venv is in
            o["revert_ok"] = rv["ok"]
            if st != "none":
                ap = P.apply(dst, [st])
                o["apply_ok"], o["apply_error"] = ap["ok"], ap.get("error")
            c = P.check(dst)
            o["names"] = c["names"]
            o["files"] = {rel: c["files"][rel] for rel in sorted({PATCHED_FILE[t][1] for t in TESTS})}
            o["state_ok"] = P.state_ok(c, () if st == "none" else (st,))
        except Exception as e:
            o["error"] = "%s: %s" % (type(e).__name__, e)
        rep["overlays"][st] = o
        rep["ok"] = rep["ok"] and o["state_ok"]
    rep["sec"] = round(time.time() - t0, 2)
    return rep


# ------------------------------------------------------------------------------------------------- run (venv python)
def _pack(t):
    a = t.detach().float().cpu().contiguous().view(-1).tolist()
    return {"shape": list(t.shape), "dtype": "float32", "b64": base64.b64encode(array.array("f", a).tobytes()).decode()}


def _pack_int(t):
    return {"shape": list(t.shape), "dtype": "int64", "values": [int(v) for v in t.detach().cpu().view(-1).tolist()]}


def _imported(res, names, overlay):
    """Record __file__ + sha256 of each module; True only if every one comes from inside the overlay."""
    root = os.path.realpath(overlay) + os.sep
    ok = True
    for n in names:
        m = sys.modules.get(n)
        f = getattr(m, "__file__", None) if m is not None else None
        inside = bool(f) and os.path.realpath(f).startswith(root)
        res["imported"][n] = {"file": f, "sha256": _sha_file(f) if f else None, "inside_overlay": inside}
        ok = ok and inside
    return ok


def _conv_source(path):
    """The overlay's own _grouped_conv (compiled as in the server) + DFlashGroupedConv, as one module text."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    seg = {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body
           if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ("_grouped_conv", "DFlashGroupedConv")}
    return "@torch.compile(dynamic=True)\n" + seg["_grouped_conv"] + "\n\n\n" + seg["DFlashGroupedConv"] + "\n"


def _conv_class(torch, res, overlay):
    """DFlashGroupedConv from the overlay module (production import). If that import fails standalone, the overlay's
    own text of _grouped_conv + DFlashGroupedConv compiled as the server does (torch.compile dynamic, inductor), so the
    verdict still comes from the overlay's code; recorded as observed.path = "ast_fallback"."""
    try:
        import sglang.srt.models.dflash as D      # noqa: E402  (from the overlay at sys.path[0])
    except Exception as e:
        res["import_error"] = "%s: %s" % (type(e).__name__, str(e)[:500])
        path = os.path.join(overlay, "sglang", "srt", "models", "dflash.py")
        ns = {"torch": torch, "F": torch.nn.functional, "nn": torch.nn}
        exec(compile(_conv_source(path), path, "exec"), ns)
        _imported(res, ("sglang",), overlay)
        res["imported"]["sglang.srt.models.dflash"] = {"file": path, "sha256": _sha_file(path), "inside_overlay": True,
                                                       "mode": "ast_fallback"}
        return ns["DFlashGroupedConv"], "ast_fallback"
    if not _imported(res, MODULES["41736"], overlay):
        raise RuntimeError("imported outside the overlay")
    return D.DFlashGroupedConv, "module"


def _test_41736(torch, res, overlay, dtype, seed):
    conv_cls, path = _conv_class(torch, res, overlay)
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    dev = torch.device("cuda")
    H, G, block, bs, taps = 256, 64, 6, 3, 2       # W6 block, our drafter's conv_kernel_size 2
    N = bs * block
    torch.manual_seed(seed)
    conv = conv_cls(H, block, taps, G).to(device=dev, dtype=dt)
    with torch.no_grad():
        conv.base_kernel.copy_((torch.randn_like(conv.base_kernel.float()) * 0.5).to(dt))
    x = torch.randn(N, H, device=dev, dtype=dt)
    delta = torch.randn(N, taps, H // G, device=dev, dtype=dt)
    cases, bug, sane = {}, False, True
    with torch.no_grad():
        for side in (0, 1):
            out = conv._convolve(x.clone(), delta.clone(), side)
            res["finite"]["conv_side%d" % side] = _pack(out)
            cases["finite_side%d" % side] = {"all_finite": bool(torch.isfinite(out).all())}
            sane = sane and cases["finite_side%d" % side]["all_finite"]
            for label, val in (("nan", float("nan")), ("inf", float("inf"))):
                xb = x.clone()
                xb[block - 1] = val                    # request 0's LAST row
                o = conv._convolve(xb, delta.clone(), side)
                c = {"neighbor_first_row_finite": bool(torch.isfinite(o[block]).all()),
                     "neighbor_rest_finite": bool(torch.isfinite(o[block + 1:]).all()),
                     "own_earlier_rows_finite": bool(torch.isfinite(o[: block - 1]).all()),
                     "own_poisoned_row_finite": bool(torch.isfinite(o[block - 1]).all())}
                cases["%s_side%d" % (label, side)] = c
                bug = bug or not c["neighbor_first_row_finite"]
                sane = sane and c["neighbor_rest_finite"] and c["own_earlier_rows_finite"]
    res["observed"] = {"bug_observed": bug, "sane": sane, "cases": cases, "path": path,
                       "config": {"hidden": H, "group_size": G, "block": block, "bs": bs, "taps": taps, "dtype": dtype,
                                  "seed": seed}}


def _greedy_state(torch, bs, slots, device):
    return dict(uniforms=torch.zeros(bs, slots, device=device), temperatures=torch.ones(bs, device=device),
                greedy_mask=torch.ones(bs, dtype=torch.bool, device=device))


def _test_41737(torch, res, overlay, seed):
    import sglang.kernels.ops.speculative.dflash as K       # noqa: E402
    import sglang.kernels.ops.speculative.lilicorr as L     # noqa: E402
    if not _imported(res, MODULES["41737"], overlay):
        raise RuntimeError("imported outside the overlay")

    def ref(log_start, log_pair, cand, state):
        return L._selector_walk_torch(candidate_ids=cand.cpu(), scores=L._lattice_scores(log_start.cpu(), log_pair.cpu()),
                                      **{k: v.cpu() for k, v in state.items()})

    def summary(tokens, q, rt, rq, cand):
        t, qq, c = tokens.cpu(), q.cpu(), cand.cpu()
        out_of_row = sum(1 for r in range(t.shape[0]) for s in range(t.shape[1]) if int(t[r, s]) not in
                         set(int(v) for v in c[r, s].tolist()))
        return {"tokens_equal": bool(torch.equal(t, rt)), "q_equal": bool(torch.equal(qq, rq)),
                "rows_equal": [bool(torch.equal(t[r], rt[r]) and torch.equal(qq[r], rq[r])) for r in range(t.shape[0])],
                "zero_q_slots": int((qq.sum(-1) == 0).sum()), "out_of_row_picks": out_of_row}

    cases, keep = {}, {}
    for nan_rows in ("all", "mixed"):                       # upstream test, seed 5, bs 3, slots 6, topk 8
        torch.manual_seed(5)
        bs, slots, topk = 3, 6, 8
        log_start = torch.randn(bs, topk, device="cuda")
        log_pair = torch.randn(bs, slots - 1, topk, topk, device="cuda")
        if nan_rows == "all":
            log_start[0] = float("nan")
            log_pair[0] = float("nan")
        else:
            log_pair[torch.rand_like(log_pair) < 0.3] = float("nan")
        cand = torch.arange(bs * slots * topk, device="cuda").view(bs, slots, topk)
        state = _greedy_state(torch, bs, slots, cand.device)
        tokens, q = L.lilicorr_sample_path(log_start, log_pair, cand, **state)
        rt, rq = ref(log_start, log_pair, cand, state)
        cases[nan_rows] = summary(tokens, q, rt, rq, cand)
        keep[nan_rows] = (log_start, log_pair, cand, state, rt, rq)
    # the production class path: CandidateSelector.sample_path -> selector_walk_triton (optional: heavier import)
    try:
        import sglang.srt.models.dflash as D                # noqa: E402
        _imported(res, ("sglang.srt.models.dflash",), overlay)
        log_start, log_pair, cand, state, rt, rq = keep["all"]
        sel = D.CandidateSelector(hidden_size=8, vocab_size=int(cand.numel()) + 8, state_rank=4, top_k=8).to("cuda")
        t2, q2 = sel.sample_path(candidate_ids=cand, scores=L._lattice_scores(log_start, log_pair), **state)
        cases["selector_class_all"] = dict(summary(t2, q2, rt, rq, cand), ran=True)
    except Exception as e:
        cases["selector_class_all"] = {"ran": False, "reason": "%s: %s" % (type(e).__name__, str(e)[:300])}
    # finite inputs (upstream test_greedy_path_matches_the_reference + the sampled path): packed for max |Δ|
    fin_ok = True
    for topk in (8, 16):
        torch.manual_seed(4)
        bs, slots = 5, 15
        ls = torch.randn(bs, topk, device="cuda")
        lp = torch.randn(bs, slots - 1, topk, topk, device="cuda")
        toks = torch.randint(0, 151936, (bs, slots, topk), device="cuda")
        st = _greedy_state(torch, bs, slots, toks.device)
        t, q = L.lilicorr_sample_path(ls, lp, toks, **st)
        rt, rq = ref(ls, lp, toks, st)
        fin_ok = fin_ok and bool(torch.equal(t.cpu(), rt))
        res["finite"]["greedy_topk%d_tokens" % topk] = _pack_int(t)
        res["finite"]["greedy_topk%d_q" % topk] = _pack(q)
    torch.manual_seed(8)
    bs, slots, k = 4, 5, 8
    a = dict(log_start=torch.randn(bs, k, device="cuda"), log_pair=torch.randn(bs, slots - 1, k, k, device="cuda"),
             candidate_tokens=torch.arange(bs * slots * k, device="cuda").view(bs, slots, k),
             uniforms=torch.rand(bs, slots, device="cuda"), temperatures=torch.rand(bs, device="cuda") + 0.5,
             greedy_mask=torch.zeros(bs, dtype=torch.bool, device="cuda"))
    t, q = L.lilicorr_sample_path(**a)
    res["finite"]["sampled_k8_tokens"] = _pack_int(t)
    res["finite"]["sampled_k8_q"] = _pack(q)
    bug = not cases["all"]["rows_equal"][0]                       # row 0 is the all-NaN row
    sane = all(cases["all"]["rows_equal"][1:]) and fin_ok          # rows without NaN agree in both versions
    res["observed"] = {"bug_observed": bug, "sane": sane, "cases": cases, "finite_greedy_matches_reference": fin_ok,
                       "same_walk_function": L.selector_walk_triton is K.selector_walk_triton}


def _alarm(sec):
    if sec and hasattr(signal, "SIGALRM"):
        def handler(signum, frame):
            raise _Timeout("timeout after %s s" % sec)
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(sec))


def run_test(test, overlay, patcher=None, cache_root=None, timeout=360, dtype="bf16", seed=1234):
    t0 = time.time()
    overlay = os.path.abspath(overlay)
    pkg = os.path.join(overlay, "sglang")
    res = {"schema": "e24_unit_run/1", "rev": REV, "test": test, "overlay": overlay, "ran": False, "reason": None,
           "patch_state": None, "expect": None, "chain": None, "clean": None, "observed": {}, "imported": {},
           "finite": {}, "versions": {"python": sys.version.split()[0], "executable": sys.executable},
           "isolated": bool(sys.flags.isolated), "pid": os.getpid()}
    # own caches before torch / sglang are imported (setdefault: the notebook's per-arm values win)
    croot = cache_root or os.path.join(os.path.dirname(overlay), "unit_cache",
                                       "%s_%s" % (test, os.path.basename(overlay)))
    for v in CACHE_VARS:
        os.environ.setdefault(v, os.path.join(croot, v.lower()))
    res["env"] = {k: os.environ.get(k) for k in CACHE_VARS + ("CUDA_HOME", "LIBRARY_PATH", "LD_LIBRARY_PATH",
                                                              "TORCH_CUDA_ARCH_LIST", "CUDA_VISIBLE_DEVICES",
                                                              "PYTHONPATH")}
    try:
        P, _ = _load_patcher(patcher)
        res["patch_state"] = P.check(pkg, [test])["names"].get(test)
    except Exception as e:
        res["patch_state"] = "unknown"
        res["patch_check_error"] = "%s: %s" % (type(e).__name__, e)
    res["expect"] = {"applied": "clean", "not_applied": "chain"}.get(res["patch_state"])
    if not os.path.isfile(os.path.join(pkg, "__init__.py")):
        res["reason"] = "overlay has no sglang package: %s" % pkg
        return res
    sys.path.insert(0, overlay)
    _alarm(timeout)
    try:
        try:
            import torch                                    # noqa: E402
        except Exception as e:
            res["reason"] = "torch import failed: %s: %s" % (type(e).__name__, e)
            return res
        res["versions"]["torch"] = getattr(torch, "__version__", None)
        res["versions"]["cuda"] = getattr(torch.version, "cuda", None)
        if not torch.cuda.is_available():
            res["reason"] = "CUDA not available"
            return res
        res["versions"]["device"] = torch.cuda.get_device_name(0)
        res["versions"]["capability"] = list(torch.cuda.get_device_capability(0))
        tri = sys.modules.get("triton")
        if test == "41736":
            _test_41736(torch, res, overlay, dtype, seed)
        else:
            _test_41737(torch, res, overlay, seed)
        res["versions"]["triton"] = getattr(tri or sys.modules.get("triton"), "__version__", None)
        res["ran"] = True
    except _Timeout as e:
        res["reason"] = str(e)
    except Exception as e:
        res["reason"] = "%s: %s" % (type(e).__name__, str(e)[:500])
        res["error"] = traceback.format_exc()[-4000:]
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        res["sec"] = round(time.time() - t0, 2)
    bug = res["observed"].get("bug_observed")
    if res["ran"] and isinstance(bug, bool):
        if res["patch_state"] == "not_applied":                # the bug, with the control rows behaving
            res["chain"] = bug and res["observed"].get("sane") is True
        elif res["patch_state"] == "applied":                  # clean = no leak AND the control rows behave
            res["clean"] = (not bug) and res["observed"].get("sane") is True
    return res


# ------------------------------------------------------------------------------------------------- compare (stdlib)
def _unpack(p):
    if p.get("dtype") == "int64":
        return [float(v) for v in p.get("values") or []]
    a = array.array("f")
    a.frombytes(base64.b64decode(p.get("b64") or ""))
    return list(a)


def _max_abs_delta(a, b):
    if len(a) != len(b):
        return None
    m = 0.0
    for x, y in zip(a, b):
        if x != x or y != y:                                   # NaN on either side: 0 if both, inf if one
            d = 0.0 if (x != x and y != y) else math.inf
        else:
            d = abs(x - y)
        m = max(m, d)
    return m


def compare(unpatched, patched, patcher=None):
    u, p = unpatched, patched
    rep = {"schema": "e24_unit_compare/1", "rev": REV, "test": u.get("test"), "ran": bool(u.get("ran") and p.get("ran")),
           "chain": u.get("chain"), "clean": p.get("clean"), "status": None, "max_abs_delta_finite": {},
           "paths": sorted({(x.get("observed") or {}).get("path") or "module" for x in (u, p)}),
           "max_abs_delta_finite_all": None,
           "details": {"unpatched": {k: u.get(k) for k in ("overlay", "ran", "reason", "patch_state", "observed",
                                                            "imported", "sec")},
                       "patched": {k: p.get(k) for k in ("overlay", "ran", "reason", "patch_state", "observed",
                                                          "imported", "sec")}}}
    test = u.get("test")
    sha_ok = None
    try:
        P, _ = _load_patcher(patcher)
        mod, rel = PATCHED_FILE[test]
        su = ((u.get("imported") or {}).get(mod) or {}).get("sha256")
        sp = ((p.get("imported") or {}).get(mod) or {}).get("sha256")
        sha_ok = su == P.REFERENCE_FILE_SHA256.get(rel) and sp == P.REFERENCE_PATCHED_SHA256.get(rel)
        rep["details"]["imported_sha"] = {"unpatched": su, "patched": sp, "ok": sha_ok}
    except Exception as e:
        rep["details"]["imported_sha"] = {"ok": None, "error": "%s: %s" % (type(e).__name__, e)}
    if test != p.get("test") or test not in TESTS:
        rep["status"] = "INVALID_TEST_MISMATCH"
    elif not rep["ran"]:
        rep["status"] = "INCONCLUSIVE_NOT_RUN"
    elif u.get("patch_state") != "not_applied" or p.get("patch_state") != "applied" or sha_ok is False:
        rep["status"] = "INVALID_OVERLAY_STATE"
    elif p.get("clean") is not True:
        rep["status"] = "FAIL_PATCHED_NOT_CLEAN"
    elif u.get("chain") is not True:
        rep["status"] = "INCONCLUSIVE_NOT_EXERCISED"
    elif rep["paths"] != ["module"]:
        rep["status"] = "PASS_AST_FALLBACK"
    else:
        rep["status"] = "PASS"
    fu, fp = u.get("finite") or {}, p.get("finite") or {}
    for k in sorted(set(fu) & set(fp)):
        try:
            rep["max_abs_delta_finite"][k] = _max_abs_delta(_unpack(fu[k]), _unpack(fp[k]))
        except Exception as e:
            rep["max_abs_delta_finite"][k] = None
            rep["details"].setdefault("delta_errors", {})[k] = repr(e)[:200]
    vals = [v for v in rep["max_abs_delta_finite"].values() if v is not None]
    rep["max_abs_delta_finite_all"] = max(vals) if vals else None
    return rep


# ------------------------------------------------------------------------------------------------- orchestrate (stdlib)
_FIND = ("import importlib.util, json\n"
         "s = importlib.util.find_spec('sglang')\n"
         "print(json.dumps(list(s.submodule_search_locations or []) if s else []))\n")


def _find_site(py=None):
    """The `sglang` package dir of `py` (default: this interpreter) by find_spec in a `-I` child: never imported."""
    cp = subprocess.run([py or sys.executable, "-I", "-c", _FIND], capture_output=True, text=True, timeout=120)
    locs = json.loads((cp.stdout or "[]").strip().splitlines()[-1] if cp.stdout.strip() else "[]")
    if not locs or not os.path.isdir(locs[0]):
        raise RuntimeError("sglang not found by %s (rc=%s %s)" % (py or sys.executable, cp.returncode, cp.stderr[-300:]))
    return os.path.realpath(locs[0])


def _compact_run(r):
    obs = r.get("observed") or {}
    return {k: r.get(k) for k in ("ran", "reason", "patch_state", "expect", "chain", "clean", "imported", "versions",
                                  "isolated", "sec", "rc", "file", "import_error")} | {
        "observed": {k: obs.get(k) for k in ("bug_observed", "sane", "cases", "config", "path",
                                             "finite_greedy_matches_reference", "same_walk_function")},
        "error_tail": (r.get("error") or "")[-600:] or None}


def orchestrate(test, root=None, site=None, patcher=None, timeout=360, dtype="bf16", seed=1234):
    """BUILD_SPEC rev 1 single call: overlays none + <test>, both versions in fresh `-I` children (same process group as
    this one, so the notebook's killpg reaches them; each also has its own timeout), then compare. Small `details`."""
    t0 = time.time()
    # default under <work>/cache/: the notebook packs every other /tmp/e24 subdir into e24_raw.tar.gz, and three
    # ~100 MB package copies do not belong there
    root = os.path.abspath(root or os.environ.get("E24_UNIT_ROOT") or os.path.join(
        os.environ.get("E24_WORK") or "/tmp/e24", "cache", "unit_overlays"))
    rep = {"schema": "e24_unit_test/1", "rev": REV, "test": test, "mode": "orchestrated", "ran": False, "chain": None,
           "clean": None, "status": None, "reason": None, "root": root, "max_abs_delta_finite": None,
           "max_abs_delta_finite_all": None, "details": {}}
    try:
        site = site or _find_site()
        prep = prepare(site, root, patcher, ("none", test))
    except Exception as e:
        rep.update(status="INCONCLUSIVE_NOT_RUN", reason="prepare: %s: %s" % (type(e).__name__, e))
        return rep
    rep["details"]["prepare"] = {"ok": prep["ok"], "site": prep["site"], "site_names": prep.get("site_names"),
                                 "overlays": {k: {x: v.get(x) for x in ("dir", "state_ok", "error", "files")}
                                              for k, v in prep["overlays"].items()}}
    if not prep["ok"]:
        rep.update(status="INVALID_OVERLAY_STATE", reason="an overlay did not reach its patch state")
        return rep
    runs = {}
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for st in ("none", test):
        rj = os.path.join(root, "unit_%s_%s.json" % (test, st))
        cmd = [sys.executable, "-I", os.path.abspath(__file__), "--test", test, "--overlay", prep["overlays"][st]["dir"],
               "--out", rj, "--timeout", str(timeout), "--dtype", dtype, "--seed", str(seed)]
        if patcher:
            cmd += ["--patcher", patcher]
        try:
            os.remove(rj)
        except OSError:
            pass
        with open(rj[:-5] + ".log", "w") as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                rc = p.wait(timeout=timeout + 60)
            except subprocess.TimeoutExpired:
                p.kill()
                rc = p.wait(30)
        try:
            with open(rj, encoding="utf-8") as f:
                runs[st] = json.load(f)
        except (OSError, ValueError):
            runs[st] = {"test": test, "ran": False, "reason": "rc=%s: no result file (%s)" % (rc, rj[:-5] + ".log")}
        runs[st].update(rc=rc, file=rj)
    cmp = compare(runs["none"], runs[test], patcher)
    rep.update(ran=cmp["ran"], chain=cmp["chain"], clean=cmp["clean"], status=cmp["status"],
               max_abs_delta_finite=cmp["max_abs_delta_finite"], max_abs_delta_finite_all=cmp["max_abs_delta_finite_all"])
    if not cmp["ran"]:
        rep["reason"] = "; ".join("%s: %s" % (st, runs[st].get("reason")) for st in runs if not runs[st].get("ran"))
    rep["details"].update(unpatched=_compact_run(runs["none"]), patched=_compact_run(runs[test]),
                          imported_sha=cmp["details"].get("imported_sha"), sec=round(time.time() - t0, 2))
    return rep


# ------------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv and argv[0] in ("prepare", "compare") else "run"
    ap = argparse.ArgumentParser(prog="e24_unit_41736_41737.py", description="E24 s6_unit (#41736 / #41737)")
    if cmd == "prepare":
        argv = argv[1:]
        ap.add_argument("--site", required=True)
        ap.add_argument("--root", required=True)
        ap.add_argument("--states", default="none,41736,41737")
    elif cmd == "compare":
        argv = argv[1:]
        ap.add_argument("--unpatched", required=True)
        ap.add_argument("--patched", required=True)
    else:
        ap.add_argument("--test", required=True, choices=TESTS)
        ap.add_argument("--overlay", help="one overlay dir (<root>/overlay_<state>); omitted = orchestrate both")
        ap.add_argument("--root", help="orchestrate: overlay root (default $E24_UNIT_ROOT, else $E24_WORK/cache/unit_overlays)")
        ap.add_argument("--site", help="orchestrate: the sglang package dir (default: find_spec in this python)")
        ap.add_argument("--cache-root")
        ap.add_argument("--timeout", type=float, default=360.0, help="per version (SIGALRM); orchestrate waits +60 s "
                        "per child, so both fit the notebook's 900 s")
        ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
        ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    ap.add_argument("--patcher")
    a = ap.parse_args(argv)                                  # exits 2 on a usage error
    try:
        if cmd == "prepare":
            rep = prepare(a.site, a.root, a.patcher, tuple(s for s in a.states.split(",") if s))
        elif cmd == "compare":
            with open(a.unpatched, encoding="utf-8") as f:
                u = json.load(f)
            with open(a.patched, encoding="utf-8") as f:
                p = json.load(f)
            rep = compare(u, p, a.patcher)
        elif a.overlay:
            rep = run_test(a.test, a.overlay, a.patcher, a.cache_root, a.timeout, a.dtype, a.seed)
        else:
            rep = orchestrate(a.test, a.root, a.site, a.patcher, a.timeout, a.dtype, a.seed)
    except Exception as e:
        rep = {"schema": "e24_unit_%s/1" % cmd, "rev": REV, "ran": False, "ok": False,
               "reason": "%s: %s" % (type(e).__name__, e), "error": traceback.format_exc()[-4000:]}
    try:
        _write_json(a.out, rep)
    except OSError as e:
        print("e24_unit: cannot write %s: %s" % (a.out, e), file=sys.stderr)
        return 1
    brief = {k: rep.get(k) for k in ("test", "ran", "reason", "patch_state", "chain", "clean", "status", "ok")
             if k in rep}
    print("E24_UNIT " + cmd + " " + json.dumps(brief, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
