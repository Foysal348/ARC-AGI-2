"""E13 (simple path, user 2026-10-20): the E12 Duck SMOKE1 notebook unchanged, only the engine swapped to Qwen3.8-Flash-Next.

Base: E12 SGLang SMOKE1 notebook (sha 0fea4d0c, PASSED on Kaggle 2026-10-19). Unchanged: CELL 0, CELL 2 (install/tokenizer;
the Flash-Next tokenizer.json is byte-identical to our RedHat INT4 one, sha 06b95093, and the EOS ids match), CELL 3 (Duck
package + adapter + sandbox), CELL 4 (tasks), CELL 6 (Duck solver), CELL 7 (report).
CELL 1: only the engine inputs and limits (model / draft path, context 139,264, running 4, SMOKE1 time budget).
CELL 5 (engine): new and Duck-only (= the E12-V cell structure):
  - e11_install (verbatim from E12);
  - + FA2 wheel (+ FA4 restore) + Franzen patch #3 (his SHAs);
  - + the user-mandated top-k test (recorded, never gating) + a host-RAM guard;
  - Franzen's launch (cell 12), verbatim except: D1 0.5.21, D4 TP4+EP4, D5 triton GDN, D6 arch 8.9, D7 no
    --gdn-mtp-cache-mode, and the memory items NEXTN off / chunk 2048 / mem 0.93 / 4 running (PLAN 3b);
  - a smoke ("!!!!" check);
  - Duck streams = min(running, KV // 60k) (pre-run capacity rule).
rev 6 (user 2026-10-26, re-test on the foysal account): the FA2 wheel is found by name + sha256 in any attached dataset
(foysalemonshanto/flash-attn-2-8-3-post1-cu13-torch2-13-cp312 first, the hosen42 copy second); after the smoke a 10-15 min
speed stage (decode tok/s at 1-4 streams on ~6k ARC prompts and the long band on ~30k prompts; KV and real running;
recorded, never gating); SMOKE1 budget 105 min.
LOADPROBE (user 2026-10-27, after the rev 6b OOM in the Marlin MoE repack): --loadprobe builds a load-only notebook,
two launches with a GPU memory trace (BF16 PLE = rev 6b exactly; FP8 PLE = the E3 v4.2.1 merge); no speed stage, no Duck.
Run: python build_e13_duck.py [--eval30 | --loadprobe]
  ->  ../Q27-XHIGH-E13_DUCK_FLASHNEXT_{SMOKE1|EVAL30}.ipynb | ../Q27-XHIGH-E13_FLASHNEXT_LOADPROBE.ipynb
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E12DIR = Path(r"D:/ARC AGI 2/experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/Experiment Notebooks")
EVAL30 = "--eval30" in sys.argv
RC1 = "--rc1" in sys.argv                    # 2026-10-30: root cause of the layer-2 MoE NaN (2 GPT audits) + vLLM 0.32 FN smoke
NANPROBE = "--nanprobe" in sys.argv or RC1   # 2026-10-30: LOADPROBE + gc fix + NaN locator + KV lever + vLLM registry check
LOADPROBE = "--loadprobe" in sys.argv or NANPROBE
assert not (EVAL30 and LOADPROBE)
BASE = E12DIR / ("Q27-XHIGH-E12_DUCK_REPL_SGLANG_EVAL30.ipynb" if EVAL30 else "Q27-XHIGH-E12_DUCK_REPL_SGLANG_SMOKE1.ipynb")
BASE_SHA = ("b65fb9dcc80b3652dc18b37d43355743311e2b55db8308ebc766d9d0b23de456" if EVAL30
            else "0fea4d0cb50fe85ec76a4ac27d6de32997aa580b0f5a2d7b36400c0be645c186")
OUT_NB = HERE.parent / f"Q27-XHIGH-E13_DUCK_FLASHNEXT_{'EVAL30' if EVAL30 else 'SMOKE1'}.ipynb"
if LOADPROBE:
    OUT_NB = HERE.parent / ("Q27-XHIGH-E13_FLASHNEXT_RC1.ipynb" if RC1 else "Q27-XHIGH-E13_FLASHNEXT_NANPROBE.ipynb" if NANPROBE
                            else "Q27-XHIGH-E13_FLASHNEXT_LOADPROBE.ipynb")
# never overwrite a notebook that ran on Kaggle (memory: never-rebuild-run-notebooks); a rebuild goes to *_REBUILD.ipynb
PUSHED = {"74411bc070e9de40df066516b4096f5549ee62251d51432ca5f04a8f4feb3b6d",   # foysalemonshanto v1 (2026-10-26, SMOKE1 rev 6b)
          "107aa18227ffbc58cc725bb4dc13020a47848dcc8e5ef9f718094d2156ebf5e0",   # foysalemonshanto v1 (2026-10-27, LOADPROBE)
          "4c11a70b4c8e57942988a9c86ed38d3977b9a021220fa96b54b45d2e53f7bdd6"}   # hosen42 v1 (2026-10-30, NANPROBE rev 2)
if OUT_NB.exists() and hashlib.sha256(OUT_NB.read_bytes()).hexdigest() in PUSHED:
    OUT_NB = OUT_NB.with_name(OUT_NB.stem + "_REBUILD.ipynb")
TOPK_SCRIPT = (HERE / "e13_scripts" / "mp_topk.py").read_text(encoding="utf-8")
MEMTRACE_SCRIPT = (HERE / "e13_scripts" / "mp_memtrace.py").read_text(encoding="utf-8").replace("\r\n", "\n")
if NANPROBE:          # the gc fix + NaN locator extend the same sitecustomize (appended; rebinds _QMProxy and _patch)
    MEMTRACE_SCRIPT += (HERE / "e13_scripts" / "mp_nanprobe.py").read_text(encoding="utf-8").replace("\r\n", "\n")
VLLM_GC_SCRIPT = (HERE / "e13_scripts" / "mp_vllm_gc.py").read_text(encoding="utf-8").replace("\r\n", "\n")
if RC1:               # weight scan, PLE mode switch, layer-2 non-Marlin switch (appended; rebinds _patch again)
    MEMTRACE_SCRIPT += (HERE / "e13_scripts" / "mp_rc1.py").read_text(encoding="utf-8").replace("\r\n", "\n")
PROBE_SRC = r'''
import math
# ==================================================================== E13 LOADPROBE (user 2026-10-27)
# rev 6b OOMed in the Marlin MoE repack (>= 20.86 GiB/rank) where E3 (0.5.19, FP8 PLE) used 19.06 GB. Two launches, the same
# flags, each with the memory trace (e13_memtrace/<arm>/trace_<pid>.jsonl: GPU memory after load_weights_only with a map of
# every CUDA tensor, per-module repack deltas, the state at an exception). Load-only: no speed stage, no Duck.
E13_MEMTRACE_SRC = _b64.b64decode("MEMTRACE_B64").decode()


def _e13_link(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:
        if not E11_HARNESS:
            raise
        __import__("shutil").copyfile(src, dst)           # local harness on Windows: no symlink privilege


def e13_fp8ple_dir():
    """The E3 v4.2.1 merge: every non-weight file from the overlay (config, index, tokenizer, chat template); each weight file
    the overlay index names, from the overlay when it has it (the 10 FP8 PLE shards), else from the Intel mount (the 15 main
    shards). Symlinks only; the names and sizes are checked against E3's pinned list; the config must declare FP8 PLE."""
    import shutil as _sh
    ov, intel = Path(E13_OVERLAY_DIR), Path(E13_INTEL_DIR)
    if not (ov / "model.safetensors.index.json").is_file():
        raise E11Fail(f"FP8 PLE overlay not attached: {ov} (foysalemonshanto/qwen38-flash-next-fp8ple-overlay)")
    dest = Path(os.environ.get("E13_FP8PLE_DST", "/tmp/e13_fp8ple_model"))
    _sh.rmtree(dest, ignore_errors=True); dest.mkdir(parents=True)
    for f in ov.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            _e13_link(f, dest / f.name)
    names = sorted(set(json.loads((dest / "model.safetensors.index.json").read_text())["weight_map"].values()))
    problems, n_ov = [], 0
    for n in names:
        src = ov / n if (ov / n).is_file() else intel / n
        if not src.is_file():
            problems.append(f"missing {n}"); continue
        n_ov += src.parent == ov
        if not E11_HARNESS and E13_OVERLAY_SHARDS.get(n) != src.stat().st_size:
            problems.append(f"{n}: {src.stat().st_size} bytes, expected {E13_OVERLAY_SHARDS.get(n)}")
        _e13_link(src, dest / n)
    if set(names) != set(E13_OVERLAY_SHARDS):
        problems.append(f"index names {len(names)} weight files, expected {len(E13_OVERLAY_SHARDS)}")
    ple = (json.loads((dest / "config.json").read_text()).get("text_config") or {}).get("ple_embedding_dtype")
    if ple != "float8_e4m3fn":
        problems.append(f"config ple_embedding_dtype={ple!r}")
    if problems:
        raise E11Fail("FP8 PLE merge: " + "; ".join(problems)[:800])
    return dest, {"weight_files": len(names), "from_overlay": n_ov, "from_intel": len(names) - n_ov, "ple_embedding_dtype": ple}


def e13_memtrace_summary(d):
    """Per process: the patch marker, GPU memory after load_weights_only (+ the largest CUDA tensor groups), the repack
    count / delta, the exception point. GiB, 2 decimals."""
    g = lambda b: round(b / 2 ** 30, 2) if isinstance(b, (int, float)) else None
    out = {}
    for f in sorted(Path(d).glob("trace_*.jsonl")):
        recs = []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
        ev = {r.get("ev"): r for r in recs}
        if not any(k in ev for k in ("load_weights_only_end", "postprocess_error", "postprocess_end")):
            out[f.stem] = {"events": sorted(ev)}; continue
        s = {"dev": (ev.get("load_weights_only_end") or {}).get("dev"), "events": sorted(ev)}
        lw = ev.get("load_weights_only_end") or {}
        sc = lw.get("scan") or {}
        s["after_load"] = {"alloc_gib": g(lw.get("alloc")), "reserved_gib": g(lw.get("reserved")), "free_gib": g(lw.get("free")),
                           "cuda_tensor_gib": g(sc.get("cuda_tensor_bytes")), "host_tensor_gib": g(sc.get("host_tensor_bytes")),
                           "by_dtype_gib": {k: g(v) for k, v in (sc.get("by_dtype") or {}).items()},
                           "top_groups_gib": {k: g(v) for k, v in list((sc.get("by_kind") or {}).items())[:12]}}
        end = ev.get("postprocess_error") or ev.get("postprocess_end") or {}
        s["postprocess"] = {"ok": "postprocess_end" in ev, "error": end.get("error"), "alloc_gib": g(end.get("alloc")),
                            "peak_gib": g(end.get("peak")), "free_gib": g(end.get("free")), "stats": end.get("stats"),
                            "cuda_tensor_gib": g((end.get("scan") or {}).get("cuda_tensor_bytes"))}
        pe = ev.get("pwal_error")
        if pe:
            s["failed_module"] = {"module": pe.get("module"), "qm": pe.get("qm"),
                                  "alloc_before_gib": g((pe.get("before") or {}).get("alloc"))}
        big = sorted((r for r in recs if r.get("ev") == "pwal"), key=lambda r: -abs(r.get("delta") or 0))[:5]
        s["largest_repack_deltas"] = [{"module": r.get("module"), "delta_mib": round((r.get("delta") or 0) / 2 ** 20, 1)} for r in big]
        gcs = [r for r in recs if r.get("ev") == "gc_collect"]          # NANPROBE: the gc fix at work
        s["gc"] = {"collections": len(gcs), "freed_gib": g(sum(max(0, (r.get("alloc_before") or 0) - (r.get("alloc_after") or 0))
                                                                  for r in gcs))}
        nf = [r for r in recs if r.get("ev") == "nan_forward"]           # NANPROBE rev 2 (fix list F15): per rank
        if nf or "nan_hooks_installed" in ev:
            lg_bad = lambda r: bool((r.get("logits") or {}).get("nonfinite"))
            fb = next((r for r in nf if r.get("first_nonfinite") or lg_bad(r)), None)
            fc = next((r for r in nf if r.get("const")), None)
            first = (fb.get("first_nonfinite") or [{}])[0] if fb else {}
            s["nan"] = {"hooks": ev.get("nan_hooks_installed"), "removed": ev.get("nan_hooks_removed"),
                        "hook_errors": [r.get("error") for r in recs if r.get("ev") == "hook_error"][:3],
                        "hooks_error": (ev.get("nan_hooks_error") or {}).get("error"),
                        "forwards_logged": len(nf), "forwards_with_nonfinite": sum(1 for r in nf if r.get("first_nonfinite") or lg_bad(r)),
                        "coverage": ("INCONCLUSIVE_NO_HOOK_COVERAGE" if "nan_hooks_installed" in ev and not nf else "OK"),
                        "first_nonfinite_forward": ({k: fb.get(k) for k in ("fw", "forward_mode", "batch_size", "seq_lens_sum",
                                                                            "input_shapes", "first_nonfinite", "logits", "layers")}
                                                    if fb else None),
                        "origin": ({"module": first.get("module"), "type": first.get("type"),
                                    "verdict": ("ORIGIN (inputs finite)" if first.get("inputs_bad") is False else
                                                f"PROPAGATED: bad between {first.get('prev_clean_module')} and {first.get('module')}"
                                                if first.get("inputs_bad") else "UNKNOWN")} if first else
                                   ({"module": "logits", "verdict": "logits only (hidden states finite)"} if fb else None)),
                        "first_const_forward": ({k: fc.get(k) for k in ("fw", "forward_mode", "const", "logits")} if fc else None),
                        "first_big": next((r.get("first_big") for r in nf if r.get("first_big")), None),
                        "indexer_first": next((r.get("indexer") for r in nf if r.get("indexer")), None),
                        "logits_first_forwards": [{"fw": r.get("fw"), "mode": r.get("forward_mode"), **(r.get("logits") or {})}
                                                  for r in nf[:12]]}
        if any(str(r.get("ev", "")).startswith("rc1_") for r in recs):    # RC1: weight scans, PLE modes, MoE inputs
            s["rc1"] = e13_rc1_summary(recs)
        out[f.stem] = s
    return out


def _e13_post(path, body, deadline, cap):
    """POST with the timeout clamped to the arm deadline (fix list F10); raises TimeoutError when no time is left."""
    left = min(cap, deadline - time.perf_counter())
    if left < 20:
        raise TimeoutError(f"arm deadline: {round(left)} s left")
    req = _urlreq.Request(SGL_URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with _urlreq.urlopen(req, timeout=left) as r:
        return json.loads(r.read())


def e13_probe_smoke(deadline, cfg):
    """NANPROBE rev 2 smoke: the count result is stored before anything else; the chat is short (cfg chat_max_tokens),
    skipped when the count already failed on the NaN arm (G1), every timeout clamped to the arm deadline. Optional drain
    request first (passes the hook window before the after-hooks count, F14). Timestamps for the trace (F12)."""
    out = {"t_start": round(time.time(), 3)}
    try:
        if cfg.get("drain_tokens"):
            d = _e13_post("/generate", {"text": "Count slowly: 1, 2, 3,", "sampling_params": {"max_new_tokens": int(cfg["drain_tokens"]),
                                                                                          "temperature": 0, "ignore_eos": True}},
                          deadline, 900)
            out["drain_tokens"] = (d.get("meta_info") or {}).get("completion_tokens")
        out["t_count"] = round(time.time(), 3)
        g = _e13_post("/generate", {"text": "1, 2, 3, 4, 5, 6, 7, 8,", "sampling_params": {"max_new_tokens": 32, "temperature": 0}},
                      deadline, 600)
        text = g.get("text") or ""
        out.update(count_text=text[:80], count_ok=text.strip().startswith("9, 10, 11"), garbage=_e13_garbage(text),
                   count_output_ids=(g.get("output_ids") or [])[:16])
        n = int(cfg.get("chat_max_tokens", 256))
        if n <= 0:
            out["chat_skipped"] = "count only"
        elif not out["count_ok"] and os.environ.get("E13_NANPROBE_SKIP_CHAT_ON_FAIL", "1") == "1" and cfg.get("nan"):
            out["chat_skipped"] = "counting already failed (NaN arm)"
        else:
            out["t_chat"] = round(time.time(), 3)
            c = _e13_post("/v1/chat/completions", {"model": "flashnext", "max_tokens": n, "messages": [
                {"role": "user", "content": "What is 2 + 2? Answer with the number only."}]}, deadline, 900)
            m = (c.get("choices") or [{}])[0].get("message") or {}
            ans = (m.get("content") or "") + " " + (m.get("reasoning_content") or "")
            out.update(chat_content=(m.get("content") or "")[:120], chat_has_4="4" in ans, usage=c.get("usage"),
                       garbage=out["garbage"] or _e13_garbage(m.get("content") or ""))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:400]
    out["t_end"] = round(time.time(), 3)
    out["ok"] = bool(out.get("count_ok")) and not out.get("garbage") and not out.get("error") and (
        out.get("chat_has_4", True) if "chat_skipped" not in out else True)
    return out


E13_PLE_FLAG_PATH = Path(os.environ.get("E13_PLE_FLAG", "/tmp/e13_ple_mode"))
_E13_PLE_PROMPT = ("1, 2, 3, 4, 5, 6, 7, 8,", "9, 10, 11")   # RC1 audit: the same token ids in every PLE mode


def _e13_set_ple_mode(mode):
    """RC1: the flag file every server process reads per forward (mp_rc1._ple_mode); 'orig' = no file."""
    try:
        if mode == "orig":
            E13_PLE_FLAG_PATH.unlink(missing_ok=True)
        else:
            E13_PLE_FLAG_PATH.write_text(mode, encoding="utf-8")
    except OSError as e:
        print("E13 RC1: PLE flag write failed:", e, flush=True)


def e13_ple_mode_runs(deadline, modes):
    """RC1 (GPT audits: PLE A/B in one server): per mode, flush the radix cache (fail closed: no prompt without a
    successful flush, so the same prompt runs cold and no NaN-poisoned mamba state is reused), set the flag file, then
    the same count prompt in every mode (token-identical input, so routing differs only by the PLE mode). Records the text, the first output ids, the
    expected continuation; the trace tags every forward with the mode it ran under."""
    runs = []
    for mode in modes:
        prompt, want = _E13_PLE_PROMPT
        r = {"mode": mode, "prompt": prompt, "t_start": round(time.time(), 3)}
        try:
            try:                                            # RC1 audit H3: fail closed, no prompt without a flushed cache
                left = min(60.0, deadline - time.perf_counter())
                if left <= 5:
                    raise TimeoutError("no time left for the cache flush")
                with _urlreq.urlopen(_urlreq.Request(SGL_URL + "/flush_cache", data=b"", method="POST"), timeout=left) as fr:
                    fr.read()
                r["flush"] = "ok"
            except Exception as fe:
                r.update(flush=f"{type(fe).__name__}: {fe}"[:160], error=f"cache isolation failed: {type(fe).__name__}: {fe}"[:300],
                         valid_comparison=False, ok=False, t_end=round(time.time(), 3))
                runs.append(r)
                print(f"E13 RC1 PLE mode {mode}: SKIPPED ({r['error']})", flush=True)
                continue
            r["valid_comparison"] = True
            _e13_set_ple_mode(mode)
            time.sleep(1)
            g = _e13_post("/generate", {"text": prompt, "sampling_params": {"max_new_tokens": 32, "temperature": 0}}, deadline, 600)
            text = g.get("text") or ""
            r.update(count_text=text[:80], count_ok=text.strip().startswith(want), garbage=_e13_garbage(text),
                     count_output_ids=(g.get("output_ids") or [])[:16])
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"[:400]
        r["t_end"] = round(time.time(), 3)
        r["ok"] = bool(r.get("count_ok")) and not r.get("garbage") and not r.get("error")
        runs.append(r)
        print(f"E13 RC1 PLE mode {mode}:", json.dumps({k: r.get(k) for k in ("ok", "count_text", "error", "flush")}), flush=True)
    _e13_set_ple_mode("orig")
    return runs


def e13_rc1_summary(recs):
    """RC1: the weight scans (after the repack / after forward 0) per layer and tensor, what changed between them, the
    NaN result per PLE mode, the MoE inputs of the first forward, and the patch events."""
    scans = {r.get("when"): r.get("layers") or {} for r in recs if r.get("ev") == "rc1_scan"}
    keep = ("dtype", "shape", "stride", "contig", "nonfinite", "bad_rows_n", "bad_rows", "min", "max", "mean_abs", "sum", "wsum")
    comp = {w: {L: {"type": v.get("type"), "quant_method": v.get("quant_method"), "scheme": v.get("scheme"),
                    "runner": v.get("runner"),
                    "meta": {k: v.get(k) for k in ("num_experts", "num_local_experts", "moe_ep_rank", "moe_ep_size", "top_k",
                                                   "intermediate_size_per_partition") if k in v},
                    "tensors": {n: {k: (t.get(k)[:8] if k == "bad_rows" and t.get(k) else t.get(k)) for k in keep if k in t}
                                for n, t in (v.get("tensors") or {}).items()}}
                for L, v in ls.items()} for w, ls in scans.items()}
    changed = []
    a, b = scans.get("after_postprocess") or {}, scans.get("after_forward_0") or {}
    for L in a:
        for n, t in ((a[L] or {}).get("tensors") or {}).items():
            u = (((b.get(L) or {}).get("tensors")) or {}).get(n) or {}
            for k in ("sum", "wsum", "nonfinite", "min", "max"):
                if u and k in t and t.get(k) != u.get(k):
                    changed.append({"layer": L, "tensor": n, "field": k, "before": t.get(k), "after": u.get(k)})
    nf = [r for r in recs if r.get("ev") == "nan_forward"]
    by_mode = {}
    for r in nf:
        m = r.get("ple_mode") or "orig"
        d = by_mode.setdefault(m, {"forwards": 0, "with_nonfinite": 0, "first_module": None, "logits_nonfinite": 0})
        d["forwards"] += 1
        bad = bool(r.get("first_nonfinite")) or bool((r.get("logits") or {}).get("nonfinite"))
        d["with_nonfinite"] += bad
        d["logits_nonfinite"] += bool((r.get("logits") or {}).get("nonfinite"))
        if bad and d["first_module"] is None:
            d["first_module"] = ((r.get("first_nonfinite") or [{}])[0]).get("module") or "logits"
    first_moe = next((r.get("moe_in") for r in nf if r.get("moe_in")), None)
    src = next((v.get("src") for ls in scans.values() for v in ls.values() if isinstance(v, dict) and v.get("src")), None)
    return {"runtime_src": src, "scans": comp, "changed_after_forward0": changed[:40], "by_ple_mode": by_mode, "first_moe_in": first_moe,
            "events": [{k: r.get(k) for k in ("ev", "mode", "module", "prefix", "qm", "error") if r.get(k) is not None}
                       for r in recs if str(r.get("ev", "")).startswith(("rc1_", "ple_mode"))][:30]}


def _e13_gptq_dequant(qweight, qzeros, scales, group=128):
    """GPTQ (auto_gptq v1 packing, AutoRound 'auto_round:auto_gptq'): qweight [K/8, N] int32 (8 nibbles along K),
    qzeros [K/g, N/8] int32 (8 nibbles along N, stored zero - 1), scales [K/g, N] -> W [K, N] float32."""
    import torch
    sh = torch.arange(0, 32, 4, dtype=torch.int32)
    q = ((qweight.unsqueeze(1) >> sh.view(1, 8, 1)) & 0xF).reshape(-1, qweight.shape[1])         # [K, N]
    zs = ((qzeros.unsqueeze(2) >> sh.view(1, 1, 8)) & 0xF).reshape(qzeros.shape[0], -1)          # [K/g, N] stored
    z = zs if bool((zs == 8).all()) else zs + 1     # v2 stores the zero (8 for sym int4); v1 stores zero - 1 (7)
    g = torch.arange(q.shape[0]) // group
    return (q.float() - z[g].float()) * scales.float()[g], int(torch.unique(zs).numel()), int(zs.flatten()[0])


def e13_rc1_reference(tdir, model_dir, deadline):
    """RC1 (GPT audit 3, test B; RC1 audit H1/H2): for every saved MoE fixture (layers 1 and 2, first forwards, per rank)
    recompute the MoE output on the CPU from the checkpoint (GPTQ dequant + PyTorch SiLU-MLP; a fused shared-expert slot
    from its bf16 weights) and compare with the Marlin output of the same tokens.
    EP semantics from SGLang 0.5.21 (token_dispatcher/standard.py): FusedMoE receives GLOBAL top-k ids; the dispatcher maps
    them with local_expert_mapping (global -> local index, -1 = not on this rank). The ownership comes only from the
    captured map (never inferred from the id range); a fixture without it is not compared. The module output is the
    rank-local sum unless reduce_results is set (then all-reduced = the global sum).
    Also scans the checkpoint's scales of layers 1-2. Layer 1 (finite Marlin output) validates the reference itself.
    Runs after all GPU arms; every loop checks `deadline` (a truncated result says so)."""
    import torch
    from safetensors import safe_open
    t0 = time.perf_counter()
    left = lambda: deadline - time.perf_counter()
    out = {"fixtures": [], "ckpt_scales": {}}
    try:
        wm = json.loads((Path(model_dir) / "model.safetensors.index.json").read_text())["weight_map"]
    except Exception as e:
        return {"error": f"index: {type(e).__name__}: {e}"[:300]}
    handles = {}

    def get(name):
        f = wm.get(name)
        if f is None:
            return None
        if f not in handles:
            handles[f] = safe_open(str(Path(model_dir) / f), framework="pt")
        return handles[f].get_tensor(name)
    pre = next((p for p in ("model.language_model.layers.", "model.layers.", "language_model.model.layers.")
                if any(k.startswith(p) for k in wm)), "model.language_model.layers.")
    for L in (1, 2):                                                 # checkpoint scales: every expert, every projection
        bad, n, mx = [], 0, 0.0
        names = [k for k in wm if k.startswith(f"{pre}{L}.mlp.experts.") and k.endswith(".scales")]
        for k in names:
            if left() < 60:
                out["truncated"] = f"deadline during the checkpoint scale scan of layer {L}"; break
            s = get(k).float(); n += 1
            if not torch.isfinite(s).all():
                bad.append(k)
            else:
                mx = max(mx, float(s.abs().max()))
        out["ckpt_scales"][str(L)] = {"tensors": n, "of": len(names), "nonfinite": bad[:10], "nonfinite_n": len(bad), "absmax": mx}
    cache = __import__("collections").OrderedDict()                  # bounded: 24 experts x ~18.75 MiB fp32

    def _put(key, val):
        cache[key] = val
        while len(cache) > 24:
            cache.popitem(last=False)

    class _Late(Exception):
        pass

    def expert(L, e):
        if (L, e) not in cache:
            if left() < 30:
                raise _Late(f"deadline before loading expert {e} of layer {L}")
            W = {}
            for p in ("gate_proj", "up_proj", "down_proj"):
                b = f"{pre}{L}.mlp.experts.{e}.{p}."
                qw, qz, sc = get(b + "qweight"), get(b + "qzeros"), get(b + "scales")
                if qw is None:
                    return None
                W[p], W[p + "_zero_unique"], W[p + "_zero0"] = _e13_gptq_dequant(qw, qz, sc)
            _put((L, e), W)
            zero_seen.update({k: v for k, v in W.items() if "zero" in k})
        cache.move_to_end((L, e))
        return cache[(L, e)]

    def shared(L):                                                   # a fused shared-expert slot: bf16 weights [out, in]
        if (L, "shared") not in cache:
            W = {}
            for p in ("gate_proj", "up_proj", "down_proj"):
                w = get(f"{pre}{L}.mlp.shared_expert.{p}.weight")
                if w is None:
                    return None
                W[p] = w.float().t()
            _put((L, "shared"), W)
        cache.move_to_end((L, "shared"))
        return cache[(L, "shared")]
    rel = lambda a, b: float((a - b).norm() / max(float(b.norm()), 1e-9))
    zero_seen = {}
    for fp in sorted(Path(tdir).glob("fixture_L*_fw*_pid*.pt")):
        if left() < 30:
            out["truncated"] = out.get("truncated") or "deadline before all fixtures"; break
        r = {"file": fp.name}
        try:
            fx = torch.load(fp, map_location="cpu", weights_only=False)
            L = int(re.search(r"layers\.(\d+)\.", fx["name"]).group(1))
            meta = fx.get("meta") or {}
            x = fx["hidden"].float()
            T = x.shape[0]
            ids = next((t for t in fx["ints"] if t is not None and t.dim() == 2 and t.shape[0] == T), None)
            wts = next((t for t in fx["floats"] if t is not None and t.dim() == 2 and ids is not None and t.shape == ids.shape), None)
            em = fx.get("expert_map")
            r.update(layer=L, fw=fx.get("fw"), rank=fx.get("rank"), meta=meta, tokens=T, hidden_from=fx.get("hidden_from"),
                     ids=None if ids is None else ids.tolist(), weights=None if wts is None else [[round(float(v), 4) for v in row] for row in wts])
            if ids is None or wts is None:
                r["error"] = f"no top-k ids/weights among ints {fx.get('int_shapes')} floats {fx.get('float_shapes')}"
                out["fixtures"].append(r); continue
            nfs = int(meta.get("num_fused_shared_experts") or 0)
            ne = int(meta.get("num_experts") or 512)
            ner = ne - nfs                                           # routed experts (checkpoint indices 0 .. ner-1)
            if em is None:
                if int(meta.get("moe_ep_size") or 1) > 1:
                    r["error"] = "missing expert_map; EP mapping unverified (not compared)"
                    out["fixtures"].append(r); continue
                em = torch.arange(ne, dtype=torch.int32)             # no EP: every expert local, identity map
            em = em.to(torch.int64).flatten()
            if int(ids.max()) >= em.numel() or int(ids.min()) < -1:
                r["error"] = f"top-k ids outside the expert map: min {int(ids.min())} max {int(ids.max())} map {em.numel()}"
                out["fixtures"].append(r); continue
            # audit extra A: the map's ownership must be one contiguous block of routed experts numbered 0.. in order
            owned = [g for g in range(min(ner, em.numel())) if int(em[g]) >= 0]
            r["mapping"] = {"owned_routed": len(owned), "owned_first": owned[:1], "owned_last": owned[-1:],
                            "contiguous_in_order": bool(owned) and owned == list(range(owned[0], owned[0] + len(owned)))
                                                   and [int(em[g]) for g in owned] == list(range(len(owned))),
                            "selected_owned": int(sum(1 for v in ids.flatten().tolist() if v >= 0 and int(em[v]) >= 0)),
                            "selected_total": int((ids >= 0).sum())}
            ref_glob = torch.zeros(T, x.shape[1]); ref_loc = torch.zeros(T, x.shape[1])
            exp_bad = []
            for t in range(T):
                for k in range(ids.shape[1]):
                    g = int(ids[t, k])
                    if g < 0:
                        continue
                    W = expert(L, g) if g < ner else shared(L)
                    if W is None:
                        r.setdefault("missing_experts", []).append(g); continue
                    xt = x[t:t + 1]
                    h = torch.nn.functional.silu(xt @ W["gate_proj"]) * (xt @ W["up_proj"])
                    y = (h @ W["down_proj"])[0] * float(wts[t, k])
                    if not torch.isfinite(y).all():
                        exp_bad.append(g)
                    ref_glob[t] += y
                    if int(em[g]) >= 0:
                        ref_loc[t] += y
            mo = fx["out"].float() if fx.get("out") is not None else None
            red = bool(meta.get("reduce_results"))
            ref = ref_glob if red else ref_loc
            r.update(compare_to="global (reduce_results)" if red else "rank-local (reduce_results False)",
                     ref_finite=bool(torch.isfinite(ref).all()), ref_experts_nonfinite=sorted(set(exp_bad))[:10],
                     ref_absmax=float(ref.abs().max()), marlin_finite=None if mo is None else bool(torch.isfinite(mo).all()),
                     zero_convention=dict(zero_seen))
            if mo is not None and mo.shape == ref.shape and bool(torch.isfinite(mo).all()):
                r.update(rel_err=round(rel(mo, ref), 5), rel_err_other_sum=round(rel(mo, ref_loc if red else ref_glob), 5))
            elif mo is not None and mo.shape == ref.shape:
                r["marlin_nonfinite_tokens"] = int((~torch.isfinite(mo)).any(1).sum())
            elif mo is not None:
                r["error"] = f"Marlin output shape {list(mo.shape)} != reference {list(ref.shape)}"
        except _Late as e:
            r["error"] = str(e); out["truncated"] = str(e)
            out["fixtures"].append(r); break
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"[:400]
        out["fixtures"].append(r)
    # verdict per the GPT audit table, only when layer 1 validates the reference (finite Marlin, rel err < 5 %)
    by = {}
    for r in out["fixtures"]:
        by.setdefault(r.get("layer"), []).append(r)
    l1 = [r for r in by.get(1, []) if r.get("rel_err") is not None]
    l2 = [r for r in by.get(2, []) if not r.get("error")]
    out["reference_validated_on_layer1"] = bool(l1) and max(r["rel_err"] for r in l1) < 0.05
    if not out["reference_validated_on_layer1"]:
        out["verdict"] = "INCONCLUSIVE: reference not validated on layer 1"
    elif not l2:
        out["verdict"] = "INCONCLUSIVE: no usable layer-2 fixture"
    elif any(r.get("marlin_finite") is False for r in l2) and all(r.get("ref_finite") for r in l2):
        out["verdict"] = "MARLIN PATH FAULT: Marlin NaN on layer 2, the checkpoint reference is finite"
    elif any(r.get("marlin_finite") is False for r in l2):
        out["verdict"] = "WEIGHTS / ROUTING: Marlin and the reference are both non-finite on layer 2"
    elif all(r.get("rel_err") is not None and r["rel_err"] < 0.05 for r in l2):
        out["verdict"] = "MOE OK on these tokens: Marlin = reference on layer 2 (look at stream / integration)"
    else:
        out["verdict"] = "MISMATCH: layer-2 Marlin finite but differs from the reference"
    out["sec"] = round(time.perf_counter() - t0, 1)
    return out


# ---- RC1 arm 2: Flash-Next on vLLM 0.32.0 (native qwen4_exp). Install = the E18 VMS installer (Kaggle-proven).
_E13_VLLM_INST = None
E13_VLLM_HOOK_SRC = __import__("zlib").decompress(_b64.b64decode("VLLMGC_B64")).decode()   # gc after each MoE repack


def e13_vllm_install(deadline=None):
    global _E13_VLLM_INST
    if _E13_VLLM_INST is not None:
        return _E13_VLLM_INST
    if os.environ.get("E13_VLLM_FAKE"):                                 # local harness: a fake OpenAI server
        _E13_VLLM_INST = {"fake": True, "launch": [sys.executable, os.environ["E13_VLLM_FAKE"]], "env": dict(os.environ), "sec": 0}
        return _E13_VLLM_INST
    _whd = Path("/kaggle/input/datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse")
    wh = sorted(_whd.rglob("install_wheelhouse.py")) if _whd.is_dir() else []
    if not wh:
        raise E11Fail("vLLM 0.32 wheelhouse not attached (foysalemonshanto/vllm-0-32-0-cu13-wheelhouse)")
    t = time.perf_counter()
    r = subprocess.run([sys.executable, str(wh[0]), "/tmp/vllm-venv"], capture_output=True, text=True,
                       timeout=max(60, min(2400, HARD_STOP - time.perf_counter() - 150,
                                           (deadline or HARD_STOP) - time.perf_counter() - 60)))
    if r.returncode:
        raise E11Fail("vLLM 0.32 install failed: " + ((r.stdout or "") + (r.stderr or ""))[-2000:])
    kv = dict(l.split("=", 1) for l in (r.stdout or "").splitlines() if l.startswith(("VENV_PYTHON=", "CUDA_HOME=")))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH", "LD_PRELOAD")}
    env.update(CUDA_HOME=kv["CUDA_HOME"], PATH=f"/tmp/vllm-venv/bin:{kv['CUDA_HOME']}/bin:{env.get('PATH', '')}",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1", VLLM_LOGGING_LEVEL="INFO",
               VLLM_CACHE_ROOT="/tmp/vllm-cache", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="8")
    _E13_VLLM_INST = {"launch": ["/tmp/vllm-venv/bin/vllm", "serve"], "env": env, "sec": round(time.perf_counter() - t),
                      "venv_python": kv.get("VENV_PYTHON")}
    return _E13_VLLM_INST


def e13_vllm_flags():
    """vLLM 0.32.0 serve flags for the first Flash-Next smoke (GPT audit 2: target only, eager, short context, low
    concurrency): TP4 + expert parallel (= SGLang's EP4 layout, full experts per rank -> Marlin), PLE table in pinned host
    memory, language model only (no vision profiling), bf16 KV (the default), no prefix caching (state-free smoke)."""
    return [str(E13_INTEL_DIR), "--served-model-name", "flashnext", "--host", "127.0.0.1", "--port", str(E13_VLLM_PORT),
            "--tensor-parallel-size", "4", "--enable-expert-parallel", "--enforce-eager",
            "--max-model-len", "8192", "--max-num-seqs", "4", "--max-num-batched-tokens", "4096",
            "--gpu-memory-utilization", "0.93", "--disable-custom-all-reduce", "--language-model-only",
            "--engram-config", json.dumps({"cpu_offload": True}), "--distributed-executor-backend", "mp",
            "--load-format", "safetensors", "--dtype", "bfloat16", "--no-enable-prefix-caching", "--seed", "0"]


def _e13_vpost(path, body, deadline, cap):
    left = min(cap, deadline - time.perf_counter())
    if left < 20:
        raise TimeoutError(f"arm deadline: {round(left)} s left")
    req = _urlreq.Request(f"http://127.0.0.1:{E13_VLLM_PORT}" + path, data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json"})
    with _urlreq.urlopen(req, timeout=left) as r:
        return json.loads(r.read())


def e13_vllm_smoke(deadline):
    out = {}
    try:
        t = time.perf_counter()
        c = _e13_vpost("/v1/completions", {"model": "flashnext", "prompt": "1, 2, 3, 4, 5, 6, 7, 8,", "max_tokens": 32,
                                           "temperature": 0, "logprobs": 1}, deadline, 600)
        ch = (c.get("choices") or [{}])[0]
        text = ch.get("text") or ""
        lp = (ch.get("logprobs") or {}).get("token_logprobs") or []
        lpv = [float(v) for v in lp if v is not None]
        out.update(count_text=text[:80], count_ok=text.strip().startswith("9, 10, 11"), garbage=_e13_garbage(text),
                   count_logprobs=[None if v is None else round(float(v), 4) for v in lp[:8]],
                   logprobs_ok=bool(lpv) and all(math.isfinite(v) for v in lpv), count_sec=round(time.perf_counter() - t, 1))
        t = time.perf_counter()
        m = _e13_vpost("/v1/chat/completions", {"model": "flashnext", "max_tokens": 64, "temperature": 0,
                                                "chat_template_kwargs": {"enable_thinking": False}, "messages": [
            {"role": "user", "content": "What is 2 + 2? Answer with the number only."}]}, deadline, 600)
        msg = (m.get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        answer = content.split("</think>")[-1].strip()           # the reply after any thinking, nothing from the reasoning
        out.update(chat_content=content[:160], chat_answer=answer[:40], chat_ok=re.fullmatch(r"\**4\**\s*\.?", answer) is not None,
                   chat_usage=m.get("usage"), chat_finish=(m.get("choices") or [{}])[0].get("finish_reason"),
                   chat_sec=round(time.perf_counter() - t, 1))
        res = [None] * 4

        def one(i):
            try:
                j = _e13_vpost("/v1/completions", {"model": "flashnext", "prompt": f"Write a long story about grid puzzle {i}.",
                                                   "max_tokens": 256, "temperature": 0.7, "ignore_eos": True}, deadline, 900)
                res[i] = (j.get("usage") or {}).get("completion_tokens")
            except Exception as e:
                res[i] = f"{type(e).__name__}: {e}"[:160]
        t = time.perf_counter()
        ths = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(4)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(max(1.0, deadline - time.perf_counter()))
        sec = time.perf_counter() - t
        toks = sum(x for x in res if isinstance(x, int))
        out.update(streams4=res, streams4_sec=round(sec, 1), streams4_tok_s=round(toks / max(sec, 1e-6), 1),
                   streams4_ok=len(res) == 4 and all(type(n) is int and n == 256 for n in res))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:400]
    # RC1 audits: correctness = count + finite logprobs + the exact answer; throughput only counts with all 4 streams
    out["smoke_correct"] = (bool(out.get("count_ok")) and not out.get("garbage") and bool(out.get("logprobs_ok"))
                            and bool(out.get("chat_ok")) and not out.get("error"))
    out["throughput_valid"] = out["smoke_correct"] and bool(out.get("streams4_ok"))
    out["ok"] = out["throughput_valid"]
    return out


_E13_VLLM_LOG_KEYS = ("PLE", "Engram", "engram", "ngram", "quantiz", "INC", "Marlin", "marlin", "MoE", "KV cache",
                      "Maximum concurrency", "QSA", "FlashAttention", "fallback", "Traceback", "Error", "ERROR", "out of memory")


def e13_vllm_arm(tag, cfg, budget):
    """RC1 arm 2. Never raises for an engine outcome (that is the measurement)."""
    global E11_SERVER
    rec = {"tag": tag, "engine": "vllm 0.32.0", "model": str(E13_INTEL_DIR), "budget_sec": round(budget)}
    rec["host_ram"] = e13_host_ram_guard()
    if not rec["host_ram"].get("ok", False):                           # RC1 audits: fail closed like the SGLang arm
        rec["outcome"] = "skipped: host RAM guard"; return rec
    rec["gpu_free_before"] = _e11_gpu_free(180)
    if rec["gpu_free_before"].get("free") is not True:
        rec["outcome"] = "skipped: GPU free status not verified"; return rec
    t_deadline = min(time.perf_counter() + budget, HARD_STOP - 5 * 60)
    try:
        inst = e13_vllm_install(t_deadline)
    except Exception as e:
        rec["outcome"] = f"install error: {type(e).__name__}: {e}"[:1500]; return rec
    if time.perf_counter() > t_deadline - 12 * 60:                     # no launch that cannot reach ready + smoke
        rec["outcome"] = f"skipped: {round(t_deadline - time.perf_counter())} s left after the install"; return rec
    rec["install_sec"] = inst.get("sec")
    rec["flags"] = e13_vllm_flags()
    _vh = Path(os.environ.get("E13_VLLM_HOOK_DIR", "/tmp/e13_vllm_hook")); _vh.mkdir(parents=True, exist_ok=True)
    (_vh / "sitecustomize.py").write_text(E13_VLLM_HOOK_SRC, encoding="utf-8")
    vtd = OUT / "e13_memtrace" / tag
    venv_env = dict(inst["env"], PYTHONPATH=str(_vh), E13_VLLM_TRACE_DIR=str(vtd))
    E11_SERVER = E11Server(inst["launch"] + rec["flags"], venv_env, f"e13_{tag}")
    E11_SERVER.log_path = OUT / f"vllm_server_e13_{tag}.log"
    E11_SERVER.start()
    print(f"E13 RC1 {tag}: vLLM launch", flush=True)
    last, last_t = -1, time.perf_counter()
    while True:
        if not E11_SERVER.alive():
            rec["outcome"] = f"exited rc={E11_SERVER.proc.returncode}"; break
        try:
            _urlreq.urlopen(f"http://127.0.0.1:{E13_VLLM_PORT}/health", timeout=10).read(); rec["outcome"] = "ready"; break
        except Exception:
            pass
        sz = E11_SERVER.log_path.stat().st_size if E11_SERVER.log_path.exists() else 0
        if sz != last:
            last, last_t = sz, time.perf_counter()
        if time.perf_counter() - last_t > 10 * 60:
            rec["outcome"] = "stall (10 min log silence)"; break
        if time.perf_counter() > t_deadline - 4 * 60:
            rec["outcome"] = "timeout"; break
        time.sleep(2 if E11_HARNESS else 5)
    rec["sec"] = round(time.perf_counter() - E11_SERVER.t0, 1)
    if rec["outcome"] == "ready":
        rec["smoke"] = e13_vllm_smoke(t_deadline)
    log = E11_SERVER.log_text()
    lines = log.splitlines()
    rec["log_lines"] = [l[:300] for l in lines if any(k in l for k in _E13_VLLM_LOG_KEYS)][:80]
    rec["kv_tokens"] = [int(x.replace(",", "")) for x in re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)][:4]
    rec["max_concurrency"] = re.findall(r"Maximum concurrency for [^\n]{0,80}", log)[:2]
    rec["oom_lines"] = [l[:300] for l in lines if "out of memory" in l.lower()][:6]
    rec["log_tail"] = log[-3000:] if rec["outcome"] != "ready" or not (rec.get("smoke") or {}).get("ok") else ""
    E11_SERVER.kill()
    if not E11_HARNESS:
        subprocess.run(["pkill", "-9", "-f", "vllm"], capture_output=True)
    rec["gpu_free_after"] = _e11_gpu_free(180)
    vrecs = []
    for _vf in sorted(vtd.glob("vtrace_*.jsonl")):
        for _l in _vf.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                vrecs.append(json.loads(_l))
            except ValueError:
                pass
    rec["vllm_gc"] = {"patched": sum(r.get("ev") == "vpatched" for r in vrecs),
                      "errors": [r for r in vrecs if "error" in str(r.get("ev"))][:4],
                      "pwal_end": [{k: r.get(k) for k in ("pid", "alloc", "moe_modules", "freed_bytes")} for r in vrecs if r.get("ev") == "vpwal_end"][:4],
                      "samples": [r for r in vrecs if r.get("ev") == "vgc"][:6]}
    print(f"E13 RC1 {tag}:", json.dumps({k: rec.get(k) for k in ("outcome", "sec", "kv_tokens", "smoke", "oom_lines")},
                                       default=str)[:2000], flush=True)
    return rec


def e13_long_test(deadline, spec):
    """NANPROBE rev 2 (F9): runtime memory headroom. `streams` concurrent requests of ~`tokens` prompt tokens each
    (ARC evaluation grids as text), max_new 16, ignore_eos. Correct output is not needed (memory only): records OOM lines
    in the server log after the start, server alive, per-request errors, and the scheduler's available_gpu_mem."""
    out = {"spec": spec}
    try:
        allt = _load_json(ARC_ROOT / "arc-agi_evaluation_challenges.json")
    except Exception:
        allt = dict(CHALLENGES)
    rows = lambda g: "\n".join(" ".join(str(v) for v in r) for r in g)
    blob = "\n".join(rows(q["input"]) for k in sorted(allt) for q in allt[k]["train"])
    ids = tokenizer.encode(blob, add_special_tokens=False)
    want = int(spec.get("tokens", 20000)) if not E11_HARNESS else 200
    if len(ids) < want:
        ids = (ids * (want // max(1, len(ids)) + 1))
    prompts = [ids[i * 97: i * 97 + want] for i in range(int(spec.get("streams", 4)))]
    off = len(E11_SERVER.log_text())
    res = [None] * len(prompts)

    def one(i):
        try:
            j = _e13_post("/generate", {"input_ids": prompts[i], "sampling_params": {"max_new_tokens": 16, "ignore_eos": True}},
                          deadline, 1200)
            j = j[0] if isinstance(j, list) else j
            res[i] = {"completion_tokens": (j.get("meta_info") or {}).get("completion_tokens"),
                      "prompt_tokens": (j.get("meta_info") or {}).get("prompt_tokens")}
        except Exception as e:
            res[i] = {"error": f"{type(e).__name__}: {e}"[:200]}
    t0 = time.perf_counter()
    ths = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(len(prompts))]
    for th in ths:
        th.start()
    for th in ths:
        th.join(max(1.0, deadline - time.perf_counter()))
    tail = E11_SERVER.log_text()[off:]
    out.update(sec=round(time.perf_counter() - t0, 1), requests=res, server_alive=E11_SERVER.alive(),
               oom_lines=[l[:300] for l in tail.splitlines() if "out of memory" in l.lower()][:4],
               avail_gpu_mem=re.findall(r"available_gpu_mem=([\d.]+) GB", tail)[-3:],
               max_running_seen=max([int(x) for x in re.findall(r"#running-req: (\d+)", tail)] or [0]),
               max_full_token_usage=max([float(x) for x in re.findall(r"full token usage: ([\d.]+)", tail)] or [0.0]))
    out["ok"] = out["server_alive"] and not out["oom_lines"] and all(r and not r.get("error") for r in res)
    return out


def e13_probe_arm(tag, model_dir, vpy, hookdir, budget):
    """One launch with the rev 6b flags on `model_dir`, the memory trace on; until ready / exit / stall / budget; then a
    smoke if ready; then kill and wait for free GPUs. Never raises for an engine outcome (that is the measurement)."""
    global SGL_MODEL_PATH, E11_SERVER
    rec = {"tag": tag, "model": str(model_dir), "budget_sec": round(budget)}
    SGL_MODEL_PATH = Path(model_dir)
    rec["host_ram"] = e13_host_ram_guard()
    if not rec["host_ram"]["ok"]:
        rec["outcome"] = "skipped: host RAM guard"; return rec
    rec["gpu_free_before"] = _e11_gpu_free(180)
    if rec["gpu_free_before"].get("free") is not True:               # fix list F17 + RC1 audit: fail closed
        rec["outcome"] = "skipped: GPU free status not verified"; return rec
    t_deadline = min(time.perf_counter() + budget, HARD_STOP - 5 * 60)  # F10: smoke + long test inside the arm too
    tdir = OUT / "e13_memtrace" / tag
    _e13_set_ple_mode("orig")                                          # RC1: every arm starts on SGLang's own PLE path
    env = dict(_E13_ENV); env["PYTHONPATH"] = str(hookdir); env["E13_MEMTRACE_DIR"] = str(tdir)
    cfg = (globals().get("E13_ARM_CFG") or {}).get(tag) or {}          # NANPROBE: per-arm env / flag changes
    env.update(cfg.get("env") or {})
    fl = e13_flags()
    for k, v in (cfg.get("set") or {}).items():
        i = fl.index(k)
        fl[i + 1] = v
    rec["flags"] = fl + list(cfg.get("add") or [])
    rec["arm_cfg"] = cfg
    launch = [str(vpy), "-m", "sglang.launch_server"] if not E11_HARNESS else [sys.executable, os.environ["E11_FAKE_SERVER"]]
    E11_SERVER = E11Server(launch + rec["flags"], env, f"e13_{tag}")
    E11_SERVER.start()
    print(f"E13 LOADPROBE {tag}: launch on {model_dir}", flush=True)
    t_end, last, last_t = t_deadline - (cfg.get("reserve_after_ready_sec") or 0), -1, time.perf_counter()
    while True:
        if not E11_SERVER.alive():
            rec["outcome"] = f"exited rc={E11_SERVER.proc.returncode}"; break
        try:
            _urlreq.urlopen(SGL_URL + "/health", timeout=10).read(); rec["outcome"] = "ready"; break
        except Exception:
            pass
        sz = E11_SERVER.log_path.stat().st_size if E11_SERVER.log_path.exists() else 0
        if sz != last:
            last, last_t = sz, time.perf_counter()
        if time.perf_counter() - last_t > 10 * 60:
            rec["outcome"] = "stall (10 min log silence)"; break
        if time.perf_counter() > t_end:
            rec["outcome"] = "timeout"; break
        time.sleep(2 if E11_HARNESS else 5)
    rec["sec"] = round(time.perf_counter() - E11_SERVER.t0, 1)
    log = E11_SERVER.log_text()
    rec["caps"] = _e13_caps(log)
    rec["oom_lines"] = [l[:300] for l in log.splitlines() if "out of memory" in l.lower()][:6]
    rec["load_weight_end_lines"] = [l[:300] for l in log.splitlines() if "Load weight end" in l][:4]
    if rec["outcome"] == "ready" and cfg.get("ple_modes"):             # RC1: one count per PLE mode, same server
        rec["ple_runs"] = e13_ple_mode_runs(t_deadline, cfg["ple_modes"])
        rec["smoke"] = (rec["ple_runs"] or [{}])[0]
    elif rec["outcome"] == "ready" and cfg:                            # NANPROBE arms (rev 2)
        rec["smoke"] = e13_probe_smoke(t_deadline, cfg)
        if cfg.get("rerun_count") and E11_SERVER.alive():              # F14: drain past the hook window, then count again
            rec["count_after_hooks"] = e13_probe_smoke(t_deadline, {**cfg, "chat_max_tokens": 0, "drain_tokens": 160})
        if cfg.get("capacity_gate"):                                   # F8: the KV figure must belong to 4 streams
            cap = rec["caps"].get("capped_running")
            rec["capacity_gate"] = "OK" if cap in (None, SGL_SERVER_MAX_RUNNING) else f"CAPACITY_GATE_FAILED (capped to {cap})"
        if cfg.get("long_test") and E11_SERVER.alive():                # F9: runtime memory headroom with long prefills
            rec["long_test"] = e13_long_test(t_deadline, cfg["long_test"])
        rec["caps_after"] = _e13_caps(E11_SERVER.log_text())
    elif rec["outcome"] == "ready":
        try:
            rec["smoke"] = e13_smoke()
        except Exception as e:
            rec["smoke"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
    E11_SERVER.kill()
    rec["gpu_free_after"] = _e11_gpu_free(180)
    if (cfg.get("env") or {}).get("E13_RC1_FIXTURE") == "1":            # RC1 test B: CPU reference, after ALL GPU arms
        E13_RC1_PENDING.append((tag, tdir, model_dir))
        rec["rc1_reference"] = {"pending": "runs after all GPU arms (RC1 audit H1)"}
    rec["memtrace"] = e13_memtrace_summary(tdir)
    print(f"E13 LOADPROBE {tag}:", json.dumps({k: rec.get(k) for k in ("outcome", "sec", "caps", "oom_lines", "smoke")},
                                             default=str)[:1500], flush=True)
    print(f"E13 LOADPROBE {tag} memtrace:", json.dumps(rec["memtrace"], default=str)[:4000], flush=True)
    return rec


def e13_vllm_registry():
    """NANPROBE: does stock vLLM 0.32 (our wheelhouse venv, the E23 install) list Flash-Next's architecture? An import
    check in that venv (no server, no GPU work): the supported-architecture list and the model config's architectures."""
    arch = (json.loads((Path(E13_INTEL_DIR) / "config.json").read_text()).get("architectures") or [None])[0]
    _whd = Path("/kaggle/input/datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse")   # the dataset only (model mounts are slow)
    wh = sorted(_whd.rglob("install_wheelhouse.py")) if _whd.is_dir() else []
    if not wh or E11_HARNESS:
        return {"skipped": "vllm 0.32 wheelhouse not attached" if not wh else "harness", "arch": arch}
    t = time.perf_counter()
    r = subprocess.run([sys.executable, str(wh[0]), "/tmp/vllm-venv"], capture_output=True, text=True,
                       timeout=max(60, min(2400, HARD_STOP - time.perf_counter() - 150)))
    kv = dict(l.split("=", 1) for l in (r.stdout or "").splitlines() if l.startswith(("VENV_PYTHON=", "CUDA_HOME=")))
    out = {"arch": arch, "install_rc": r.returncode, "install_sec": round(time.perf_counter() - t)}
    if r.returncode or "VENV_PYTHON" not in kv:
        out["install_tail"] = ((r.stdout or "") + (r.stderr or ""))[-1500:]
        return out
    code = ("import json, re, vllm\nfrom vllm.model_executor.models.registry import ModelRegistry\n"
            "s = sorted(ModelRegistry.get_supported_archs())\n"
            f"a = {arch!r}\n"
            "info = {'vllm': vllm.__version__, 'supported': a in s, 'qwen_archs': [x for x in s if 'Qwen' in x]}\n"
            "try:\n"
            "    from vllm.model_executor.models.registry import _VLLM_MODELS as M\n"   # (module, class) of the implementation
            "    mod = M.get(a)\n"
            "    info['impl'] = mod\n"
            "    if mod:\n"
            "        import importlib.util as u\n"
            "        sp = u.find_spec('vllm.model_executor.models.' + mod[0])\n"
            "        src = open(sp.origin).read() if sp and sp.origin else ''\n"
            "        info['impl_mentions'] = {k: len(re.findall(k, src, re.I)) for k in ('ple', 'offload', 'pin_memory', 'hyper', 'indexer', 'sparse')}\n"
            "except Exception as e:\n"
            "    info['impl_error'] = repr(e)[:200]\n"
            "print('E13VLLM ' + json.dumps(info))\n")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_PRELOAD")}
    env.update(CUDA_HOME=kv.get("CUDA_HOME", ""), VLLM_NO_USAGE_STATS="1")
    c = subprocess.run([kv["VENV_PYTHON"], "-c", code], capture_output=True, text=True,
                       timeout=max(60, min(900, HARD_STOP - time.perf_counter() - 60)), env=env)
    m = re.search(r"^E13VLLM (.*)$", c.stdout or "", re.M)
    out.update(json.loads(m.group(1)) if m else {"check_rc": c.returncode, "check_tail": ((c.stdout or "") + (c.stderr or ""))[-1500:]})
    return out


_E13_ENV = None
_t_engine = time.perf_counter()
E13_RC1_PENDING = []          # RC1: (arm tag, trace dir, model dir) whose CPU reference runs after all GPU arms
E13_PROBE = {"arms": [], "budget_hard_stop_in_sec": round(HARD_STOP - time.perf_counter())}
ENGINE_INFO["e13"]["probe"] = E13_PROBE
if not E10_SANDBOX_SELFTEST.get("ok", False):
    ENGINE_INFO["e13"]["stage"] = "skipped: sandbox self-test failed"
else:
    try:
        _info = ENGINE_INFO["e13"]
        _info["stage"] = "install"
        _vpy, _E13_ENV, _info["install"] = e11_install()
        _E13_ENV.update({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
                         "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_NUMA_BIND_V2": "false", "SGLANG_MM_PREPROCESS_DEVICE": "cpu",
                         "OMP_NUM_THREADS": "8", "TORCH_CUDA_ARCH_LIST": "8.9"})                      # = e13_bring_up
        _info["stage"] = "fa2_patch3"; _info["extras"] = e13_extras(_vpy)
        _info["stage"] = "topk_test"; _info["topk"] = e13_topk_test(_vpy)
        print("E13 top-k test:", json.dumps(_info["topk"], default=str)[:1500], flush=True)
        _hook = Path(os.environ.get("E13_HOOK_DIR", "/tmp/e13_memtrace_hook")); _hook.mkdir(parents=True, exist_ok=True)
        (_hook / "sitecustomize.py").write_text(E13_MEMTRACE_SRC, encoding="utf-8")
        _arms = list(E13_PROBE_ARMS)
        while _arms:
            _tag = _arms.pop(0)
            _room = HARD_STOP - time.perf_counter() - 5 * 60
            if _room < 8 * 60:
                E13_PROBE["arms"].append({"tag": _tag, "outcome": f"skipped: {round(_room)} s left"}); continue
            _info["stage"] = f"probe {_tag}"
            _acfg = (globals().get("E13_ARM_CFG") or {}).get(_tag) or {}
            if _room < float(_acfg.get("min_room_sec") or 0):                 # RC1: the optional third arm
                E13_PROBE["arms"].append({"tag": _tag, "outcome": f"skipped: {round(_room)} s left (< {_acfg['min_room_sec']})"}); continue
            try:
                if _acfg.get("engine") == "vllm":                                  # RC1 arm 2: Flash-Next on vLLM 0.32
                    E13_PROBE["arms"].append(e13_vllm_arm(_tag, _acfg, min(float(_acfg.get("budget_sec") or 30 * 60), _room)))
                    (OUT / "e13_loadprobe.json").write_text(json.dumps(E13_PROBE, indent=1, default=str), encoding="utf-8")
                    continue
                if _tag == "bf16ple" or _acfg.get("model") == "intel":
                    _md = E13_INTEL_DIR
                else:
                    _md, E13_PROBE["fp8ple_merge"] = e13_fp8ple_dir()
                _r = e13_probe_arm(_tag, _md, _vpy, _hook, min(float(_acfg.get("budget_sec") or 25 * 60), _room))
                E13_PROBE["arms"].append(_r)
                # NANPROBE rev 2 (F13): an OOM before ready on the NaN arm -> one retry with gc after every module; the KV
                # arm is dropped then (it would OOM the same way and Q1 matters more)
                if _acfg.get("oom_retry") and _r.get("outcome") != "ready" and (_r.get("oom_lines") or _r.get("caps", {}).get("oom")):
                    _arms = [_acfg["oom_retry"]] + [a for a in _arms if a not in ("kv_lean",)]
                    E13_PROBE["oom_retry"] = {"after": _tag, "next": _acfg["oom_retry"], "dropped": "kv_lean"}
            except Exception as _ae:
                E13_PROBE["arms"].append({"tag": _tag, "outcome": f"error: {type(_ae).__name__}: {_ae}"[:800]})
                if E11_SERVER is not None:
                    E11_SERVER.kill()
            (OUT / "e13_loadprobe.json").write_text(json.dumps(E13_PROBE, indent=1, default=str), encoding="utf-8")
        # RC1 audits H1: the CPU reference after every GPU arm, with its own capped deadline; the GPU evidence is on disk
        # already (e13_loadprobe.json is written after each arm), so a truncated reference loses nothing
        for _tag, _tdir, _md in E13_RC1_PENDING:
            _arm = next((a for a in E13_PROBE["arms"] if a.get("tag") == _tag), {})
            _left = HARD_STOP - time.perf_counter() - 4 * 60
            _info["stage"] = f"rc1 reference {_tag}"
            if _left < 90:
                _arm["rc1_reference"] = {"skipped": f"{round(_left)} s left before the hard stop"}
            else:
                try:
                    _arm["rc1_reference"] = e13_rc1_reference(_tdir, _md, time.perf_counter() + min(15 * 60, _left))
                except Exception as _re:
                    _arm["rc1_reference"] = {"error": f"{type(_re).__name__}: {_re}"[:400]}
            print(f"E13 RC1 {_tag} reference:", json.dumps(_arm.get("rc1_reference"), default=str)[:3000], flush=True)
            (OUT / "e13_loadprobe.json").write_text(json.dumps(E13_PROBE, indent=1, default=str), encoding="utf-8")
        # rev 2 (F11): CPU-only, after the arms, its own guard (a timeout here never costs an arm)
        if globals().get("E13_VLLM_CHECK"):
            _info["stage"] = "vllm_registry"
            if HARD_STOP - time.perf_counter() < 6 * 60:
                E13_PROBE["vllm_registry"] = {"skipped": "less than 6 min left"}
            else:
                try:
                    E13_PROBE["vllm_registry"] = e13_vllm_registry()
                except Exception as _ve:
                    E13_PROBE["vllm_registry"] = {"error": f"{type(_ve).__name__}: {_ve}"[:400]}
            print("E13 vLLM 0.32 registry:", json.dumps(E13_PROBE["vllm_registry"], default=str)[:1500], flush=True)
            (OUT / "e13_loadprobe.json").write_text(json.dumps(E13_PROBE, indent=1, default=str), encoding="utf-8")
        _info["stage"] = "probe done"
    except Exception as _e:
        ENGINE_INFO["e13"]["error"] = f"{type(_e).__name__}: {_e}"[:3000]
        print("E13 LOADPROBE stopped:", ENGINE_INFO["e13"]["stage"], "|", ENGINE_INFO["e13"]["error"][:1500], flush=True)
        if E11_SERVER is not None:
            E11_SERVER.kill()
ENGINE_READY = False; CONCURRENCY = 0                 # load-only: the Duck never runs; CELL 0's fallback is the submission
ENGINE_INFO["fatal"] = "LOADPROBE: load-only notebook (no Duck)"
(OUT / "e13_loadprobe.json").write_text(json.dumps(E13_PROBE, indent=1, default=str), encoding="utf-8")
ENGINE_INFO["load_sec"] = round(time.perf_counter() - _t_engine, 1)
ENGINE_INFO["concurrency"] = 0
print("E13 LOADPROBE:", json.dumps([{k: a.get(k) for k in ("tag", "outcome", "sec")} for a in E13_PROBE["arms"]]), flush=True)
'''


def src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: {text.count(old)}"
    return text.replace(old, new)


def between(text, start, end, label):
    assert text.count(start) == 1 and (end is None or text.count(end) >= 1), label
    i = text.index(start); j = text.index(end, i) if end else len(text)
    return text[i:j]


raw = BASE.read_bytes(); assert hashlib.sha256(raw).hexdigest() == BASE_SHA, "E12 base notebook changed"
nb = json.loads(raw)
cells = [src(c) for c in nb["cells"]]
ENG12 = cells[6]

# ---------------------------------------------------------------- CELL 1: engine inputs and limits only
c1 = cells[2]
c1 = once(c1, 'SGL_MODEL_PATH = Path(os.environ.get("E11_INT4", "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1"))',
          'SGL_MODEL_PATH = Path(os.environ.get("E11_INT4", "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"))   # E13',
          "model path")
c1 = once(c1, 'SGL_DRAFT_PATH = Path(os.environ.get("E11_DFLASH", "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-dflash2/pytorch/bf16/1"))',
          'SGL_DRAFT_PATH = Path(os.environ.get("E11_DFLASH", "/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1"))   # E13: precache only (NEXTN off)',
          "draft path")
c1 = once(c1, 'and (c.get("quantization_config") or {}).get("quant_method") == "compressed-tensors")',
          'and (c.get("quantization_config") or c.get("text_config", {}).get("quantization_config") or {}).get("quant_method") == "auto-round")   # E13',
          "model predicate")
c1 = once(c1, 'and c.get("architectures") == ["DFlash2DraftModel"])',
          'and "mtp_routed_experts" in ((c.get("quantization_config") or {}).get("config_groups") or {}))   # E13: Franzen\'s draft test',
          "draft predicate")
c1 = once(c1, "SGL_CONTEXT_LEN = 155648 ", "SGL_CONTEXT_LEN = 139264 ", "context")          # E13: Franzen (116+12+8)k
c1 = once(c1, "SGL_MAX_RUNNING = 12 ", "SGL_MAX_RUNNING = 4  ", "running")               # E13: PLAN 3b memory item
c1 = once(c1, "SGL_SERVER_MAX_RUNNING = 13 ", "SGL_SERVER_MAX_RUNNING = 4  ", "server running")
if not EVAL30:   # E13: Flash-Next load (about 25 min incl. the top-k test) + 15 min solving does not fit 50 min
    c1 = once(c1, '"SMOKE1":    {"source": "eval", "task_list": ["2ba387bc"], "time_budget_sec": 50 * 60,',
              '"SMOKE1":    {"source": "eval", "task_list": ["2ba387bc"], "time_budget_sec": '
              + ("120" if RC1 else "90" if NANPROBE else "75" if LOADPROBE else "105") + ' * 60,', "smoke budget")   # rev 6: +15 min speed stage; LOADPROBE 75 min (ceiling; ~35-40 min expected)
c1 += r'''
# ---- E13 (Flash-Next on 4xL4, PLAN rev 4.1 simple path): engine constants. Everything else in this notebook = E12.
E13_FA2_WHEEL = "flash_attn-2.8.3.post1+cu13torch2.13cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
E13_FA2_SHA = "74984f628eaa9b22de271c787d82c4ef16daffcfb9684286a7a15ff6ad41ae84"
# rev 6: the foysal copy first, then the hosen42 copy, then any directory under /kaggle/input holding the wheel (the name with
# or without '+': Kaggle drops it); the wheel's sha256 is checked before use either way
E13_FA2_DIRS = ([Path(os.environ["E13_FA2_DIR"])] if os.environ.get("E13_FA2_DIR") else []) + [
    Path("/kaggle/input/datasets/foysalemonshanto/flash-attn-2-8-3-post1-cu13-torch2-13-cp312"),
    Path("/kaggle/input/datasets/hosen42/flash-attn-2-8-3-post1-cu13-torch2-13-cp312")]
E13_FA2_DIR = next((d for d in E13_FA2_DIRS if d.is_dir()), E13_FA2_DIRS[0])
E13_PATCH3 = ("135e004f7613a59aa0765cbe3d779c9bb0c459f3a199b868358759e020aaeb5d",   # Franzen BEFORE_SHA (= 0.5.21 file)
              "47e65140e2480a8637b9ea5bb746ff77e9c09c01e448fd2056c0b485e4f3a10a")   # Franzen AFTER_SHA (= Pennyroyal wheel)
E13_MEM_FRACTION = "0.93"        # PLAN 3b: about 1.5 GiB outside the static pools (Franzen 0.96 on a 96 GB card)
E13_CHUNK = 2048                 # PLAN 3b: E3-proven on 4xL4 (Franzen 8192 on 96 GB)
E13_KV_PER_STREAM = 60000        # pre-run capacity rule: resident Duck contexts <= KV / 60k (E12 EVAL30 prompts p50 47.9k)
E13_TOPK_TEST = os.environ.get("E13_TOPK_TEST", "1") == "1"   # user: the fast_topk<512> correctness test always runs
E13_SKIP_CAT_PREFETCH = True      # rev 6b (audit B-H1): no CELL 2 background `cat` of 178 GB (it competes with the loader's cache)
CONFIG["e13"] = {k: globals()[k] for k in ("E13_FA2_SHA", "E13_MEM_FRACTION", "E13_CHUNK", "E13_KV_PER_STREAM", "E13_TOPK_TEST",
                                           "SGL_CONTEXT_LEN", "SGL_MAX_RUNNING")}
CONFIG["e13"]["fa2_dir"] = str(E13_FA2_DIR)
'''
if LOADPROBE:
    c1 += r'''
# ---- E13 LOADPROBE (user 2026-10-27): load-only, two launches with a GPU memory trace; no speed stage, no Duck
E13_LOADPROBE = True
E13_PROBE_ARMS = ("bf16ple", "fp8ple")          # 1 = rev 6b exactly (Franzen's BF16 PLE); 2 = the E3 v4.2.1 FP8 PLE merge
E13_INTEL_DIR = SGL_MODEL_PATH
_e13_ov = [Path(os.environ["E13_OVERLAY"])] if os.environ.get("E13_OVERLAY") else [
    Path("/kaggle/input/models/foysalemonshanto/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1")]
_e13_ovroot = Path("/kaggle/input/models/foysalemonshanto/qwen38-flash-next-fp8ple-overlay")
_e13_ov += sorted(_e13_ovroot.glob("*/*/*")) if _e13_ovroot.is_dir() else []
E13_OVERLAY_DIR = next((d for d in _e13_ov if (d / "config.json").is_file()), _e13_ov[0])
E13_OVERLAY_SHARDS = {"model-00001-of-00017.safetensors": 2899311784, "model-00003-of-00017.safetensors": 5370548192, "model-00004-of-00017.safetensors": 5370910416, "model-00005-of-00017.safetensors": 5370140704, "model-00006-of-00017.safetensors": 5370205184, "model-00007-of-00017.safetensors": 5370927288, "model-00008-of-00017.safetensors": 5344445352, "model-00009-of-00017.safetensors": 5370355648, "model-00010-of-00017.safetensors": 5370927288, "model-00011-of-00017.safetensors": 5324173744, "model-00012-of-00017.safetensors": 5370438520, "model-00013-of-00017.safetensors": 5370927288, "model-00014-of-00017.safetensors": 5342689856, "model-00015-of-00017.safetensors": 5050474736, "model-00017-of-00017.safetensors": 1284526608, "model-plefp8-00000.safetensors": 5200027199, "model-plefp8-00001.safetensors": 5200027202, "model-plefp8-00002.safetensors": 5200027198, "model-plefp8-00003.safetensors": 5200027189, "model-plefp8-00004.safetensors": 5200027190, "model-plefp8-00005.safetensors": 5200027190, "model-plefp8-00006.safetensors": 5200027190, "model-plefp8-00007.safetensors": 5200027190, "model-plefp8-00008.safetensors": 5200027190, "model-plefp8-00009.safetensors": 4400023163}                  # E3 v4.2.1 pinned sizes: 15 Intel main shards + 10 FP8 PLE shards
CONFIG["e13"]["loadprobe"] = {"arms": list(E13_PROBE_ARMS), "intel": str(E13_INTEL_DIR), "overlay": str(E13_OVERLAY_DIR)}
'''
if NANPROBE:
    c1 += r'''
# ---- E13 NANPROBE (user 2026-10-30): the two Flash-Next blockers on 4xL4 in one session.
# Arm 1 nan_eager: BF16 PLE, gc fix, eager (no CUDA graphs) + the NaN / collapse locator (instance-forward wrapper, per
#   rank and forward: first non-finite module + origin vs propagated, layer [min, max], logits stats, QSA indexer ints);
#   count smoke, short chat only when the count passes; then a drain past the hook window and the count again (hooks off).
#   An OOM before ready -> one retry with gc after every module (nan_eager_gc0), and kv_lean is dropped.
# Arm 2 kv_lean: BF16 PLE, gc fix, graphs on, mamba cache 20 slots (= 4 running x 5 for extra_buffer + overlap: 8 slots
#   would cap running to 1) and ONLY mem fraction 0.93 -> 0.95 (LOADPROBE: mamba 0.29 GB, KV 0.54 GB, 1.05 GB free after
#   graphs at 0.93); capacity gate (running cap must stay 4) + 4 concurrent ~20k-token prompts (OOM / headroom).
# FP8 PLE is not an arm: LOADPROBE showed it changes host memory only (GPU tensors byte-identical, KV unchanged).
# After the arms (CPU only): a vLLM 0.32 registry check (our wheelhouse venv): is Flash-Next's architecture there?
E13_PROBE_ARMS = ("nan_eager", "kv_lean")
_e13_nan_env = {"E13_NANPROBE": "1", "E13_NAN_FORWARDS": "120", "E13_NAN_ABSMAX": "1e4", "E13_GC_MIB": "256"}
E13_ARM_CFG = {
    "nan_eager": {"model": "intel", "env": _e13_nan_env, "add": ["--disable-cuda-graph"], "nan": True, "chat_max_tokens": 256,
                  "rerun_count": True, "oom_retry": "nan_eager_gc0", "budget_sec": 30 * 60},
    "nan_eager_gc0": {"model": "intel", "env": {**_e13_nan_env, "E13_GC_MIB": "0"}, "add": ["--disable-cuda-graph"], "nan": True,
                      "chat_max_tokens": 256, "rerun_count": True, "budget_sec": 30 * 60},
    "kv_lean": {"model": "intel", "env": {"E13_GC_MIB": "256"}, "set": {"--mem-fraction-static": "0.95"},
                "chat_max_tokens": 256, "capacity_gate": True, "long_test": {"streams": 4, "tokens": 20000},
                "budget_sec": 30 * 60}}
E13_VLLM_CHECK = True
CONFIG["e13"]["nanprobe"] = {"arms": list(E13_PROBE_ARMS), "cfg": E13_ARM_CFG, "vllm_check": E13_VLLM_CHECK}
'''
if RC1:
    c1 += r'''
# ---- E13 RC1 (user 2026-10-30): root cause of the layer-2 MoE NaN (NANPROBE rev 2 + two GPT audits) and Flash-Next on
# vLLM 0.32. One notebook, three arms in order (each with its own deadline; a GPU-free gate between arms):
# Arm 1 rc1 (SGLang, eager, NANPROBE flags): hooks on layers 0-3 only; expert weight scan of layers 0-3 after the repack and
#   after forward 0 (non-finite scales + the local experts holding them, int checksums, expert maps, quant method); every
#   MoE input incl. the integer top-k ids (dtype, shape, stride, min / max); then the count prompt under three PLE modes
#   in the same server (flag file; cache flushed + a different prompt per mode): orig, sync (side stream synchronized),
#   skip (no prefetch, main-stream gather).
# Arm 2 vllm_fn (vLLM 0.32.0, native qwen4_exp): TP4 + EP, eager, 8k context, PLE in pinned host memory (--engram-config),
#   language model only; count + 2+2 chat + 4 x 256-token streams (tok/s); PLE / quant method / KV lines from the log.
# Arm 3 l2_wna16 (SGLang, only with >= 22 min left): arm 1 with layer 2's experts on the non-Marlin GPTQ path (MoeWNA16).
E13_PROBE_ARMS = ("rc1", "vllm_fn", "l2_wna16")
_e13_rc1_env = {"E13_NANPROBE": "1", "E13_NAN_FORWARDS": "240", "E13_NAN_ABSMAX": "1e4", "E13_GC_MIB": "256",
                "E13_NAN_ONLY": r"(^|\.)layers\.[0-3](\.|$)", "E13_RC1_SCAN": "1", "E13_RC1_LAYERS": "0,1,2,3",
                "E13_PLE_FLAG": os.environ.get("E13_PLE_FLAG", "/tmp/e13_ple_mode"), "E13_RC1_FIXTURE": "1", "E13_RC1_FIX_LAYERS": "1,2",
                "E13_RC1_FIX_FORWARDS": "2", "E13_RC1_FIX_TOKENS": "6",
                "E13_SYNC_RE": r"layers\.2\.mlp\.experts$"}   # RC1 audit extra B: device-wide sync around layer 2's MoE
E13_ARM_CFG = {
    "rc1": {"model": "intel", "env": _e13_rc1_env, "add": ["--disable-cuda-graph"], "nan": True, "chat_max_tokens": 0,
            "ple_modes": ["orig", "sync", "skip"], "budget_sec": 32 * 60},
    "vllm_fn": {"engine": "vllm", "budget_sec": 38 * 60},
    "l2_wna16": {"model": "intel", "env": {**_e13_rc1_env, "E13_L2_NO_MARLIN": "1"}, "add": ["--disable-cuda-graph"], "nan": True,
                 "chat_max_tokens": 0, "ple_modes": ["orig"], "min_room_sec": 22 * 60, "budget_sec": 25 * 60}}
E13_VLLM_CHECK = False            # arm 2 runs vLLM for real
E13_VLLM_PORT = int(os.environ.get("E13_VLLM_PORT", 8000))
CONFIG["e13"]["rc1"] = {"arms": list(E13_PROBE_ARMS), "cfg": E13_ARM_CFG}
'''

# ---------------------------------------------------------------- CELL 5: Flash-Next engine (Duck-only)
REUSED = (between(ENG12, "class E11Fail(Exception):", "# ---- launch: the E11-04 W1 command", "install block") + "\n"
          + between(ENG12, "class E11Server:", "# ---- the shim:", "E11Server + kill") + "\n"
          + between(ENG12, "_E11_KV_RE = re.compile", "def e11_smoke", "decode summary"))
for must in ("def e11_install():", "class E11Server:", "def e11_kill_server(", "def e11_decode_summary(", "_E11_KV_RE"):
    assert must in REUSED, must
# rev 6b (audits A-L1 / B-L1): with --ep-size 4 SGLang prefixes rank-0 lines "TP0 EP0]" (scheduler.py:6007-6010)
REUSED = once(REUSED, 'if "Decode batch" in line and "TP0]" in line:',
              'if "Decode batch" in line and re.search(r"TP0[ \]]", line):', "decode summary prefix")

C5 = r'''# ==================================================================== CELL 5 — E13 engine: SGLang 0.5.21 + Qwen3.8-Flash-Next (Franzen's launch)
# Duck-only (no Poetiq shim). Reused verbatim from E12 (Kaggle-proven): E11Fail .. e11_install, E11Server, e11_kill_server,
# e11_decode_summary. New: FA2 + patch #3, the top-k test, a host-RAM guard, Franzen's launch with the L4 deltas, a smoke.
import queue as _queue, signal as _signal, statistics as _statistics, types as _types, urllib.request as _urlreq, base64 as _b64
E11_HARNESS = os.environ.get("E11_HARNESS") == "1"            # local CPU harness only: fake server, no install, no GPU
SGL_PORT = int(os.environ.get("E11_PORT", 30000))
SGL_URL = f"http://127.0.0.1:{SGL_PORT}"
E11_SERVER = None
ENGINE_READY = False; llm = None; engine = None; CONCURRENCY = 0
ENGINE_INFO = {"backend": None, "e13": {"stage": "not started"}, "kv_tokens": None, "concurrency": None, "fatal": None,
               "load_sec": None, "tokenizer_check": E11_TOKENIZER_CHECK}

''' + REUSED + r'''

def metrics_snapshot():                                  # SGLang /metrics (CELL 6's heartbeat matches name substrings)
    if not ENGINE_READY:
        return []
    try:
        txt = _urlreq.urlopen(SGL_URL + "/metrics", timeout=10).read().decode("utf-8", "replace")
        rows = []
        for line in txt.splitlines():
            m = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)", line) if line and not line.startswith("#") else None
            if m and not m.group(1).endswith(("_bucket", "_created")) and any(
                    k in m.group(1) for k in ("num_running_reqs", "num_queue_reqs", "gen_throughput", "token_usage", "cache_hit_rate")):
                try:
                    rows.append({"name": m.group(1), "labels": m.group(2), "value": float(m.group(3))})
                except ValueError:
                    pass
        return rows
    except Exception as e:
        return [{"error": repr(e)[:300]}]


def _e13_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def e13_extras(vpy):
    """FA2 (QSA decode on sm89, upstream PR #36968) + FA4 restore + Franzen patch #3 with his SHAs."""
    if E11_HARNESS:
        return {"harness": True}
    names = (E13_FA2_WHEEL, E13_FA2_WHEEL.replace("+", ""))           # Kaggle drops '+' from file names
    cands = [d / n for d in E13_FA2_DIRS for n in names]
    if Path("/kaggle/input").is_dir():                                 # rev 6: any attached copy (found by name, checked by sha)
        cands += [f for n in names for f in sorted(Path("/kaggle/input").rglob(n))]
    w = next((c for c in cands if c.is_file() and _e13_sha(c) == E13_FA2_SHA), None)
    if w is None:
        raise E11Fail("FA2 wheel (sha256 74984f62...) not found under " + ", ".join(str(d) for d in E13_FA2_DIRS)
                      + " or /kaggle/input: attach foysalemonshanto/flash-attn-2-8-3-post1-cu13-torch2-13-cp312")
    fl = Path("/tmp/e13-fa2"); fl.mkdir(exist_ok=True)
    if not (fl / E13_FA2_WHEEL).exists():
        (fl / E13_FA2_WHEEL).symlink_to(w)
    pip = [sys.executable, "-m", "pip", "--python", str(vpy), "install", "--no-index", "--no-deps", "-q", "--disable-pip-version-check"]
    r = subprocess.run(pip + ["--find-links", "/tmp/e11-wheels", str(fl / E13_FA2_WHEEL)], capture_output=True, text=True)
    if r.returncode:
        raise E11Fail("FA2 install failed: " + (r.stderr or r.stdout)[-800:])
    fa4 = sorted(Path("/tmp/e11-wheels").glob("flash_attn_4-*.whl"))   # FA2 ships 21 old flash_attn/cute files: restore FA4's
    if fa4:
        r = subprocess.run(pip + ["--force-reinstall", str(fa4[0])], capture_output=True, text=True)
        if r.returncode:
            raise E11Fail("FA4 restore failed: " + (r.stderr or r.stdout)[-800:])
    site = Path(subprocess.run([str(vpy), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                               capture_output=True, text=True, check=True).stdout.strip())
    g = site / "sglang/srt/layers/quantization/gptq/schemes/gptq_moe.py"
    h = _e13_sha(g)
    if h == E13_PATCH3[0]:
        t = g.read_bytes(); assert t.count(b"dtype=torch.half") == 2
        tmp = g.with_name(g.name + ".e13tmp"); tmp.write_bytes(t.replace(b"dtype=torch.half", b"dtype=params_dtype")); os.replace(tmp, g)
        h = _e13_sha(g)
    if h != E13_PATCH3[1]:
        raise E11Fail(f"patch #3 after-sha mismatch {h[:12]}")
    probe = ("import json, torch, flash_attn\nfrom sglang.srt.layers.attention.qwen_sparse_attn_backend import "
             "_resolve_flash_attn_varlen_func as r\nprint(json.dumps({'fa2': flash_attn.__version__, 'resolves_to': r().__module__,"
             "'cxx11abi': torch.compiled_with_cxx11_abi()}))")
    r = subprocess.run([str(vpy), "-c", probe], capture_output=True, text=True, env=_E13_ENV, timeout=900)
    info = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"probe_error": (r.stderr or "")[-800:]}
    if not str(info.get("resolves_to", "")).startswith("flash_attn.") or "cute" in str(info.get("resolves_to")):
        raise E11Fail("QSA decode does not resolve to classic FA2: " + json.dumps(info)[:600])
    return info


E13_TOPK_SRC = _b64.b64decode("TOPK_B64").decode()

def e13_topk_test(vpy):
    """User-mandated: SGLang's JIT fast_topk<512> (the QSA block top-k) vs a tie-aware torch oracle (#36807). Recorded only."""
    if not E13_TOPK_TEST or E11_HARNESS:
        return {"skipped": True}
    p = OUT / "mp_topk.py"; p.write_text(E13_TOPK_SRC, encoding="utf-8")
    try:
        r = subprocess.run([str(vpy), "-I", str(p), "--out", str(OUT / "e13_topk.json"), "--fixtures", str(OUT / "topk_fixtures")],
                           capture_output=True, text=True, env=_E13_ENV, timeout=1500)
        (OUT / "e13_topk.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
        rep = json.loads((OUT / "e13_topk.json").read_text())
        return {k: (rep.get(k) or {}).get("summary") if isinstance(rep.get(k), dict) else rep.get(k) for k in ("t1", "t2")} | {
            "graph": rep.get("graph"), "errors": [e[:300] for e in rep.get("errors", [])]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:600]}


def e13_host_ram_guard():
    ple = sum(f.stat().st_size for f in SGL_MODEL_PATH.glob("model-ple*.safetensors"))   # BF16 model-ple-*, FP8 model-plefp8-*
    avail = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                avail = int(line.split()[1]) * 1024
    except Exception:
        pass
    cg = None                                          # rev 6b (B-L2): the cgroup limit can be below MemAvailable
    try:
        mx = open("/sys/fs/cgroup/memory.max").read().strip()
        if mx.isdigit():
            cg = int(mx) - int(open("/sys/fs/cgroup/memory.current").read().strip())
    except Exception:
        pass
    eff = min(x for x in (avail, cg) if x is not None) if (avail is not None or cg is not None) else None
    need = ple + 20 * 10 ** 9
    return {"ple_gb": round(ple / 1e9, 1), "mem_available_gb": round(avail / 1e9, 1) if avail else None,
            "cgroup_free_gb": round(cg / 1e9, 1) if cg is not None else None,
            "need_gb": round(need / 1e9, 1), "ok": E11_HARNESS or (eff is not None and eff >= need)}


def e13_flags():
    """Franzen's launch (his notebook cell 12) with the L4 deltas only (PLAN rev 4.1 sections 2 and 3b)."""
    run = SGL_SERVER_MAX_RUNNING
    gb = [n for n in sorted({1, 2, 4, 7, 8, 9, 10, run}) if n <= run]
    f = ["--model-path", str(SGL_MODEL_PATH), "--load-format", "safetensors",
         "--model-loader-extra-config", json.dumps({"enable_multithread_load": False}),
         "--served-model-name", "flashnext", "--host", "127.0.0.1", "--port", str(SGL_PORT),
         "--tensor-parallel-size", "4", "--ep-size", "4",                                               # D4
         "--dtype", "bfloat16", "--quantization", "auto-round", "--kv-cache-dtype", "fp8_e4m3",
         "--mem-fraction-static", E13_MEM_FRACTION, "--context-length", str(SGL_CONTEXT_LEN), "--page-size", "64",
         "--max-running-requests", str(run), "--chunked-prefill-size", str(E13_CHUNK), "--max-prefill-tokens", "16384",
         "--cuda-graph-max-bs-decode", str(run), "--cuda-graph-bs-decode", *map(str, gb),
         "--mamba-ssm-dtype", "bfloat16", "--max-mamba-cache-size", str(run * 5),                     # extra_buffer: 5 slots
         "--mamba-radix-cache-strategy", "extra_buffer", "--mamba-track-interval", "64",
         "--mamba-backend", "triton", "--linear-attn-decode-backend", "triton", "--linear-attn-prefill-backend", "triton",   # D5
         "--moe-runner-backend", "auto", "--ple-offload-embedding", "--trust-remote-code",
         "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
         "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
         "--default-chat-template-kwargs", json.dumps({"preserve_thinking": True}),
         "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
         "--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging",
         "--weight-loader-prefetch-checkpoints", "--weight-loader-prefetch-num-threads", "1",   # rev 6b (B-H1): 4 shards in flight
         "--weight-loader-drop-cache-after-load"]
    if (SGL_MODEL_PATH / "chat_template.jinja").exists():
        f += ["--chat-template", str(SGL_MODEL_PATH / "chat_template.jinja")]
    return f


def _e13_caps(log):
    tp0 = "\n".join(l for l in log.splitlines() if "TP0" in l) or log
    kv = re.findall(r"KV Cache is allocated\. dtype: ([\w.]+), #tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB", tp0)
    lw = re.findall(r"Load weight end.*?avail mem=([\d.]+) GB, mem usage=([\d.]+) GB", tp0)
    cap = re.findall(r"max_running_requests is capped to (\d+)", log)
    gr = re.findall(r"Capture [\w ]*?(?:cuda|CUDA) graph end\..*?mem usage=([\d.]+) GB, avail mem=([\d.]+) GB", tp0)
    mb = re.findall(r"Mamba Cache is allocated\. max_mamba_cache_size: (\d+), conv_state size: ([\d.]+)GB, ssm_state size: ([\d.]+)GB", tp0)
    pe = re.findall(r"Memory pool end\. avail mem=([\d.]+) GB", tp0)
    return {"mamba_slots": int(mb[0][0]) if mb else None, "mamba_gb": round(float(mb[0][1]) + float(mb[0][2]), 2) if mb else None,
            "kv_gb": round(float(kv[0][2]) + float(kv[0][3]), 2) if kv else None, "pool_end_avail_gb": float(pe[0]) if pe else None,
            "kv_dtype": kv[0][0] if kv else None, "kv_tokens": int(kv[0][1]) if kv else None,
            "post_weight_avail_gb": float(lw[0][0]) if lw else None, "weight_gb": float(lw[0][1]) if lw else None,
            "capped_running": min(int(x) for x in cap) if cap else None,
            "post_graph_avail_gb": float(gr[-1][1]) if gr else None, "oom": "out of memory" in log.lower()}


def e13_smoke():
    """'!!!!' (the E3 symptom) or a wrong continuation = not ready: raw greedy counting + one chat answer."""
    def post(path, body, timeout):
        req = _urlreq.Request(SGL_URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with _urlreq.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    g = post("/generate", {"text": "1, 2, 3, 4, 5, 6, 7, 8,", "sampling_params": {"max_new_tokens": 32, "temperature": 0}}, 600)
    text = g.get("text") or ""
    count_ok = text.strip().startswith("9, 10, 11")
    c = post("/v1/chat/completions", {"model": "flashnext", "max_tokens": 4096,
                                      "messages": [{"role": "user", "content": "What is 2 + 2? Answer with the number only."}]}, 1200)
    m = (c.get("choices") or [{}])[0].get("message") or {}
    ans = (m.get("content") or "") + " " + (m.get("reasoning_content") or "")
    garbage = lambda t: bool(t.strip()) and max(t.count(ch) for ch in set(t.strip())) / len(t.strip()) > 0.5 and len(t.strip()) > 8
    out = {"count_text": text[:80], "count_ok": count_ok, "chat_content": (m.get("content") or "")[:120],
           "chat_has_4": "4" in ans, "garbage": garbage(text) or garbage(m.get("content") or ""), "usage": c.get("usage")}
    out["ok"] = count_ok and out["chat_has_4"] and not out["garbage"]
    return out


E13_SPEED_REF = {"27b_dflash_w6_s16_8k": 583, "27b_dflash_w6_s16_40k": 521,     # EMS SAFE 2026-10-26 (decode lines)
                 "27b_dflash_w6_s16_xl90k": 370,
                 "note": "27B at 16 streams with DFlash; Flash-Next here runs <= 4 streams without NEXTN: not comparable 1:1"}


def _e13_garbage(t):
    t = (t or "").strip()
    return bool(t) and len(t) > 8 and max(t.count(ch) for ch in set(t)) / len(t) > 0.5


def e13_speed():
    """rev 6 (user 2026-10-26): a 10-15 min speed stage right after the smoke, before the Duck run. Recorded, never gating.
    Short band: 1 / 2 / 3 / 4 concurrent requests (as KV and the server cap allow) on ~6k-token ARC evaluation prompts;
    long band: as many ~30k-token prompts as KV holds (max 4). Each request: max_new 1024, ignore_eos. Per level: wall,
    completion tokens, aggregate tok/s (incl. prefill), the median decode-line rate and running count from the server log
    after the level started, and the max running seen on /metrics."""
    if RERUN or PROFILE != "SMOKE1":                   # rev 6b (A-L5): never in a rerun / other profile
        return {"skipped": f"profile {PROFILE} rerun {RERUN}"}
    # rev 6b (A-L2): never into the Duck window: the stage ends 2 min before (solve window + stop margin) before HARD_STOP
    _room = HARD_STOP - time.perf_counter() - float(globals().get("E12_SOLVE_WINDOW_SEC") or 15 * 60) \
        - float(globals().get("E12_STOP_BEFORE_HARD_SEC") or 180) - 120
    budget = max(0.0, min(15 * 60 if not E11_HARNESS else 90, _room))
    if budget < (120 if not E11_HARNESS else 10):
        return {"skipped": f"no room before the Duck window ({round(_room)} s)"}
    out = {"budget_sec": round(budget), "levels": {}, "long": {}, "ref_27b": E13_SPEED_REF}
    t_end = time.perf_counter() + budget
    kv = (ENGINE_INFO["e13"].get("caps") or {}).get("kv_tokens") or 0
    cap = min(SGL_SERVER_MAX_RUNNING, (ENGINE_INFO["e13"].get("caps") or {}).get("capped_running") or SGL_SERVER_MAX_RUNNING)
    try:
        allt = _load_json(ARC_ROOT / "arc-agi_evaluation_challenges.json")
    except Exception:
        allt = dict(CHALLENGES)

    def rows(g):
        return "\n".join(" ".join(str(v) for v in r) for r in g)

    def text(tk):
        s = "".join(f"\nExample {i + 1}\nInput:\n{rows(q['input'])}\nOutput:\n{rows(q['output'])}\n" for i, q in enumerate(tk["train"]))
        return s + "".join(f"\nTest input {j + 1}:\n{rows(q['input'])}\n" for j, q in enumerate(tk["test"]))

    def ids_of(content):
        return tokenizer.encode(tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                              add_generation_prompt=True), add_special_tokens=False)
    ask = "Find the transformation rule of this ARC-AGI-2 task, then give the test output grids.\n"
    keys = sorted(allt)
    short = [ids_of(ask + text(allt[k])) for k in keys[:4]]
    blk, n, longp = [], 0, []
    for k in keys[4:]:
        blk.append(text(allt[k]))
        n += len(tokenizer.encode(blk[-1], add_special_tokens=False))
        if n >= (30000 if not E11_HARNESS else 3000):
            longp.append(ids_of("Reference tasks:\n" + "\n".join(blk) + "\n" + ask))
            blk, n = [], 0
            if len(longp) == 4:
                break
    out["prompt_tokens"] = {"short": [len(x) for x in short], "long": [len(x) for x in longp]}
    new = 1024 if not E11_HARNESS else 64

    def one(ids, res, i):
        body = {"input_ids": ids, "sampling_params": {"max_new_tokens": new, "ignore_eos": True}}
        req = _urlreq.Request(SGL_URL + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with _urlreq.urlopen(req, timeout=max(30.0, t_end - time.perf_counter())) as r:
                j = json.loads(r.read())
            j = j[0] if isinstance(j, list) else j
            res[i] = {"tokens": (j.get("meta_info") or {}).get("completion_tokens") or len(j.get("output_ids") or []),
                      "sec": round(time.perf_counter() - t0, 1), "text": (j.get("text") or "")[:300],   # rev 6b (B-H2)
                      "garbage": _e13_garbage(j.get("text") or "")}
        except Exception as e:
            res[i] = {"error": repr(e)[:200], "sec": round(time.perf_counter() - t0, 1)}

    def level(prompts, tag):
        if time.perf_counter() > t_end - 30:
            return {"skipped": "speed budget"}
        off = len(E11_SERVER.log_text()) if E11_SERVER is not None else 0
        res, ths, mx = [None] * len(prompts), [], [0]
        stop = threading.Event()

        def sampler():                                    # /metrics directly (metrics_snapshot() is empty before ENGINE_READY)
            while not stop.is_set():
                try:
                    txt = _urlreq.urlopen(SGL_URL + "/metrics", timeout=5).read().decode("utf-8", "replace")
                    for v in re.findall(r"^sglang:num_running_reqs(?:\{[^}]*\})?\s+(\S+)", txt, re.M):
                        mx[0] = max(mx[0], int(float(v)))
                except Exception:
                    pass
                stop.wait(1.0)
        st = threading.Thread(target=sampler, daemon=True); st.start()
        t0 = time.perf_counter()
        for i, p in enumerate(prompts):
            th = threading.Thread(target=one, args=(p, res, i), daemon=True); th.start(); ths.append(th)
        for th in ths:
            th.join(max(1.0, t_end - time.perf_counter()))
        wall = time.perf_counter() - t0
        stop.set()
        lines = (E11_SERVER.log_text()[off:] if E11_SERVER is not None else "").splitlines()
        gen = [float(x) for l in lines if "TP0" in l and "Decode batch" in l
               for x in re.findall(r"gen throughput \(token/s\): ([\d.]+)", l)]
        run = [int(x) for l in lines if "TP0" in l and "Decode batch" in l for x in re.findall(r"#running-req: (\d+)", l)]
        tok = sum((r or {}).get("tokens") or 0 for r in res)
        o = {"streams": len(prompts), "wall_sec": round(wall, 1), "completion_tokens": tok,
             "agg_tok_s_incl_prefill": round(tok / wall, 1) if wall > 0 else None,
             "decode_line_tok_s_p50": _statistics.median(gen) if gen else None, "decode_lines": len(gen),
             "running_p50": _statistics.median(run) if run else None, "metrics_running_max": mx[0],
             "errors": [r.get("error") for r in res if r and r.get("error")], "unfinished": sum(1 for r in res if r is None),
             "garbage": sum(1 for r in res if r and r.get("garbage")), "text_sample": ((res[0] or {}).get("text") or "")[:300]}
        print(f"E13 speed {tag}: {json.dumps(o)}", flush=True)
        return o
    _w = [None]                                           # rev 6b (A-L3): one unmeasured warm-up (JIT / autotune)
    one(short[0][:2000], _w, 0) if short else None
    out["warmup"] = _w[0]
    for L in (1, 2, 3, 4):
        if L > cap or (kv and L * (max(out["prompt_tokens"]["short"]) + new + 512) > kv):
            out["levels"][str(L)] = {"skipped": f"cap {cap} / KV {kv}"}
            continue
        out["levels"][str(L)] = level(short[:L], f"short x{L}")
    if longp:
        Ll = max(0, min(len(longp), cap, kv // (max(len(x) for x in longp) + new + 512) if kv else len(longp)))
        out["long"] = level(longp[:Ll], f"long x{Ll}") if Ll else {"skipped": f"KV {kv} holds no long prompt"}
    out["sec"] = round(time.perf_counter() - (t_end - budget), 1)
    return out


def e13_bring_up():
    global E11_SERVER, _E13_ENV
    info = ENGINE_INFO["e13"]
    info["stage"] = "install"
    vpy, _E13_ENV, info["install"] = e11_install()
    _E13_ENV.update({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
                     "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_NUMA_BIND_V2": "false", "SGLANG_MM_PREPROCESS_DEVICE": "cpu",
                     "OMP_NUM_THREADS": "8", "TORCH_CUDA_ARCH_LIST": "8.9"})                          # Franzen env; D6
    info["stage"] = "fa2_patch3"; info["extras"] = e13_extras(vpy)
    info["stage"] = "topk_test"; info["topk"] = e13_topk_test(vpy)
    print("E13 top-k test:", json.dumps(info["topk"], default=str)[:1500], flush=True)
    info["stage"] = "host_ram"; info["host_ram"] = e13_host_ram_guard()
    if not info["host_ram"]["ok"]:
        raise E11Fail("host RAM guard: " + json.dumps(info["host_ram"]))
    info["gpu_free_before"] = _e11_gpu_free(180)
    info["stage"] = "launch"; info["flags"] = e13_flags()
    launch = [str(vpy), "-m", "sglang.launch_server"] if not E11_HARNESS else [sys.executable, os.environ["E11_FAKE_SERVER"]]
    E11_SERVER = E11Server(launch + info["flags"], _E13_ENV, "e13")
    E11_SERVER.start()
    print("E13: starting SGLang Flash-Next:", " ".join(info["flags"]), flush=True)
    deadline = time.perf_counter() + max(300, min(55 * 60, ADMIT_DEADLINE - time.perf_counter() - 20 * 60))   # rev 6b: 45 -> 55
    last, last_t = -1, time.perf_counter()
    while True:                                          # /health only while the launcher lives; 15 min log silence = stall
        if not E11_SERVER.alive():
            raise E11Fail(f"server exited rc={E11_SERVER.proc.returncode}: " + E11_SERVER.log_text()[-3000:])
        try:
            _urlreq.urlopen(SGL_URL + "/health", timeout=10).read(); break
        except Exception:
            pass
        sz = E11_SERVER.log_path.stat().st_size if E11_SERVER.log_path.exists() else 0
        if sz != last:
            last, last_t = sz, time.perf_counter()
        if time.perf_counter() > deadline or time.perf_counter() - last_t > 15 * 60:
            raise E11Fail("server not ready (deadline or 15 min log silence): " + E11_SERVER.log_text()[-3000:])
        time.sleep(2 if E11_HARNESS else 5)
    info["ready_sec"] = round(time.perf_counter() - E11_SERVER.t0, 1)
    info["caps"] = _e13_caps(E11_SERVER.log_text())
    print("E13 caps:", json.dumps(info["caps"]), flush=True)
    info["stage"] = "smoke"; info["smoke"] = e13_smoke()
    print("E13 smoke:", json.dumps(info["smoke"], default=str)[:800], flush=True)
    if not info["smoke"]["ok"]:
        raise E11Fail("smoke failed: " + json.dumps(info["smoke"], default=str)[:600])
    info["stage"] = "speed"
    try:                                                  # rev 6: recorded, never gating (a failure here never stops the Duck run)
        info["speed"] = e13_speed()
    except Exception as _se:
        info["speed"] = {"error": f"{type(_se).__name__}: {_se}"[:600]}
    print("E13 speed:", json.dumps(info["speed"], default=str)[:2500], flush=True)
    # rev 6b (A-H1 / B-L4): the Duck run never starts on a dead or broken server
    if not info["speed"].get("skipped"):
        info["post_speed_smoke"] = e13_smoke() if E11_SERVER is not None and E11_SERVER.alive() else {"ok": False, "dead": True}
        print("E13 post-speed smoke:", json.dumps(info["post_speed_smoke"], default=str)[:600], flush=True)
        if not info["post_speed_smoke"].get("ok"):
            raise E11Fail("engine lost or broken during the speed stage: " + json.dumps(info["post_speed_smoke"], default=str)[:400])
    kv = info["caps"].get("kv_tokens") or 0
    run = min(SGL_MAX_RUNNING, info["caps"].get("capped_running") or SGL_MAX_RUNNING)
    return max(1, min(run, kv // E13_KV_PER_STREAM))          # pre-run capacity rule: resident contexts <= KV / 60k


_E13_ENV = None
_t_engine = time.perf_counter()
if not E10_SANDBOX_SELFTEST.get("ok", False):
    ENGINE_INFO["e13"]["stage"] = "skipped: sandbox self-test failed"
else:
    try:
        CONCURRENCY = e13_bring_up()
        ENGINE_READY = True
        ENGINE_INFO.update(backend="sglang-0.5.21-flashnext-w4a16-ep4-fa2", kv_tokens=ENGINE_INFO["e13"]["caps"].get("kv_tokens"),
                           max_model_len=SGL_CONTEXT_LEN)
        ENGINE_INFO["e13"]["stage"] = "ready"
    except Exception as _e:
        ENGINE_INFO["e13"]["error"] = f"{type(_e).__name__}: {_e}"[:3000]
        ENGINE_INFO["fatal"] = ENGINE_INFO["e13"]["error"][:300]
        print("E13 engine NOT ready:", ENGINE_INFO["e13"]["stage"], "|", ENGINE_INFO["e13"]["error"][:1500], flush=True)
        if E11_SERVER is not None:
            ENGINE_INFO["e13"]["server_log_tail"] = E11_SERVER.log_text()[-4000:]
        ENGINE_READY = False; CONCURRENCY = 0
        # keep E11_SERVER (CELL 7 bundles its log); the process is stopped by CELL 7's e11_kill_server()
ENGINE_INFO["load_sec"] = round(time.perf_counter() - _t_engine, 1)
ENGINE_INFO["concurrency"] = int(CONCURRENCY)
print("engine:", json.dumps(ENGINE_INFO, default=str)[:3000], flush=True)
'''
_enc = ((lambda b: __import__("base64").b64encode(__import__("zlib").compress(b, 9)).decode()) if RC1 else
        (lambda b: __import__("base64").b64encode(b).decode()))     # RC1: zlib payloads (notebook < 400 kB, commit limit)
if RC1:
    C5 = once(C5, 'E13_TOPK_SRC = _b64.b64decode("TOPK_B64").decode()',
              'E13_TOPK_SRC = __import__("zlib").decompress(_b64.b64decode("TOPK_B64")).decode()', "topk zlib")
    PROBE_SRC = once(PROBE_SRC, 'E13_MEMTRACE_SRC = _b64.b64decode("MEMTRACE_B64").decode()',
                     'E13_MEMTRACE_SRC = __import__("zlib").decompress(_b64.b64decode("MEMTRACE_B64")).decode()', "memtrace zlib")
    # unused in RC1 (no FP8 PLE arm, no registry check, no long test): keep the notebook small
    for _a, _b in (("def e13_fp8ple_dir():", "def e13_memtrace_summary(d):"), ("def e13_vllm_registry():", "_E13_ENV = None"),
                   ("def e13_long_test(deadline, spec):", "def e13_probe_arm("),
                   ("def e13_probe_smoke(deadline, cfg):", "E13_PLE_FLAG_PATH = ")):   # every RC1 arm uses ple_modes
        _i, _j = PROBE_SRC.index(_a), PROBE_SRC.index(_b)
        PROBE_SRC = PROBE_SRC[:_i] + PROBE_SRC[_j:]
C5 = C5.replace("TOPK_B64", _enc(TOPK_SCRIPT.encode()))
PROBE_SRC = PROBE_SRC.replace("VLLMGC_B64", __import__("base64").b64encode(__import__("zlib").compress(VLLM_GC_SCRIPT.encode(), 9)).decode())
if LOADPROBE:   # the probe replaces the bring-up block (from `_E13_ENV = None` to the end); every helper above stays
    _cut = C5.index("\n_E13_ENV = None\n_t_engine = time.perf_counter()")
    C5 = C5[:_cut] + "\n" + PROBE_SRC.replace("MEMTRACE_B64", _enc(MEMTRACE_SCRIPT.encode()))

nb["cells"][0]["source"] = ((("# E13 RC1 — Qwen3.8-Flash-Next on 4xL4: root cause of the layer-2 MoE NaN (SGLang 0.5.21) + vLLM 0.32 smoke\n"
                              "NANPROBE rev 2 put the first NaN in model.layers.2.mlp.experts (GPTQMarlinMoE, finite float inputs) "
                              "on all ranks. This notebook (RC1 fix list, 3 GPT audits): arm 1 SGLang eager with hooks on layers 0-3, "
                              "expert weight scans after the repack and after forward 0, every MoE input incl. top-k ids, MoE "
                              "fixtures of layers 1-2 recomputed on the CPU from the checkpoint's GPTQ weights (Marlin vs reference), "
                              "the runtime gptq_moe source sha, and the count prompt under three PLE modes (orig / sync / skip) in the "
                              "same server; arm 2 Flash-Next on vLLM 0.32.0 (native qwen4_exp, TP4 + EP, eager, PLE in pinned host "
                              "memory): count, 2+2 chat, 4 streams; arm 3 (time permitting) layer 2 on the non-Marlin GPTQ path. "
                              "No Duck; the submission is CELL 0's fallback.\n\n") if RC1 else
                             ("# E13 NANPROBE — Qwen3.8-Flash-Next on SGLang 0.5.21 (4xL4): output correctness + KV capacity\n"
                              "This version (rev 2, NANPROBE_FIX_LIST 2026-10-30: 2 GPT + 2 agent audits): the LOADPROBE "
                              "notebook plus the gc fix (repack OOM); an eager launch with a NaN / collapse locator (instance-forward "
                              "wrapper, per rank: first non-finite module with origin vs propagated, layer [min, max], logits "
                              "stats, QSA indexer ints; the count again after the hooks); a KV launch changing only mem fraction "
                              "0.93 -> 0.95 (mamba 20 = 4 x 5) with a running-cap gate and 4 x 20k-token prompts; after the arms a "
                              "vLLM 0.32 registry check. No Duck; the submission is CELL 0's fallback.\n\n") if NANPROBE else
                             ("# E13 LOADPROBE — Qwen3.8-Flash-Next load-only memory probe on SGLang 0.5.21 (4xL4)\n"
                              "Two launches with the rev 6b flags and a GPU memory trace: BF16 PLE (Franzen's package, rev 6b exactly) "
                              "and FP8 PLE (the E3 v4.2.1 merge). No speed stage, no Duck; the submission is CELL 0's fallback.\n\n"))
                            if LOADPROBE else "") + ("# E13 — Duck REPL agent for ARC-AGI-2 on SGLang 0.5.21 + Qwen3.8-Flash-Next (Intel W4A16, BF16 PLE) "
                            f"({'EVAL30' if EVAL30 else 'SMOKE1'})\n"
                            "The E12 Duck notebook unchanged except the engine (CELL 5) and its inputs/limits in CELL 1. "
                            "Engine = Franzen's ARC-AGI-3 milestone-2 launch with the 4xL4 deltas (PLAN rev 4.1 simple path): "
                            "0.5.21, TP4+EP4, Triton GDN, arch 8.9, classic FA2 for QSA decode, Franzen patch #3; memory items "
                            "NEXTN off, chunk 2048, mem 0.93, 4 running; Duck streams = min(4, KV // 60k).")
nb["cells"][2]["source"] = c1
_c3 = cells[3]
_c3 = once(_c3, '# S1: read the model files into the page cache in the background while pip runs (read-only, best effort)\nif os.environ.get("Q27X_SKIP_INSTALL") != "1":',
           '# S1: read the model files into the page cache in the background while pip runs (read-only, best effort)\n'
           '# E13 rev 6b (audit B-H1): skipped; with 102 GB of pinned PLE the 178 GB cat evicts the loader\'s prefetched pages\n'
           'if os.environ.get("Q27X_SKIP_INSTALL") != "1" and not globals().get("E13_SKIP_CAT_PREFETCH"):', "cat skip")
nb["cells"][3]["source"] = _c3
nb["cells"][6]["source"] = C5
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []; c["execution_count"] = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        compile(src(c), f"cell{i}", "exec")
unchanged = [i for i in range(len(cells)) if src(nb["cells"][i]) == cells[i]]
data = json.dumps(nb, indent=1).encode("utf-8")
OUT_NB.write_bytes(data)
print("wrote", OUT_NB.name, "sha256", hashlib.sha256(data).hexdigest(), "| unchanged cells", unchanged, "of", len(cells))
