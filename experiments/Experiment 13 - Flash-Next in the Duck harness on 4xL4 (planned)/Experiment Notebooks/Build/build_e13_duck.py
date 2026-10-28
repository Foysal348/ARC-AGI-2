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
LOADPROBE = "--loadprobe" in sys.argv
assert not (EVAL30 and LOADPROBE)
BASE = E12DIR / ("Q27-XHIGH-E12_DUCK_REPL_SGLANG_EVAL30.ipynb" if EVAL30 else "Q27-XHIGH-E12_DUCK_REPL_SGLANG_SMOKE1.ipynb")
BASE_SHA = ("b65fb9dcc80b3652dc18b37d43355743311e2b55db8308ebc766d9d0b23de456" if EVAL30
            else "0fea4d0cb50fe85ec76a4ac27d6de32997aa580b0f5a2d7b36400c0be645c186")
OUT_NB = HERE.parent / f"Q27-XHIGH-E13_DUCK_FLASHNEXT_{'EVAL30' if EVAL30 else 'SMOKE1'}.ipynb"
if LOADPROBE:
    OUT_NB = HERE.parent / "Q27-XHIGH-E13_FLASHNEXT_LOADPROBE.ipynb"
# never overwrite a notebook that ran on Kaggle (memory: never-rebuild-run-notebooks); a rebuild goes to *_REBUILD.ipynb
PUSHED = {"74411bc070e9de40df066516b4096f5549ee62251d51432ca5f04a8f4feb3b6d",   # foysalemonshanto v1 (2026-10-26, SMOKE1 rev 6b)
          "107aa18227ffbc58cc725bb4dc13020a47848dcc8e5ef9f718094d2156ebf5e0"}   # foysalemonshanto v1 (2026-10-27, LOADPROBE)
if OUT_NB.exists() and hashlib.sha256(OUT_NB.read_bytes()).hexdigest() in PUSHED:
    OUT_NB = OUT_NB.with_name(OUT_NB.stem + "_REBUILD.ipynb")
TOPK_SCRIPT = (HERE / "e13_scripts" / "mp_topk.py").read_text(encoding="utf-8")
MEMTRACE_SCRIPT = (HERE / "e13_scripts" / "mp_memtrace.py").read_text(encoding="utf-8").replace("\r\n", "\n")
PROBE_SRC = r'''
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
        out[f.stem] = s
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
    tdir = OUT / "e13_memtrace" / tag
    env = dict(_E13_ENV); env["PYTHONPATH"] = str(hookdir); env["E13_MEMTRACE_DIR"] = str(tdir)
    rec["flags"] = e13_flags()
    launch = [str(vpy), "-m", "sglang.launch_server"] if not E11_HARNESS else [sys.executable, os.environ["E11_FAKE_SERVER"]]
    E11_SERVER = E11Server(launch + rec["flags"], env, f"e13_{tag}")
    E11_SERVER.start()
    print(f"E13 LOADPROBE {tag}: launch on {model_dir}", flush=True)
    t_end, last, last_t = time.perf_counter() + budget, -1, time.perf_counter()
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
    if rec["outcome"] == "ready":
        try:
            rec["smoke"] = e13_smoke()
        except Exception as e:
            rec["smoke"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
    E11_SERVER.kill()
    rec["gpu_free_after"] = _e11_gpu_free(180)
    rec["memtrace"] = e13_memtrace_summary(tdir)
    print(f"E13 LOADPROBE {tag}:", json.dumps({k: rec.get(k) for k in ("outcome", "sec", "caps", "oom_lines", "smoke")},
                                             default=str)[:1500], flush=True)
    print(f"E13 LOADPROBE {tag} memtrace:", json.dumps(rec["memtrace"], default=str)[:4000], flush=True)
    return rec


_E13_ENV = None
_t_engine = time.perf_counter()
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
        for _tag in E13_PROBE_ARMS:
            _room = HARD_STOP - time.perf_counter() - 5 * 60
            if _room < 8 * 60:
                E13_PROBE["arms"].append({"tag": _tag, "outcome": f"skipped: {round(_room)} s left"}); continue
            _info["stage"] = f"probe {_tag}"
            try:
                if _tag == "bf16ple":
                    _md = E13_INTEL_DIR
                else:
                    _md, E13_PROBE["fp8ple_merge"] = e13_fp8ple_dir()
                E13_PROBE["arms"].append(e13_probe_arm(_tag, _md, _vpy, _hook, min(25 * 60, _room)))
            except Exception as _ae:
                E13_PROBE["arms"].append({"tag": _tag, "outcome": f"error: {type(_ae).__name__}: {_ae}"[:800]})
                if E11_SERVER is not None:
                    E11_SERVER.kill()
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
              + ("75" if LOADPROBE else "105") + ' * 60,', "smoke budget")   # rev 6: +15 min speed stage; LOADPROBE 75 min (ceiling; ~35-40 min expected)
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
    return {"kv_dtype": kv[0][0] if kv else None, "kv_tokens": int(kv[0][1]) if kv else None,
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
C5 = C5.replace("TOPK_B64", __import__("base64").b64encode(TOPK_SCRIPT.encode()).decode())
if LOADPROBE:   # the probe replaces the bring-up block (from `_E13_ENV = None` to the end); every helper above stays
    _cut = C5.index("\n_E13_ENV = None\n_t_engine = time.perf_counter()")
    C5 = C5[:_cut] + "\n" + PROBE_SRC.replace("MEMTRACE_B64", __import__("base64").b64encode(MEMTRACE_SCRIPT.encode()).decode())

nb["cells"][0]["source"] = ("# E13 LOADPROBE — Qwen3.8-Flash-Next load-only memory probe on SGLang 0.5.21 (4xL4)\n"
                            "Two launches with the rev 6b flags and a GPU memory trace: BF16 PLE (Franzen's package, rev 6b exactly) "
                            "and FP8 PLE (the E3 v4.2.1 merge). No speed stage, no Duck; the submission is CELL 0's fallback.\n\n"
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
