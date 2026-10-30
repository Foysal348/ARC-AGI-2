"""E24 SGLang speed-track GPU notebook builder, rev 1 (DESIGN.md rev 2 + BUILD_SPEC rev 2, 2026-10-30).

Cloned from the E23 builder (build_e23_notebook.py rev 3b: CELL 0 markdown, CELL 1 config / inputs / modules, CELL 2
engines, CELL 3 prompts, CELL 4 step runner, CELL 5 the plan, CELL 6 report; MODE / PART baked per notebook because a
Kaggle commit cannot set env; RERUN guard; compile() of every code cell; nbformat 4.5; sha8 + kB printed; < 400 kB).
E23 defects not copied (e23_template.md §13.6): TMP is created; the flag check is ENFORCED (exact 0.5.21 option strings);
R["errors"] is written on every caught exception and an engine install failure blocks only that engine's steps; every
embedded module sha is recorded; no DONE default status; the full argv is stored; per-arm cache isolation; label-aware
metrics (lab); repo-relative paths only; budget-gated steps only; find() refuses ambiguous mounts.
Engine layer = the EMS / E15B engine cell (venv install = e11_install, E14 /dev/shm staging, e11_flags + EMS arm edits,
asserted against DESIGN §2 BASE); vLLM 0.32 oracle arms = E23 install_stock + stock_flags(seqs=16, spec=0, prefix=False).

Outputs (identical code, only the baked MODE / PART differ):
  Experiment Notebooks/Q27-XHIGH-E24_SGLANG_SPEED_SMOKE.ipynb    (MODE=SMOKE, PART=ALL; 210 min)
  Experiment Notebooks/Q27-XHIGH-E24_SGLANG_SPEED_FULL_A.ipynb   (MODE=FULL,  PART=A: sections 1 + 6; 480 min)
  Experiment Notebooks/Q27-XHIGH-E24_SGLANG_SPEED_FULL_B.ipynb   (MODE=FULL,  PART=B: sections 2-5; 690 min)
  Experiment Notebooks/Build/plan_argvs_e24.json                 (every planned server argv per variant, both branches)
  Experiment Notebooks/Build/kernels/<variant>/kernel-metadata.json

usage: /opt/homebrew/bin/python3.12 build_e24_notebook.py [--allow-missing]
  --allow-missing builds even when a parallel work-package module is absent (the notebook then stops at CELL 1 with
  MODULE_MISSING_AT_BUILD); without it a missing module is a build error naming the files.
"""
import ast
import base64
import gzip
import hashlib
import io
import json
import sys
import textwrap
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
E24 = HERE.parents[1]
SRC = E24 / "src"
E14MOD = E24 / "Base Engine" / "src_e14_modules"
ALLOW_MISSING = "--allow-missing" in sys.argv[1:]
MODULES = {                     # key in R["shas"] -> (file written next to the lab at runtime, repo source)
    "lab": ("e24_lab.py", SRC / "e24_lab.py"),
    "patches": ("e24_sglang_patches.py", SRC / "e24_sglang_patches.py"),
    "hook": ("e24_hook.py", SRC / "e24_hook.py"),
    "unit": ("e24_unit_41736_41737.py", SRC / "e24_unit_41736_41737.py"),
    "loadpath": ("e14_loadpath.py", E14MOD / "e14_loadpath.py"),
    "trace_summary": ("e14_trace_summary.py", E14MOD / "e14_trace_summary.py"),
}
MISSING = [str(p) for _, p in MODULES.values() if not p.is_file()]
if MISSING and not ALLOW_MISSING:
    raise SystemExit("E24 builder: missing module(s) (parallel work packages not finished?):\n  " + "\n  ".join(MISSING)
                     + "\nRe-run when they exist, or pass --allow-missing for a structure-only build.")


def _text(p):
    return p.read_text(encoding="utf-8").replace("\r\n", "\n") if p.is_file() else None


SOURCES = {k: _text(p) for k, (_, p) in MODULES.items()}
SHAS = {k: (hashlib.sha256(v.encode("utf-8")).hexdigest() if v is not None else "MISSING") for k, v in SOURCES.items()}
SHAS["notebook_builder"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
# Every module is embedded as gzip+base64 with its sha256 checked on decode. The lab too: BUILD_SPEC asked for plain text
# (repr), but the rev-2 lab is ~145 kB (E23's was 43 kB) and plain text pushes the notebook past the 400 kB commit limit.
BLOBS = {k: (MODULES[k][0], SHAS[k], base64.b64encode(gzip.compress(SOURCES[k].encode("utf-8"), 9, mtime=0)).decode("ascii")
             if SOURCES[k] is not None else "")
         for k in ("lab", "patches", "hook", "unit", "loadpath", "trace_summary")}

MD0 = """# E24 — SGLang speed track on 4×L4 (SGLang 0.5.21, Qwen3.8-27B) · rev 1 (DESIGN rev 2) · __VARIANT__

Goal: 1500+ tok/s sustained accepted-output decode for Qwen3.8-27B on Kaggle 4×L4 with SGLang 0.5.21 (RedHat INT4 W4A16
+ DFlash2), with the SGLang accuracy gap explained first. Six sections (README / DESIGN.md rev 2):
1. **Accuracy-gap decomposition**: vLLM 0.32 FP8 / INT4 target-only oracles, SGLang target-only (fp16 / fp32 SSM),
   DFlash W6 / W4 / forced-reject, cold / warm / cache-off, multi-boot A/A; M1 greedy rows, M2 c16, M3 FP8-generated
   teacher-forced traces up to 120k, M4 decode vs prefill, M5 rank probe, M6 tool-call probe, M7 cache isolation.
2. **DFlash width × streams** (W4 / W6 × S16 / S20, W5 or W7 by rule), equal-N windows, parity rows.
3. **TP4 collective critical path** (NCCL CTAs / ALGO / PROTO / NTHREADS) with profiles and env proofs.
4. **Capacity census at 90k / 116k** (Mamba slots 68 / 84 (+88 / 92 if Mamba-bound), mem-fraction 0.80 / 0.83 / 0.86).
5. **Prefill / decode interference** (chunk 2048 / 4096 / 8192, prefill-decode-interval 1 / 4; D0-D3; starvation).
6. **Correctness backports** #43567+#43559, #43029, #41736, #41737 (unit + negative tests, trigger evidence required).

Attach (DESIGN §2; FP8 in every notebook: prompt ids come from its tokenizer.json):

| Input | Mount |
|---|---|
| SGLang wheelhouse | `/kaggle/input/datasets/foysalemonshanto/sglang-0-5-21-cu13-wheelhouse` |
| vLLM 0.32 wheelhouse (SMOKE, FULL_A) | `/kaggle/input/datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse` |
| INT4 model | `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1` |
| DFlash2 model | `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-dflash2/pytorch/bf16/1` |
| FP8 model | `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1` |
| ARC tasks | `/kaggle/input/competitions/arc-prize-2026-arc-agi-2` |

This copy runs `MODE=__MODE__`, `PART=__PART__` by default (the code is identical in the three E24 notebooks).
Every number is a measured counter; a missing measurement is INCONCLUSIVE / SKIPPED, never PASS or zero, and nothing
defaults to GO. Outputs: `/kaggle/working/e24/` (`e24_results.json`, `e24_report.md`, server logs, per-arm summaries,
`e24_raw.tar.gz`).
"""

CELL1 = r'''# ==================================================================== CELL 1 — config, inputs, modules
import os, sys, json, time, re, shutil, subprocess, hashlib, threading, traceback, base64, gzip, tarfile, inspect
import random, math, signal, socket, urllib.request as _U
from pathlib import Path
T_START = time.time()
MODE = os.environ.get("Q27X_E24_MODE", "__MODE__").upper()        # SMOKE | FULL (baked per notebook)
PART = os.environ.get("Q27X_E24_PART", "__PART__").upper()        # FULL: A (sections 1 + 6) | B (sections 2-5) | ALL
assert MODE in ("SMOKE", "FULL") and PART in ("A", "B", "ALL"), (MODE, PART)
SM = MODE == "SMOKE"
VARIANT = "SMOKE" if SM else {"A": "FULL_A", "B": "FULL_B", "ALL": "FULL_ALL"}[PART]
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
FAKE = os.environ.get("E24_FAKE_SERVER", "").strip().lower() in {"1", "true", "yes"}       # local / CPU kit only
TS = float(os.environ.get("E24_TIME_SCALE") or 1.0) if FAKE else 1.0     # FAKE only: windows, settles, waits, est_min
FORCE_COND = FAKE and os.environ.get("E24_FORCE_CONDITIONAL", "").strip() == "1"        # FAKE / kit: both branches
BUDGET_MIN = 210 if SM else {"A": 480, "B": 690, "ALL": 690}[PART]
BUDGET = int(float(os.environ.get("E24_BUDGET_SEC") or BUDGET_MIN * 60 * TS))
HARD_STOP = T_START + BUDGET
MAX_LEN, SGL_CTX = 131072, 155648                                  # user rule: thinking up to 120k (vLLM / SGLang)
OUT = Path(os.environ.get("E24_OUT", "/kaggle/working/e24"))
OUT.mkdir(parents=True, exist_ok=True)
LOGS = OUT / "logs"
LOGS.mkdir(exist_ok=True)
ARMS_OUT = OUT / "arms"
ARMS_OUT.mkdir(exist_ok=True)
TMP = Path(os.environ.get("E24_TMP", "/tmp"))
TMP.mkdir(parents=True, exist_ok=True)                             # E23 kit v1 lesson: TMP is always created
WORK = TMP / "e24"                                                 # raw per arm: WORK/<arm>/ ; caches: WORK/cache/<arm>/
WORK.mkdir(parents=True, exist_ok=True)
INP = Path(os.environ.get("E24_INPUT", "/kaggle/input"))
R = {"mode": MODE, "part": PART, "variant": VARIANT, "rev": "1 (DESIGN rev 2)", "budget_min": round(BUDGET / 60, 1),
     "fake": FAKE, "time_scale": TS, "force_conditional": FORCE_COND, "max_len": MAX_LEN, "sglang_context": SGL_CTX,
     "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "inputs": {}, "identity": {}, "setup": {},
     "sections": {s: {} for s in ("1", "2", "3", "4", "5", "6")}, "comparisons": {}, "argv": {}, "errors": [], "order": [],
     "shas": {}, "api_drift": {}, "gates": {}, "not_measured": {
         "coverage": "needs the Duck / Poetiq harness (paired ARC sample session, DESIGN §5.1)",
         "adoption_correct_outputs_per_gpu_hour": "needs the paired ARC sample (DESIGN §5.1)",
         "agent_end_to_end_denominator": "needs the Duck / Poetiq harness (Y11 third denominator)"}}
_SAVE_LOCK = threading.Lock()
ABORT = {"why": None}


def save():
    with _SAVE_LOCK:
        tmp = OUT / "e24_results.json.tmp"
        tmp.write_text(json.dumps(R, indent=1, default=str))
        tmp.replace(OUT / "e24_results.json")


def left():
    return HARD_STOP - time.time()


def log(*a):
    print(f"[{(time.time() - T_START) / 60:6.1f} min]", *a, flush=True)


def err(where, e=None, **kw):
    """Every caught exception is recorded (E23 defect: R["errors"] was never written)."""
    rec = {"where": where, "t_min": round((time.time() - T_START) / 60, 2), **kw}
    if e is not None:
        rec.update(error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2500:])
    R["errors"].append(rec)
    try:
        save()
    except Exception:
        pass
    return rec


def find(pattern, root=INP, must=True):
    """E23 find() fixed (e23_template §13.6 #13): the shallowest hit under `root`; two different hits at that depth
    (two attached versions) are refused instead of silently taking the first."""
    hits = sorted(Path(root).rglob(pattern)) if Path(root).exists() else []
    if hits:
        d0 = min(len(h.parts) for h in hits)
        top = [h for h in hits if len(h.parts) == d0]
        if len(top) > 1:
            raise FileNotFoundError(f"AMBIGUOUS: {len(top)} hits for {pattern} under {root}: {[str(x) for x in top[:4]]}")
        return top[0]
    if must:
        raise FileNotFoundError(f"{pattern} not under {root}: attach the input")
    return None


def _dir_size(p):
    try:
        fs = [f for f in Path(p).rglob("*") if f.is_file()]
        return {"files": len(fs), "bytes": sum(f.stat().st_size for f in fs)}
    except Exception as e:
        return {"error": repr(e)[:200]}


if RERUN:      # a competition rerun writes the minimal valid submission and skips cells 2-6 (no exception, as E23)
    tasks = json.load(open(find("arc-agi_test_challenges.json")))
    sub = {k: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in v["test"]] for k, v in tasks.items()}
    _sp = Path(os.environ.get("E24_SUBMISSION_PATH") or "/kaggle/working/submission.json")   # env: local tests only
    _sp.parent.mkdir(parents=True, exist_ok=True)
    _sp.write_text(json.dumps(sub))
    print("E24 lab notebook: competition rerun -> fallback submission.json only")
else:
    # ---- embedded modules: gzip+base64, sha256 checked on decode (DESIGN §7); every sha recorded in R["shas"]
    MODS = WORK / "mods"
    MODS.mkdir(parents=True, exist_ok=True)
    BUILD_SHAS = __SHAS__
    BLOBS = __BLOBS__
    MISSING_AT_BUILD = [k for k, v in BUILD_SHAS.items() if v == "MISSING"]
    for _k, (_name, _sha, _b64) in BLOBS.items():
        if _sha == "MISSING":
            continue
        _data = gzip.decompress(base64.b64decode(_b64))
        _got = hashlib.sha256(_data).hexdigest()
        if _got != _sha:
            raise RuntimeError(f"embedded {_name}: sha256 {_got[:12]} != build {_sha[:12]} (corrupt notebook)")
        (MODS / _name).write_bytes(_data)
        R["shas"][_k] = _got
    R["shas"]["notebook_builder"] = BUILD_SHAS["notebook_builder"]
    if MISSING_AT_BUILD:
        R["gates"]["modules"] = "MODULE_MISSING_AT_BUILD: " + ",".join(MISSING_AT_BUILD)
        save()
        raise RuntimeError("E24 notebook built with missing modules " + str(MISSING_AT_BUILD) + ": rebuild before running")
    HOOK_SITE = WORK / "e24_hook_site"                              # server PYTHONPATH only (sitecustomize)
    HOOK_SITE.mkdir(parents=True, exist_ok=True)
    (HOOK_SITE / "sitecustomize.py").write_bytes((MODS / "e24_hook.py").read_bytes())
    R["shas"]["hook_site"] = hashlib.sha256((HOOK_SITE / "sitecustomize.py").read_bytes()).hexdigest()
    assert R["shas"]["hook_site"] == R["shas"]["hook"]
    UNIT = MODS / "e24_unit_41736_41737.py"
    sys.path.insert(0, str(MODS))
    if not FAKE:
        os.environ.pop("E24_TIME_SCALE", None)                     # the lab reads it at import: FAKE only
    for _m in ("e24_lab", "e24_sglang_patches", "e14_loadpath", "e14_trace_summary"):
        sys.modules.pop(_m, None)
    import e24_lab as L
    import e24_sglang_patches as P
    import e14_loadpath as E14L
    if FAKE:                                     # FAKE: fake servers from the repo harness (E24_FAKE_DIR)
        _fd = os.environ.get("E24_FAKE_DIR")
        _cands = [Path(_fd)] if _fd else [p / "experiments" / "Experiment 24 - SGLang speed track (planned)" / "src" / "harness"
                                         for p in [Path.cwd(), *Path.cwd().parents]] + [p / "src" / "harness" for p in [Path.cwd(), *Path.cwd().parents]]
        FAKE_DIR = next((c for c in _cands if (c / "fake_sglang_server.py").is_file()), None)
        if FAKE_DIR is None:
            raise FileNotFoundError("E24_FAKE_SERVER=1 needs E24_FAKE_DIR (the dir with fake_sglang_server.py / fake_vllm_server.py)")
        R["setup"]["fake_dir"] = str(FAKE_DIR)
    # ---- inputs (fail closed: a missing or ambiguous input is recorded and stops the measured steps, never guessed)
    M = INP / "models" / "foysalemonshanto"
    D = INP / "datasets" / "foysalemonshanto"
    NEED_VLLM = SM or PART in ("A", "ALL")
    INPUT_ERR = {}

    def _inp(key, fn):
        try:
            return fn()
        except Exception as e:
            INPUT_ERR[key] = f"{type(e).__name__}: {e}"[:400]
            err("inputs/" + key, e)
            return None
    FP8 = _inp("FP8", lambda: find("config.json", M / "qwen3-8-27b-fp8-repacked-v1").parent)
    INT4 = _inp("INT4", lambda: find("config.json", M / "qwen3-8-27b-int4-redhat").parent)
    DFLASH = _inp("DFLASH", lambda: find("config.json", M / "qwen3-8-27b-dflash2").parent)
    SGL_WH = _inp("SGL_WHEELHOUSE", lambda: find("requirements.lock", D / "sglang-0-5-21-cu13-wheelhouse").parent)
    VLLM_WH = _inp("VLLM_WHEELHOUSE", lambda: find("install_wheelhouse.py", D / "vllm-0-32-0-cu13-wheelhouse")) if NEED_VLLM else None
    ARC = _inp("ARC", lambda: find("arc-agi_evaluation_challenges.json"))
    R["inputs"] = {k: (str(v) if v else None) for k, v in dict(FP8=FP8, INT4=INT4, DFLASH=DFLASH, SGL_WHEELHOUSE=SGL_WH,
                                                               VLLM_WHEELHOUSE=VLLM_WH, ARC=ARC).items()}
    R["inputs"]["errors"] = INPUT_ERR
    R["inputs"]["sizes"] = {k: _dir_size(v) for k, v in dict(FP8=FP8, INT4=INT4, DFLASH=DFLASH).items() if v}
    _need = ["FP8", "INT4", "DFLASH", "SGL_WHEELHOUSE", "ARC"] + (["VLLM_WHEELHOUSE"] if NEED_VLLM else [])
    _miss = [k for k in _need if not R["inputs"].get(k)]
    if _miss:
        ABORT["why"] = f"NO_GO_INPUT_MISSING: {_miss} (attach the inputs; a wrong model variation drops the mount silently)"
        R["gates"]["inputs"] = ABORT["why"]
    else:
        R["gates"]["inputs"] = "GO"
    # ---- checkpoint identity (config / tokenizer / index sha) + the DFlash2 drafter facts; shard sha256 from staging
    for _k, _v in dict(fp8=FP8, int4=INT4, dflash2=DFLASH).items():
        if _v:
            try:
                R["identity"][_k] = L.identity(_v)
            except Exception as e:
                err("identity/" + _k, e)
            _tj = Path(_v) / "tokenizer.json"
            R["identity"].setdefault("tokenizer_sha256", {})[_k] = hashlib.sha256(_tj.read_bytes()).hexdigest()[:16] if _tj.is_file() else None
    try:
        _dc = json.loads((DFLASH / "config.json").read_text()) if DFLASH else {}
        R["identity"]["dflash2_config"] = {"architectures": _dc.get("architectures"), "num_target_layers": _dc.get("num_target_layers"),
                                           "target_layer_ids": (_dc.get("dflash_config") or {}).get("target_layer_ids"),
                                           "block_size": (_dc.get("dflash_config") or {}).get("block_size"),
                                           "taps_ok": (_dc.get("dflash_config") or {}).get("target_layer_ids") == [5, 19, 33, 47, 61]
                                           and _dc.get("num_target_layers") == 64}
        _ic = json.loads((INT4 / "config.json").read_text()) if INT4 else {}
        VOCAB = int((_ic.get("text_config") or {}).get("vocab_size") or _ic.get("vocab_size") or 248320)
        _gc = json.loads((FP8 / "generation_config.json").read_text()) if FP8 and (FP8 / "generation_config.json").is_file() else {}
        EOS_IDS = sorted({int(x) for x in (_gc.get("eos_token_id") if isinstance(_gc.get("eos_token_id"), list) else [_gc.get("eos_token_id")]) if x is not None}) or [248046, 248044]
    except Exception as e:
        err("identity/configs", e)
        VOCAB, EOS_IDS = 248320, [248046, 248044]
    R["identity"]["vocab_size"], R["identity"]["eos_ids"] = VOCAB, EOS_IDS
    L.start_watchdog(HARD_STOP, lambda: (ABORT.update(why="HARD_STOP reached: watchdog killed live servers"), save()))
    log("E24", VARIANT, "| budget", round(BUDGET / 60, 1), "min | fake", FAKE, "ts", TS, "|", json.dumps(R["inputs"])[:900])
    save()
'''

# ---------------------------------------------------------------------------------------------------------------------
# The argv builders, the arm catalog and the plans. This SAME text runs inside CELL 2 and in this builder (to assert
# BASE and to write plan_argvs_e24.json), so the JSON can never drift from what the notebook launches.
ARGV_CODE = r'''# ---- argv builders + arm catalog + plans (the builder runs this same text: plan_argvs_e24.json, BASE assert)
SGL_PORT, VLLM_PORT = 30000, 8200
SPEC_FLAGS = ("--speculative-algorithm", "--speculative-draft-model-path", "--speculative-draft-model-quantization",
              "--speculative-num-draft-tokens", "--speculative-num-steps", "--speculative-eagle-topk")
TGT_DROP = SPEC_FLAGS + ("--min-free-slots-delay",)               # target-only = EMS drop=_SPEC + delay (cell [8] L768)
NCCL_DBG = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,ENV"}
PATCH_ALL5 = ("43567", "43559", "43029", "41736", "41737")
PATCH_ALL7 = PATCH_ALL5 + ("44401", "41872")


def edit_flags(flags, drop=(), set_=None, add=()):
    """EMS e14_edit_flags (cell [8] L10-25): every flag in drop | set_ is removed with its values (up to the next --),
    then set_ is appended ({flag: value | [values] | None = bare flag}), then add."""
    out, i, set_ = [], 0, dict(set_ or {})
    names = set(drop) | set(set_)
    while i < len(flags):
        if flags[i] in names:
            i += 1
            while i < len(flags) and not str(flags[i]).startswith("--"):
                i += 1
            continue
        out.append(str(flags[i]))
        i += 1
    for k, v in set_.items():
        out += [k] + ([str(x) for x in v] if isinstance(v, (list, tuple)) else ([str(v)] if v is not None else []))
    return out + [str(a) for a in add]


def _streams(n, slots=4):
    """n client streams: cap n + 1, decode graphs 1..n+1 (with --min-free-slots-delay 1 a shorter graph list crashes
    every TP rank, E14A), slots x (n + 1) Mamba slots (extra_buffer_lazy + overlap = 4 per request; 0.5.21 caps
    max_running at max_mamba_cache_size // 4, the source-derived minimum)."""
    return {"--max-running-requests": n + 1, "--cuda-graph-bs-decode": list(range(1, n + 2)),
            "--max-mamba-cache-size": slots * (n + 1)}


def prod_flags(model, draft, port=None):
    """E15B R3K2 production argv (S16, DFlash W6) = the EMS / E15B engine cell e11_flags() W1 base (cell [7] L165-189,
    E15B: mrr 17, Mamba 68) + the E14 submit-arm override (L195-223: graphs 1..17, width 6, delay 1, max-bs dropped)."""
    f = ["--model-path", str(model), "--tp-size", "4", "--kv-cache-dtype", "fp8_e4m3", "--context-length", "155648",
         "--max-running-requests", "17", "--chunked-prefill-size", "4096", "--mamba-ssm-dtype", "float16", "--enable-metrics",
         "--disable-flashinfer-autotune", "--random-seed", "20261005", "--host", "127.0.0.1",
         "--port", str(SGL_PORT if port is None else port), "--log-level", "info", "--cuda-graph-max-bs-decode", "16", "--watchdog-timeout", "1800", "--decode-log-interval", "40",
         "--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", str(draft),
         "--speculative-draft-model-quantization", "unquant", "--speculative-num-draft-tokens", "8",
         "--disable-prefill-cuda-graph", "--mamba-radix-cache-strategy", "extra_buffer_lazy", "--max-mamba-cache-size", "68",
         "--mem-fraction-static", "0.83", "--incremental-streaming-output", "--stream-interval", "1",
         "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
         "--default-chat-template-kwargs", json.dumps({"preserve_thinking": True}), "--schedule-policy", "lpm",
         "--enable-cache-report"]
    return edit_flags(f, drop=("--cuda-graph-max-bs-decode",),
                      set_={"--cuda-graph-bs-decode": list(range(1, 18)), "--speculative-num-draft-tokens": 6,
                            "--min-free-slots-delay": 1})


def sgl(model, draft, streams=16, width=6, spec=True, set_=None, drop=()):
    """An SGLang arm = production + _streams(n) (+ width) + the arm's edits; target-only drops the six speculative
    flags and --min-free-slots-delay (DESIGN §2 rules)."""
    s = dict(_streams(streams))
    if spec and width != 6:
        s["--speculative-num-draft-tokens"] = width
    s.update(set_ or {})
    return edit_flags(prod_flags(model, draft), drop=tuple(drop) + (() if spec else TGT_DROP), set_=s)


def stock_flags(model, port=None, *, tp=4, seqs=16, spec=0, kv="fp8_e4m3", backend="FLASHINFER", prefix=False,
                mem=0.90, batched=4096, max_len=131072, extra=()):
    """E23 stock_flags (build_e23_notebook.py L300-315) with the oracle values: seqs 16, no speculation, prefix off."""
    f = [str(model), "--tensor-parallel-size", str(tp), "--kv-cache-dtype", kv, "--max-model-len", str(max_len),
         "--max-num-seqs", str(seqs), "--max-num-batched-tokens", str(batched), "--gpu-memory-utilization", str(mem),
         "--disable-custom-all-reduce", "--seed", "20261005", "--host", "127.0.0.1",
         "--port", str(VLLM_PORT if port is None else port), "--served-model-name", "q27", "--attention-backend", backend, "--trust-remote-code", "--language-model-only"]
    f += ["--enable-prefix-caching"] if prefix else ["--no-enable-prefix-caching"]
    if spec:
        f += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": int(spec)})]
    return f + list(extra)


# DESIGN §2 BASE (the EMS dflash_w6_s20 argv, replayed against the logged server_args); asserted as sorted
# flag/value pairs with placeholders for the paths and the port.
BASE_DESIGN = ("--model-path <INT4> --tp-size 4 --kv-cache-dtype fp8_e4m3 --context-length 155648 --chunked-prefill-size 4096 "
               "--mamba-ssm-dtype float16 --enable-metrics --disable-flashinfer-autotune --random-seed 20261005 --host 127.0.0.1 "
               "--port <P> --log-level info --watchdog-timeout 1800 --decode-log-interval 40 --speculative-algorithm DFLASH "
               "--speculative-draft-model-path <DFLASH> --speculative-draft-model-quantization unquant "
               "--disable-prefill-cuda-graph --mamba-radix-cache-strategy extra_buffer_lazy --mem-fraction-static 0.83 "
               "--incremental-streaming-output --stream-interval 1 --reasoning-parser qwen3 --tool-call-parser qwen3_coder "
               "--default-chat-template-kwargs <CTK> --schedule-policy lpm --enable-cache-report "
               "--speculative-num-draft-tokens 6 --min-free-slots-delay 1 --max-running-requests 21 --cuda-graph-bs-decode "
               + " ".join(str(b) for b in range(1, 22)) + " --max-mamba-cache-size 84").split()


def flag_pairs(argv):
    out, cur = [], None
    for a in argv:
        a = str(a)
        if a.startswith("--"):
            cur = [a]
            out.append(cur)
        elif cur is not None:
            cur.append(a)
        else:
            out.append(["<positional>", a])
    return sorted(tuple(x) for x in out)


def base_check(model, draft):
    base = sgl(model, draft, streams=20)
    sub = {"<INT4>": str(model), "<DFLASH>": str(draft), "<P>": str(SGL_PORT), "<CTK>": json.dumps({"preserve_thinking": True})}
    want = [sub.get(t, t) for t in BASE_DESIGN]
    return {"ok": flag_pairs(base) == flag_pairs(want), "exact_order": base == want, "n_tokens": len(base)}


def small(interval, slots=4):
    """E17 e17_small: the upstream small-slot recipe (4 Mamba slots, 1 running)."""
    return {"--max-mamba-cache-size": slots, "--max-running-requests": 1, "--mamba-track-interval": interval,
            "--chunked-prefill-size": 2048, "--cuda-graph-bs-decode": [1]}


NGRAM_SET = {"--speculative-algorithm": "NGRAM", "--speculative-num-draft-tokens": 4, "--speculative-ngram-max-bfs-breadth": 1,
             "--mamba-radix-cache-strategy": "extra_buffer", "--max-mamba-cache-size": 32, "--max-running-requests": 4,
             "--cuda-graph-bs-decode": [1, 2, 3, 4], "--random-seed": 1, "--grammar-backend": "xgrammar"}
NGRAM_DROP = ("--speculative-draft-model-path", "--speculative-draft-model-quantization", "--min-free-slots-delay")
GDF_SET = {"--mamba-radix-cache-strategy": "extra_buffer", "--max-mamba-cache-size": 32, "--max-running-requests": 4,
           "--cuda-graph-bs-decode": [1, 2, 3, 4], "--random-seed": 1, "--grammar-backend": "xgrammar"}
S2_CHALLENGERS = ("s2_w4s20", "s2_w6s16", "s2_w4s16", "s2_w5s20", "s2_w7s20")
S3_ARMS = ("s3_ctrl", "s3_cta2", "s3_cta4", "s3_cta8", "s3_ring", "s3_tree", "s3_simple", "s3_ll", "s3_nthreads512")
S3_ENV = {"s3_ctrl": {}, "s3_cta2": {"NCCL_MIN_CTAS": "2", "NCCL_MAX_CTAS": "2"}, "s3_cta4": {"NCCL_MIN_CTAS": "4", "NCCL_MAX_CTAS": "4"},
          "s3_cta8": {"NCCL_MIN_CTAS": "8", "NCCL_MAX_CTAS": "8"}, "s3_ring": {"NCCL_ALGO": "allreduce:ring"},
          "s3_tree": {"NCCL_ALGO": "allreduce:tree"}, "s3_simple": {"NCCL_PROTO": "Simple"}, "s3_ll": {"NCCL_PROTO": "LL"},
          "s3_nthreads512": {"NCCL_NTHREADS": "512"}}


def catalog(int4, dflash, fp8, variant):
    """Every arm a variant can launch (DESIGN rev 2 §4): engine, server flags (after the launch prefix), arm env, cache
    policy (§3.3: fresh / ("reuse", arm) / ("copy", arm) / "shared"), expected patch set (§3.4), hook options."""
    A = {}
    full_b = variant in ("FULL_B", "FULL_ALL")
    warm_d6 = "shared" if full_b and variant == "FULL_B" else ("copy", "s1_d6")    # SMOKE speed arms: s1_d6's cache

    def add(arm, sec, engine, flags, env=None, cache="fresh", patches=(), hook=None, **kw):
        A[arm] = dict(arm=arm, section=sec, engine=engine, flags=[str(x) for x in flags], env=dict(env or {}), cache=cache,
                      patches=tuple(patches), hook=dict(hook or {}), **kw)

    def S(**k):
        return sgl(int4, dflash, **k)
    probe = {"probe": 1}
    # ---- section 1 (fresh isolated caches; c_warm / c_nocache relaunch on s1_c's root)
    add("s1_a", "1", "vllm", stock_flags(fp8), hook=probe)          # M5 on vLLM: GPUModelRunner probe (informational)
    add("s1_b", "1", "vllm", stock_flags(int4), hook=probe)
    add("s1_d6", "1", "sglang", S(), hook=probe)
    add("s1_d6fr", "1", "sglang", S(), env={"SGLANG_SIMULATE_ACC_LEN": "1", "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token"},
        hook=probe, greedy_only=True)
    add("s1_d4", "1", "sglang", S(width=4), hook=probe)
    add("s1_c", "1", "sglang", S(spec=False), hook=probe)
    add("s1_c_warm", "1", "sglang", S(spec=False), cache=("reuse", "s1_c"), hook=probe)
    add("s1_c_nocache", "1", "sglang", S(spec=False, set_={"--disable-radix-cache": None}), cache=("reuse", "s1_c"), hook=probe)
    add("s1_c32", "1", "sglang", S(spec=False, set_={"--mamba-ssm-dtype": "float32"}), hook=probe)
    add("s1_d6_r2", "1", "sglang", S(), hook=probe)
    add("s1_c_r2", "1", "sglang", S(spec=False), hook=probe)
    # ---- section 2 (S20 control = DESIGN BASE)
    sec2 = {"s2_w6s20": S(streams=20), "s2_w6s20_aa": S(streams=20), "s2_w4s20": S(streams=20, width=4),
            "s2_w6s16": S(streams=16), "s2_w4s16": S(streams=16, width=4), "s2_w5s20": S(streams=20, width=5),
            "s2_w7s20": S(streams=20, width=7)}
    for a, f in sec2.items():
        add(a, "2", "sglang", f, cache=warm_d6)
    for c in S2_CHALLENGERS:
        add("s2_best_long__" + c, "2", "sglang", sec2[c], cache=warm_d6, base_arm=c)
    for i in (1, 2, 3):
        add(f"aa_ctrl_r{i}", "2", "sglang", sec2["s2_w6s20"], cache=warm_d6, base_arm="s2_w6s20")
        for c in S2_CHALLENGERS:
            add(f"aa_best_r{i}__{c}", "2", "sglang", sec2[c], cache=warm_d6, base_arm=c)
    # ---- section 3 (fresh launches on the warm cache; NCCL INFO in every arm)
    for a in S3_ARMS:
        add(a, "3", "sglang", S(streams=20), env=dict(NCCL_DBG, **S3_ENV[a]), cache=warm_d6)
        add("s3_conf__" + a, "3", "sglang", S(streams=20), env=dict(NCCL_DBG, **S3_ENV[a]), cache=warm_d6, base_arm=a)
    add("s3_ctrl_rerun", "3", "sglang", S(streams=20), env=dict(NCCL_DBG), cache=warm_d6, base_arm="s3_ctrl")
    # ---- section 4 (counter hook on: natural #43567 exposure; throughput labelled hook-on)
    cnt = {"counters": 1}
    add("s4_m84", "4", "sglang", S(streams=20), cache=warm_d6, hook=cnt)
    add("s4_m68", "4", "sglang", S(streams=16), cache=warm_d6, hook=cnt)
    add("s4_mem080", "4", "sglang", S(streams=20, set_={"--mem-fraction-static": "0.80"}), cache=warm_d6, hook=cnt)
    add("s4_mem086", "4", "sglang", S(streams=20, set_={"--mem-fraction-static": "0.86"}), cache=warm_d6, hook=cnt)
    add("s4_m88", "4", "sglang", S(streams=21), cache=warm_d6, hook=cnt)
    add("s4_m92", "4", "sglang", S(streams=22), cache=warm_d6, hook=cnt)
    add("s4_lowcap", "4", "sglang", S(streams=20, set_={"--max-total-tokens": 300000}), cache=warm_d6, hook=cnt)
    # ---- section 5 (control = chunk 4096; PDI arms trade prefill against decode)
    add("s5_c4096", "5", "sglang", S(streams=20), cache=warm_d6)
    add("s5_c2048", "5", "sglang", S(streams=20, set_={"--chunked-prefill-size": 2048}), cache=warm_d6)
    add("s5_c8192", "5", "sglang", S(streams=20, set_={"--chunked-prefill-size": 8192}), cache=warm_d6)
    add("s5_pdi1", "5", "sglang", S(streams=20, set_={"--prefill-decode-interval": 1}), cache=warm_d6)
    add("s5_pdi4", "5", "sglang", S(streams=20, set_={"--prefill-decode-interval": 4}), cache=warm_d6)
    add("s5_mixed", "5", "sglang", S(streams=20, set_={"--enable-mixed-chunk": None}), cache=warm_d6)
    # ---- section 6 (copies of the matching section-1 root: s1_c for target-only / NGRAM, s1_d6 for DFlash / combined)
    c_c, c_d = ("copy", "s1_c"), ("copy", "s1_d6")
    ff = {"SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL": "1"}
    add("s6_unit", "6", "sglang", [], cache=c_d, kind="unit")
    add("s6_ffu", "6", "sglang", S(spec=False, set_=small(128)), env=ff, cache=c_c, hook=cnt)
    add("s6_ffp", "6", "sglang", S(spec=False, set_=small(128)), env=ff, cache=c_c, patches=("43567", "43559"), hook=cnt)
    add("s6_np", "6", "sglang", S(spec=False, set_=small(128)), cache=c_c, patches=("43567", "43559"), hook=cnt)
    add("s6_ffud", "6", "sglang", S(set_=small(64)), env=ff, cache=c_d, hook=cnt)
    add("s6_ffpd", "6", "sglang", S(set_=small(64)), env=ff, cache=c_d, patches=("43567", "43559"), hook=cnt)
    ng = edit_flags(prod_flags(int4, dflash), drop=NGRAM_DROP, set_=NGRAM_SET)
    add("s6_gu", "6", "sglang", ng, cache=c_c, hook=cnt)
    add("s6_gp", "6", "sglang", ng, cache=c_c, patches=("43029",), hook=cnt)
    gdf = edit_flags(prod_flags(int4, dflash), set_=GDF_SET)
    add("s6_gu_df", "6", "sglang", gdf, cache=c_d, hook=cnt)
    add("s6_gp_df", "6", "sglang", gdf, cache=c_d, patches=("43029",), hook=cnt)
    gl = S(streams=3, set_={"--random-seed": 1, "--grammar-backend": "xgrammar"})
    add("s6_gl_u", "6", "sglang", gl, cache=c_d, hook=cnt)
    add("s6_gl_p", "6", "sglang", gl, cache=c_d, patches=("43029",), hook=cnt)
    eager = S(set_={"--disable-decode-cuda-graph": None})
    inj_d = {"E24_INJECT_DRAFT_NAN": "1"}
    inj_s = {"E24_INJECT_SELECTOR_NAN": "1"}
    add("s6_sent_u", "6", "sglang", eager, env=inj_d, cache=c_d, patches=("41737",), hook=dict(cnt, eager=1))
    add("s6_sent_p", "6", "sglang", eager, env=inj_d, cache=c_d, patches=("41737", "41736"), hook=dict(cnt, eager=1))
    add("s6_nan_u", "6", "sglang", eager, env=inj_s, cache=c_d, hook=dict(cnt, eager=1))
    add("s6_nan_p", "6", "sglang", eager, env=inj_s, cache=c_d, patches=("41737",), hook=dict(cnt, eager=1))
    add("s6_combined", "6", "sglang", S(), cache=c_d, patches=PATCH_ALL5, hook=probe)
    return A


# Steps per variant in run order: (section, arm, est_min). Conditional arms carry a tag; DESIGN rev 2 §5 minutes.
PLANS = {
    "SMOKE": [("1", "s1_a", 19), ("1", "s1_b", 16), ("1", "s1_d6", 21), ("1", "s1_d6fr", 12), ("1", "s1_d4", 11),
              ("1", "s1_c", 21), ("6", "s6_unit", 6), ("6", "s6_ffu", 8), ("6", "s6_ffp", 8), ("2", "s2_w6s20", 25),
              ("2", "s2_w4s20", 7), ("3", "s3_cta4", 8)],
    "FULL_A": [("1", "s1_a", 42), ("1", "s1_b", 24), ("1", "s1_d6", 38), ("1", "s1_d6fr", 17), ("1", "s1_d4", 35),
               ("1", "s1_c", 40), ("1", "s1_c_warm", 13), ("1", "s1_c_nocache", 19), ("1", "s1_c32", 38),
               ("1", "s1_d6_r2", 19), ("6", "s6_unit", 6), ("6", "s6_ffu", 8), ("6", "s6_ffp", 8), ("6", "s6_np", 8),
               ("6", "s6_ffud", 8), ("6", "s6_ffpd", 9), ("6", "s6_gu", 10), ("6", "s6_gp", 7),
               ("6", "s6_gu_df?", 8), ("6", "s6_gp_df?", 8), ("6", "s6_sent_u", 8), ("6", "s6_sent_p", 8),
               ("6", "s6_nan_u", 6), ("6", "s6_nan_p", 6), ("6", "s6_combined", 10), ("1", "s1_c_r2", 17),
               ("6", "s6_gl_u?", 12), ("6", "s6_gl_p?", 6)],
    "FULL_B": [("2", "s2_w6s20", 57), ("2", "s2_w6s20_aa", 18), ("2", "s2_w4s20", 23), ("2", "s2_w6s16", 19),
               ("2", "s2_w4s16", 19), ("2", "s2_w5s20|s2_w7s20", 19), ("2", "s2_best_long?", 24)]
              + [("3", a, 10) for a in S3_ARMS] + [("3", "s3_conf__best1?", 20), ("3", "s3_conf__best2?", 20),
                                                     ("3", "s3_ctrl_rerun", 9)]                     # section 3 = 139
              + [("4", "s4_m84", 35), ("4", "s4_m68", 26), ("4", "s4_mem080", 26), ("4", "s4_mem086", 29),
                 ("4", "s4_m88?", 26), ("4", "s4_m92?", 26)]
              + [("5", "s5_c4096", 25), ("5", "s5_c2048", 25), ("5", "s5_c8192", 25), ("5", "s5_pdi1", 25), ("5", "s5_pdi4", 25)]
              + [("2", "aa_ctrl_r%d" % i, 18) for i in (1, 2, 3)]
              # optional "if time remains" (DESIGN §4.4 / §4.5): after the A/A rounds, which confirm a section-2 best
              + [("4", "s4_lowcap?", 14), ("5", "s5_mixed?", 25)],
}
PLANS["FULL_ALL"] = PLANS["FULL_A"] + PLANS["FULL_B"]


def plan_arms(variant):
    """Every arm id a variant can launch, both branches of every conditional (for plan_argvs_e24.json / the kit)."""
    out = []
    for sec, a, _ in PLANS[variant]:
        a = a.rstrip("?")
        if a == "s2_w5s20|s2_w7s20":
            out += ["s2_w5s20", "s2_w7s20"]
        elif a == "s2_best_long":
            out += ["s2_best_long__" + c for c in S2_CHALLENGERS]
        elif a.startswith("s3_conf__best"):
            out += [] if "s3_conf__s3_ctrl" in out else ["s3_conf__" + x for x in S3_ARMS]
        elif a.startswith("aa_ctrl_r"):
            out += [a] + [f"aa_best_r{a[-1]}__{c}" for c in S2_CHALLENGERS]
        else:
            out.append(a)
    return list(dict.fromkeys(out))
'''

exec(compile(ARGV_CODE, "<ARGV_CODE>", "exec"))          # builder-side: BASE assert + plan_argvs (same text as CELL 2)
_bc = base_check("<INT4>", "<DFLASH>")
assert _bc["ok"], ("BASE argv drifted from DESIGN §2", _bc)
assert _bc["exact_order"], ("BASE argv order differs from the EMS replay", _bc)

CELL2 = r'''# ==================================================================== CELL 2 — engines (SGLang 0.5.21 venv + /dev/shm staging, vLLM 0.32)
''' + ARGV_CODE + r'''
import shlex
if FAKE and os.environ.get("E24_PORT_BASE"):            # FAKE only: parallel local runs must not share ports
    SGL_PORT, VLLM_PORT = int(os.environ["E24_PORT_BASE"]), int(os.environ["E24_PORT_BASE"]) + 1
PLAN = PLANS[VARIANT]
PLANNED = set(plan_arms(VARIANT))
R["plan"] = [list(x) for x in PLAN]


def run(name, cmd, timeout=3600, env=None):
    t = time.time()
    try:
        with open(LOGS / f"{name}.log", "w") as f:
            r = subprocess.run(["bash", "-c", "set -o pipefail; " + cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                               env=env or dict(os.environ))
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = -9
    txt = (LOGS / f"{name}.log").read_text(errors="replace")
    R["setup"][name] = {"rc": rc, "sec": round(time.time() - t, 1), "tail": txt[-1500:]}
    log(name, "rc", rc, f"{time.time() - t:.0f}s")
    save()
    return rc, txt


SHIM = WORK / "e24_libcuda"          # FlashInfer JIT links -lcuda; Kaggle has only libcuda.so.1 (E10 run 1 / E23)
SHIM.mkdir(exist_ok=True)
for _d in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/compat", "/usr/lib64"):
    _c = sorted(Path(_d).glob("libcuda.so.1")) if Path(_d).is_dir() else []
    if _c and not (SHIM / "libcuda.so").exists():
        (SHIM / "libcuda.so").symlink_to(_c[0])
_KNOBS = ("SGLANG_TEST_", "SGLANG_SIMULATE_", "E24_INJECT_", "E24_HOOK_", "E24_SENTINEL")
R["setup"]["inherited_knobs_dropped"] = sorted(k for k in os.environ if k.startswith(_KNOBS))
R["setup"]["inherited_nccl"] = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}


def clean_env(src):
    """The server env base: no inherited PYTHONPATH / CUDA_* / LD_PRELOAD (EMS e11_install) and never an inherited
    test / simulate / injection / hook knob (DESIGN §2: those exist only in the arms that set them)."""
    return {k: v for k, v in src.items() if k not in ("PYTHONPATH", "CUDA_HOME", "CUDA_PATH", "LD_PRELOAD")
            and not k.startswith(_KNOBS)}


SGL_FIXED = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
             "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1", "TORCH_CUDA_ARCH_LIST": "8.9", "MAX_JOBS": "4",
             "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "0",       # set explicitly as in E17
             "SGLANG_LOGPROB_CHUNK_SIZE": "128"}             # rev 2 blocker #1: input-logprob OOM; same GEMM shapes in every arm
VLLM_FIXED = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
              "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"}


def sgl_install():
    """= the EMS / E15B engine cell e11_install() (cell [7] L73-156): a --without-pip venv from the wheelhouse lock
    (--no-deps, --only-binary), '+'-stripped wheel names rebuilt from SHA256SUMS (sha256-checked), the pip CUDA 13
    runtime as CUDA_HOME with the libcudart.so / lib64 / -lcuda shims. Raises on failure (the caller caches it)."""
    if FAKE:
        fs = str(FAKE_DIR / "fake_sglang_server.py")
        return {"vpy": sys.executable, "cmd": [sys.executable, fs], "env": dict(clean_env(os.environ), **SGL_FIXED),
                "fake_server": fs, "fake": True}
    shims, t0 = {}, time.perf_counter()
    venv = TMP / "sglang-venv"
    vpy = venv / "bin" / "python"
    fl = TMP / "e24-wheels"
    fl.mkdir(parents=True, exist_ok=True)
    renamed = []
    for line in (SGL_WH / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        h, name = line.split("  ", 1)
        name = name.strip()
        if not name.endswith(".whl"):
            continue
        s = SGL_WH / name if (SGL_WH / name).exists() else SGL_WH / name.replace("+", "")
        if not s.exists():
            raise RuntimeError(f"wheel missing from the dataset: {name}")
        if s.name != name:
            d = hashlib.sha256()
            with open(s, "rb") as f:
                for b in iter(lambda: f.read(16 * 1024 * 1024), b""):
                    d.update(b)
            if d.hexdigest() != h:
                raise RuntimeError(f"sha256 mismatch for {s.name}")
            renamed.append(f"{s.name} -> {name}")
        if not (fl / name).exists():
            (fl / name).symlink_to(s)
    shims["renamed_wheels"] = renamed
    if not vpy.exists():
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=300)
        r = subprocess.run([sys.executable, "-m", "pip", "--python", str(vpy), "install", "--no-index", "--find-links", str(fl),
                            "--requirement", str(SGL_WH / "requirements.lock"), "--no-deps", "--only-binary", ":all:",
                            "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts", "-q"],
                           capture_output=True, text=True, timeout=1800)
        (LOGS / "install_sglang.log").write_text((r.stdout or "") + (r.stderr or ""))
        if r.returncode != 0:
            raise RuntimeError("offline SGLang install failed: " + (r.stderr or r.stdout)[-1500:])
    site = Path(subprocess.run([str(vpy), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                               capture_output=True, text=True, check=True, timeout=120).stdout.strip())
    cuda_home = site / "nvidia" / "cu13"
    if not (cuda_home / "lib" / "libcudart.so").exists():
        (cuda_home / "lib" / "libcudart.so").symlink_to("libcudart.so.13")
        shims["libcudart.so"] = "created"
    if not (cuda_home / "lib64").exists():
        (cuda_home / "lib64").symlink_to("lib")
        shims["lib64"] = "created"
    libcuda = TMP / "e24-libcuda"
    have = [d for d in ("/usr/local/cuda/lib64", "/usr/local/cuda/lib64/stubs", "/usr/lib/x86_64-linux-gnu", "/usr/lib64")
            if Path(d, "libcuda.so").exists()]
    if have:
        shims["libcuda.so"] = f"exists in {have[0]}"
    else:
        real = None
        try:
            for line in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=30).stdout.splitlines():
                if "libcuda.so.1" in line and "=>" in line:
                    real = line.split("=>")[-1].strip()
                    break
        except Exception:
            pass
        for d in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
            if real is None and Path(d, "libcuda.so.1").exists():
                real = str(Path(d, "libcuda.so.1"))
        if real:
            libcuda.mkdir(parents=True, exist_ok=True)
            if (libcuda / "libcuda.so").is_symlink() or (libcuda / "libcuda.so").exists():
                (libcuda / "libcuda.so").unlink()
            (libcuda / "libcuda.so").symlink_to(Path(real).resolve())
            shims["libcuda.so"] = f"shim -> {real}"
        else:
            shims["libcuda.so"] = "libcuda.so.1 NOT FOUND"
    env = clean_env(os.environ)
    env.update({"PATH": os.pathsep.join([str(venv / "bin"), str(cuda_home / "bin"), os.environ.get("PATH", "")]),
                "CUDA_HOME": str(cuda_home), "CUDA_PATH": str(cuda_home),
                "LIBRARY_PATH": os.pathsep.join(x for x in (str(libcuda), str(SHIM), str(cuda_home / "lib"),
                                                            os.environ.get("LIBRARY_PATH", "")) if x),
                "LD_LIBRARY_PATH": os.pathsep.join(x for x in (str(cuda_home / "lib"), os.environ.get("LD_LIBRARY_PATH", "")) if x)})
    env.update(SGL_FIXED)
    ver = subprocess.run([str(vpy), "-c", "import importlib.metadata as m; print(m.version('sglang'))"], capture_output=True,
                         text=True, timeout=120, env=env).stdout.strip()
    return {"vpy": str(vpy), "cmd": [str(vpy), "-m", "sglang.launch_server"], "env": env, "version": ver, "site": str(site),
            "cuda_home": str(cuda_home), "shims": shims, "sec": round(time.perf_counter() - t0, 1)}


def vllm_install():
    """= E23 install_stock (the dataset's install_wheelhouse.py -> VENV_PYTHON / CUDA_HOME), failure cached by the caller."""
    if FAKE:
        fs = str(FAKE_DIR / "fake_vllm_server.py")
        return {"vpy": sys.executable, "cmd": [sys.executable, fs, "serve"], "env": dict(clean_env(os.environ), **VLLM_FIXED),
                "fake_server": fs, "fake": True}
    rc, txt = run("install_vllm", f"{shlex.quote(sys.executable)} {shlex.quote(str(VLLM_WH))} {shlex.quote(str(TMP / 'vllm-venv'))}",
                  timeout=2400)
    kv = dict(l.split("=", 1) for l in txt.splitlines() if l.startswith(("VENV_PYTHON=", "CUDA_HOME=")))
    if rc or "CUDA_HOME" not in kv or "VENV_PYTHON" not in kv:
        raise RuntimeError(f"vLLM 0.32 install failed rc={rc} (logs/install_vllm.log)")
    env = clean_env(os.environ)
    env.update(CUDA_HOME=kv["CUDA_HOME"], PATH=f"{TMP}/vllm-venv/bin:{kv['CUDA_HOME']}/bin:{env.get('PATH', '')}",
               LIBRARY_PATH=f"{SHIM}:{env.get('LIBRARY_PATH', '')}", **VLLM_FIXED)
    ver = subprocess.run([kv["VENV_PYTHON"], "-c", "import importlib.metadata as m; print(m.version('vllm'))"],
                         capture_output=True, text=True, env=env, timeout=300).stdout.strip()
    return {"vpy": kv["VENV_PYTHON"], "cmd": [f"{TMP}/vllm-venv/bin/vllm", "serve"], "env": env, "version": ver}


# ---- staging (E14 load path: INT4 + DFlash2 to /dev/shm with ranged readers, verified, full shard sha256) runs while
# the venv installs; FAKE: E14_LOADPATH=0 (the module's off switch)
if FAKE:
    os.environ["E14_LOADPATH"] = "0"
INT4_RUN, DFLASH_RUN = INT4, DFLASH
STAGING = None
if not ABORT["why"]:
    try:
        STAGING = E14L.Staging([INT4, DFLASH], workers=24, stall_sec=600.0, deadline=time.perf_counter() + 40 * 60, log=log,
                               dest_candidates=tuple(x for x in os.environ.get("E14_DEST", "/dev/shm,/tmp").split(",") if x))
        STAGING.start()
    except Exception as e:
        err("staging/start", e)
ENGINES = {"sglang": {"failed": "not installed"}, "vllm": {"failed": "not installed"}}
if not ABORT["why"]:
    try:
        ENGINES["sglang"] = sgl_install()
        R["setup"]["sglang_install"] = {k: v for k, v in ENGINES["sglang"].items() if k not in ("env",)}
    except Exception as e:
        ENGINES["sglang"] = {"failed": f"{type(e).__name__}: {e}"[:600]}
        R["setup"]["sglang_install"] = ENGINES["sglang"]
        err("install/sglang", e)
    save()
if STAGING is not None:
    STAGING.join(timeout=max(0.0, 45 * 60 - (time.time() - T_START)))
    res = STAGING.result
    if res is None:
        R["setup"]["staging"] = {"still_running": True, "note": "launch from the /kaggle/input paths (fail open, as EMS)"}
    else:
        _paths, _rep = res
        INT4_RUN = Path(_paths.get(str(INT4), str(INT4)))
        DFLASH_RUN = Path(_paths.get(str(DFLASH), str(DFLASH)))
        R["setup"]["staging"] = json.loads(json.dumps(_rep, default=str))
        for _k, _v in (("int4_shard_sha256", INT4), ("dflash2_shard_sha256", DFLASH)):     # the checkpoint pins (DESIGN §2)
            R["identity"][_k] = ((_rep.get("copies") or {}).get(str(_v)) or {}).get("sha256") or "NOT_STAGED"
    R["setup"]["launch_paths"] = {"int4": str(INT4_RUN), "dflash2": str(DFLASH_RUN)}
    save()
if NEED_VLLM and not ABORT["why"]:
    if "s1_a" in PLANNED and not FAKE and FP8:          # rev 2: after the staging join, only when s1_a is planned
        def _fp8_prefetch():
            res = L.prefetch([FP8])
            with _SAVE_LOCK:                              # never mutate R while save() serializes it (review: real platform #7)
                R["setup"]["fp8_prefetch"] = res
        threading.Thread(target=_fp8_prefetch, daemon=True).start()
    try:
        ENGINES["vllm"] = vllm_install()
        R["setup"]["vllm_install"] = {k: v for k, v in ENGINES["vllm"].items() if k not in ("env",)}
    except Exception as e:
        ENGINES["vllm"] = {"failed": f"{type(e).__name__}: {e}"[:600]}
        R["setup"]["vllm_install"] = ENGINES["vllm"]
        err("install/vllm", e)
elif not NEED_VLLM:
    ENGINES["vllm"] = {"failed": "no vLLM arm in this variant"}
CAT = catalog(INT4_RUN, DFLASH_RUN, FP8, VARIANT)
R["setup"]["base_check"] = base_check(INT4_RUN, DFLASH_RUN)
assert R["setup"]["base_check"]["ok"], R["setup"]["base_check"]
# ---- patch site: the INSTALLED sglang package (find_spec, never imported); FAKE: E24_FAKE_SGLANG_SITE or SKIPPED_FAKE
PSITE, PATCH_BROKEN = None, None
try:
    if FAKE:
        PSITE = os.environ.get("E24_FAKE_SGLANG_SITE") or None
    elif not ENGINES["sglang"].get("failed"):
        PSITE = P.site_dir(ENGINES["sglang"]["vpy"])
    R["setup"]["patch_site"] = PSITE or ("SKIPPED_FAKE" if FAKE else None)
    if PSITE:
        _chk = P.check(PSITE, names=list(PATCH_ALL7))
        R["setup"]["patch_state_start"] = _chk
        if not P.state_ok(_chk, expect_applied=()):
            R["setup"]["patch_revert_start"] = P.revert(PSITE, names=[n for n, s in (_chk.get("names") or {}).items() if s != "not_applied"])
            if not P.state_ok(P.check(PSITE, names=list(PATCH_ALL7)), expect_applied=()):
                PATCH_BROKEN = "the installed sglang is not the pristine 0.5.21 at start and the revert failed"
except Exception as e:
    PATCH_BROKEN = f"patch check failed: {type(e).__name__}: {e}"[:300]
    err("patches/start", e)
R["setup"]["patch_broken"] = PATCH_BROKEN
log("engines:", {k: (v.get("failed") or v.get("version") or "ok") for k, v in ENGINES.items()}, "| staged:",
    R["setup"].get("launch_paths"), "| patch site:", R["setup"].get("patch_site"))
save()
'''

CELL3 = r'''# ==================================================================== CELL 3 — prompts (token ids from the FP8 tokenizer.json)
import types


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class FakeTok:
    """FAKE only, when the local python has no `tokenizers`: a deterministic byte-level tokenizer built from the
    tokenizer.json it is given (added tokens = single special ids; every other UTF-8 byte -> its byte-level vocab id).
    On the mock tree's byte-level BPE (no merges) this gives the ids HF tokenizers would give. Never used on Kaggle."""
    def __init__(self, path):
        j = json.loads(Path(path).read_text(encoding="utf-8"))
        self.special = {t["content"]: int(t["id"]) for t in (j.get("added_tokens") or [])}
        voc = (j.get("model") or {}).get("vocab") or {}
        b2u = _bytes_to_unicode()
        self.byte_id = [int(voc.get(b2u[b], b)) if isinstance(voc, dict) else b for b in range(256)]
        alt = "|".join(re.escape(s) for s in sorted(self.special, key=len, reverse=True))
        self._re = re.compile("(" + alt + ")") if alt else None

    def token_to_id(self, t):
        return self.special.get(t)

    def encode(self, text, add_special_tokens=False):
        ids = []
        for p in (self._re.split(text) if self._re else [text]):
            if p in self.special:
                ids.append(self.special[p])
            elif p:
                ids.extend(self.byte_id[b] for b in p.encode("utf-8"))
        return types.SimpleNamespace(ids=ids)


PROMPTS, MAX_NEW, ORACLE, C16, TRACE_P, M4_P, EMS, SHARED4, P2K = {}, {}, [], [], {}, [], {}, [], []
BANDS = {"8k": 8192, "40k": 40000, "90k": 90000, "120k": 116000}   # E23 rev 3b: the 120k band starts at 116k
NSTREAM = 24                                                       # section 4 offers 24 distinct streams per band
NTOK = 512 if not SM else 256                                      # M1 greedy row length
TRACE_NEW = 30000 if not SM else 4000                              # M3 reference continuation (FP8, production sampling)
M4_NEW = 15000 if not SM else 4000                                 # M4 decode-vs-prefill continuation per stream
TOK_INFO = {}
if not ABORT["why"]:
    try:
        try:
            from tokenizers import Tokenizer as _HFTok
            TOK, TOK4 = _HFTok.from_file(str(FP8 / "tokenizer.json")), _HFTok.from_file(str(INT4 / "tokenizer.json"))
            TOK_INFO["kind"] = "tokenizers"
        except ImportError:
            if not FAKE:
                raise
            TOK, TOK4 = FakeTok(FP8 / "tokenizer.json"), FakeTok(INT4 / "tokenizer.json")
            TOK_INFO["kind"] = "FAKE_byte_level (no `tokenizers` in this python)"
        _PIECES = {}

        def enc(text):
            """FP8 ids; the INT4 tokenizer must give the same ids on every piece (so on every prompt built from them)."""
            a = TOK.encode(text, add_special_tokens=False).ids
            if text not in _PIECES:
                _PIECES[text] = a == TOK4.encode(text, add_special_tokens=False).ids
            return list(a)
        _tasks = json.load(open(ARC))
        _ids = sorted(_tasks)

        def _grid(g):
            return "\n".join("".join(str(v) for v in row) for row in g)

        def _task_text(k):
            t = _tasks[k]
            parts = [f"## Task {k}"]
            for i, ex in enumerate(t["train"]):
                parts.append(f"Example {i + 1} input:\n{_grid(ex['input'])}\nExample {i + 1} output:\n{_grid(ex['output'])}")
            for ex in t["test"]:
                parts.append(f"Test input:\n{_grid(ex['input'])}")
            return "\n\n".join(parts)
        _HEAD = enc("<|im_start|>user\nFind the transformation rule of each ARC task below and solve the test inputs.\n\n")
        _TAIL = enc("\n<|im_end|>\n<|im_start|>assistant\n<think>\n")
        for _t in ("<|im_start|>", "<|im_end|>", "<think>"):          # the chat markers are single special tokens
            _id = TOK.token_to_id(_t)
            assert _id is not None and _id in (_HEAD + _TAIL), (_t, _id)
        _TASK_IDS = {k: enc(_task_text(k) + "\n\n") for k in _ids}

        def prompt(band, variant):
            """E23 CELL 3: exactly `band` tokens of DISTINCT ARC tasks (stream `variant` starts at its own task offset)."""
            body, i = [], (variant * 7) % len(_ids)
            need = band - len(_HEAD) - len(_TAIL)
            while len(body) < need:
                body += _TASK_IDS[_ids[i % len(_ids)]]
                i += 1
            return _HEAD + body[:need] + _TAIL
        PROMPTS = {b: [prompt(n, v) for v in range(NSTREAM)] for b, n in BANDS.items()}
        MAX_NEW = {b: MAX_LEN - n - 512 for b, n in BANDS.items()}          # a stream never ends inside a window
        P2K = [prompt(2048, v) for v in range(16)]                          # section-6 sentinel / NaN loads (eager)
        # ---- section 1 sets (DESIGN §3.5). Variant v of every band starts at the same task, so the 8k prompt of v is a
        # prefix of the 40k / 90k / 116k prompts of v: each set gets its OWN variants (no radix hit on a "cold" prompt).
        _ov = {"8k": (0, 1, 2, 3), "40k": (4, 5), "90k": (6, 7), "120k": (8,)} if not SM else {"8k": (0, 1, 2), "40k": (4,)}
        ORACLE = [(f"{b}#{v}", PROMPTS[b][v]) for b, vs in _ov.items() for v in vs]
        C16 = [(f"8k#{v}", PROMPTS["8k"][v]) for v in (0, 1, 2, 3) + tuple(range(9, 21))]    # superset of the serial 8k
        TRACE_P = {"t8k_a": PROMPTS["8k"][21], "t8k_b": PROMPTS["8k"][22]}
        if not SM:
            TRACE_P["t90k"] = PROMPTS["90k"][23]                            # 90k + 30k continuation = 120k context
        M4_P = [p for _, p in C16]
        PREFLIGHT = (PROMPTS["8k"][21], PROMPTS["8k"][20][-4096:])          # a 4,096-row extend (flushed after)
        # ---- EMS replica sets (EMS cell [8] ems_prompts: short = one task per stream; long / XL = a shared reference
        # block of other eval tasks up to 36k / 85k tokens + the task), built with this notebook's chat markers
        _ASK = "Find the transformation rule of this ARC-AGI-2 task and describe it, then give the test output grids."

        def _ems_text(k):
            t = _tasks[k]
            rows = lambda g: "\n".join(" ".join(str(v) for v in r) for r in g)
            s = "".join(f"\nExample {i + 1}\nInput:\n{rows(p['input'])}\nOutput:\n{rows(p['output'])}\n" for i, p in enumerate(t["train"]))
            return s + "".join(f"\nTest input {j + 1}:\n{rows(q['input'])}\n" for j, q in enumerate(t["test"]))

        def _chat(user, system=None):
            ids = enc("<|im_start|>system\n" + system + "<|im_end|>\n") if system else []
            return ids + enc("<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n<think>\n")
        _stream_tasks = _ids[:20]
        _bg = [k for k in _ids if k not in _stream_tasks]

        def _block(limit, start=0):
            txt, n, i = [], 0, start
            while n < limit and i < start + len(_bg) * 3:
                t = _ems_text(_bg[i % len(_bg)])
                txt.append(t)
                n += len(enc(t))
                i += 1
            return "Reference tasks (not to be solved):\n" + "\n".join(txt)
        _b36, _b85 = _block(36000), _block(85000)
        EMS = {"short": [_chat(_ASK + "\n" + _ems_text(k)) for k in _stream_tasks],
               "long": [_chat(_ASK + "\n" + _ems_text(k), _b36) for k in _stream_tasks],
               "xl": [_chat(_ASK + "\n" + _ems_text(k), _b85) for k in _stream_tasks]}
        # section 4 shared groups: 4 groups x 4 streams on one ~85k prefix each (distinct blocks per group)
        SHARED4 = [_chat(_ASK + "\n" + _ems_text(_stream_tasks[(4 * g + s) % len(_stream_tasks)]), _block(85000, start=7 * (g + 1)))
                   for g in range(4) for s in range(4)]
        # ---- section 6 recipes (E17 / upstream; DESIGN §3.5): P255 for every DFlash forced-fail / natural arm
        S6P = {"W44": [44] * 128, "W42": [42] * 128, "P255": [42] * 128 + [43] * 127, "P511": [42] * 128 + [43] * 383,
               "P252": [42] * 128 + [43] * 124}
        G_PHRASE = enc("The answer is yes. ")                                # upstream spec_grammar_at_43029.py
        G_REGEX = "The answer is yes\\. "
        G_FILLER = enc(("The history of the printing press begins in the fifteenth century, when movable type transformed "
                        "how books were made and shared across Europe. ") * 12)
        G_FOLLOW = enc(" In summary, the most important consequence of this invention was")
        G_TRACK = 256                                                        # default --mamba-track-interval grid
        TOK_INFO.update(pieces=len(_PIECES), pieces_equal=sum(_PIECES.values()), fp8_int4_ids_identical=all(_PIECES.values()))
        if not TOK_INFO["fp8_int4_ids_identical"]:
            ABORT["why"] = "NO_GO_TOKENIZER_MISMATCH: the INT4 tokenizer gives different ids than FP8 on " \
                           f"{len(_PIECES) - TOK_INFO['pieces_equal']} prompt pieces"
        R["gates"]["tokenizer"] = "GO" if TOK_INFO["fp8_int4_ids_identical"] else "NO_GO_TOKENIZER_MISMATCH"
        _sha = lambda p: hashlib.sha256(json.dumps(p).encode()).hexdigest()[:12]
        R["prompts"] = {"tokenizer": TOK_INFO,
                        "bands": {b: {"n": len(p), "len": len(p[0]), "sha": _sha(p[:4]),
                                      "distinct_first_task": len({tuple(x[len(_HEAD):len(_HEAD) + 50]) for x in p})}
                                  for b, p in PROMPTS.items()},
                        "oracle": [[k, len(p)] for k, p in ORACLE], "c16_sha": _sha([p for _, p in C16]),
                        "traces": {k: len(p) for k, p in TRACE_P.items()}, "trace_new": TRACE_NEW, "m4_new": M4_NEW,
                        "ems": {k: {"n": len(v), "p50": sorted(len(x) for x in v)[len(v) // 2]} for k, v in EMS.items()},
                        "shared4": {"n": len(SHARED4), "lens": [len(x) for x in SHARED4][:16]},
                        "s6": {k: len(v) for k, v in S6P.items()}, "grammar": {"phrase": len(G_PHRASE), "filler": len(G_FILLER),
                                                                              "follow": len(G_FOLLOW)}}
    except Exception as e:
        ABORT["why"] = ABORT["why"] or f"ERROR_PROMPTS: {type(e).__name__}: {e}"[:300]
        err("prompts", e)
log("prompts", json.dumps(R.get("prompts", {}).get("bands", {}))[:600], "| tokenizer", TOK_INFO)
save()
'''

CELL4 = r'''# ==================================================================== CELL 4 — the step runner (budget, gates, server lifecycle)
WIN = max(2.0, (60.0 if not SM else 30.0) * TS)     # steady window (FULL 60 s ~ 400 verify steps x 20 streams at 8k)
SEV = ("ERROR_", "NO_GO_", "INVALID_", "INCONCLUSIVE_")
DEAD_GATES = {"NO_GO_LAUNCH", "NO_GO_SERVER_DIED", "NO_GO_LOGPROB_PREFLIGHT", "NO_GO_TOOL_PROBE"}   # what a watchdog kill produces
GOOD = {"STEADY", "CAPACITY_LIMITED", "MEASURED", "ADMITTED_ALL", "GO", "EXACT", "EXACT_TO_N", "NEAR_TIE", "PASS", "ISOLATED",
        "OK", "PLATEAU", "AGGREGATE_STEADY"}
PREFLIGHT_DONE = {"done": False, "ok": None, "arm": None}
SHARED = {"first": True}
ERR_SIGNS = ("Scheduler hit an exception", "Can not alloc mamba cache", "Mamba release hit lock_ref=0", "lock receipt anchored on node",
             "alloc_req_slots runs out of memory", "illegal memory access", "CUDA error", "no kernel image", "invalid device function")


def ts(sec, floor=1.0):
    return max(floor, float(sec) * TS)


def call(fn, *a, **kw):
    """Call a lab / patcher function with only the keywords its signature accepts. A dropped keyword is recorded in
    R["api_drift"] (parallel work packages: drift must show in the results, never silently change a measurement)."""
    try:
        sig = inspect.signature(fn)
        if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            bad = [k for k in kw if k not in sig.parameters]
            if bad:
                d = R["api_drift"].setdefault(getattr(fn, "__qualname__", str(fn)), [])
                d += [k for k in bad if k not in d]
                kw = {k: v for k, v in kw.items() if k not in bad}
    except (TypeError, ValueError):
        pass
    return fn(*a, **kw)


def small(obj, limit=20000):
    """obj itself when its JSON is at most `limit` bytes, else a marker (never a truncated JSON text)."""
    try:
        n = len(json.dumps(obj, default=str))
    except Exception as e:
        return {"unserializable": repr(e)[:200]}
    return obj if n <= limit else {"truncated": True, "bytes": n}


def rawdir(arm):
    d = WORK / arm
    d.mkdir(parents=True, exist_ok=True)
    return d


def jdump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, default=str))
    tmp.replace(path)


def http(port, path, body=None, timeout=60, method=None):
    """(status, text) of a local request; HTTP errors return their status, transport errors status None."""
    data = json.dumps(body).encode() if body is not None else None
    req = _U.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method or ("POST" if data is not None else "GET"),
                     headers={"Content-Type": "application/json"})
    try:
        with _U.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except _U.HTTPError as e:
        try:
            return e.code, e.read().decode("utf-8", "replace")
        except Exception:
            return e.code, ""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"[:300]


def flag_value(flags, name):
    return flags[flags.index(name) + 1] if name in flags and flags.index(name) + 1 < len(flags) else None


def worst(statuses):
    """The most severe gate: ERROR_ > NO_GO_ > INVALID_ > INCONCLUSIVE_ > GO (SKIPPED_FAKE is neutral, FAKE only)."""
    best, rank = "GO", 0
    for s in statuses:
        if not s or s == "GO" or str(s).startswith("SKIPPED_FAKE"):
            continue
        r = next((len(SEV) - i for i, p in enumerate(SEV) if str(s).startswith(p)), 1)
        if r > rank:
            best, rank = str(s), r
    return best


def mgate(st):
    """A measurement status as a gate: valid -> GO, FAIL_x -> NO_GO_x, anything else unknown -> INCONCLUSIVE_x."""
    if st is None:
        return "INCONCLUSIVE_MISSING"
    s = str(st)
    if s in GOOD or s.split("(")[0] in GOOD:
        return "GO"
    if s.startswith("FAIL_"):
        return "NO_GO_" + s[5:]
    if s.startswith(SEV):
        return s
    if s == "ERROR":
        return "ERROR_MEASUREMENT"
    return "INCONCLUSIVE_" + s


def step(sec, arm, est_min, fn, dep=None):
    """One planned step: SKIPPED_ABORT / SKIPPED_DEPENDENCY / SKIPPED_BUDGET (unless est_min + 5 min remain, scaled),
    any exception -> ERROR_<Type> + R["errors"]; the status is always set explicitly (no DONE default)."""
    rec = R["sections"].setdefault(sec, {}).setdefault(arm, {"status": "NOT_RUN"})
    R["order"].append(f"{sec}/{arm}")
    rec["est_min"] = est_min
    if ABORT["why"]:
        rec.update(status="SKIPPED_ABORT", why=ABORT["why"])
        save()
        return rec
    why = dep() if callable(dep) else dep
    if why:
        rec.update(status="SKIPPED_DEPENDENCY", why=why)
        save()
        return rec
    if left() < (est_min * 60 + 300) * TS:
        rec.update(status="SKIPPED_BUDGET", left_min=round(left() / 60, 1))
        save()
        return rec
    t = time.time()
    log(">>>", sec, arm, f"(est {est_min} min, left {left() / 60:.0f} min)")
    try:
        out = fn() or {}
        rec.update(out)
        if rec.get("status") in (None, "NOT_RUN"):
            rec["status"] = "ERROR_NO_STATUS"
    except Exception as e:
        rec.update(status="ERROR_" + type(e).__name__, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2500:])
        err(f"{sec}/{arm}", e)
    rec["minutes"] = round((time.time() - t) / 60, 2)
    if ABORT["why"] and "HARD_STOP" in ABORT["why"]:
        # the watchdog killed the server inside this step: the gates its kill produces (ERROR_*, a dead server, refused
        # requests, a launch that never came up) are ours; a NO_GO_ measured before the stop still stands (review #10)
        g = rec.get("gates") or {}
        new = worst([v for v in g.values() if not str(v).startswith("ERROR_") and v not in DEAD_GATES] + ["INCONCLUSIVE_HARD_STOP"])
        if (g or rec["status"] == "GO" or str(rec["status"]).startswith("ERROR_")) and new != rec["status"]:
            rec["status_before_hard_stop"], rec["status"] = rec["status"], new
    try:
        jdump(ARMS_OUT / f"{arm}.json", rec)
    except Exception as e:
        err(f"{sec}/{arm}/summary", e)
    try:
        raw_snapshot(sec, arm)
    except Exception as e:
        err(f"{sec}/{arm}/raw_snapshot", e)
    log("<<<", sec, arm, rec.get("status"), f"{rec['minutes']:.1f} min")
    save()
    return rec


LAUNCH_FAILS = {"sglang": [], "vllm": []}
LAUNCH_RISKY = re.compile(r"^(s6_g|s4_mem086|s4_lowcap|s4_m88|s4_m92|s3_(?!ctrl))")   # arms whose own knob may stop a launch
RAW_ARMS = OUT / "raw_arms"


def launch_streak(eng, arm, gate, why=None, argv=None):
    """Quota guard (review: real platform #4): startup TIMEOUTS (900 s each) of three DISTINCT argvs in a row, on arms that
    vary no risky launch knob, mean the engine itself does not start; later arms of that engine become SKIPPED_DEPENDENCY.
    A fast exit costs seconds and is the arm's own config, and one config failing three times (s6_ffu / s6_ffp / s6_np
    share an argv) is one config (review 2 #5). Any launch that comes up resets the count."""
    if ABORT["why"] or gate != "NO_GO_LAUNCH" or LAUNCH_RISKY.match(arm) or "timeout" not in str(why or ""):
        return
    LAUNCH_FAILS[eng].append((arm, json.dumps(argv, default=str)))      # flags + the arm's declared env (not cache paths)
    if len({k for _, k in LAUNCH_FAILS[eng]}) >= 3:
        ENGINES[eng]["stopped"] = "startup timeouts of three distinct argvs: " + ", ".join(a for a, _ in LAUNCH_FAILS[eng])
        err("launch_streak/" + eng, None, note=ENGINES[eng]["stopped"])


def raw_snapshot(sec, arm):
    """Result-loss guard (review: real platform #6): a section-1 / section-6 arm's JSON raw files (rows, teacher forcing,
    M4, hook records; never profiles) go to /kaggle/working right after the arm, so a kernel death keeps them; CELL6
    folds them into e24_raw.tar.gz and removes these."""
    d = WORK / arm
    files = [p for p in d.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl") and "profile" not in p.parts] \
        if sec in ("1", "6") and d.is_dir() else []
    if files:
        RAW_ARMS.mkdir(parents=True, exist_ok=True)
        with tarfile.open(RAW_ARMS / f"{arm}.tar.gz.tmp", "w:gz") as tf:
            for p in files:
                tf.add(p, arcname=str(p.relative_to(WORK)))
        (RAW_ARMS / f"{arm}.tar.gz.tmp").replace(RAW_ARMS / f"{arm}.tar.gz")


def dep_engine(eng):
    def f():
        if ENGINES[eng].get("failed"):
            return f"{eng} engine unavailable: {ENGINES[eng]['failed']}"[:300]
        if ENGINES[eng].get("stopped"):
            return f"{eng} launches stopped: {ENGINES[eng]['stopped']}"[:300]
        if eng == "sglang" and PATCH_BROKEN:
            return "PATCH_BROKEN: " + PATCH_BROKEN
        return None
    return f


def cache_root(arm, spec):
    """DESIGN §3.3: section 1 fresh roots (c_warm / c_nocache reuse s1_c's root); section 6 + SMOKE speed arms run on a
    COPY of the matching section-1 root; FULL_B builds one shared cache with its first launch."""
    pol = spec.get("cache") or "fresh"
    base = WORK / "cache"
    if pol == "fresh":
        root = base / arm
        shutil.rmtree(root, ignore_errors=True)
        return root, {"policy": "fresh", "root": str(root)}
    if pol == "shared":
        first = SHARED["first"]
        SHARED["first"] = False
        return base / "shared", {"policy": "shared", "root": str(base / "shared"), "first_launch": first}
    kind, src = pol
    if kind == "reuse":
        return base / src, {"policy": "reuse", "source": src, "root": str(base / src), "source_exists": (base / src).is_dir()}
    root, s = base / arm, base / src
    shutil.rmtree(root, ignore_errors=True)
    info = {"policy": "copy", "source": src, "root": str(root), "source_exists": s.is_dir()}
    if s.is_dir():
        try:
            shutil.copytree(s, root, symlinks=True, ignore_dangling_symlinks=True)
            info["bytes"] = _dir_size(root).get("bytes")
        except Exception as e:
            info["copy_error"] = f"{type(e).__name__}: {e}"[:300]
            err(f"cache_copy/{arm}", e)
    return root, info


def targets(spec):
    """The arm's expected hook targets (E24_HOOK_TARGETS, e24_hook TARGETS names): a target missing on any TP rank ->
    HOOK_MISSING. vLLM arms: none required (the GPUModelRunner probe is informational)."""
    if spec["engine"] != "sglang" or spec.get("kind") == "unit":
        return []
    f, h, env = spec["flags"], spec.get("hook") or {}, spec.get("env") or {}
    dflash = flag_value(f, "--speculative-algorithm") == "DFLASH"
    eager = bool(h.get("eager")) and "--disable-decode-cuda-graph" in f
    t = ["Scheduler"]
    if h.get("counters"):
        t += ["ScheduleBatch", "SchedulerBatchResultProcessor", "MambaComponent"] + (["DFlashWorkerV2"] if dflash else [])
    if eager and env.get("E24_INJECT_DRAFT_NAN") == "1":
        t += ["DFlashWorkerV2", "_grouped_conv"]
    if eager and env.get("E24_INJECT_SELECTOR_NAN") == "1":
        t += ["DFlashWorkerV2", "CandidateSelector"]
    return list(dict.fromkeys(t))


def arm_env(spec, cenv, hookdir):
    E = ENGINES[spec["engine"]]
    env = dict(E["env"])
    env.update(cenv)
    if spec["engine"] == "sglang":
        env["SGLANG_TORCH_PROFILER_DIR"] = str(rawdir(spec["arm"]) / "profile")
    env.update(spec.get("env") or {})
    hookdir.mkdir(parents=True, exist_ok=True)
    env.update(PYTHONPATH=str(HOOK_SITE), E24_HOOK_DIR=str(hookdir), E24_HOOK_TARGETS=",".join(targets(spec)),
               E24_ARM=spec["arm"], E24_RAW_DIR=str(rawdir(spec["arm"])))
    h = spec.get("hook") or {}
    if h.get("counters"):
        env["E24_HOOK_COUNTERS"] = "1"
    if h.get("probe"):
        env["E24_HOOK_PROBE"] = "1"
    if h.get("eager") and "--disable-decode-cuda-graph" in spec["flags"]:
        env["E24_HOOK_EAGER"] = "1"                              # injection wrappers only on decode-graph-off arms
    return env


def env_static(env, spec):
    """DESIGN §2: SGLANG_TEST_* / SGLANG_SIMULATE_* / E24_INJECT_* exist only in the arms that set them."""
    declared = set(spec.get("env") or {})
    bad = sorted(k for k in env if k.startswith(("SGLANG_TEST_", "SGLANG_SIMULATE_", "E24_INJECT_")) and k not in declared)
    fixed = {k: env.get(k) for k in ("SGLANG_LOGPROB_CHUNK_SIZE", "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK", "HF_HUB_OFFLINE",
                                     "TRANSFORMERS_OFFLINE")}
    ok = not bad and (spec["engine"] != "sglang" or (fixed["SGLANG_LOGPROB_CHUNK_SIZE"] == "128"
                                                     and fixed["SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK"] == "0"))
    return {"unexpected": bad, "fixed": fixed}, ("GO" if ok else "INVALID_ENV_UNEXPECTED")


def flag_check(spec, env):
    """ENFORCED (DESIGN §3.1): every --flag an exact 0.5.21 option string + prepare_server_args parse; FAKE: the fake's
    --e24-flag-check (same option list). resolve_once() standalone runs in SMOKE only and is ADVISORY: launch_server.py:29
    resolves the same args before anything loads, so a real conflict fails the launch within seconds anyway, while a
    standalone failure from the environment must not block an arm (FULL also saves ~15 s per distinct argv)."""
    eng = spec["engine"]
    E = ENGINES[eng]
    kw = {"env": env, "timeout": int(ts(300, 30))}
    if FAKE:
        kw["fake_server"] = E["fake_server"]
    elif eng == "sglang":
        kw.update(resolve=MODE == "SMOKE", model_path=flag_value(spec["flags"], "--model-path"))
    fc = call(L.flag_check, eng, E["vpy"], list(spec["flags"]), **kw) or {}
    if not fc.get("ran"):
        return fc, "ERROR_FLAG_CHECK_NOT_RUN"
    if fc.get("status") == "NO_GO_FLAG_RESOLUTION":
        fc["resolve_advisory"] = "resolve_once failed standalone (recorded, not blocking): the launch re-resolves"
        return fc, "GO"
    return fc, ("GO" if fc.get("ok") else (fc.get("status") if str(fc.get("status")).startswith("NO_GO") else "NO_GO_FLAG_UNSUPPORTED"))


def ensure_patches(expect):
    """Bring the installed sglang to exactly `expect` (revert what is extra, apply the expected set; apply() refuses
    43559 without 43567), then check all 7 known names against the expectation (DESIGN §3.4)."""
    global PATCH_BROKEN
    if PSITE is None:
        return {"state": "SKIPPED_FAKE" if FAKE else "NO_SITE"}, ("SKIPPED_FAKE" if FAKE else "NO_GO_PATCH_SITE")
    rec = {}
    chk = P.check(PSITE, names=list(PATCH_ALL7))
    cur = sorted(n for n, s in (chk.get("names") or {}).items() if s != "not_applied")
    rec["before"] = chk.get("names")
    if set(cur) != set(expect):
        if cur:
            rec["revert"] = P.revert(PSITE, names=cur)
        if expect:
            rec["apply"] = P.apply(PSITE, names=list(expect))
    chk2 = P.check(PSITE, names=list(PATCH_ALL7))
    rec.update(after=chk2.get("names"), files=chk2.get("files"))
    ok = bool(P.state_ok(chk2, expect_applied=tuple(expect)))
    if not ok and not expect:
        PATCH_BROKEN = "revert to the pristine 0.5.21 failed: no unpatched arm may run on a patched build"
    return rec, ("GO" if ok else "NO_GO_PATCH_STATE")


def compact_facts(f):
    f = dict(f or {})
    for k in ("error_lines", "oom_lines", "fp8_scale_lines"):
        if isinstance(f.get(k), list):
            f[k] = f[k][-12:]
    sa = f.pop("server_args", None)
    if isinstance(sa, dict):
        f["server_args_keys"] = len(sa)
        f["server_args_sel"] = {k: sa.get(k) for k in ("max_running_requests", "max_mamba_cache_size", "mamba_radix_cache_strategy",
                                                       "speculative_algorithm", "speculative_num_draft_tokens", "chunked_prefill_size",
                                                       "mem_fraction_static", "mamba_ssm_dtype", "prefill_decode_interval",
                                                       "enable_mixed_chunk", "disable_radix_cache", "attention_backend",
                                                       "linear_attn_backend", "context_length", "max_total_tokens") if k in sa}
    return f


def compact_info(si):
    if not isinstance(si, dict):
        return si
    keep = {k: si.get(k) for k in ("version", "max_total_num_tokens", "max_req_input_len", "launch_command", "prefill_decode_interval",
                                   "enable_mixed_chunk", "chunked_prefill_size", "max_running_requests") if k in si}
    st = (si.get("internal_states") or [{}])[0] if isinstance(si.get("internal_states"), list) else {}
    keep["internal"] = {k: st.get(k) for k in ("effective_max_running_requests_per_dp", "memory_usage", "decode_cuda_graph_bs")
                        if isinstance(st, dict) and k in st}
    return keep


def hook_files(hdir):
    """The e24_hook outputs of one launch: counters_<pid>.json, trace_<pid>.jsonl events, probe_<pid>.json."""
    hdir = Path(hdir)
    out = {"counters": [], "trace": [], "probe": []}
    for p in sorted(hdir.glob("*")) if hdir.is_dir() else []:
        try:
            if p.name.startswith("counters_") and p.suffix == ".json":
                out["counters"].append(json.loads(p.read_text()))
            elif p.name.startswith("probe_") and p.suffix == ".json":
                out["probe"].append(json.loads(p.read_text()))
            elif p.suffix == ".jsonl":
                out["trace"] += [json.loads(l) for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        except Exception:
            continue
    return out


TP = 4                                                    # every E24 server is TP4


def rank_files(hdir, tp=TP):
    """The TP scheduler ranks' counters files ONLY (hook rev 2b: top-level tp_rank + role_tag). The launcher, the
    detokenizer, API servers and trackers import target modules but run no rank: never counted as a rank."""
    by, ignored, dup = {}, 0, 0
    for c in hook_files(hdir)["counters"]:
        r = c.get("tp_rank")
        if c.get("role_tag") != "scheduler" or not isinstance(r, int) or isinstance(r, bool) or not 0 <= r < tp:
            ignored += 1
            continue
        if r in by:
            dup += 1
            if (c.get("t") or 0) < (by[r].get("t") or 0):
                continue
        by[r] = c
    return by, {"ranks": sorted(by), "ignored_files": ignored, "duplicate_ranks": dup}


def hook_selftest(hdir, spec, wait=60.0):
    """DESIGN §4.6: the arm's targets live on EXACTLY the TP ranks (one scheduler counters file per rank 0..tp-1 whose
    self_test is ok and lists every expected target); fewer ranks is fail-closed. FAKE servers have no sglang classes:
    a FAKE miss is INCONCLUSIVE_HOOK_FAKE, never GO."""
    need = targets(spec)
    if not need:
        return {"required": []}, "GO"
    end = time.time() + ts(wait, 5)
    while True:
        by, info = rank_files(hdir)
        okr = sorted(r for r, c in by.items() if (c.get("self_test") or {}).get("ok") and set(need) <= set(c.get("patched") or []))
        if len(okr) == TP or time.time() > end:
            break
        time.sleep(1.0)
    rep = dict(info, required=need, ranks_ok=okr, ranks_wanted=TP,
               missing=sorted({m for c in by.values() for m in ((c.get("self_test") or {}).get("missing") or [])}),
               failed=[(c.get("self_test") or {}).get("failed") for c in by.values() if (c.get("self_test") or {}).get("failed")][:4])
    if len(okr) == TP and len(by) == TP:
        return rep, "GO"
    return rep, ("INCONCLUSIVE_HOOK_FAKE" if FAKE else "INVALID_HOOK_MISSING")


def hook_summary(hdir, spec):
    """Env records (roles, guarded knobs vs the arm's declaration), per-rank counters (min / max: identical on the 4
    ranks?), M5 probe records."""
    try:
        recs = call(L.hook_records, str(hdir)) or []
        cnt = call(L.hook_counters, str(hdir)) or {}
    except Exception as e:
        err("hook_summary/" + spec["arm"], e)
        recs, cnt = [], {}
    knobs = sorted({k for r in recs for k in list(r.get("env") or {}) + list(r) if isinstance(k, str)
                    and k.startswith(("SGLANG_TEST_", "SGLANG_SIMULATE_", "E24_INJECT_"))})
    roles = {}
    for r in recs:
        k = str((r.get("role") or {}).get("kind") if isinstance(r.get("role"), dict) else r.get("role"))
        roles[k] = roles.get(k, 0) + 1
    by, rinfo = rank_files(hdir)                       # per-rank counters: the TP scheduler ranks only
    rc = [c.get("counters") or {} for c in by.values()]
    keys = sorted({k for c in rc for k, v in c.items() if isinstance(v, (int, float))})
    mn = {k: min(c.get(k, 0) for c in rc) for k in keys}
    mx = {k: max(c.get(k, 0) for c in rc) for k in keys}
    probes = []
    for pr in hook_files(hdir)["probe"]:
        probes.append({k: pr.get(k) for k in ("tp_rank", "where", "target", "draft", "mamba", "server_args", "modules")})
    return {"env_records": len(recs), "roles": roles, "knobs_seen": knobs,
            "knobs_undeclared": [k for k in knobs if k not in (spec.get("env") or {})], "counter_files": cnt.get("files"),
            "rank_files": rinfo, "counters_max": mx, "counters_identical": (mn == mx) if mx else None,
            "self_test_ok": {str(r): (c.get("self_test") or {}).get("ok") for r, c in by.items()},
            "probe": small(probes, 30000) if probes else []}


def gpu_gate(arm, rec):
    rel = L.wait_gpu_release(int(ts(180, 5)), fake=FAKE, gpus="0,1,2,3")
    if rel is False:
        try:
            rec["orphans_killed"] = call(L.kill_orphans, fake=FAKE)
        except Exception as e:
            err(arm + "/kill_orphans", e)
        rel = L.wait_gpu_release(int(ts(60, 5)), fake=FAKE, gpus="0,1,2,3")
    rec["gpu_released"] = rel
    if rel is False:
        rec["gates"]["gpu_release"] = "ERROR_GPU_RELEASE"
        ABORT["why"] = f"GPUs still held after {arm} (release gate)"


def with_server(arm, body, spec=None, wait=900.0, expect_errors=False):
    """One fresh server for one arm (DESIGN §3): env assert -> flag check (enforced) -> patch state (enforced) -> cache
    env (+ dir stats before) -> hook dir -> start -> wait (900 s) with the 600 s startup gate on EVERY launch -> facts /
    server_info / hook self-test -> body -> kill -> GPU release gate -> hook outputs -> cache verdict (section 1)."""
    spec = dict(spec or CAT[arm])
    spec["arm"] = arm
    eng = spec["engine"]
    rec = {"engine": eng, "gates": {}, "measured": {}}
    raw = rawdir(arm)
    hookdir = raw / "hook"
    shutil.rmtree(hookdir, ignore_errors=True)
    root, rec["cache"] = cache_root(arm, spec)
    cenv = call(L.cache_env, eng, str(root)) or {}
    env = arm_env(spec, cenv, hookdir)
    argv = list(ENGINES[eng]["cmd"]) + list(spec["flags"])
    R["argv"][arm] = argv
    rec.update(argv=argv, env_arm=spec.get("env"), patches_expected=list(spec.get("patches") or ()), hook_targets=targets(spec))
    rec["env_static"], rec["gates"]["env_static"] = env_static(env, spec)
    iso = spec["section"] == "1"
    paths = list(L.default_cache_dirs()) if iso else []
    before = call(L.dir_stats, paths) if iso else None
    rec["flag_check"], rec["gates"]["flags"] = flag_check(spec, env)
    if rec["gates"]["flags"] != "GO":
        rec["status"] = worst(rec["gates"].values())
        return rec
    if eng == "sglang":
        rec["patch_state"], rec["gates"]["patches"] = ensure_patches(spec.get("patches") or ())
        if rec["gates"]["patches"] not in ("GO", "SKIPPED_FAKE"):
            rec["status"] = worst(rec["gates"].values())
            return rec
    port = SGL_PORT if eng == "sglang" else VLLM_PORT
    srv = call(L.Server, arm, argv, env, port, str(LOGS), gpus="0,1,2,3", engine=eng)
    try:
        srv.start()
        ok, why = srv.wait_ready(min(ts(wait, 30), max(60.0, left() - 120)))
        rec.update(ready=ok, ready_why=why, ready_sec=srv.ready_sec)
        if not ok:
            rec["log_tail"] = srv.log()[-4000:]
            try:
                fx = srv.facts() or {}
            except Exception:
                fx = {}
            rec["facts"] = compact_facts(fx)
            rec["gates"]["launch"] = ("NO_GO_SM89_KERNEL" if fx.get("cuda_invalid_kernel") else
                                      "NO_GO_OOM" if fx.get("oom_lines") else "NO_GO_LAUNCH")
            launch_streak(eng, arm, rec["gates"]["launch"], why, [argv, sorted((spec.get("env") or {}).items())])
            return rec
        LAUNCH_FAILS[eng] = []                            # a launch that came up proves the engine starts
        rec["gates"]["launch"] = "GO" if (srv.ready_sec or 0) <= ts(600) else "NO_GO_STARTUP_600"
        if rec["gates"]["launch"] != "GO":
            rec["diagnostic"] = "startup gate failed: measurements kept (startup time does not change numerics), adoption blocked"
        if rec["cache"].get("policy") == "shared" and not rec["cache"].get("first_launch") and (srv.ready_sec or 0) > ts(300):
            rec["cache_miss"] = "CACHE_MISS"
        fx = srv.facts() or {}
        rec["facts"] = compact_facts(fx)
        if fx.get("cuda_invalid_kernel"):
            rec["gates"]["sm89"] = "NO_GO_SM89_KERNEL"
        if eng == "sglang":
            try:
                rec["server_info"] = compact_info(srv.server_info())
            except Exception as e:
                rec["server_info"] = {"error": repr(e)[:200]}
                err(arm + "/server_info", e)
        rec["hook_selftest"], rec["gates"]["hook"] = hook_selftest(hookdir, spec)
        rec["gpu_mem_after_ready"] = L.gpu_mem()
        rec["model"] = getattr(srv, "model_name", None)
        try:
            out = body(srv) or {}
            for k, v in (out.pop("gates", None) or {}).items():
                rec["gates"][k] = v
            rec.update(out)
        except Exception as e:
            rec["gates"]["body"] = "ERROR_BODY_" + type(e).__name__
            rec["body_error"] = f"{type(e).__name__}: {e}"[:800]
            err(arm + "/body", e)
        try:
            fx2 = srv.facts() or {}
            rec["facts_end"] = {k: fx2.get(k) for k in ("error_lines", "oom_lines", "allocator_oom_warnings") if fx2.get(k)}
        except Exception as e:
            err(arm + "/facts_end", e)
        text = srv.log()
        pre_kill = text.partition("SIGTERM received")[0]          # kill noise after our own SIGTERM is not an engine error
        rec["error_sign_hits"] = {s: pre_kill.count(s) for s in ERR_SIGNS if s in pre_kill}
        if rec["error_sign_hits"] and not expect_errors:
            rec["gates"]["engine_errors"] = "NO_GO_ENGINE_ERROR"
        if rec.get("sampled") and "falling back to greedy argmax verification" in text:
            rec["gates"]["greedy_fallback"] = "NO_GO_GREEDY_FALLBACK"           # a T>0 measurement silently greedy
        if not srv.alive():
            rec["gates"].setdefault("alive", "ERROR_SERVER_DIED" if not expect_errors else "INCONCLUSIVE_SERVER_DIED")
            rec["log_tail"] = text[-4000:]
            if rec["gates"].get("engine_errors") == "NO_GO_ENGINE_ERROR":
                # the engine logged a known failure sign and died: the dead server and the measurements that lost their
                # connection are its consequence, not harness errors -> NO_GO (the candidate failed); originals kept
                rec["engine_consequence_gates"] = {k: v for k, v in rec["gates"].items() if str(v).startswith("ERROR_")}
                for k in rec["engine_consequence_gates"]:
                    rec["gates"][k] = "NO_GO_ENGINE_ERROR"
        rec["gpu_mem_end"], rec["host_rss_mb"] = L.gpu_mem(), srv.rss_mb()
        return rec
    finally:
        try:
            srv.kill()
        except Exception as e:
            err(arm + "/kill", e)
        gpu_gate(arm, rec)
        try:
            rec["hook"] = hook_summary(hookdir, spec)
            if rec["hook"].get("knobs_undeclared"):
                rec["gates"]["env_records"] = "INVALID_ENV_UNEXPECTED"
        except Exception as e:
            err(arm + "/hook_summary", e)
        if iso:
            try:
                after = call(L.dir_stats, sorted(set(paths) | set(L.default_cache_dirs())))   # a NEW /tmp/torchinductor_* too
                cv = call(L.cache_verdict, eng, str(root), str(hookdir), before, after,
                          dflash=flag_value(spec["flags"], "--speculative-algorithm") == "DFLASH") or {}
                rec["cache_verdict"] = cv
                rec["gates"]["cache_isolation"] = "GO" if cv.get("status") == "ISOLATED" else "INVALID_CACHE_ISOLATION"
            except Exception as e:
                rec["gates"]["cache_isolation"] = "INVALID_CACHE_ISOLATION"
                err(arm + "/cache_verdict", e)
        rec["status"] = worst(rec["gates"].values())


# ---- shared measurement helpers
def flush(eng, port):
    """/flush_cache on an idle server; valid = flushed AND idle afterwards (E17 rule)."""
    if eng != "sglang":
        return {"valid": None, "why": "vLLM oracle arms run with prefix caching off"}
    f = call(L.flush_cache, eng, port, wait=ts(60, 3)) or {}
    if not f.get("ok"):
        time.sleep(ts(2.0, 0.2))
        f = call(L.flush_cache, eng, port, wait=ts(30, 3)) or {}
    f["idle_after"] = call(L.wait_idle, eng, port, wait=ts(30, 3))
    f["valid"] = bool(f.get("ok") and f["idle_after"])
    return f


def between_loads(eng, port):
    """DESIGN §4.4 / §4.5 (review #51 / #60): wait_idle + /flush_cache (valid only when idle) + a baseline snapshot."""
    f = flush(eng, port)
    return {"flush": f, "snapshot": L.snapshot(eng, port) if eng == "sglang" else None}


def tmo():
    return int(min(3600, max(120, left() / 4)))


def row_summary(rows):
    rows = rows or []
    return {"n": len(rows), "errors": [r.get("error") for r in rows if r.get("error")][:4],
            # a failed request is an error, not garbage output (a dead or killed server must not read as repeated garbage)
            "garbage": [None if r.get("error") else L.degenerate(r.get("text")) for r in rows], "finish": [r.get("finish_reason") for r in rows],
            "cached_tokens": [r.get("cached_tokens") for r in rows], "lens": [len(r.get("tokens") or []) for r in rows]}


def keyed(rows, keys, ns):
    for k, r in zip(keys, rows or []):
        if isinstance(r, dict):
            r["key"] = ns + k
    return rows or []


def meas(o, name, fn):
    """One measurement inside an arm: an exception is recorded (gate ERROR_<name>) and never loses the others."""
    t = time.time()
    try:
        o["measured"][name] = fn()
    except Exception as e:
        o["measured"][name] = {"error": f"{type(e).__name__}: {e}"[:500]}
        o["gates"][name] = "ERROR_" + name.upper()
        err(f"measure/{name}", e)
    if isinstance(o["measured"][name], dict):
        o["measured"][name]["sec"] = round(time.time() - t, 1)
    save()
    return o["measured"][name]


'''

CELL5 = r'''# ==================================================================== CELL 5 — the plan (DESIGN rev 2 §4-§5)
ROWS, TFD, SELF, M4D, TRACES = {}, {}, {}, {}, {}      # section-1 data per arm (raw copies under WORK/<arm>/)
C16F = {}                                                # arm -> [flush valid before c16a, before c16b] (vLLM: None = no cache)
S2 = {"pool": None, "ctrl_rows": None, "aa_rows": None}
S5 = {"control": {}}
SENT = os.environ.get("E24_SENTINEL_PREFIX", "e24-sentinel-")
TOL, AA_MAX = 5e-3, 1e-3                                 # section 6 KL: fixed tolerance; reference A/A ceiling (E17 rev 2)
BINS8 = ((0, 2000), (2000, 8000), (8000, 16000), (16000, 30000))
BINS90 = ((90000, 100000), (100000, 110000), (110000, 120000))   # context bins of the 90k trace (offset = prompt length)


def skip_cond(sec, arm, why):
    rec = R["sections"].setdefault(sec, {}).setdefault(arm, {})
    rec.update(status="SKIPPED_CONDITION", why=why)
    R["order"].append(f"{sec}/{arm}")
    save()
    return rec


# ------------------------------------------------------------------------------------------------ section 1
S1_MEAS = {   # arm: (FULL, SMOKE) measurement list (DESIGN §4.1 + the §5 minute budgets)
    "s1_a": ("traces m1 m2 m3", "traces m1 self m3"), "s1_b": ("m1 m2 m3", "m1 self m3"),
    "s1_d6": ("pre m1 self p2 m2 m3 m4", "pre m1 self p2 m2 m3 m4"), "s1_d6fr": ("m1 self m3", "m1 self"),
    "s1_d4": ("m1 self p2 m2 m3 m4", "m1 self p2"), "s1_c": ("m1 self p2 m2 m3 m4", "m1 self p2 m2 m3 m4"),
    "s1_c_warm": ("m1 self m3", "m1"), "s1_c_nocache": ("m1 self p2 m3", "m1"), "s1_c32": ("m1 self p2 m2 m3 m4", "m1"),
    "s1_d6_r2": ("m1 self p2 m2 m3", "m1"), "s1_c_r2": ("m1 self p2 m3", "m1")}


def m_rows(eng, srv, arm, tag, items, n_tok, ns, concurrent=False):
    keys = [k for k, _ in items]
    rows = call(L.greedy, eng, srv.port, [p for _, p in items], n_tok=n_tok, top=2, timeout=tmo(), model=srv.model_name,
                concurrent=concurrent, keys=keys)
    rows = keyed(rows, keys, ns)
    ROWS.setdefault(arm, {})[tag] = rows
    jdump(rawdir(arm) / f"rows_{tag}.json", rows)
    return row_summary(rows)


def m_traces(eng, srv, arm):
    """M3 references (s1_a, vLLM FP8): production sampling (T 1, top_p 0.95, top_k 20), fixed seeds, ignore_eos, the
    natural EOS position recorded; generated concurrently (one batch of 2-3)."""
    keys = list(TRACE_P)
    gens = call(L.generate_logged_many, eng, srv.port, [TRACE_P[k] for k in keys], max_new=TRACE_NEW, temperature=1.0,
                top_p=0.95, top_k=20, seeds=[20261005 + i for i in range(len(keys))], ignore_eos=True, eos_ids=EOS_IDS, top=1,
                timeout=int(min(7200, max(300, left() - 600))), model=srv.model_name) or []
    for k, g in zip(keys, gens):
        g = g or {}
        TRACES[k] = {"prompt": TRACE_P[k], "cont": list(g.get("tokens") or []), "natural_eos_pos": g.get("natural_eos_pos"),
                     "error": g.get("error"), "seed": 20261005 + keys.index(k)}
    jdump(rawdir(arm) / "traces.json", TRACES)
    return {k: {"n": len(v["cont"]), "want": TRACE_NEW, "natural_eos_pos": v["natural_eos_pos"], "error": v["error"]}
            for k, v in TRACES.items()}


def m_score(eng, srv, arm):
    """M3: every arm teacher-forces the FP8 traces, one request at a time, top-2 (SGLang: logprob chunk 128)."""
    if not TRACES:
        return {"status": "INCONCLUSIVE_NO_TRACES"}
    out = {}
    for k, tr in TRACES.items():
        if not tr["cont"]:
            out[k] = {"status": "INCONCLUSIVE_NO_TRACE"}
            continue
        tf = call(L.teacher_forced, eng, srv.port, tr["prompt"], tr["cont"], top=2, timeout=tmo(), model=srv.model_name) or {}
        TFD.setdefault(arm, {})[k] = tf
        jdump(rawdir(arm) / f"tf_{k}.json", tf)
        out[k] = {"n": tf.get("n"), "status": tf.get("status"), "error": tf.get("error")}
        if not srv.alive():
            out[k]["server_died"] = True
            break
    return out


def m_pre(eng, srv, arm):
    """The first SGLang arm scores one 4,096-row extend before M3 (rev 2 blocker #1: logprob OOM)."""
    tf = call(L.teacher_forced, eng, srv.port, PREFLIGHT[0], PREFLIGHT[1], top=2, timeout=tmo(), model=srv.model_name) or {}
    ok = srv.alive() and not tf.get("error") and tf.get("status") != "INCONCLUSIVE_TF_ALIGN"
    PREFLIGHT_DONE.update(done=True, ok=ok, arm=arm)
    fl = flush(eng, srv.port) if ok else None                # the pre-flight prompt is a trace prompt: leave no radix entry
    return {"rows": len(PREFLIGHT[1]), "n": tf.get("n"), "status": tf.get("status"), "error": tf.get("error"), "alive_after": srv.alive(),
            "flush_after": fl, "gate": "GO" if ok else "NO_GO_LOGPROB_PREFLIGHT"}


def m_self(eng, srv, arm):
    """Prefill re-score of each serial M1 row along its own greedy path (top-2 at every position): with both arms'
    self scores, a FLIP at k is attributed decode/verify-path specific (prefills agree) or engine-level."""
    out = 0
    for (key, p), r in zip(ORACLE, (ROWS.get(arm) or {}).get("m1") or []):
        if r.get("tokens"):
            tf = call(L.teacher_forced, eng, srv.port, p, r["tokens"], top=2, timeout=tmo(), model=srv.model_name) or {}
            SELF.setdefault(arm, {})["s:" + key] = tf
            out += 1
    jdump(rawdir(arm) / "self_tf.json", SELF.get(arm) or {})
    return {"rows": out}


def m_m2(eng, srv, arm, sampled):
    """M2: the c16 set twice, each batch after a flush, so c16b is a pure repeat of c16a (the A/A) and never a radix re-run
    of it; a restore inside the A/A would turn restore errors into A/A allowance (review: decision logic #4)."""
    fa = flush(eng, srv.port)
    a = m_rows(eng, srv, arm, "c16a", C16, 256, "c:", concurrent=True)
    fb = flush(eng, srv.port)
    out = {"c16a": a, "c16b": m_rows(eng, srv, arm, "c16b", C16, 256, "c:", concurrent=True),
           "flush_valid": [fa.get("valid"), fb.get("valid")]}
    C16F[arm] = out["flush_valid"]
    if sampled:      # a smoke garbage check, no pair verdict (16 rows cannot detect EOS-rate differences)
        gens = call(L.generate_logged_many, eng, srv.port, [p for _, p in C16], max_new=2048, temperature=1.0, top_p=0.95,
                    top_k=20, seeds=[7000 + i for i in range(16)], ignore_eos=False, eos_ids=EOS_IDS, top=1, timeout=tmo(),
                    model=srv.model_name) or []
        jdump(rawdir(arm) / "t1_batch.json", gens)
        out["t1_smoke_garbage"] = {"label": "smoke garbage check (no pair verdict)", "n": len(gens),
                                   "garbage": [L.degenerate(g.get("text")) for g in gens if isinstance(g, dict)],
                                   "natural_eos_pos": [g.get("natural_eos_pos") for g in gens if isinstance(g, dict)],
                                   "lens": [len(g.get("tokens") or []) for g in gens if isinstance(g, dict)]}
    return out


def m_m4(eng, srv, arm):
    """M4: production-sampled c16 (fixed seeds, ignore_eos, output logprobs), then the same server re-scores every
    trace by prefill (after a flush: prefill path only)."""
    gens = call(L.generate_logged_many, eng, srv.port, M4_P, max_new=M4_NEW, temperature=1.0, top_p=0.95, top_k=20,
                seeds=[30000 + i for i in range(len(M4_P))], ignore_eos=True, eos_ids=EOS_IDS, top=1,
                timeout=int(min(7200, max(300, left() - 600))), model=srv.model_name) or []
    fl = flush(eng, srv.port)
    tfs = []
    for p, g in zip(M4_P, gens):
        toks = list((g or {}).get("tokens") or [])
        tfs.append(call(L.teacher_forced, eng, srv.port, p, toks, top=2, timeout=tmo(), model=srv.model_name) if toks
                   else {"error": "no generated tokens"})
    M4D[arm] = {"gen": gens, "tf": tfs}
    jdump(rawdir(arm) / "m4.json", M4D[arm])
    own = call(L.decode_vs_prefill, gens, tfs)
    return {"flush": fl, "streams": len(gens), "tokens": [len((g or {}).get("tokens") or []) for g in gens], "own": own}


def s1(arm):
    """A section-1 arm: fresh isolated cache, its measurement list, M5 rank probe (hook), M6 tool probe (SGLang)."""
    spec = CAT[arm]
    eng = spec["engine"]

    def body(srv):
        o = {"gates": {}, "measured": {}}
        plan = [m for m in (S1_MEAS[arm][1 if SM else 0] or "").split() if m != "pre"]
        if eng == "vllm" and not SM and "self" in plan:
            plan.remove("self")
        heavy = [m for m in plan if m in ("m3", "m4")]       # >= 4,096-row extends with logprobs: the OOM the pre-flight guards
        if eng == "sglang" and heavy and not PREFLIGHT_DONE["done"]:
            plan.insert(plan.index(heavy[0]), "pre")         # DESIGN §3.1: "before M3" (review 2: never before M1)
        try:
            o["probes"] = call(L.chat_probes, eng, srv.port, model=srv.model_name)
        except Exception as e:
            o["probes"] = {"error": repr(e)[:200]}
            err(arm + "/chat_probes", e)
        for m in plan:
            if m in ("m3", "m4") and eng == "sglang" and PREFLIGHT_DONE["done"] and not PREFLIGHT_DONE["ok"]:
                o["measured"][{"m3": "m3_score", "m4": "m4"}[m]] = {"status": "NOT_MEASURED", "why": "the logprob pre-flight "
                                                                    f"failed on {PREFLIGHT_DONE['arm']}: no >= 4,096-row extend"}
                continue
            if m == "pre":
                r = meas(o, "pre", lambda: m_pre(eng, srv, arm))
                o["gates"]["preflight"] = (r or {}).get("gate", "ERROR_PREFLIGHT")
            elif m == "traces":
                meas(o, "m3_traces", lambda: m_traces(eng, srv, arm))
                o["sampled"] = True
            elif m == "m1":
                meas(o, "m1", lambda: m_rows(eng, srv, arm, "m1", ORACLE, NTOK, "s:"))
            elif m == "self":
                meas(o, "self_tf", lambda: m_self(eng, srv, arm))
            elif m == "p2":
                meas(o, "m1_pass2", lambda: m_rows(eng, srv, arm, "p2", ORACLE, NTOK, "s:"))
            elif m == "m2":
                meas(o, "m2", lambda: m_m2(eng, srv, arm, sampled=not spec.get("greedy_only")))
                o["sampled"] = o.get("sampled") or not spec.get("greedy_only")
            elif m == "m3":
                meas(o, "m3_score", lambda: m_score(eng, srv, arm))
            elif m == "m4":
                meas(o, "m4", lambda: m_m4(eng, srv, arm))
                o["sampled"] = True
            if not srv.alive():
                o["gates"]["alive"] = "ERROR_SERVER_DIED"
                break
        if eng == "sglang" and srv.alive():
            o["m6_tool"] = call(L.tool_probe, eng, srv.port, model=srv.model_name)
            o["gates"]["m6_tool"] = "GO" if o["m6_tool"].get("ok") else "NO_GO_TOOL_PROBE"
        else:
            o["m6_tool"] = {"status": "NOT_APPLICABLE", "why": "vLLM oracle argv has no tool-call parser"}
        for k in ("m1", "m1_pass2"):
            g = [x for x in ((o["measured"].get(k) or {}).get("garbage") or []) if x]
            if len(g) >= 2:
                o["gates"]["garbage_" + k] = "NO_GO_REPEATED_GARBAGE"
        return o

    def f():
        rec = with_server(arm, body)
        rec["m5_probe"] = (rec.get("hook") or {}).get("probe") or "INCONCLUSIVE_NO_PROBE_RECORD"
        try:
            R["comparisons"]["1"] = s1_compare_all(final=False)
        except Exception as e:
            err("comparisons/1", e)
        return rec
    return f


def _cls_counts(c):
    cnt = dict(c.get("counts") or c.get("class_counts") or {})
    if not cnt:
        for r in c.get("rows") or []:
            k = r.get("class") or r.get("status")
            cnt[k] = cnt.get(k, 0) + 1
    return cnt


def hm(c, kind="same_weights"):
    """High-margin count of a compare() result: FLIP + STATE_FLIP (same weights) or STATE_FLIP (cross weights)."""
    c = c or {}
    if kind == "cross_weights":
        return int(c.get("state_flips") if c.get("state_flips") is not None else _cls_counts(c).get("STATE_FLIP", 0))
    if isinstance(c.get("high"), int):
        return c["high"]
    cnt = _cls_counts(c)
    return int(cnt.get("STATE_FLIP", 0)) + int(cnt.get("FLIP", 0))


def _rows(arm, *tags):
    out = []
    for t in tags:
        out += list((ROWS.get(arm) or {}).get(t) or [])
    return out


def _tp(a, b):
    """T_pair = max(0.5, q99.9 of the pair's |delta(top1 - top2)| on identical prefixes), from M3; per trace, the max."""
    vals = []
    for k in TRACES:
        x, y = (TFD.get(a) or {}).get(k), (TFD.get(b) or {}).get(k)
        if x and y and not x.get("error") and not y.get("error"):
            try:
                vals.append(float(call(L.tpair, x, y)))
            except Exception as e:
                err(f"tpair/{a}~{b}/{k}", e)
    return (max([0.5] + vals), "m3" if vals else "default_0.5")


def _attr(ref, cand, c):
    """FLIP attribution by both arms' prefill re-scores at the first differing token."""
    out = []
    for r in c.get("rows") or []:
        if (r.get("class") or r.get("status")) not in ("FLIP", "STATE_FLIP"):
            continue
        key, k = r.get("key"), r.get("first_diff")
        pf = []
        for arm in (ref, cand):
            tf = (SELF.get(arm) or {}).get(key) or {}
            pos = tf.get("positions") or []
            pf.append(pos[k][1] if isinstance(k, int) and k < len(pos) and pos[k] else None)
        out.append({"key": key, "k": k, "class": r.get("class"), "prefill_top1": pf,
                    "attribution": ("partial" if None in pf else ("decode_verify_path" if pf[0] == pf[1] else "engine_level"))})
    return out


S1_PAIRS = [("s1_a", "s1_b", "cross_weights", "weights + KV scale"), ("s1_b", "s1_c32", "same_weights", "engine (both fp32 SSM)"),
            ("s1_c32", "s1_c", "same_weights", "SSM dtype"), ("s1_c", "s1_d6", "same_weights", "speculation W6"),
            ("s1_c", "s1_d4", "same_weights", "speculation W4"), ("s1_c", "s1_d6fr", "same_weights", "target-verify path vs plain decode"),
            ("s1_c", "s1_c_warm", "same_weights", "cold vs warm cache"),
            ("s1_c", "s1_c_nocache", "same_weights", "radix / Mamba checkpoint reuse"),
            ("s1_b", "s1_c", "same_weights", "mixed engine + SSM (README literal b vs c, report only)")]
# A/A = pure repeats only: c16 <-> c16 (each batch after a flush) and the multi-boot r2 arms. Cold <-> warm and pass 1 <->
# pass 2 are test pairs with their own verdicts: as allowance they would hide the very restore / cache errors they test
# (review: decision logic #4; DESIGN §4.1 "NEAR_TIE is expected", §3.6 restore gate)
S1_AA_BOOT = [("s1_d6", "s1_d6_r2"), ("s1_c", "s1_c_r2")]
S1_CHAIN = [("s1_a", "s1_b"), ("s1_b", "s1_c32"), ("s1_c32", "s1_c"), ("s1_c", "s1_d6"), ("s1_c", "s1_d4")]
SPEC_PAIRS = {("s1_c", "s1_d6"), ("s1_c", "s1_d4"), ("s1_c", "s1_d6fr")}
S1_TARGET_ONLY = {"s1_c32", "s1_c", "s1_c_warm", "s1_c_nocache", "s1_c_r2"}   # README gate: target-only high-margin drift
S1_REPORT_ONLY = {("s1_b", "s1_c")}                                         # README literal b vs c: reported, never a verdict
M4_BASE = {"s1_d6": "s1_c", "s1_d4": "s1_c", "s1_c": "s1_c32"}


def _same_restore(a_rows, b_rows):
    """A/A rows only where both runs resumed from the same cached prefix (cached_tokens equal): the ARC prompts share a
    64-token header, so "cold" rows are partial restores whose hit pattern depends on admission order; a row that hit in
    one run and not in the other compares two different paths, not A/A noise (second validation round, FAKE scenario
    restore_corruption). Returns (a_rows, b_rows, n_excluded)."""
    cb = {r.get("key"): r.get("cached_tokens") or 0 for r in b_rows or []}
    keep = {r.get("key") for r in a_rows or [] if r.get("key") in cb and (r.get("cached_tokens") or 0) == cb[r.get("key")]}
    return ([r for r in a_rows or [] if r.get("key") in keep], [r for r in b_rows or [] if r.get("key") in keep],
            len({r.get("key") for r in a_rows or []} & set(cb)) - len(keep))


def _n_eval(c):
    """Rows a compare() result judged (aligned keys minus INCONCLUSIVE rows): the denominator of an A/A rate."""
    c = c or {}
    return max(0, int(c.get("n_pairs") or 0) - int(_cls_counts(c).get("INCONCLUSIVE", 0)))


def _rate(cmps, state=False):
    """Max per-row rate of FLIP + STATE_FLIP (state: STATE_FLIP only) over usable A/A comparisons; None without one."""
    v = [(hm(c, "cross_weights") if state else hm(c)) / _n_eval(c) for c in cmps or () if isinstance(c, dict) and _n_eval(c) > 0]
    return max(v) if v else None


S1_ALPHA = 0.05      # one-sided Fisher test per link / drift verdict: null chain false-decisive <= ~5 % at p <= 0.05 (MC)


def _fisher_greater(x, n1, y, n2):
    """One-sided Fisher exact test: P(X >= x) for X ~ Hypergeometric(n1 + n2 rows, x + y high-margin rows, n1 drawn),
    the chance that a pair with no real difference shows x of its n1 judged rows high-margin when y of n2 pure A/A rows
    did. None without rows on either side."""
    if not n1 or not n2 or x is None or y is None:
        return None
    k_all, m = x + y, n1 + n2
    return sum(math.comb(k_all, k) * math.comb(m - k_all, n1 - k) for k in range(x, min(k_all, n1) + 1)) / math.comb(m, n1)


def _aa_pool(aa, *arms):
    """(FLIP + STATE_FLIP rows, judged rows) over the distinct pure A/A comparisons of the arms; (None, None) without one."""
    seen = {id(c): c for a in arms for c in (aa.get(a) or ()) if isinstance(c, dict) and _n_eval(c) > 0}
    return (sum(hm(c) for c in seen.values()), sum(_n_eval(c) for c in seen.values())) if seen else (None, None)


def _allow(rates, n):
    """A/A allowance scaled to the rows a pair judged: ceil(rate x n) (review #12: a 16-row A/A against a 25-row pair is
    not the same count); None when neither arm has a usable A/A."""
    r = [x for x in rates if x is not None]
    return int(math.ceil(max(r) * n - 1e-9)) if r else None


def _pair(ref_rows, cand_rows, kind, t_pair, rates, state_rates):
    """compare() with the scaled allowance: one pass for the judged rows, one with aa_high = the allowance (cross weights:
    the STATE_FLIP allowance). Returns (result, allowance FLIP + STATE_FLIP, allowance STATE_FLIP, judged rows)."""
    c0 = call(L.compare, ref_rows, cand_rows, kind=kind, t_pair=t_pair, aa_high=None) or {}
    n = _n_eval(c0)
    ah, ast = _allow(rates, n), _allow(state_rates, n)
    use = ast if kind == "cross_weights" else ah
    c = (call(L.compare, ref_rows, cand_rows, kind=kind, t_pair=t_pair, aa_high=use) or {}) if use is not None else c0
    return c, ah, ast, n


def _usable(arm):
    st = str((R["sections"].get("1", {}).get(arm) or {}).get("status") or "")
    gates = (R["sections"].get("1", {}).get(arm) or {}).get("gates") or {}
    return bool(ROWS.get(arm)) and gates.get("cache_isolation") != "INVALID_CACHE_ISOLATION" and not st.startswith("SKIPPED")


def s1_compare_all(final=False):
    """DESIGN §3.2 + §4.1: A/A per arm from pure repeats (c16 <-> c16 after flushes, r2 boots) as per-row rates; every pair
    aligned by key with its calibrated T_pair and the A/A allowance scaled to its judged rows; the within-arm test pairs
    (pass 1 <-> pass 2 restore, serial vs c16) and cold <-> warm with their own verdicts; FLIP attribution, M3 per trace,
    M4 with the length-matched baseline, the decisive-pair chain a -> b -> c32 -> c -> d6 / d4 (unevaluated links are
    never clean), and the arm verdicts written into the arm statuses. Recomputed in the report cell with every A/A."""
    out = {"final": final, "aa": {}, "within": {}, "pairs": {}, "m4": {}, "chain": [], "decisive": None}
    aa = {}
    for arm in ROWS:
        w = {}
        if _rows(arm, "c16a") and _rows(arm, "c16b"):
            xa, xb, w["c16_aa_rows_restore_mismatch"] = _same_restore(_rows(arm, "c16a"), _rows(arm, "c16b"))
            w["c16_aa"] = call(L.compare, xa, xb, kind="same_weights", t_pair=0.5, aa_high=None)
            fl = C16F.get(arm) or []
            w["c16_aa_pure"] = bool(fl) and all(x in (True, None) for x in fl)   # both batches after a valid flush (vLLM: no cache)
        out["within"][arm] = w
        aa[arm] = [w["c16_aa"]] if w.get("c16_aa_pure") and isinstance(w.get("c16_aa"), dict) else []
    for a, b in S1_AA_BOOT:
        if _usable(a) and _usable(b):
            tags = ("m1", "c16a") if all((C16F.get(x) or [True])[0] in (True, None) for x in (a, b)) else ("m1",)
            xa, xb, nx = _same_restore(_rows(a, *tags), _rows(b, *tags))
            c = call(L.compare, xa, xb, kind="same_weights", t_pair=0.5, aa_high=None)
            if isinstance(c, dict):
                c["rows_restore_mismatch"] = nx
            out["aa"][f"{a}~{b}"] = c
            aa.setdefault(a, []).append(c)
            aa.setdefault(b, []).append(c)
    out["aa_rate"] = {k: _rate(v) for k, v in aa.items()}
    out["aa_state_rate"] = {k: _rate(v, state=True) for k, v in aa.items()}
    out["aa_high"] = {k: L.aa_high_of(v) for k, v in aa.items()}          # raw A/A counts (display; verdicts use the rates)
    out["aa_state"] = {k: (max(hm(c, "cross_weights") for c in v if not str(c.get("status", "")).startswith("INCONCLUSIVE"))
                           if any(not str(c.get("status", "")).startswith("INCONCLUSIVE") for c in v) else None) for k, v in aa.items()}
    for arm, w in out["within"].items():
        rr, rs = [out["aa_rate"].get(arm)], [out["aa_state_rate"].get(arm)]
        if _rows(arm, "m1") and _rows(arm, "p2"):              # restore test: a radix hit through a Mamba checkpoint restore
            c, ah, ast, n = _pair(_rows(arm, "m1"), _rows(arm, "p2"), "same_weights", 0.5, rr, rs)
            cached = [r.get("cached_tokens") for r in _rows(arm, "p2")]
            hy, ny = _aa_pool(aa, arm)
            w.update(pass1_pass2=c, pass1_pass2_allow={"high": ah, "state": ast, "rows": n}, pass2_cached_tokens=cached,
                     pass1_pass2_p=_fisher_greater(hm(c), n, hy, ny),
                     pass2_restore="NOT_EXERCISED" if not any(isinstance(x, int) and x > 0 for x in cached) else "EXERCISED")
        if _rows(arm, "m1") and _rows(arm, "c16a"):            # README "serial vs c16": the shared 8k keys
            ser = [dict(r, key=str(r.get("key", "")).replace("s:", "")) for r in _rows(arm, "m1") if "8k#" in str(r.get("key"))]
            c16 = [dict(r, key=str(r.get("key", "")).replace("c:", "")) for r in _rows(arm, "c16a")]
            c16 = [r for r in c16 if r["key"] in {x["key"] for x in ser}]
            c, ah, ast, n = _pair(ser, c16, "same_weights", 0.5, rr, rs)
            w.update(serial_vs_c16=c, serial_vs_c16_allow={"high": ah, "state": ast, "rows": n})
    for ref, cand, kind, label in S1_PAIRS:
        key = f"{ref}~{cand}"
        if not (_usable(ref) and _usable(cand)):
            out["pairs"][key] = {"label": label, "kind": kind, "status": "INCONCLUSIVE_ARM_MISSING_OR_INVALID",
                                 "ref_status": (R["sections"]["1"].get(ref) or {}).get("status"),
                                 "cand_status": (R["sections"]["1"].get(cand) or {}).get("status")}
            continue
        tp, tsrc = _tp(ref, cand)
        rs = [out["aa_state_rate"].get(ref), out["aa_state_rate"].get(cand)]
        rr = rs if kind == "cross_weights" else [out["aa_rate"].get(ref), out["aa_rate"].get(cand)]   # cross: only STATE_FLIPs count
        c, ah, ast, n = _pair(_rows(ref, "m1", "c16a"), _rows(cand, "m1", "c16a"), kind, tp, rr, rs)
        aah = ast if kind == "cross_weights" else ah
        hy, ny = _aa_pool(aa, ref, cand)
        pv = _fisher_greater(hm(c, kind), n, hy, ny) if kind == "same_weights" else None
        p = {"label": label, "kind": kind, "t_pair": tp, "t_pair_source": tsrc, "aa_high": aah, "aa_state_allow": ast,
             "rows_judged": n, "hm": hm(c, kind), "state_flips": c.get("state_flips"), "garbage": c.get("garbage"), "p_value": pv,
             "inconclusive_rows": _cls_counts(c).get("INCONCLUSIVE", 0),
             "status": c.get("status"), "counts": _cls_counts(c), "positions_examined": c.get("positions_examined"),
             "attribution": _attr(ref, cand, c),
             "diagnostic": [x for x in (ref, cand) if ((R["sections"]["1"].get(x) or {}).get("gates") or {}).get("launch") == "NO_GO_STARTUP_600"]}
        p["rows"] = [{k: r.get(k) for k in ("key", "first_diff", "class", "m_sum", "swap", "garbage", "status") if k in r}
                     for r in (c.get("rows") or [])][:40]
        m3 = {}
        for k in TRACES:
            x, y = (TFD.get(ref) or {}).get(k), (TFD.get(cand) or {}).get(k)
            if x and y:
                m3[k] = call(L.compare_tf, x, y, bins=BINS90 if k == "t90k" else BINS8,
                             offset=len(TRACES[k]["prompt"]) if k == "t90k" else 0)
        p["m3"] = m3 or {"status": "INCONCLUSIVE_NO_M3"}
        if (ref, cand) in SPEC_PAIRS:
            p["m3_label"] = "NOT_APPLICABLE_PREFILL_ONLY for DFlash state (server-equivalence sanity); M4 is the DFlash test"
        out["pairs"][key] = p
    m4own = {}
    for arm, d in M4D.items():
        try:
            m4own[arm] = call(L.decode_vs_prefill, d["gen"], d["tf"])
        except Exception as e:
            m4own[arm] = {"error": repr(e)[:200]}
            err(f"m4/{arm}", e)
    for arm, own in m4own.items():
        base = M4_BASE.get(arm)
        res = {"own": own, "base_arm": base}
        if base and base in m4own and not (m4own[base] or {}).get("error"):
            try:
                res["vs_base"] = call(L.decode_vs_prefill, M4D[arm]["gen"], M4D[arm]["tf"], base=m4own[base])
            except Exception as e:
                res["vs_base"] = {"error": repr(e)[:200]}
                err(f"m4/{arm}/vs_base", e)
        else:
            res["vs_base"] = {"status": "INCONCLUSIVE_NO_BASELINE", "why": f"baseline {base} has no M4" if base else "no baseline (gross rule only)"}
        effect = arm == "s1_c"                          # c vs c32: the decode-time SSM fp16 accumulation effect, reported
        res["vs_base_role"] = "effect (reported, not a verdict)" if effect else "STATE_DRIFT baseline"
        res["state_drift"] = any(str((x or {}).get("status")) == "STATE_DRIFT" for x in ((own,) if effect else (own, res["vs_base"])))
        out["m4"][arm] = res
    for ref, cand in S1_CHAIN:
        p = out["pairs"].get(f"{ref}~{cand}") or {}
        norm, note = None, None                         # same weights: count - (A/A + 1) (EMS); cross: STATE_FLIPs - A/A's
        g = p.get("garbage") or 0                       # garbage (reference clean) fails a link with or without an A/A
        if "hm" not in p:
            note = "pair missing (an arm missing or invalid)"
        elif str(p.get("status") or "").startswith("INCONCLUSIVE") or not p.get("positions_examined"):
            note = "pair not evaluated (INCONCLUSIVE or no positions examined): never a clean link"
        elif p.get("aa_high") is None and not g:
            note = "no usable A/A: not normalisable"
        else:
            base = p["hm"] - (p["aa_high"] + (0 if p.get("kind") == "cross_weights" else 1)) if p.get("aa_high") is not None else None
            norm = max(x for x in (base, g or None) if x is not None)
            if norm <= 0 and p.get("inconclusive_rows"):
                norm, note = None, f"{p['inconclusive_rows']} rows unjudged: a clean link is not proven"
        # significance (second validation round): with ~25 rows the EMS rule alone names a false decisive pair in 15-22 %
        # of null chains at FLIP rates 3-10 %; a one-sided Fisher test of the pair against the pooled pure A/A at alpha
        # 0.05 keeps that <= ~5-7 % with 88-98 % power at +30 % FLIP rows. STATE_FLIP links and garbage need no test.
        sig = norm is not None and norm > 0 and (p.get("kind") == "cross_weights" or g > 0 or
                                                  (p.get("p_value") is not None and p["p_value"] < S1_ALPHA))
        out["chain"].append({"pair": f"{ref}~{cand}", "kind": p.get("kind"), "hm": p.get("hm"), "aa_high": p.get("aa_high"),
                             "normalised": norm, "p_value": p.get("p_value"), "significant": sig, "status": p.get("status"), "note": note})
    dec = next((x for x in out["chain"] if x["significant"]), None)
    sug = next((x for x in out["chain"] if x["normalised"] is not None and x["normalised"] > 0 and not x["significant"]), None)
    lead = dec or sug
    upstream_open = [x["pair"] for x in out["chain"][:out["chain"].index(lead)] if x["normalised"] is None] if lead else \
        [x["pair"] for x in out["chain"] if x["normalised"] is None]
    tail = "_UPSTREAM_UNCERTAIN" if upstream_open else ""
    out["decisive"] = {"pair": dec["pair"] if dec else None, "link": dec, "suggestive_pair": sug["pair"] if sug else None,
                       "suggestive_link": sug, "alpha": S1_ALPHA, "upstream_not_normalisable": upstream_open,
                       "verdict": ("DECISIVE_PAIR_FOUND" + tail if dec else "DECISIVE_PAIR_SUGGESTIVE" + tail if sug else
                                   "NO_HIGH_MARGIN_LINK" if not upstream_open else "INCONCLUSIVE_CHAIN_INCOMPLETE")}
    # §4.1 STOP: STATE_FLIP (beyond the STATE_FLIP-only allowance) or STATE_DRIFT in a same-weights pair -> NO-GO for the arm
    # under test; README gate: target-only high-margin drift; §3.6: a wrong restore (pass 1 <-> pass 2, restore exercised).
    # FLIP-count verdicts (drift, restore) also need the Fisher test; without it they are recorded as notes, never NO-GO.
    arm_v, notes = {}, {}
    for key, p in out["pairs"].items():
        ref, cand = key.split("~")
        if p.get("kind") != "same_weights" or (ref, cand) in S1_REPORT_ONLY or str(p.get("status") or "").startswith("INCONCLUSIVE"):
            continue
        if (p.get("state_flips") or 0) > (p.get("aa_state_allow") or 0):
            arm_v.setdefault(cand, []).append("NO_GO_STATE_FLIP vs " + ref)
        if cand in S1_TARGET_ONLY and p.get("status") == "FAIL_HIGH_MARGIN":
            if (p.get("garbage") or 0) > 0 or (p.get("p_value") is not None and p["p_value"] < S1_ALPHA):
                arm_v.setdefault(cand, []).append("NO_GO_HIGH_MARGIN_DRIFT vs " + ref)
            else:
                notes.setdefault(cand, []).append(f"SUGGESTIVE_HIGH_MARGIN_DRIFT vs {ref} (p={p.get('p_value')})")
    for arm, w in out["within"].items():
        for k in ("pass1_pass2", "serial_vs_c16"):
            c = w.get(k)
            if not isinstance(c, dict) or str(c.get("status") or "").startswith("INCONCLUSIVE") or \
                    (k == "pass1_pass2" and w.get("pass2_restore") != "EXERCISED"):
                continue
            if (c.get("state_flips") or 0) > ((w.get(k + "_allow") or {}).get("state") or 0):
                arm_v.setdefault(arm, []).append(f"NO_GO_STATE_FLIP ({k})")
            if k == "pass1_pass2" and c.get("status") == "FAIL_HIGH_MARGIN":
                pv = w.get("pass1_pass2_p")
                if (c.get("garbage") or 0) > 0 or (pv is not None and pv < S1_ALPHA):
                    arm_v.setdefault(arm, []).append("NO_GO_RESTORE_DRIFT (pass1_pass2)")
                else:
                    notes.setdefault(arm, []).append(f"SUGGESTIVE_RESTORE_DRIFT (pass1_pass2, p={pv})")
        if w.get("c16_aa_pure") and ((w.get("c16_aa") or {}).get("state_flips") or 0) > 0:
            arm_v.setdefault(arm, []).append("NO_GO_STATE_FLIP (c16 A/A)")
    for key, c in out["aa"].items():
        if isinstance(c, dict) and (c.get("state_flips") or 0) > 0 and not str(c.get("status") or "").startswith("INCONCLUSIVE"):
            for a in key.split("~"):
                arm_v.setdefault(a, []).append(f"NO_GO_STATE_FLIP (boot A/A {key})")
    for arm, r in out["m4"].items():
        if r.get("state_drift"):
            arm_v.setdefault(arm, []).append("NO_GO_STATE_DRIFT")
    out["arm_notes"] = notes
    out["arm_verdicts"] = arm_v
    if not SM and PART in ("A", "ALL"):                 # README gate: missing cold / warm controls (FULL_A plans both)
        cc = {a: (R["sections"]["1"].get(a) or {}).get("status") for a in ("s1_c_warm", "s1_c_nocache")}
        out["cache_controls"] = dict(cc, verdict="OK" if all(_usable(a) for a in cc) else "NO_GO_MISSING_COLD_WARM_CONTROLS")
    s1_apply_verdicts(out)
    return out


def s1_apply_verdicts(cmp):
    """The section-1 arm verdicts reach the arm status (review #5: printing them was not enough). Recomputed on every
    call from the measured status (status_measured), so A/A data that arrives later can add or lift a verdict."""
    for arm, rec in (R["sections"].get("1") or {}).items():
        st = str((rec or {}).get("status") or "") if isinstance(rec, dict) else ""
        if st in ("", "NOT_RUN") or st.startswith("SKIPPED"):
            continue
        base = rec.setdefault("status_measured", rec["status"])
        rec["s1_verdicts"] = list((cmp.get("arm_verdicts") or {}).get(arm) or [])
        rec["status"] = worst([base] + [v.split(" ")[0] for v in rec["s1_verdicts"]])


# ------------------------------------------------------------------------------------------------ sections 2 and 3
def _pool(srv):
    try:
        return int((srv.facts() or {}).get("kv_tokens") or 0) or None
    except Exception as e:
        err("pool/" + srv.name, e)
        return None


def equal_n(band, streams, own_pool):
    """DESIGN §4.2: N = floor((pool - margin) / (band + growth + 2.9k)) from the CONTROL's pool, the same for every
    compared arm; an arm whose own pool cannot hold N records NOT_EQUAL_N (and runs at what it can hold)."""
    growth = int(30 * (ts(60) + WIN + ts(60)))           # ~30 tok/s per stream over settle + window + 60 s
    pool = S2["pool"] or own_pool
    info = {"pool_source": "control" if S2["pool"] else "own", "pool": pool, "growth": growth}
    if not pool:
        return streams, dict(info, label="POOL_UNKNOWN")
    n = int(call(L.equal_n, pool, BANDS[band], growth) or 0)
    nn = max(1, min(streams, n))
    own = int(call(L.equal_n, own_pool, BANDS[band], growth) or 0) if own_pool else nn
    if own < nn:
        return max(1, own), dict(info, n_equal=nn, label="NOT_EQUAL_N", own_n=own)
    return nn, dict(info, n_equal=nn)


def window(srv, arm, band, n, prompts=None, label=None, max_new=None, warm_ids=None):
    """flush + idle + snapshot, the optional shared-prefix warm request (AFTER the flush), then one steady window."""
    bl = between_loads("sglang", srv.port)
    if warm_ids is not None:
        bl["warm_http"] = warm(srv, warm_ids)
    # the admission budget (an upper bound: settle returns as soon as calm) at 1,000 tok/s: EMS prefill falls from 1,615
    # to 1,280 tok/s by 86k, ~1,130 at 116k, so 1,300 left the 120k band ~40 s to settle (FAKE runs read NO_SETTLE)
    w = call(L.steady, "sglang", srv.port, prompts or PROMPTS[band], n, secs=WIN, max_new=max_new or MAX_NEW.get(band, 16384),
             temperature=1.0, model=srv.model_name, server=srv, min_depth=1000, prefill_tok_s=1000.0,
             raw_dir=str(rawdir(arm) / f"win_{label or band}")) or {}
    w["between"] = bl
    w["n_offered"] = n
    return w


def warm(srv, ids):
    """EMS steady_window warm request: one 1-token request so a shared prefix is in the radix cache first."""
    code, txt = http(srv.port, "/generate", {"input_ids": list(ids), "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
                     timeout=tmo())
    return code


def parity(srv, arm, role):
    """8k x 3 greedy rows (256 tokens): the control saves them, the A/A arm gives the noise count, challengers compare."""
    rows = keyed(call(L.greedy, "sglang", srv.port, PROMPTS["8k"][:3], n_tok=256, top=2, timeout=tmo(), model=srv.model_name,
                      keys=[f"8k#{i}" for i in range(3)]), [f"8k#{i}" for i in range(3)], "p:")
    jdump(rawdir(arm) / "parity_rows.json", rows)
    if role == "control":
        S2["ctrl_rows"] = rows
        return {"saved": len(rows), "status": "MEASURED"}
    if not S2["ctrl_rows"]:
        return {"status": "INCONCLUSIVE_NO_CONTROL_ROWS"}
    if role == "aa":
        S2["aa_rows"] = rows
    aah = hm(call(L.compare, S2["ctrl_rows"], S2["aa_rows"], kind="same_weights", t_pair=0.5, aa_high=None)) if S2["aa_rows"] else None
    c = call(L.compare, S2["ctrl_rows"], rows, kind="same_weights", t_pair=0.5, aa_high=aah) or {}
    return {"status": c.get("status"), "hm": hm(c), "aa_high": aah, "counts": _cls_counts(c)}


def profile_load(srv, arm):
    """A separate 30-step profile under its own 8k load (never overlapping a measured window): breakdown only."""
    pdir = rawdir(arm) / "profile"
    shutil.rmtree(pdir, ignore_errors=True)
    return call(L.profiled_load, "sglang", srv.port, PROMPTS["8k"], 20, str(pdir), num_steps=30, max_new=MAX_NEW["8k"],
                model=srv.model_name, prefill_tok_s=1000.0)


def parity_gate(p):
    """Section-2 parity as a gate (§3.2 rule): FAIL -> NO_GO_PARITY; EXACT_TO_N / NEAR_TIE (control: rows saved) -> GO;
    BOUNDARY is GO only within a usable A/A allowance; anything else (no control rows, INCONCLUSIVE, BOUNDARY without an
    A/A, missing) -> INCONCLUSIVE_PARITY_* (review #1: never GO by default)."""
    st = str((p or {}).get("status") or "MISSING")
    if st.startswith("FAIL"):
        return "NO_GO_PARITY"
    if st in ("MEASURED", "EXACT", "EXACT_TO_N", "NEAR_TIE") or (st == "BOUNDARY" and (p or {}).get("aa_high") is not None):
        return "GO"
    return "INCONCLUSIVE_PARITY_" + (re.sub(r"^INCONCLUSIVE_?", "", st) or "UNJUDGED")


def s2_arm(arm, bands, ems=(), role="challenger"):
    spec = CAT[arm]
    streams = int(flag_value(spec["flags"], "--max-running-requests")) - 1

    def body(srv):
        o = {"gates": {}, "measured": {}, "windows": {}, "sampled": True, "streams": streams}
        pool = _pool(srv)
        if role == "control" and pool:
            S2["pool"] = pool
        for b in bands:
            n, info = equal_n(b, streams, pool)
            w = o["windows"][b] = window(srv, arm, b, n)
            w["equal_n"] = info
        for e in ems:                                   # EMS replica windows on the control (README baseline)
            prompts = EMS[e]
            lv = streams if e != "xl" else max(1, min(streams, (pool or 700000) // 100000))   # EMS: XL at kv // 100k
            o["windows"]["ems_" + e] = window(srv, arm, "8k", lv, prompts=prompts, label="ems_" + e, max_new=16384,
                                              warm_ids=prompts[0] if e != "short" else None)
        if SM and role == "control":                    # SMOKE: one server for the W6 S20 control checks
            o["measured"]["profile"] = profile_load(srv, arm)
            bl = between_loads("sglang", srv.port)
            o["measured"]["census_90k"] = census_load(srv, arm, "90k")
            o["measured"]["census_90k"]["between"] = bl
            o["measured"]["interference_D2"] = interference_load(srv, arm, "D2", d0=None)
        o["parity_between"] = between_loads("sglang", srv.port)
        o["parity"] = parity(srv, arm, role)
        o["gates"]["parity"] = parity_gate(o["parity"])
        o["gates"]["windows"] = worst([mgate(w.get("status")) for w in o["windows"].values()])
        return o
    return lambda: with_server(arm, body)


def _wnum(w, k):
    v = (w or {}).get(k)
    return v.get("mean") if isinstance(v, dict) else v


def _s2_win_ok(w, strict=True):
    """A section-2 window usable in a comparison: a GO measurement status, a tok/s, the common N (no NOT_EQUAL_N), and
    (strict, every gain claim) STEADY, i.e. the offered N actually running: a CAPACITY_LIMITED window ran fewer streams
    (review 2 audit #2). The long-band hold check is lenient: running fewer streams IS the XL collapse it looks for."""
    return isinstance(w, dict) and mgate(w.get("status")) == "GO" and bool(_wnum(w, "tok_s")) and \
        (w.get("equal_n") or {}).get("label") != "NOT_EQUAL_N" and (not strict or w.get("status") == "STEADY")


def _s2_gates_ok(rec, ref=False):
    """Every gate GO except the windows summary (each compared band is checked on its own). A reference (the control, an
    A/A boot) may have failed only the startup gate (startup time does not change numerics); a challenger may not
    (NO_GO_STARTUP_600 blocks adoption, §3.6)."""
    st = str((rec or {}).get("status") or "")
    if not isinstance(rec, dict) or st.startswith(("SKIPPED", "ERROR")) or st == "INCONCLUSIVE_HARD_STOP":
        return False
    g = {k: v for k, v in (rec.get("gates") or {}).items() if k != "windows" and not (ref and k == "launch" and v == "NO_GO_STARTUP_600")}
    return bool(g) and worst(g.values()) == "GO"


def _s2_mean_ratio(rec, ref, bands=("8k", "40k")):
    """Mean tok/s ratio of an arm's windows over a reference's: (ratio, None), or (None, why) unless every band is usable
    on both sides."""
    ws, rs = (rec or {}).get("windows") or {}, (ref or {}).get("windows") or {}
    bad = [b for b in bands if not _s2_win_ok(ws.get(b)) or not _s2_win_ok(rs.get(b))]
    if bad:
        return None, "window not usable at " + ",".join(bad)
    r = [_wnum(ws[b], "tok_s") / _wnum(rs[b], "tok_s") for b in bands]
    return round(sum(r) / len(r), 4), None


def s2_compare():
    """Section 2 (§4.2): every arm vs the control at each band (equal N): tok/s ratio, tokens / step, step ms, running p50.
    The A/A spread comes from control boots whose compared windows are valid on BOTH sides. A challenger is eligible only
    with every gate GO (startup included) and usable 8k / 40k windows on it and on the control; better = mean ratio - 1
    beyond the spread (§4.2 STOP: "net tok/s not better (beyond the A/A spread)"). best = the top better challenger that
    also holds 90k / 116k in s2_best_long (no XL collapse beyond the spread) and is reproduced in every evaluable A/A-round
    repeat (review: decision logic #1 / #2); best_candidate (8k / 40k only) picks the arm s2_best_long runs."""
    sec = R["sections"].get("2") or {}
    crec = sec.get("s2_w6s20") or {}
    ctrl = crec.get("windows") or {}
    ctrl_ok = _s2_gates_ok(crec, ref=True)
    sp = []
    for a in ["s2_w6s20_aa"] + sorted(x for x in sec if x.startswith("aa_ctrl_r")):    # every control A/A boot so far
        aa = (sec.get(a) or {}).get("windows") or {}
        for b in ("8k", "40k"):
            if _s2_win_ok(ctrl.get(b)) and _s2_win_ok(aa.get(b)):
                x, y = _wnum(ctrl[b], "tok_s"), _wnum(aa[b], "tok_s")
                sp.append(abs(x - y) / x)
    spread = max(sp) if sp else None
    out = {"aa_spread": spread, "aa_points": len(sp), "control_ok": ctrl_ok, "arms": {}}
    for arm, rec in sec.items():
        if arm == "s2_w6s20" or not isinstance(rec, dict) or not rec.get("windows"):
            continue
        rows = {}
        for b, w in rec["windows"].items():
            c = ctrl.get(b)
            t1, t0 = _wnum(w, "tok_s"), _wnum(c, "tok_s")
            rows[b] = {"tok_s": t1, "ctrl_tok_s": t0, "ratio": round(t1 / t0, 4) if t1 and t0 else None,
                       "tokens_per_step": _wnum(w, "tokens_per_step"), "step_ms": _wnum(w, "step_ms"),
                       "running_p50": (((w.get("decode_lines") or {}).get("running")) or {}).get("p50"),
                       "client_step_ms_mean": (w.get("client_step_ms") or {}).get("mean"), "status": w.get("status"), "ctrl_status": (c or {}).get("status"),
                       "n": w.get("n_offered"), "ctrl_n": (c or {}).get("n_offered"), "equal_n": (w.get("equal_n") or {}).get("label")}
        r = [rows[b]["ratio"] for b in ("8k", "40k") if b in rows and rows[b]["ratio"]]
        mr, why = _s2_mean_ratio(rec, crec)
        no = (([] if ctrl_ok else ["control gates not GO"]) + ([] if _s2_gates_ok(rec) else ["arm gates not GO: " + _bad_gate_list(rec)])
              + ([why] if why else []) + ([] if spread is not None else ["no usable A/A spread"]))
        out["arms"][arm] = {"bands": rows, "mean_ratio_8k_40k": mr, "raw_mean_ratio_8k_40k": round(sum(r) / len(r), 4) if len(r) == 2 else None,
                            "parity": (rec.get("parity") or {}).get("status"), "threshold": spread, "eligible": not no,
                            "not_eligible": no, "better": (mr - 1 > spread) if not no else None}
    w4r, w4why = _s2_mean_ratio(sec.get("s2_w4s20"), crec)      # where the width optimum lies: windows, not adoption gates
    if w4r and spread is not None:
        gain, thr = 1 / w4r - 1, max(0.05, 2 * spread)          # W6 S20 over W4 S20 on the mean of 8k and 40k
        out["branch"] = {"w6_over_w4": round(gain, 4), "threshold": thr, "run": "s2_w7s20" if gain >= thr else "s2_w5s20"}
    else:
        out["branch"] = {"run": None, "why": "rule not evaluable: " + (w4why or "no usable A/A spread")}
    cand = [(a, v["mean_ratio_8k_40k"]) for a, v in out["arms"].items() if a in S2_CHALLENGERS and v.get("eligible")]
    cand.sort(key=lambda x: -x[1])
    out["ranking"] = cand
    top = next((a for a, _ in cand if out["arms"][a].get("better")), None)
    out["best_candidate"], out["best_or_top"] = top, top or (cand[0][0] if cand else None)
    out["best"], out["best_check"] = None, None
    if top:
        bl = sec.get("s2_best_long__" + top) or {}
        lr = {}
        for b in ("90k", "120k"):
            w, c = (bl.get("windows") or {}).get(b), ctrl.get(b)
            ok = _s2_win_ok(w, strict=False) and _s2_win_ok(c, strict=False)
            lr[b] = {"ratio": round(_wnum(w, "tok_s") / _wnum(c, "tok_s"), 4) if ok else None, "status": (w or {}).get("status"),
                     "ctrl_status": (c or {}).get("status"), "running_p50": (((w or {}).get("decode_lines") or {}).get("running") or {}).get("p50"),
                     "ctrl_running_p50": (((c or {}).get("decode_lines") or {}).get("running") or {}).get("p50")}
        if not bl.get("windows"):
            lv = "INCONCLUSIVE_LONG_BANDS_NOT_RUN"
        elif not _s2_gates_ok(bl) or any(v["ratio"] is None for v in lr.values()):
            lv = "INCONCLUSIVE_LONG_BANDS"
        else:
            lv = "STOP_XL_COLLAPSE" if any(v["ratio"] < 1 - spread for v in lr.values()) else "HOLDS"
        reps = {}
        for a in sorted(x for x in sec if x.startswith("aa_best_r") and x.endswith("__" + top)):
            i = a[len("aa_best_r"):].split("__")[0]
            ref, rep = sec.get(f"aa_ctrl_r{i}"), sec[a]
            use = f"aa_ctrl_r{i}" if _s2_gates_ok(ref, ref=True) and _s2_mean_ratio(ref, crec)[0] is not None else "s2_w6s20"
            rr, rwhy = _s2_mean_ratio(rep, ref if use != "s2_w6s20" else crec)
            st = str(rep.get("status") or "")
            ran = not (st.startswith(("SKIPPED", "ERROR")) or st == "INCONCLUSIVE_HARD_STOP" or not rep.get("windows"))
            bad = [f"{k}={v}" for k, v in (rep.get("gates") or {}).items() if k != "windows" and str(v).startswith(("NO_GO", "INVALID"))]
            # a repeat that ran and failed a gate (crash, greedy fallback, parity, startup) did not reproduce (review 2 #1)
            v = ("NOT_REPRODUCED_GATE" if ran and bad else "UNEVALUABLE" if not ran or rr is None or not _s2_gates_ok(rep) else
                 "REPRODUCED" if rr - 1 > spread else "NOT_REPRODUCED")
            reps[a] = {"ratio": rr, "vs": use, "why": rwhy or (", ".join(bad) or None), "verdict": v}
        nr = [a for a, v in reps.items() if v["verdict"].startswith("NOT_REPRODUCED")]
        ok_r = [a for a, v in reps.items() if v["verdict"] == "REPRODUCED"]
        # >= 1 reproduced repeat (second validation round, null Monte Carlo of 4 challengers: false best 13.5 % with the
        # 2-point selection spread alone, 1.5 % with one repeat, 0.1 % with two)
        out["best"] = top if lv == "HOLDS" and ok_r and not nr else None
        out["best_check"] = {"candidate": top, "long": lr, "long_verdict": lv, "repeats": reps,
                             "verdict": "CONFIRMED" if out["best"] else ("NOT_REPRODUCED" if nr and lv == "HOLDS" else
                                                                         "UNCONFIRMED_NO_REPEAT" if lv == "HOLDS" else lv)}
    return out


def _bad_gate_list(rec):
    return ", ".join(f"{k}={v}" for k, v in ((rec or {}).get("gates") or {}).items() if v not in ("GO", None, "SKIPPED_FAKE"))[:200] or \
        str((rec or {}).get("status"))


def s3_arm(arm, conf=False, rerun=False):
    spec = CAT[arm]
    nccl = {k: v for k, v in spec["env"].items() if k.startswith("NCCL_") and k not in ("NCCL_DEBUG", "NCCL_DEBUG_SUBSYS")}

    def body(srv):
        o = {"gates": {}, "measured": {}, "windows": {}, "sampled": True}
        pool = _pool(srv)
        if conf or rerun:
            bands = ("40k",) if rerun else ("40k", "90k")
            for b in bands:
                n, info = equal_n(b, 20, pool)
                o["windows"][b] = window(srv, arm, b, n)
                o["windows"][b]["equal_n"] = info
            if rerun:                                   # cold-prefill seconds of one 40k prompt after a flush
                between_loads("sglang", srv.port)
                t = time.time()
                code = warm(srv, PROMPTS["40k"][23])
                o["measured"]["cold_prefill_40k_sec"] = round(time.time() - t, 2) if code == 200 else None
        else:
            o["windows"]["8k"] = window(srv, arm, "8k", 20)
            between_loads("sglang", srv.port)
            o["measured"]["profile"] = profile_load(srv, arm)
        o["nccl"] = call(L.nccl_info, srv.log())
        trace, tsrc = (o["measured"].get("profile") or {}).get("trace"), arm
        if not trace and arm.startswith("s3_conf__"):     # no profile here: the base arm (same env) shows the kernels; this
            tsrc = arm.split("__", 1)[1]                  # arm's own log must still show the variables read (review #11)
            trace = ((((R["sections"].get("3") or {}).get(tsrc) or {}).get("measured") or {}).get("profile") or {}).get("trace")
        o["env_proof"] = call(L.nccl_env_proof, spec["env"], o["nccl"], trace=trace) or {}
        o["env_proof"]["trace_from"] = tsrc if trace else None
        ps = o["env_proof"].get("status")
        o["gates"]["env_proof"] = "GO" if (ps == "APPLIED" or not nccl) else (ps if str(ps).startswith(SEV) else "INCONCLUSIVE_" + str(ps))
        o["gates"]["windows"] = worst([mgate(w.get("status")) for w in o["windows"].values()])
        return o
    return lambda: with_server(arm, body)


def s3_compare():
    sec = R["sections"].get("3") or {}
    rows = {}
    for arm in S3_ARMS:
        rec = sec.get(arm) or {}
        w = (rec.get("windows") or {}).get("8k") or {}
        cs = w.get("client_step_ms") or {}
        tps = _wnum(w, "tokens_per_step")
        rows[arm] = {"status": rec.get("status"), "window": w.get("status"), "env_proof": (rec.get("gates") or {}).get("env_proof"),
                     "client_step_ms_mean": cs.get("mean") if isinstance(cs, dict) else None,
                     "client_step_ms_p95": cs.get("p95") if isinstance(cs, dict) else None, "tok_s": _wnum(w, "tok_s"),
                     "tok_s_per_token_per_step": round(_wnum(w, "tok_s") / tps, 2) if _wnum(w, "tok_s") and tps else None,
                     "running_p50": (((w.get("decode_lines") or {}).get("running")) or {}).get("p50")}
    # the two arms to confirm: every gate GO (startup included) and faster than the control at 8k (review #11)
    cr = sec.get("s3_ctrl") or {}
    cms = rows.get("s3_ctrl", {}).get("client_step_ms_mean") if _s2_gates_ok(cr, ref=True) and mgate(rows.get("s3_ctrl", {}).get("window")) == "GO" else None
    valid = [(a, r["client_step_ms_mean"]) for a, r in rows.items() if a != "s3_ctrl" and r["client_step_ms_mean"]
             and _s2_gates_ok(sec.get(a)) and mgate(r["window"]) == "GO" and r["env_proof"] == "GO" and cms and r["client_step_ms_mean"] < cms]
    valid.sort(key=lambda x: x[1])
    ctrl40r, ctrl90r = sec.get("s3_ctrl_rerun") or {}, (R["sections"].get("2") or {}).get("s2_w6s20") or {}
    ctrl40, ctrl90 = (ctrl40r.get("windows") or {}).get("40k") or {}, (ctrl90r.get("windows") or {}).get("90k") or {}
    spread = (R["comparisons"].get("2") or {}).get("aa_spread")
    conf = {}
    for a, rec in sec.items():
        if a.startswith("s3_conf__") and isinstance(rec, dict):
            ws = rec.get("windows") or {}
            bands = {b: {"tok_s": _wnum(ws.get(b), "tok_s"), "ctrl_tok_s": _wnum(c, "tok_s"), "status": (ws.get(b) or {}).get("status"),
                         "ctrl_status": c.get("status"), "n": (ws.get(b) or {}).get("n_offered"), "ctrl_n": c.get("n_offered"),
                         "ratio": round(_wnum(ws.get(b), "tok_s") / _wnum(c, "tok_s"), 4)
                         if _s2_win_ok(ws.get(b)) and _s2_win_ok(c) and _s2_gates_ok(cref, ref=True) else None,
                         "control": "s3_ctrl_rerun (same session)" if b == "40k" else "s2_w6s20 90k window"}
                     for b, c, cref in (("40k", ctrl40, ctrl40r), ("90k", ctrl90, ctrl90r))}
            rs = [v["ratio"] for v in bands.values()]
            # §4.3: gains only from these windows; STOP "slower end to end" (beyond the section-2 A/A spread)
            conf[a] = {"bands": bands, "verdict": ("INCONCLUSIVE_ARM_" + str(rec.get("status")) if not _s2_gates_ok(rec) else
                                                   "INCONCLUSIVE_WINDOWS" if None in rs else "INCONCLUSIVE_NO_AA_SPREAD" if spread is None else
                                                   "STOP_SLOWER" if min(rs) < 1 - spread else
                                                   "GAIN_CONFIRMED" if min(rs) > 1 + spread else "NO_CLEAR_GAIN")}
    return {"arms": rows, "control_step_ms_mean": cms, "ranking_by_mean_step_ms": valid, "best_two": [a for a, _ in valid[:2]],
            "confirmations": conf, "aa_spread_from_section2": spread,
            "cold_prefill_40k_sec": ((sec.get("s3_ctrl_rerun") or {}).get("measured") or {}).get("cold_prefill_40k_sec"),
            "note": "gains are claimed only from the 40k / 90k confirmation windows + the control re-run (same session)"}
'''

CELL5 += r'''

# ------------------------------------------------------------------------------------------------ sections 4 and 5
CENSUS = {"90k": ("90k", 24), "116k": ("120k", 24), "shared": (None, 16)}


def max_running(srv):
    """The resolved running cap for census(): /server_info effective_max_running_requests_per_dp, else its
    max_running_requests, else the capacity log line (WP2 contract: without it a capped load has no plateau)."""
    try:
        si = srv.server_info() or {}
        st = (si.get("internal_states") or [{}])[0] if isinstance(si.get("internal_states"), list) else {}
        for v in ((st or {}).get("effective_max_running_requests_per_dp"), si.get("max_running_requests")):
            if isinstance(v, int) and v > 0:
                return v
    except Exception as e:
        err("max_running/" + srv.name, e)
    return (srv.facts() or {}).get("max_running")


def census_load(srv, arm, load):
    """DESIGN §4.4: offer the full logical count (never a client cap); plateau = prefill counter flat AND no new Prefill
    batch line AND queue unchanged AND a blocking reason, held >= max(60, 1.5 x band / 1300) s; max_wait = offered x
    band / 1300 + 120 s (rev 2 blockers #2 / #4)."""
    band_key, offered = CENSUS[load]
    prompts = PROMPTS[band_key] if band_key else SHARED4
    band = BANDS[band_key] if band_key else max(len(p) for p in SHARED4)
    if not band_key:
        for g in range(4):                                # one warm request per shared ~85k prefix
            warm(srv, SHARED4[4 * g])
    mr = max_running(srv)
    off = len(srv.log() or "")                            # prefill accounting starts after the warm requests
    c = call(L.census, "sglang", srv.port, prompts, offered, secs=WIN, plateau_sec=ts(max(60.0, 1.5 * band / 1300.0)),
             max_wait=ts(offered * band / 1300.0 + 120.0), server=srv, model=srv.model_name, band=band, max_running=mr,
             raw_dir=str(rawdir(arm) / f"census_{load}"), tag=load) or {}
    c["offered"], c["band"], c["max_running_passed"] = offered, band, mr
    try:                                                  # §4.4 "repeated prefix recompute": TP0 Prefill-line new tokens
        pls = L.prefill_lines(srv.log() or "", off)       # per admitted stream, against the band (shared: a reused prefix
        adm, new = c.get("admitted_client") or 0, sum(x.get("new_token") or 0 for x in pls)   # leaves only the tails)
        c["prefill_accounting"] = {"prefill_lines": len(pls), "new_tokens": new, "admitted": adm,
                                   "cached_tokens_logged": sum(x.get("cached_token") or 0 for x in pls),
                                   "new_per_admitted_over_band": round(new / (adm * band), 4) if adm and band else None}
    except Exception as e:
        err(f"census/{arm}/{load}/prefill_accounting", e)
    return c


def _blocking(c):
    return str((c or {}).get("blocking") or (c or {}).get("blocking_reason") or "").lower()


def s4_arm(arm, loads):
    def body(srv):
        o = {"gates": {}, "measured": {}, "sampled": True,
             "label": "hook-on: the counter hook is live (natural #43567 exposure); tok/s at plateau labelled hook-on"}
        if arm == "s4_mem086":                            # gated by an 8k probe (TP0 near OOM at 0.83 already)
            rows = call(L.greedy, "sglang", srv.port, PROMPTS["8k"][:20], n_tok=32, top=2, timeout=tmo(), model=srv.model_name,
                        concurrent=True) or []
            oom = bool(L.log_errors(srv.log()).get("oom_lines"))   # never the recovered allocator warning (review: real #1)
            o["probe_8k"] = {"alive": srv.alive(), "oom": oom, "errors": [r.get("error") for r in rows if r.get("error")][:3]}
            if oom or not srv.alive():
                o["gates"]["probe_8k"] = "NO_GO_OOM" if oom else "NO_GO_SERVER_DIED"
                if oom:
                    o["expected_negative"] = "mem-fraction 0.86 OOM: an expected-negative result, not ERROR"
                return o
        for ld in loads:
            bl = between_loads("sglang", srv.port)
            c = meas(o, "census_" + ld, lambda ld=ld: census_load(srv, arm, ld))
            if isinstance(c, dict):
                c["between"] = bl
            st = str((c or {}).get("status"))
            o["gates"]["census_" + ld] = mgate(st)
            if st.startswith("FAIL_") or not srv.alive():          # STOP: Can not alloc mamba cache / OOM / crash
                o["gates"]["census_" + ld] = mgate(st) if st.startswith("FAIL_") else "NO_GO_SERVER_DIED"
                break
            pa = (c or {}).get("prefill_accounting") or {}
            if ld == "shared" and isinstance(pa.get("new_per_admitted_over_band"), (int, float)) and pa["new_per_admitted_over_band"] > 0.5:
                o["gates"]["census_shared_prefix"] = "NO_GO_PREFIX_RECOMPUTE"   # §4.4 STOP: each stream re-prefilled its ~85k prefix
        if arm == "s4_lowcap":                          # §4.4 optional negative test: "the census must report the cap"
            c90 = o["measured"].get("census_90k") or {}
            cap = ((c90.get("gauges_end") or {}).get("max_total_num_tokens") or (c90.get("gauges_mean") or {}).get("max_total_num_tokens"))
            o["lowcap_check"] = {"max_total_num_tokens": cap, "blocking": _blocking(c90)}
            o["gates"]["lowcap"] = ("GO" if isinstance(cap, (int, float)) and cap <= 300000 * 1.02 and "kv" in _blocking(c90)
                                    else "INCONCLUSIVE_LOWCAP_NOT_REPORTED")
        return o

    def f():
        rec = with_server(arm, body)
        cnt = (rec.get("hook") or {}).get("counters_max") or {}
        b = cnt.get("optional_alloc_attempts")
        rec["natural_exposure"] = {"optional_alloc_failed": cnt.get("optional_alloc_failed"),
                                   "stale_donation_candidate": cnt.get("stale_donation_candidate"),
                                   "per_1000_boundaries": (round(1000.0 * (cnt.get("optional_alloc_failed") or 0) / b, 3) if b else None),
                                   "boundaries": b}
        return rec
    return f


LOADS5 = {"D0": (20, 0, False), "D1": (19, 1, True), "D2": (19, 1, False), "D3": (16, 4, False)}


def s5_windows():
    """One fixed wall window per load, identical in every arm: preset from the planned prefill work at a conservative
    1,100 tok/s (EMS: 1,615 -> 1,280 tok/s by 86k context) + 30 s, so slower arms still finish inside it and read as
    slower, not INCONCLUSIVE (D0 / D1: 60 s); the control's observed latest injected first token is recorded against it."""
    return {"D0": ts(60), "D1": ts(60), "D2": ts(90000 / 1100 + 30), "D3": ts(4 * 90000 / 1100 + 30)}


def interference_load(srv, arm, load, d0):
    nres, ninj, warm_ = LOADS5[load]
    W = s5_windows()
    res = PROMPTS["8k"][:nres]                            # residents: variants 0-19; injected 90k: variants 20-23 (no shared prefix)
    inj = [PROMPTS["90k"][23]] if warm_ else [PROMPTS["90k"][20 + i] for i in range(ninj)]
    bl = between_loads("sglang", srv.port)
    r = call(L.interference, "sglang", srv.port, res, inj if ninj else [], nres, warm_prompt=inj[0] if warm_ else None,
             secs_after=ts(30), server=srv, model=srv.model_name, window_sec=W[load], d0_median_step=d0,
             raw_dir=str(rawdir(arm) / f"load_{load}")) or {}
    r.update(window_preset_sec=W[load], between=bl)
    return r


def _d0_step(r):
    """D0's median client step (ms): every later load of the arm counts stalls against it."""
    v = (r or {}).get("d0_median_step_ms")
    if v is None:
        v = ((r or {}).get("pre_injection_step_ms") or {}).get("p50")
    return v if isinstance(v, (int, float)) else None


def s5_arm(arm):
    def body(srv):
        o = {"gates": {}, "measured": {}, "sampled": True, "windows_preset": s5_windows()}
        d0 = None
        for ld in LOADS5:
            r = meas(o, ld, lambda ld=ld: interference_load(srv, arm, ld, d0))
            if ld == "D0":
                d0 = _d0_step(r)
            st = (r or {}).get("status")
            g = mgate(st)
            if (r or {}).get("starvation"):
                g = "NO_GO_STARVATION"
            o["gates"]["load_" + ld] = g
        if arm == "s5_c4096":                        # injection -> latest injected first token, vs the preset windows
            S5["control"] = {ld: max([x for x in ((o["measured"].get(ld) or {}).get("prefill_sec") or []) if x is not None] or [None],
                                     key=lambda x: -1 if x is None else x) for ld in LOADS5}
        o["control_latest_completion_sec"] = S5["control"]
        return o
    return lambda: with_server(arm, body)


def s5_compare():
    sec = R["sections"].get("5") or {}
    ctrl = (sec.get("s5_c4096") or {}).get("measured") or {}
    out = {"windows_preset": s5_windows(), "control_latest_completion": S5["control"], "arms": {}, "stop": {}}
    order = sorted(sec, key=lambda a: a != "s5_c4096")       # the control first: every other verdict needs it
    for arm in order:
        rec = sec[arm]
        m = (rec or {}).get("measured") or {}
        out["arms"][arm] = {ld: {"decode_tok_s": (m.get(ld) or {}).get("tok_s_window"),
                                 "ctrl_decode_tok_s": (ctrl.get(ld) or {}).get("tok_s_window"),
                                 "starvation": (m.get(ld) or {}).get("starvation"), "max_gap_s": (m.get(ld) or {}).get("max_gap_s"),
                                 "stall_s": (m.get(ld) or {}).get("stall_s_max"),
                                 "prefill_s": max([x for x in ((m.get(ld) or {}).get("prefill_sec") or []) if x is not None] or [0]) or None,
                                 "decode_tokens_prefill_phase": (m.get(ld) or {}).get("resident_tokens_prefill_phase"),
                                 "decode_flat_s": (m.get(ld) or {}).get("decode_flat_sec"),
                                 "tpot_p95": ((m.get(ld) or {}).get("tpot_ms") or {}).get("p95"), "status": (m.get(ld) or {}).get("status"),
                                 "ctrl_status": (ctrl.get(ld) or {}).get("status")}
                            for ld in LOADS5}
        L5 = out["arms"][arm]
        starv = [l for l, v in L5.items() if v["starvation"]]
        # NOT_WORSE needs D1-D3 measured on BOTH sides, starvation evaluated, and the arm's other gates GO (review #3);
        # a starvation seen on any load, or a worse ratio on the loads that did measure, still stops the arm
        bad = [l for l in ("D1", "D2", "D3") if mgate(L5[l]["status"]) != "GO" or mgate(L5[l]["ctrl_status"]) != "GO"
               or L5[l]["starvation"] is None or not L5[l]["decode_tok_s"] or not L5[l]["ctrl_decode_tok_s"]]
        r_ = [L5[l]["decode_tok_s"] / L5[l]["ctrl_decode_tok_s"] for l in ("D1", "D2", "D3") if l not in bad]
        mean_r = round(sum(r_) / len(r_), 4) if r_ else None
        g5 = (rec or {}).get("gates") or {}
        startup = g5.get("launch") == "NO_GO_STARTUP_600"   # blocks adoption, never the measured verdict (§3.6, fuzz finding)
        other = {k: v for k, v in g5.items() if not (k.startswith("load_") and v == "NO_GO_STARVATION")
                 and not (k == "launch" and v == "NO_GO_STARTUP_600")}
        ost = worst(other.values()) if other else ("GO" if str((rec or {}).get("status")) in ("GO", "NO_GO_STARTUP_600")
                                                   else str((rec or {}).get("status")))
        cv = (out["stop"].get("s5_c4096") or {}).get("verdict")
        if starv:
            v = "STOP_STARVATION"
        elif ost.startswith("NO_GO"):
            v = "STOP_" + ost
        elif arm == "s5_c4096":
            v = "CONTROL" if ost == "GO" and not bad else "INCONCLUSIVE_CONTROL"
        elif cv not in ("CONTROL", "STOP_STARVATION"):  # a ratio needs a valid control (review 2 #6); a starving control
            v = "INCONCLUSIVE_CONTROL_" + str(cv)       # still measured every load (its starvation is its own STOP)
        elif mean_r is not None and mean_r < 0.97:      # README STOP: worse mixed-window throughput than the 4096 control (3 %)
            v = "STOP_WORSE_MIXED" + ("_PARTIAL" if bad else "")
        elif bad:
            v = "INCONCLUSIVE_LOADS_" + "_".join(bad)
        elif ost != "GO":
            v = "INCONCLUSIVE_ARM_" + ost
        else:
            v = "NOT_WORSE"
        out["stop"][arm] = {"starvation_loads": starv, "mixed_ratio_d1_d3": mean_r if not bad else None,
                            "mixed_ratio_measured_loads": mean_r, "unusable_loads": bad, "verdict": v,
                            "adoption_blocked": "NO_GO_STARTUP_600" if startup else None}
    return out


# ------------------------------------------------------------------------------------------------ section 6
def gen_raw(port, body, timeout=None):
    code, txt = http(port, "/generate", body, timeout=timeout or tmo())
    try:
        r = json.loads(txt) if code == 200 else {"error": f"HTTP {code}: {txt[:300]}"}
    except Exception as e:
        r = {"error": f"{type(e).__name__}: {txt[:200]}"}
    return r[0] if isinstance(r, list) and r else r


def s6_req(port, ids, max_new=1, score=False, ignore_eos=True, rid=None):
    """score=True: the full-vocab next-token distribution (lab next_token_dist, E17 method); else a greedy run (rid)."""
    if score:
        d = call(L.next_token_dist, port, list(ids), VOCAB, timeout=tmo()) or {}
        return {"rid": None, "cached": d.get("cached"), "lp": d.get("lp"), "error": d.get("error"), "sum_raw": d.get("sum_raw")}
    body = {"rid": rid or f"e24-{random.randrange(10 ** 12)}", "input_ids": list(ids),
            "sampling_params": {"temperature": 0.0, "max_new_tokens": int(max_new), "ignore_eos": ignore_eos}}
    r = gen_raw(port, body)
    meta = r.get("meta_info") or {}
    fr = meta.get("finish_reason")
    return {"rid": body["rid"], "cached": meta.get("cached_tokens"), "n_out": len(r.get("output_ids") or []),
            "finish": fr.get("type") if isinstance(fr, dict) else fr, "error": r.get("error")}


def kl_rec(a, b):
    k = call(L.kl, a, b) if a and b else None
    d = [abs(x - y) for x, y in zip(a, b)] if a and b else []
    return {"kl": float(k) if k is not None else None, "max_abs_dlogp": max(d) if d else None,
            "top1_changed": (max(range(len(a)), key=a.__getitem__) != max(range(len(b)), key=b.__getitem__)) if a and b else None}


def s6_t1(port, pname, seed_new, aa=3, cycles=3):
    """Upstream Class 1 (test_mamba_cache_kl.py, E17 e17_t1): reference = flush, warm W44 + W42, score P (aa times = the
    A/A); cycle = flush, warm, seed (P + seed_new greedy), score P (the reuse). Persisted per cycle."""
    Pp, pref = S6P[pname], [S6P["W44"], S6P["W42"]]
    tag = f"{pname}-{random.randrange(10 ** 9)}"
    out = {"prompt": pname, "len": len(Pp), "seed_new": seed_new, "safe_depth": max(len(p) for p in pref if Pp[:len(p)] == p),
           "ref_cached": [], "ref_flush_valid": [], "aa": [], "cycles": [], "tol": TOL}
    refs = []
    for _ in range(aa):
        f = flush("sglang", port)
        out["ref_flush_valid"].append(f["valid"])
        for p in pref:
            s6_req(port, p, 1)
        r = s6_req(port, Pp, 1, score=True)
        refs.append(r.get("lp"))
        out["ref_cached"].append(r.get("cached"))
    good = [x for x in refs if x]
    out["aa"] = [kl_rec(good[0], x) for x in good[1:]] if len(good) > 1 else []
    kls = [x["kl"] for x in out["aa"]]
    out["aa_kl_max"] = max(kls) if kls and None not in kls else float("nan")      # a missing A/A KL = no valid reference
    out["reference_valid"] = bool(len(good) == aa and all(out["ref_flush_valid"]) and all(c == out["safe_depth"] for c in out["ref_cached"])
                                  and math.isfinite(out["aa_kl_max"]) and out["aa_kl_max"] <= AA_MAX)
    for c in range(cycles):
        f = flush("sglang", port)
        for p in pref:
            s6_req(port, p, 1)
        srid, prid = f"e24-seed-{tag}-{c}", f"e24-probe-{tag}-{c}"
        s = s6_req(port, Pp, seed_new, rid=srid)
        r = s6_req(port, Pp, 1, score=True, rid=prid)
        k = kl_rec(good[0], r["lp"]) if good and r.get("lp") else {"kl": None, "top1_changed": None}
        out["cycles"].append({"cycle": c, "seed_rid": srid, "probe_rid": prid, "flush_valid": f["valid"], "seed_cached": s.get("cached"),
                              "seed_finish": s.get("finish"), "probe_cached": r.get("cached"), "probe_kl": k["kl"],
                              "top1_changed": k.get("top1_changed"), "error": s.get("error") or r.get("error")})
        jdump(WORK / "s6_cycles" / f"{tag}_{c}.json", out["cycles"][-1])     # atomic per cycle (E17 run 1 lost data)
    out["_ref"] = good[0] if good else None
    return out


def s6_events(hdir):
    """Per-rid hook evidence: optional-slot failures, stale donations (decode path confirmed at donation; spec path
    confirmed by a later spec_resolve crossed=True), lock releases, alloc asserts, grammar / injection events."""
    h = hook_files(hdir)
    ev = {"failed": set(), "confirmed": {}, "events": {}, "by_rid": {}}
    pend = {}
    for e in h["trace"]:
        k, rid = e.get("ev"), e.get("rid")
        ev["events"][k] = ev["events"].get(k, 0) + 1
        if rid:
            ev["by_rid"].setdefault(rid, {}).setdefault(k, 0)
            ev["by_rid"][rid][k] += 1
        if k in ("alloc_failed", "optional_alloc_failed") and rid:
            ev["failed"].add(rid)
        elif k == "donation" and e.get("stale_candidate") and rid:
            ln = e.get("returned") if isinstance(e.get("returned"), int) else e.get("cache_len")
            if ((e.get("marked") or {}).get("path")) == "decode":
                ev["confirmed"].setdefault(rid, []).append(ln)
            else:
                pend.setdefault(rid, []).append(ln)
        elif k == "spec_resolve" and rid and e.get("confirmed"):
            ev["confirmed"].setdefault(rid, []).extend(pend.pop(rid, []))
    by, ev["ranks"] = rank_files(hdir)               # counters / per-rid / allocator: the TP scheduler ranks only
    ranks = [by[r] for r in sorted(by)]
    ev["counters"] = [c.get("counters") or {} for c in ranks]
    ev["alloc"] = [c.get("alloc") for c in ranks if c.get("alloc")]
    per = {}
    for c in ranks:
        for rid, d in (c.get("per_rid") or {}).items():
            for k, v in (d or {}).items():
                per.setdefault(rid, {})[k] = max(per.get(rid, {}).get(k, 0), v or 0)
    ev["per_rid"] = per
    ev["failed"] |= {rid for rid, d in per.items() if d.get("optional_alloc_failed")}
    return ev


S6_AGREE = {"t1": ("optional_alloc_failed", "stale_donation_candidate", "prefix_lock_released"),
            "g": ("grammar_steps", "grammar_truncated_steps", "false_crossing_possible", "false_crossing_returned"),
            "sent": ("draft_nan_injections", "neighbor_rows_nonfinite"), "nan": ("selector_nan_injections", "zero_q_rows", "first_candidate_picks")}


def _ranks_agree(ev, names):
    """DESIGN §4.6 "with identical counters": the SPMD schedulers of the TP ranks take the same Mamba / grammar / sampling
    decisions, so a verdict counter seen on one rank only is a desync, never evidence (review 2 audit #4)."""
    cs = ev.get("counters") or []
    return len(cs) == TP and all(len({int((c or {}).get(n) or 0) for c in cs}) == 1 for n in names)


def s6_events_settled(hdir, kind, tries=3):
    """s6_events after the hook's last flushes: re-read (2 s apart) until the verdict counters agree across the ranks."""
    ev = s6_events(hdir)
    for _ in range(tries - 1):
        if _ranks_agree(ev, S6_AGREE[kind]):
            break
        time.sleep(2.0)
        ev = s6_events(hdir)
    ev["ranks_agree"] = _ranks_agree(ev, S6_AGREE[kind])
    return ev


def _csum(ev, name):
    vals = [int(c.get(name) or 0) for c in ev.get("counters") or []]
    return max(vals) if vals else None


def s6_chain(t, cyc, ev):
    lens = [x for x in ev["confirmed"].get(cyc["seed_rid"], []) if isinstance(x, int)]
    link = isinstance(cyc["probe_cached"], int) and cyc["probe_cached"] > t["safe_depth"] and cyc["probe_cached"] in lens
    return {"failed": cyc["seed_rid"] in ev["failed"], "stale_confirmed": bool(lens), "stale_lens": lens, "restore_link": link,
            "hot": isinstance(cyc["probe_kl"], float) and cyc["probe_kl"] > TOL}


def s6_verdict_unpatched(t1s, ev):
    """E17 e17_verdict: BUG_EXPOSED only when ONE valid cycle holds the whole chain (failure -> confirmed stale donation
    -> restore link -> KL above the fixed tolerance); evidence from different cycles never adds up."""
    if not t1s or not all(t["reference_valid"] for t in t1s):
        return "INCONCLUSIVE_INVALID_REFERENCE"
    ch = [s6_chain(t, c, ev) for t in t1s for c in t["cycles"] if c["flush_valid"]]
    if not ch:
        return "INCONCLUSIVE_INVALID_COLD_CONTROL"
    if any(x["failed"] and x["stale_confirmed"] and x["restore_link"] and x["hot"] for x in ch):
        return "BUG_EXPOSED" if ev.get("ranks_agree", True) else "INCONCLUSIVE_COUNTERS_DESYNC"
    return "INCONCLUSIVE_NOT_EXERCISED"


def s6_verdict_patched(t1s, ev, safety, trigger="failed", need_safety=False):
    """Patched arm: CLEAN needs the trigger evidence in THIS arm (every valid cycle's seed rid failed its optional slot;
    the probe hit the safe depth), no stale donation, KL <= tol, no leak, and (FF-P / FF-P-D / N-P) the E17 safety cases
    with a valid allocator baseline; missing trigger -> INCONCLUSIVE_NOT_EXERCISED."""
    if not t1s or not all(t["reference_valid"] for t in t1s):
        return "INCONCLUSIVE_INVALID_REFERENCE"
    cyc = [(t, c) for t in t1s for c in t["cycles"] if c["flush_valid"]]
    if not cyc:
        return "INCONCLUSIVE_INVALID_COLD_CONTROL"
    stale = any(ev["confirmed"].get(c["seed_rid"]) for _, c in cyc) or (_csum(ev, "stale_donation_candidate") or 0) > 0
    hot = any(isinstance(c["probe_kl"], float) and c["probe_kl"] > TOL for _, c in cyc)
    if stale or hot or (safety or {}).get("stale") or ((safety or {}).get("leaks") and (safety or {}).get("status") == "OK"):
        return "FAIL"
    if any(c["probe_kl"] is None for _, c in cyc):
        return "INCONCLUSIVE_PROBE_MISSING"                    # a cycle without its KL is never clean
    if need_safety and (safety or {}).get("status") != "OK":   # review #8b / #8c: no safety cases, or no allocator baseline
        return "INCONCLUSIVE_SAFETY_" + str((safety or {}).get("status") or "NOT_RUN").replace("INCONCLUSIVE_", "")
    if trigger == "failed":
        ok = all(c["seed_rid"] in ev["failed"] and c["probe_cached"] == t["safe_depth"] for t, c in cyc)
    else:
        ok = (_csum(ev, trigger) or 0) >= 1
    if ok and not ev.get("ranks_agree", True):
        return "INCONCLUSIVE_COUNTERS_DESYNC"
    return "CLEAN" if ok else "INCONCLUSIVE_NOT_EXERCISED"


def s6_safety(port, ref_t1, hdir):
    """E17 e17_safety: EOS, abort mid-decode, max_new, 5x reuse, cold after flush; after each the allocator's
    available slots vs the baseline (a leak = fewer) and a warm probe (deeper than safe with KL > tol = stale)."""
    Pp, pref, ref = S6P["P255"], [S6P["W44"], S6P["W42"]], ref_t1.get("_ref")

    def avail():
        time.sleep(3.0)
        vals = [a.get("available") for a in s6_events(hdir)["alloc"] if isinstance(a, dict) and a.get("available") is not None]
        return max(vals) if vals else None
    f0 = flush("sglang", port)
    base, rows = avail(), []

    def after(case, extra=None):
        for p in pref:
            s6_req(port, p, 1)
        r = s6_req(port, Pp, 1, score=True)
        k = kl_rec(ref, r["lp"]) if ref and r.get("lp") else {"kl": None}
        f = flush("sglang", port)
        a = avail()
        rows.append(dict({"case": case, "flush_valid": f["valid"], "available": a, "available_base": base,
                          "leak": bool(f["valid"]) and a is not None and base is not None and a < base,
                          "probe_cached": r.get("cached"), "probe_kl": k["kl"],
                          "stale": bool(r.get("cached") and r["cached"] > 128 and isinstance(k["kl"], float) and k["kl"] > TOL)},
                         **(extra or {})))
    r = s6_req(port, Pp, 64, ignore_eos=False)
    after("eos", {"finish": r.get("finish")})
    rid = f"e24-abort-{random.randrange(10 ** 9)}"
    th = threading.Thread(target=lambda: s6_req(port, Pp, 400, rid=rid), daemon=True)
    th.start()
    time.sleep(0.8)
    http(port, "/abort_request", {"rid": rid}, timeout=30)
    th.join(60)
    after("abort")
    r = s6_req(port, Pp, 48)
    after("max_new", {"n_out": r.get("n_out")})
    for _ in range(5):
        s6_req(port, Pp, 8)
    after("reuse5")
    flush("sglang", port)
    r = s6_req(port, Pp, 1, score=True)
    after("cold", {"cold_cached": r.get("cached")})
    out = {"rows": rows, "leaks": sum(1 for x in rows if x["leak"]), "stale": sum(1 for x in rows if x["stale"]),
           "available_base": base, "base_flush_valid": f0.get("valid"), "cold_cached_zero": rows[-1].get("cold_cached") == 0}
    # review #8c / review 2 #2: a leak can only be seen against a real allocator baseline taken after a valid flush, with
    # valid flushes after each case and a cold control; a reading above the baseline means the baseline was not empty
    out["status"] = ("INCONCLUSIVE_NO_ALLOCATOR_SNAPSHOT" if base is None or any(x["available"] is None for x in rows) else
                     "INCONCLUSIVE_FLUSH_INVALID" if not f0.get("valid") or not all(x["flush_valid"] for x in rows) else
                     "INCONCLUSIVE_BASELINE_INVALID" if any(x["available"] > base for x in rows) else
                     "INCONCLUSIVE_COLD_NOT_ZERO" if not out["cold_cached_zero"] else "OK")
    return out


def _strip(t):
    return {k: v for k, v in t.items() if not k.startswith("_")}


def s6_t1_arm(arm, patched, prompts=("P255",), seed_new=32, safety=False, natural=False):
    """FF-U / FF-P (target-only, interval 128), FF-U-D / FF-P-D (DFlash W6, interval 64, P255), N-P (natural)."""
    def body(srv):
        o = {"gates": {}, "measured": {}}
        hdir = rawdir(arm) / "hook"
        if natural:                                       # the #43559 path: flush, warm W42, generate P255 + 32
            flush("sglang", srv.port)
            s6_req(srv.port, S6P["W42"], 1)
            o["measured"]["natural_seed"] = s6_req(srv.port, S6P["P255"], 32, rid="e24-natural-seed")
            o["measured"]["alive_after_seed"] = srv.alive()
            if not srv.alive():                           # §4.6: an assert / traceback on a patched arm is a FAIL (review #8a)
                hits = {s: srv.log().count(s) for s in ERR_SIGNS if s in srv.log()}
                o["verdict"], o["log_error_hits"] = ("FAIL" if hits else "INCONCLUSIVE_SERVER_DIED"), hits
                o["gates"]["natural"] = ("NO_GO_ASSERT_43559" if "Can not alloc mamba cache" in srv.log() else
                                         "NO_GO_PATCHED_ARM_FAILED" if hits else "ERROR_SERVER_DIED")
                return o
        t1s = []
        for pn in prompts:
            t1s.append(s6_t1(srv.port, pn, seed_new))
            if not srv.alive():
                break
        if safety and t1s and srv.alive():
            o["measured"]["safety"] = s6_safety(srv.port, t1s[0], hdir)
        time.sleep(3.0)                                   # the hook flushes on change (>= 1-2 s cadence)
        ev = s6_events_settled(hdir, "t1")
        o["measured"]["t1"] = [_strip(t) for t in t1s]
        o["measured"]["rid_evidence"] = {"failed": sorted(ev["failed"])[:40], "confirmed": dict(list(ev["confirmed"].items())[:40]),
                                         "events": ev["events"], "counters": ev["counters"][:4]}
        for t in t1s:
            for c in t["cycles"]:
                if c["flush_valid"]:
                    c["chain"] = s6_chain(t, c, ev)
        o["measured"]["t1"] = [_strip(t) for t in t1s]
        log_hits = {s: srv.log().count(s) for s in ERR_SIGNS if s in srv.log()}
        if patched:
            v = s6_verdict_patched(t1s, ev, o["measured"].get("safety"), trigger="prefix_lock_released" if natural else "failed",
                                   need_safety=safety)
            if log_hits:
                v = "FAIL"
        else:
            v = s6_verdict_unpatched(t1s, ev)
        o["verdict"], o["log_error_hits"] = v, log_hits
        o["gates"]["verdict"] = {"CLEAN": "GO", "BUG_EXPOSED": "GO", "FAIL": "NO_GO_PATCHED_ARM_FAILED"}.get(v, v if v.startswith(SEV) else "INCONCLUSIVE_" + v)
        return o
    return lambda: with_server(arm, body, expect_errors=not patched)


def ngram_launch_failed(rec):
    """§4.6 fallback trigger: the NGRAM arm ran but never reached its body (a refused flag, a failed or OOM launch, an
    invalid sm89 kernel), so #43029 has no verdict (review #12: not only NO_GO_LAUNCH*); a skipped arm is no trigger."""
    if not isinstance(rec, dict) or str(rec.get("status") or "").startswith("SKIPPED") or rec.get("verdict"):
        return False
    g = rec.get("gates") or {}
    return str(g.get("flags", "GO")) != "GO" or str(g.get("launch")) not in ("GO", "NO_GO_STARTUP_600")


def s6_g_arm(arm, patched):
    """#43029 (upstream spec_grammar_at_43029.py port): a grammar-constrained NGRAM run whose last verify accepts past
    the terminator, swept across the 256 boundary; follow-up scored with the cache (hit) and after a flush (miss)."""
    def body(srv):
        o = {"gates": {}, "measured": {}}
        port = srv.port
        r0 = gen_raw(port, {"input_ids": G_FILLER[:64] + G_PHRASE * 4,
                            "sampling_params": {"max_new_tokens": 32, "regex": G_REGEX, "temperature": 0}})
        ids0 = r0.get("output_ids") or []
        if r0.get("error") or not ids0:
            o["gates"]["recipe"] = "INCONCLUSIVE_RECIPE_STOP_TOKEN"
            o["measured"]["stop_run"] = {"error": r0.get("error")}
            return o
        pattern = (G_PHRASE + [ids0[-1]]) * 6
        rows = []
        for plen in range(G_TRACK - 10, G_TRACK + 2):
            flush("sglang", port)
            for k in range(2):
                gen_raw(port, {"text": f"Write a short story about planet number {2 * plen + k}. ", "sampling_params": {"max_new_tokens": 24, "temperature": 0}})
            prompt = G_FILLER[:max(0, plen - len(pattern))] + pattern
            con = gen_raw(port, {"rid": f"e24-g-{plen}", "input_ids": prompt,
                                 "sampling_params": {"max_new_tokens": 32, "regex": G_REGEX, "temperature": 0}})
            follow = prompt + list(con.get("output_ids") or []) + G_FOLLOW

            def score():
                r = gen_raw(port, {"input_ids": follow, "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                                   "return_logprob": True, "logprob_start_len": G_TRACK})
                m = r.get("meta_info") or {}
                return [x[0] for x in (m.get("input_token_logprobs") or []) if x and x[0] is not None], m.get("cached_tokens")
            hit, cached = score()
            flush("sglang", port)
            miss, _ = score()
            cm = con.get("meta_info") or {}
            rows.append({"prompt_len": plen, "finish": (cm.get("finish_reason") or {}).get("type") if isinstance(cm.get("finish_reason"), dict) else cm.get("finish_reason"),
                         "spec_verify_ct": cm.get("spec_verify_ct"), "cached": cached, "resumed": cached == G_TRACK,
                         "delta": round(sum(abs(a - b) for a, b in zip(hit, miss)), 4) if len(hit) == len(miss) and hit else None,
                         "len_ok": len(hit) == len(miss)})
            jdump(rawdir(arm) / "g_rows.json", rows)
            if not srv.alive():
                break
        time.sleep(3.0)
        ev = s6_events_settled(rawdir(arm) / "hook", "g")
        c = {k: _csum(ev, k) for k in ("grammar_steps", "grammar_truncated_steps", "false_crossing_possible", "false_crossing_returned")}
        resumed = sum(1 for r in rows if r["resumed"])
        big = [r for r in rows if r["delta"] is not None and r["delta"] >= 3.0]
        o["measured"].update(rows=rows, counters=c, resumed_at_checkpoint=resumed, events=ev["events"])
        if patched:
            trig = (c["grammar_truncated_steps"] or 0) >= 1 and (c["false_crossing_possible"] or 0) >= 1 and resumed > 0
            if (c["false_crossing_returned"] or 0) > 0 or any(r["resumed"] for r in big):
                v = "FAIL"
            elif any(r["resumed"] and r["delta"] is None for r in rows):
                v = "INCONCLUSIVE_DELTA_MISSING"            # §4.6 "all delta < 3" needs a measured delta (review #8d)
            else:
                v = "CLEAN" if trig else "INCONCLUSIVE_NOT_EXERCISED"
        else:
            chain = [r for r in big if r["resumed"] and (ev["per_rid"].get(f"e24-g-{r['prompt_len']}") or {}).get("false_crossing_returned")]
            v = "BUG_EXPOSED" if chain and (c["false_crossing_returned"] or 0) >= 1 else (
                "INCONCLUSIVE_NOT_EXERCISED" if not (c["grammar_truncated_steps"] or 0) else "INCONCLUSIVE_NO_CHAIN")
        if v in ("CLEAN", "BUG_EXPOSED") and not ev.get("ranks_agree"):
            v = "INCONCLUSIVE_COUNTERS_DESYNC"
        o["verdict"] = v
        o["gates"]["verdict"] = {"CLEAN": "GO", "BUG_EXPOSED": "GO", "FAIL": "NO_GO_PATCHED_ARM_FAILED"}.get(v, v if v.startswith(SEV) else "INCONCLUSIVE_" + v)
        return o
    return lambda: with_server(arm, body, expect_errors=not patched)


def greedy_rid_rows(port, prompts, rids, n_tok=256, conc=True):
    """Native /generate greedy rows WITH rids (T 0, top-2), in the lab's row shape, all requests concurrent."""
    rows = [None] * len(prompts)

    def one(i):
        t = time.time()
        r = gen_raw(port, {"rid": rids[i], "input_ids": list(prompts[i]), "return_logprob": True, "top_logprobs_num": 2,
                           "sampling_params": {"max_new_tokens": n_tok, "temperature": 0.0, "top_k": 1, "ignore_eos": True}})
        m = r.get("meta_info") or {}
        tops = m.get("output_top_logprobs") or []
        top2, marg = [], []
        for e in tops:
            e = sorted([x for x in (e or []) if x and x[0] is not None], key=lambda x: -x[0])
            top2.append([e[0][1], e[0][0], e[1][1], e[1][0]] if len(e) > 1 else None)
            marg.append(e[0][0] - e[1][0] if len(e) > 1 else None)
        rows[i] = {"key": rids[i], "tokens": list(r.get("output_ids") or []), "margins": marg, "top2": top2,
                   "lps": [x[0] for x in (m.get("output_token_logprobs") or [])], "text": r.get("text") or "", "error": r.get("error"),
                   "finish_reason": m.get("finish_reason"), "cached_tokens": m.get("cached_tokens"), "elapsed": round(time.time() - t, 2)}
    ths = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(len(prompts))]
    for th in ths:
        th.start()
        if not conc:
            th.join()
    for th in ths:
        th.join(tmo())
    return [r or {"error": "NO_ROW", "tokens": []} for r in rows]


def sentinel_load(port, conc, inject, tag):
    """The sentinel is submitted first and streams its first token before the other conc-1 requests are submitted
    (merge appends: it sits at batch index 0); inject=False uses a non-sentinel rid (the same load, no injection)."""
    srid = (SENT if inject else "e24-ctrl-") + f"{tag}-c{conc}"
    first, box = threading.Event(), {}

    def stream():
        body = {"rid": srid, "input_ids": list(P2K[0]), "stream": True, "return_logprob": True, "top_logprobs_num": 2,
                "sampling_params": {"max_new_tokens": 256, "temperature": 0.0, "top_k": 1, "ignore_eos": True}}
        req = _U.Request(f"http://127.0.0.1:{port}/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        ids = []
        try:
            with _U.urlopen(req, timeout=tmo()) as r:
                for line in r:
                    line = line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                        continue
                    d = json.loads(line[5:])
                    ids += list(d.get("output_ids") or [])
                    if ids:
                        first.set()
        except Exception as e:
            box["error"] = repr(e)[:200]
            err("sentinel_stream/" + srid, e)
        first.set()
        box["tokens"] = ids
    th = threading.Thread(target=stream, daemon=True)
    th.start()
    first.wait(tmo())
    rows = greedy_rid_rows(port, P2K[1:conc], [f"e24-nb-{tag}-c{conc}-{i}" for i in range(1, conc)]) if conc > 1 else []
    th.join(tmo())
    return {"sentinel_rid": srid, "sentinel_tokens": len(box.get("tokens") or []), "sentinel_error": box.get("error"), "rows": rows}


def s6_inject_verdict(kind, patched, injected, nb, zq, fcp, neighbour_logged, cmp, alive, hits, agree=True):
    """#41736 sentinels / #41737 selector NaN. Unpatched: BUG_EXPOSED = the exact counter (non-finite neighbour rows /
    an all-zero q row) with the injection on every rank. Patched FAIL: that counter, a STATE_FLIP or garbage between the
    uninjected and injected neighbours, or an error sign in the log; a plain FLIP without an uninjected A/A is BOUNDARY,
    not FAIL (§3.2; review #9) -> INCONCLUSIVE_FLIP_NO_AA; a death without an error sign is not a verdict."""
    chain = (nb or 0) > 0 if kind == "sent" else (zq or 0) > 0
    if not patched:
        return ("BUG_EXPOSED" if agree else "INCONCLUSIVE_COUNTERS_DESYNC") if chain and injected else "INCONCLUSIVE_NOT_EXERCISED"
    if chain or hits or any(((c or {}).get("state_flips") or 0) or ((c or {}).get("garbage") or 0) for c in cmp.values()):
        return "FAIL"
    if not alive:
        return "INCONCLUSIVE_SERVER_DIED"
    if not injected or (kind == "sent" and not neighbour_logged) or (kind == "nan" and not (fcp or 0) > 0):
        return "INCONCLUSIVE_NOT_EXERCISED"
    return "INCONCLUSIVE_FLIP_NO_AA" if sum(hm(c) for c in cmp.values()) else "CLEAN" if agree else "INCONCLUSIVE_COUNTERS_DESYNC"


def s6_inject_arm(arm, kind, patched):
    """#41736 sentinels (kind 'sent') / #41737 selector NaN (kind 'nan'): eager DFlash W6, T 0 everywhere, each
    concurrency with and without injection; neighbours compared with compare() (fails only at the calibrated margin)."""
    concs = (1, 8, 16) if kind == "sent" else (1, 8)

    def body(srv):
        o = {"gates": {}, "measured": {}, "loads": {}}
        caps = "Capture" in srv.log() and "CUDA graph begin" in srv.log()
        o["measured"]["decode_graphs_captured"] = caps
        if caps:
            o["gates"]["eager"] = "INVALID_DECODE_GRAPHS_CAPTURED"
        cmp, rows = {}, {}
        for c in concs:
            for inj in (False, True):
                flush("sglang", srv.port)
                ld = sentinel_load(srv.port, c, inj, f"{arm}")
                jdump(rawdir(arm) / f"load_c{c}_{int(inj)}.json", ld)
                rows[(c, inj)] = ld.pop("rows")
                o["loads"][f"c{c}_{'inj' if inj else 'base'}"] = dict(ld, n_rows=len(rows[(c, inj)]),
                                                                      row_errors=[r.get("error") for r in rows[(c, inj)] if r.get("error")][:3])
            b, i = rows[(c, False)], rows[(c, True)]
            for r_b, r_i in zip(b, i):
                r_i["key"] = r_b["key"] = r_b["key"].rsplit("-", 1)[-1]
            if b and i:
                cmp[f"c{c}"] = call(L.compare, b, i, kind="same_weights", t_pair=0.5, aa_high=None)
        time.sleep(3.0)
        ev = s6_events_settled(rawdir(arm) / "hook", kind)
        per_rank = [int((c or {}).get("draft_nan_injections" if kind == "sent" else "selector_nan_injections") or 0)
                    for c in ev["counters"]]          # one entry per TP scheduler rank (rank_files)
        nb = _csum(ev, "neighbor_rows_nonfinite") if kind == "sent" else None
        zq = _csum(ev, "zero_q_rows") if kind == "nan" else None              # #41737: exact (all-zero q row)
        fcp = _csum(ev, "first_candidate_picks") if kind == "nan" else None
        oor = _csum(ev, "out_of_row_picks") if kind == "nan" else None         # diagnostic only (repetitive grids hide it)
        o["measured"].update(per_rank_injections=per_rank, ranks=ev["ranks"], neighbor_rows_nonfinite=nb, zero_q_rows=zq,
                             first_candidate_picks=fcp, out_of_row_picks_diagnostic=oor, events=ev["events"],
                             neighbour_logged=(ev["events"].get("neighbor") or 0) > 0,
                             compare={k: {"status": v.get("status"), "hm": hm(v)} for k, v in cmp.items()})
        if len(per_rank) != TP:
            o["gates"]["injection"] = "INCONCLUSIVE_RANKS_MISSING"             # fewer than tp scheduler ranks: fail closed
        elif len(set(per_rank)) > 1:
            o["gates"]["injection"] = "INVALID_INJECTION_DESYNC"
        injected = len(per_rank) == TP and min(per_rank) >= 1
        hits = {s: srv.log().count(s) for s in ERR_SIGNS if s in srv.log()}
        v = s6_inject_verdict(kind, patched, injected, nb, zq, fcp, o["measured"]["neighbour_logged"], cmp, srv.alive(), hits,
                              agree=ev.get("ranks_agree", False))
        o["verdict"], o["log_error_hits"] = v, hits
        o["gates"]["verdict"] = {"CLEAN": "GO", "BUG_EXPOSED": "GO", "FAIL": "NO_GO_PATCHED_ARM_FAILED"}.get(v, v if v.startswith(SEV) else "INCONCLUSIVE_" + v)
        return o
    return lambda: with_server(arm, body, expect_errors=not patched)


def s6_combined():
    def body(srv):
        o = {"gates": {}, "measured": {}}
        meas(o, "m1", lambda: m_rows("sglang", srv, "s6_combined", "m1", ORACLE, NTOK, "s:"))
        o["m6_tool"] = call(L.tool_probe, "sglang", srv.port, model=srv.model_name)
        ref = _rows("s1_d6", "m1")
        s1c = s1_compare_all()                                         # d6's pure A/A: c16 repeats + the d6 <-> d6_r2 boots
        tp, tsrc = _tp("s1_d6", "s1_d6_r2")                            # A/A-calibrated T_pair (the d6 boot pair's M3)
        c, aa, ast, n = _pair(ref, _rows("s6_combined", "m1"), "same_weights", tp, [(s1c.get("aa_rate") or {}).get("s1_d6")],
                              [(s1c.get("aa_state_rate") or {}).get("s1_d6")]) if ref else ({}, None, None, 0)
        pre_kill = srv.log().partition("SIGTERM received")[0]
        hits = {s: pre_kill.count(s) for s in ERR_SIGNS if s in pre_kill}
        own = _rows("s6_combined", "m1")
        o["measured"]["vs_s1_d6"] = {"status": c.get("status"), "hm": hm(c), "aa_high": aa, "aa_state_allow": ast, "rows_judged": n,
                                     "t_pair": tp, "t_pair_source": tsrc, "counts": _cls_counts(c), "state_flips": c.get("state_flips")}
        if not ref or not c:
            v = "INCONCLUSIVE_NO_REFERENCE_ROWS"
        elif hits or str(c.get("status")).startswith("FAIL") or (c.get("state_flips") or 0) > (ast or 0):
            v = "FAIL"                                  # an error sign, a high-margin finding, or a STATE_FLIP (review 2 #3a)
        elif not srv.alive() or not own or all(r.get("error") for r in own):
            v = "INCONCLUSIVE_SERVER_DIED"              # missing rows -> INCONCLUSIVE (§4.6), never FAIL (review 2 #3b)
        elif not o["m6_tool"].get("ok"):
            v = "FAIL"
        elif c.get("status") in ("EXACT", "EXACT_TO_N", "NEAR_TIE"):
            v = "PASS"
        else:
            v = "INCONCLUSIVE_" + str(c.get("status"))
        o["verdict"], o["log_error_hits"] = v, hits
        o["gates"]["verdict"] = {"PASS": "GO", "FAIL": "NO_GO_COMBINED"}.get(v, v)
        return o
    return lambda: with_server("s6_combined", body)


def s6_unit():
    """GPU unit tests of #41736 / #41737 on overlay copies (never the venv), with the server env minus PYTHONPATH, own
    Triton / Inductor dirs, a timeout and killpg; trigger evidence = the imported module __file__ + sha recorded."""
    E = ENGINES["sglang"]
    raw = rawdir("s6_unit")
    root, cinfo = cache_root("s6_unit", CAT["s6_unit"])
    env = {k: v for k, v in E["env"].items() if k != "PYTHONPATH"}
    env.update(call(L.cache_env, "sglang", str(root)) or {})
    env.update(TRITON_CACHE_DIR=str(raw / "triton"), TORCHINDUCTOR_CACHE_DIR=str(raw / "inductor"), E24_WORK=str(WORK))
    out = {"gates": {}, "tests": {}, "cache": cinfo}
    for t in ("41736", "41737"):
        res_p = raw / f"unit_{t}.json"
        res_p.unlink(missing_ok=True)
        tsec = int(ts(360, 10))                        # per version (the script default); none + <test> run as -I children
        cmd = [E["vpy"], str(UNIT), "--test", t, "--out", str(res_p), "--patcher", str(MODS / "e24_sglang_patches.py"),
               "--timeout", str(tsec)] + (["--site", str(PSITE)] if PSITE else [])
        R["argv"]["s6_unit_" + t] = cmd
        with open(LOGS / f"unit_{t}.log", "w") as fh:
            p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(raw))
            try:
                rc = p.wait(timeout=2 * (tsec + 120) + 180)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    pass
                p.wait(30)
                rc = -9
        try:
            res = json.loads(res_p.read_text())
        except Exception:
            res = {"ran": False, "reason": f"rc={rc}: no result file (logs/unit_{t}.log)"}
        res["rc"] = rc
        # the script's own status (PASS | FAIL_PATCHED_NOT_CLEAN | INCONCLUSIVE_NOT_EXERCISED | INCONCLUSIVE_NOT_RUN |
        # INVALID_OVERLAY_STATE | INVALID_TEST_MISMATCH); never derived from chain / clean here
        v = res.get("status") or ("INCONCLUSIVE_NOT_RUN" if not res.get("ran") else "INCONCLUSIVE_NO_STATUS")
        out["tests"][t] = {"verdict": v, **{k: res.get(k) for k in ("ran", "chain", "clean", "reason", "rc", "max_abs_delta_finite", "paths")},
                           "details": small(res.get("details")) if res.get("details") else None}
        if v == "PASS_AST_FALLBACK":                  # a weaker PASS: one side ran the overlay's source text of _grouped_conv
            out["tests"][t]["caveat"] = ("PASS_AST_FALLBACK: `import sglang.srt.models.dflash` failed standalone, so one side "
                                         "ran the overlay's _grouped_conv source text; not the plain module path")
        out["gates"]["unit_" + t] = ("GO" if v in ("PASS", "PASS_AST_FALLBACK") else ("NO_GO_UNIT_" + t + "_" + v) if str(v).startswith("FAIL")
                                     else v if str(v).startswith(SEV) else "INCONCLUSIVE_" + str(v))
    out["status"] = worst(out["gates"].values())
    gpu_gate("s6_unit", out)
    return out


def s6_pairs():
    sec = R["sections"].get("6") or {}
    P6 = {"43567+43559 forced-fail (target-only)": ("s6_ffu", "s6_ffp"), "43567+43559 forced-fail (DFlash W6)": ("s6_ffud", "s6_ffpd"),
          "43029 grammar (NGRAM)": ("s6_gu", "s6_gp"), "43029 grammar (DFlash extra_buffer fallback)": ("s6_gu_df", "s6_gp_df"),
          "43029 grammar (production lazy, G-L)": ("s6_gl_u", "s6_gl_p"), "41736 sentinels": ("s6_sent_u", "s6_sent_p"),
          "41737 selector NaN": ("s6_nan_u", "s6_nan_p")}
    def vd(a):                                        # an arm that ran without a verdict still shows (review #8a)
        r = sec.get(a)
        if not isinstance(r, dict) or (not r.get("verdict") and str(r.get("status") or "").startswith("SKIPPED")):
            return None
        return r.get("verdict") or f"NO_VERDICT ({r.get('status')})"
    out = {}
    for name, (u, p) in P6.items():
        vu, vp = vd(u), vd(p)
        if vu is None and vp is None:
            continue
        out[name] = {"unpatched": vu, "patched": vp, "verdict": "FAIL" if vp == "FAIL" else
                     ("PASS" if vu == "BUG_EXPOSED" and vp == "CLEAN" else "INCONCLUSIVE")}
    vnp = vd("s6_np")
    if vnp:
        out["43559 natural (unpatched half = E17 run 1's recorded crash, README measured negative)"] = {
            "unpatched": "E17_RUN1_CRASH_PATH_43559 (recorded, not repeated)", "patched": vnp,
            "verdict": "PASS" if vnp == "CLEAN" else ("FAIL" if vnp == "FAIL" else "INCONCLUSIVE")}
    for t, r in ((sec.get("s6_unit") or {}).get("tests") or {}).items():
        out[f"unit {t}"] = {"verdict": r.get("verdict"), **({"caveat": r["caveat"]} if r.get("caveat") else {})}
    if sec.get("s6_combined"):
        out["combined (all five)"] = {"verdict": sec["s6_combined"].get("verdict")}
    return out


# ------------------------------------------------------------------------------------------------ run order
def run_plan():
    sgl, vll = dep_engine("sglang"), dep_engine("vllm")
    for sec, a, est in PLAN:
        opt = a.endswith("?")
        a = a.rstrip("?")
        try:
            if a in S1_MEAS:
                step(sec, a, est, s1(a), dep=vll if CAT[a]["engine"] == "vllm" else sgl)
            elif a == "s6_unit":
                step(sec, a, est, s6_unit, dep=sgl)
            elif a in ("s6_ffu", "s6_ffp"):
                step(sec, a, est, s6_t1_arm(a, a == "s6_ffp", safety=a == "s6_ffp"), dep=sgl)
            elif a == "s6_np":
                step(sec, a, est, s6_t1_arm(a, True, safety=True, natural=True), dep=sgl)
            elif a in ("s6_ffud", "s6_ffpd"):
                step(sec, a, est, s6_t1_arm(a, a == "s6_ffpd", safety=a == "s6_ffpd"), dep=sgl)   # FF-P-D: FF-P's safety cases (review #8b)
            elif a in ("s6_gu", "s6_gp", "s6_gl_u", "s6_gl_p"):
                step(sec, a, est, s6_g_arm(a, a.endswith("p")), dep=sgl)
            elif a in ("s6_gu_df", "s6_gp_df"):
                ng = {x: ngram_launch_failed(R["sections"]["6"].get(x)) for x in ("s6_gu", "s6_gp")}
                if FORCE_COND or any(ng.values()):
                    step(sec, a, est, s6_g_arm(a, a == "s6_gp_df"), dep=sgl)
                else:
                    skip_cond(sec, a, f"NGRAM arms launched ({ng}): the DFLASH extra_buffer fallback is not needed")
            elif a in ("s6_sent_u", "s6_sent_p"):
                step(sec, a, est, s6_inject_arm(a, "sent", a.endswith("p")), dep=sgl)
            elif a in ("s6_nan_u", "s6_nan_p"):
                step(sec, a, est, s6_inject_arm(a, "nan", a.endswith("p")), dep=sgl)
            elif a == "s6_combined":
                step(sec, a, est, s6_combined(), dep=sgl)
            elif a == "s2_w6s20":
                bands = ("8k",) if SM else ("8k", "40k", "90k", "120k")
                step(sec, a, est, s2_arm(a, bands, ems=() if SM else ("short", "long", "xl"), role="control"), dep=sgl)
            elif a == "s2_w6s20_aa":
                step(sec, a, est, s2_arm(a, ("8k", "40k"), role="aa"), dep=sgl)
            elif a in ("s2_w4s20", "s2_w6s16", "s2_w4s16"):
                bands = ("8k",) if SM else ("8k", "40k")
                step(sec, a, est, s2_arm(a, bands, ems=("long",) if a == "s2_w4s20" and not SM else ()), dep=sgl)
                R["comparisons"]["2"] = s2_compare()
            elif a == "s2_w5s20|s2_w7s20":
                cmpd = R["comparisons"]["2"] = s2_compare()
                run = ["s2_w5s20", "s2_w7s20"] if FORCE_COND else ([cmpd["branch"]["run"]] if cmpd["branch"].get("run") else [])
                for x in ("s2_w5s20", "s2_w7s20"):
                    if x in run:
                        step(sec, x, est, s2_arm(x, ("8k", "40k")), dep=sgl)
                    else:
                        skip_cond(sec, x, "branch rule: " + json.dumps(cmpd["branch"]))
                R["comparisons"]["2"] = s2_compare()
            elif a == "s2_best_long":
                cmpd = R["comparisons"]["2"] = s2_compare()
                best = (cmpd["best_or_top"] or S2_CHALLENGERS[0]) if FORCE_COND else cmpd["best_candidate"]
                if best:
                    step(sec, "s2_best_long__" + best, est, s2_arm("s2_best_long__" + best, ("90k", "120k")), dep=sgl)
                    R["comparisons"]["2"] = s2_compare()
                else:
                    skip_cond(sec, "s2_best_long", "no eligible challenger beats the control beyond the A/A spread: " + json.dumps(
                        {a: v.get("not_eligible") or v.get("better") for a, v in cmpd["arms"].items() if a in S2_CHALLENGERS})[:400])
            elif a in S3_ARMS:
                step(sec, a, est, s3_arm(a), dep=sgl)
                R["comparisons"]["3"] = s3_compare()
            elif a.startswith("s3_conf__best"):
                c3 = R["comparisons"]["3"] = s3_compare()
                i = int(a[-1]) - 1
                pool = c3["best_two"] or ([x for x in S3_ARMS if x != "s3_ctrl"][:2] if FORCE_COND else [])
                if i < len(pool):
                    step(sec, "s3_conf__" + pool[i], est, s3_arm("s3_conf__" + pool[i], conf=True), dep=sgl)
                else:
                    skip_cond(sec, a, "fewer than two valid section-3 arms to confirm")
            elif a == "s3_ctrl_rerun":
                step(sec, a, est, s3_arm(a, rerun=True), dep=sgl)
            elif a in ("s4_m84", "s4_m68", "s4_mem080", "s4_mem086", "s4_lowcap"):
                loads = ("90k", "116k", "shared") if a == "s4_m84" else ("90k",) if a == "s4_lowcap" else ("90k", "116k")
                step(sec, a, est, s4_arm(a, loads), dep=sgl)
            elif a in ("s4_m88", "s4_m92"):
                m84 = (R["sections"]["4"].get("s4_m84") or {}).get("measured") or {}
                mb = [ld for ld in ("census_90k", "census_116k") if "mamba" in _blocking(m84.get(ld))]
                if FORCE_COND or mb:
                    step(sec, a, est, s4_arm(a, ("90k", "116k")), dep=sgl)
                else:
                    skip_cond(sec, a, "the control is not Mamba-bound by the direct gauges: " + json.dumps({k: _blocking(m84.get(k)) for k in ("census_90k", "census_116k")}))
            elif a in ("s5_c4096", "s5_c2048", "s5_c8192", "s5_pdi1", "s5_pdi4", "s5_mixed"):
                step(sec, a, est, s5_arm(a), dep=sgl)
                R["comparisons"]["5"] = s5_compare()
            elif a.startswith("aa_ctrl_r"):
                i = int(a[-1])
                cmpd = s2_compare()
                best = cmpd["best_or_top"] or (S2_CHALLENGERS[0] if FORCE_COND else None)
                if left() < (2 * 18 + 5 + 10) * 60 * TS:
                    skip_cond(sec, a, "A/A rounds stop: less than one round + the report reserve left")
                    continue                                  # later entries still get their own skip record
                step(sec, a, est, s2_arm(a, ("8k", "40k"), role="aa_repeat"), dep=sgl)
                if best:
                    step(sec, f"aa_best_r{i}__{best}", est, s2_arm(f"aa_best_r{i}__{best}", ("8k", "40k"), role="aa_repeat"), dep=sgl)
            else:
                skip_cond(sec, a, "no runner for this plan entry (builder bug)")
                err("plan/" + a, None, note="no runner")
        except Exception as e:
            err(f"plan/{a}", e)
            R["sections"].setdefault(sec, {}).setdefault(a, {}).update(status="ERROR_" + type(e).__name__)
        if sec == "6":
            R["comparisons"]["6"] = s6_pairs()
        save()


if not ABORT["why"]:
    run_plan()
else:
    for _sec, _a, _est in PLAN:
        step(_sec, _a.rstrip("?"), _est, lambda: {"status": "SKIPPED_ABORT"})
'''

CELL6 = r'''# ==================================================================== CELL 6 — report (comparisons recomputed with all A/A data)
for _k, _fn in (("1", lambda: s1_compare_all(final=True)), ("2", s2_compare), ("3", s3_compare), ("5", s5_compare), ("6", s6_pairs)):
    if not R["sections"].get(_k):                       # a section this variant never stepped prints no table
        continue
    try:
        R["comparisons"][_k] = _fn()
    except Exception as e:
        err("report/comparisons/" + _k, e)
for _a, _r in (R["sections"].get("1") or {}).items():   # the final A/A recompute may add or lift a section-1 verdict
    if isinstance(_r, dict) and _r.get("status_measured"):
        try:
            jdump(ARMS_OUT / f"{_a}.json", _r)
        except Exception as e:
            err(f"report/arm_summary/{_a}", e)
try:
    if not FAKE and not L.LIVE:                         # FAKE: never pkill (parallel local runs share the host)
        R["orphans_killed_at_end"] = L.kill_orphans(fake=False)
except Exception as e:
    err("report/kill_orphans", e)


def _f(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _bad_gates(rec):
    return ", ".join(f"{k}={v}" for k, v in ((rec or {}).get("gates") or {}).items() if v not in ("GO", None, "SKIPPED_FAKE"))[:160] or "-"


md = [f"# E24 {VARIANT} report ({R['start_utc']}, {round((time.time() - T_START) / 60, 1)} min, fake={FAKE})", "",
      f"abort: {ABORT['why']} | errors: {len(R['errors'])} | api drift: {json.dumps(R['api_drift'])[:300]}", "",
      "## Steps", "| section | arm | status | ready s | min | not-GO gates |", "|---|---|---|---|---|---|"]
for _o in R["order"]:
    _s, _a = _o.split("/", 1)
    _r = R["sections"].get(_s, {}).get(_a) or {}
    md.append(f"| {_s} | {_a} | {_r.get('status')} | {_f(_r.get('ready_sec'))} | {_f(_r.get('minutes'), 2)} | {_bad_gates(_r)} |")
c1 = R["comparisons"].get("1") or {}
if c1.get("pairs"):
    md += ["", "## Section 1: oracle pairs (aligned by key; T_pair calibrated; A/A allowance scaled to the judged rows)",
           "| pair | kind | status | high-margin | A/A allowance | Fisher p vs A/A | T_pair | positions | attribution |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _k, _p in c1["pairs"].items():
        _at = ",".join(sorted({x.get("attribution") for x in _p.get("attribution") or []})) or "-"
        md.append(f"| {_k} ({_p.get('label')}) | {_p.get('kind')} | {_p.get('status')} | {_f(_p.get('hm'))} | {_f(_p.get('aa_high'))} | "
                  f"{_f(_p.get('p_value'), 4)} | {_f(_p.get('t_pair'), 3)} ({_p.get('t_pair_source', '-')}) | "
                  f"{_f(_p.get('positions_examined'))} | {_at} |")
    md += ["", "Decisive chain a -> b -> c32 -> c -> d6 / d4: " + json.dumps(c1.get("decisive"))[:600],
           "", "M4 decode vs prefill: " + json.dumps({a: {"own": (v.get("own") or {}).get("status"), "vs_base": (v.get("vs_base") or {}).get("status"),
                                                           "state_drift": v.get("state_drift")} for a, v in (c1.get("m4") or {}).items()}),
           "", "Arm verdicts (applied to the arm statuses; STATE_FLIP / STATE_DRIFT / target-only drift / restore): "
           + json.dumps(c1.get("arm_verdicts")), "", "Suggestive only (EMS rule without Fisher significance; not NO-GO): "
           + json.dumps(c1.get("arm_notes")), "", "Cold / warm controls: " + json.dumps(c1.get("cache_controls")),
           "", "| arm | pass 1 <-> pass 2 (restore) | restore | serial vs c16 | c16 A/A | A/A rate (FLIP+STATE / STATE per row) |",
           "|---|---|---|---|---|---|"]
    for _a, _w in (c1.get("within") or {}).items():
        md.append(f"| {_a} | {(_w.get('pass1_pass2') or {}).get('status', '-')} | {_w.get('pass2_restore', '-')} | "
                  f"{(_w.get('serial_vs_c16') or {}).get('status', '-')} | {(_w.get('c16_aa') or {}).get('status', '-')}"
                  f"{'' if _w.get('c16_aa_pure', True) else ' (not pure: a flush failed)'} | "
                  f"{_f((c1.get('aa_rate') or {}).get(_a), 3)} / {_f((c1.get('aa_state_rate') or {}).get(_a), 3)} |")
c2 = R["comparisons"].get("2") or {}
if c2.get("arms"):
    md += ["", f"## Section 2: width x streams vs W6 S20 (A/A spread {_f(c2.get('aa_spread'), 4)}; branch {json.dumps(c2.get('branch'))})",
           f"best (confirmed: 8k / 40k beyond the spread + 90k / 116k hold + reproduced): {c2.get('best')} | candidate: "
           f"{c2.get('best_candidate')} | check: {json.dumps(c2.get('best_check'))[:700]}",
           "eligibility: " + json.dumps({_a: (_v.get('not_eligible') or 'eligible') for _a, _v in c2['arms'].items()})[:1200], "",
           "| arm | band | tok/s | control | ratio | tok/step | step ms | running p50 | status | N |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _a, _v in c2["arms"].items():
        for _b, _w in (_v.get("bands") or {}).items():
            md.append(f"| {_a} | {_b} | {_f(_w.get('tok_s'))} | {_f(_w.get('ctrl_tok_s'))} | {_f(_w.get('ratio'), 3)} | {_f(_w.get('tokens_per_step'), 2)} | "
                      f"{_f(_w.get('step_ms'))} | {_f(_w.get('running_p50'))} | {_w.get('status')} | {_w.get('n')} {_w.get('equal_n') or ''} |")
c3 = R["comparisons"].get("3") or {}
if c3.get("arms"):
    md += ["", "## Section 3: NCCL arms (8k, N = 20; ranking by mean client step ms)", "| arm | status | window | env proof | step ms mean | p95 | tok/s | tok/s per tok/step |",
           "|---|---|---|---|---|---|---|---|"]
    for _a, _v in c3["arms"].items():
        md.append(f"| {_a} | {_v.get('status')} | {_v.get('window')} | {_v.get('env_proof')} | {_f(_v.get('client_step_ms_mean'))} | "
                  f"{_f(_v.get('client_step_ms_p95'))} | {_f(_v.get('tok_s'))} | {_f(_v.get('tok_s_per_token_per_step'))} |")
    md.append("control mean step ms: " + _f(c3.get("control_step_ms_mean")) + " | best two (GO, faster than the control): "
              + json.dumps(c3.get("best_two")) + " | confirmations: " + json.dumps(c3.get("confirmations"))[:1200]
              + " | control cold 40k prefill s: " + str(c3.get("cold_prefill_40k_sec")))
_s4 = R["sections"].get("4") or {}
_cen = [(a, k, v) for a, r in _s4.items() for k, v in ((r or {}).get("measured") or {}).items() if k.startswith("census_") and isinstance(v, dict)]
if SM:
    _cen += [("s2_w6s20", "census_90k", ((R["sections"]["2"].get("s2_w6s20") or {}).get("measured") or {}).get("census_90k") or {})]
if _cen:
    md += ["", "## Section 4: capacity census (offered = the full logical count)",
           "| arm | load | status | running p50 | blocking | KV pool | retractions | tok/s | new prefill / admitted / band |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _a, _k, _v in _cen:
        _rp = ((_v.get("running_lines") or {}).get("p50") if (_v.get("running_lines") or {}).get("n") else (_v.get("running_gauge") or {}).get("p50"))
        md.append(f"| {_a} | {_k} | {_v.get('status')} | {_f(_rp)} | {_v.get('blocking_reason')} | "
                  f"{_f((_v.get('gauges_end') or {}).get('max_total_num_tokens') or (_v.get('gauges_mean') or {}).get('max_total_num_tokens'))} | "
                  f"{_f(_v.get('retractions'))} | {_f(_v.get('tok_s'))} | "
                  f"{_f((_v.get('prefill_accounting') or {}).get('new_per_admitted_over_band'), 3)} |")
c5 = R["comparisons"].get("5") or {}
if c5.get("arms"):
    md += ["", "## Section 5: interference (fixed windows " + json.dumps({k: round(v, 1) for k, v in c5['windows_preset'].items()}) + ")",
           "| arm | load | decode tok/s | control | starvation | max gap s | stall s | prefill s | TPOT p95 | status |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _a, _v in c5["arms"].items():
        for _l, _w in _v.items():
            md.append(f"| {_a} | {_l} | {_f(_w.get('decode_tok_s'))} | {_f(_w.get('ctrl_decode_tok_s'))} | {_w.get('starvation')} | {_f(_w.get('max_gap_s'), 2)} | "
                      f"{_f(_w.get('stall_s'), 2)} | {_f(_w.get('prefill_s'), 1)} | {_f(_w.get('tpot_p95'), 4)} | {_w.get('status')} |")
    md.append("STOP verdicts (starvation on any load, or mixed-window throughput < 0.97 x the 4096 control): "
              + json.dumps(c5.get("stop")))
c6 = R["comparisons"].get("6") or {}
if c6:
    md += ["", "## Section 6: backports (PASS = unpatched chain + patched clean WITH trigger evidence in the patched arm)",
           "| test | unpatched | patched | verdict |", "|---|---|---|---|"]
    for _k, _v in c6.items():
        md.append(f"| {_k} | {_v.get('unpatched', '-')} | {_v.get('patched', '-')} | {_v.get('verdict')}"
                  + (f" (caveat: {_v['caveat'][:90]})" if _v.get("caveat") else "") + " |")
# ---- the three throughput denominators (Y11), reported separately
_sat = [(s, a, k, w.get("tok_s"), w.get("status")) for s in ("2", "3") for a, r in (R["sections"].get(s) or {}).items()
        for k, w in ((r or {}).get("windows") or {}).items() if isinstance(w, dict)]
_mix = [(a, l, w.get("decode_tok_s"), w.get("status")) for a, v in (c5.get("arms") or {}).items() for l, w in v.items()]
R["denominators"] = {"saturated_decode": [list(x) for x in _sat], "mixed_serving_window": [list(x) for x in _mix],
                     "agent_end_to_end": "NOT_MEASURED: " + R["not_measured"]["agent_end_to_end_denominator"]}
md += ["", "## Throughput denominators (Y11)", f"- saturated decode (steady windows): {len(_sat)} windows",
       f"- mixed serving window (section 5 fixed windows): {len(_mix)} loads", "- agent end to end: NOT_MEASURED",
       "", "NOT_MEASURED: " + json.dumps(R["not_measured"])]
# ---- SMOKE gate summary: which machinery worked (fail-closed: missing = not proven)
_s = lambda sec, a: (R["sections"].get(sec) or {}).get(a) or {}
_flag = [r.get("flag_check") or {} for sec in R["sections"].values() for r in sec.values() if isinstance(r, dict) and r.get("flag_check")]
R["machinery"] = {
    "inputs": R["gates"].get("inputs"), "tokenizer_ids_identical": (R.get("prompts") or {}).get("tokenizer", {}).get("fp8_int4_ids_identical"),
    "sglang_install": not ENGINES["sglang"].get("failed"), "vllm_install": not ENGINES["vllm"].get("failed") if NEED_VLLM else "not needed",
    "staging_sha256": bool(R["identity"].get("int4_shard_sha256")) and R["identity"].get("int4_shard_sha256") != "NOT_STAGED",
    "base_argv_matches_design": (R["setup"].get("base_check") or {}).get("ok"),
    "flag_checks_ran_ok": f"{sum(1 for f in _flag if f.get('ran') and f.get('ok'))}/{len(_flag)}",
    "flag_resolve_advisory_failures": sorted({str(f.get("resolve_error"))[:160] for f in _flag if f.get("resolve_advisory")}),
    "patch_site": R["setup"].get("patch_site"), "patch_broken": PATCH_BROKEN,
    "hook_selftest_go": sum(1 for sec in R["sections"].values() for r in sec.values() if isinstance(r, dict) and (r.get("gates") or {}).get("hook") == "GO"),
    "cache_isolated": {a: (r.get("cache_verdict") or {}).get("status") for a, r in (R["sections"].get("1") or {}).items() if isinstance(r, dict)},
    "tf_preflight": PREFLIGHT_DONE, "traces": {k: len(v.get("cont") or []) for k, v in TRACES.items()},
    "m4_arms": sorted(M4D), "unit_tests": {t: v.get("verdict") for t, v in (_s("6", "s6_unit").get("tests") or {}).items()},
    "census_90k": (((_s("2", "s2_w6s20").get("measured") or {}).get("census_90k")) or {}).get("status") if SM else None,
    "interference_D2": (((_s("2", "s2_w6s20").get("measured") or {}).get("interference_D2")) or {}).get("status") if SM else None,
    "profile": {a: (lambda p: isinstance(p, dict) and not p.get("error") and len(p.get("ranks") or []) >= TP and isinstance(p.get("trace"), dict)
                       and not p["trace"].get("error"))((_s(sec, a).get("measured") or {}).get("profile")) for sec, a in (("2", "s2_w6s20"), ("3", "s3_cta4"))},
    "nccl_env_proof_cta4": {k: (_s("3", "s3_cta4").get("env_proof") or {}).get(k) for k in ("status", "checks")},
}
md += ["", "## Machinery (SMOKE gate summary)", "```", json.dumps(R["machinery"], indent=1, default=str)[:6000], "```"]
R["report_md"] = "\n".join(md)
(OUT / "e24_report.md").write_text(R["report_md"])
# ---- raw files: /tmp/e24/<arm>/ -> one e24_raw.tar.gz (DESIGN §3.6); OUT keeps < 300 files
_skip = {"cache", "mods", "e24_hook_site", "e24_libcuda"}
_budget, _used, _skipped = 2 * 1024 ** 3, 0, []
try:
    with tarfile.open(OUT / "e24_raw.tar.gz.tmp", "w:gz") as _tf:
        for _d in sorted(WORK.iterdir()):
            if not _d.is_dir() or _d.name in _skip:
                continue
            for _p in sorted(_d.rglob("*")):
                if _p.is_file():
                    _sz = _p.stat().st_size
                    if _used + _sz > _budget:
                        _skipped.append(str(_p))
                        continue
                    _tf.add(_p, arcname=str(_p.relative_to(WORK)))
                    _used += _sz
    (OUT / "e24_raw.tar.gz.tmp").replace(OUT / "e24_raw.tar.gz")
    R["raw_tar"] = {"bytes_in": _used, "skipped_over_budget": _skipped[:50], "n_skipped": len(_skipped)}
    if not _skipped:                                  # the per-arm snapshots are inside e24_raw.tar.gz now
        shutil.rmtree(RAW_ARMS, ignore_errors=True)
except Exception as e:
    err("report/raw_tar", e)
_n = sum(1 for _p in OUT.rglob("*") if _p.is_file())
if _n >= 300:                                       # consolidate the per-arm summaries before asserting
    with tarfile.open(OUT / "e24_arms.tar.gz", "w:gz") as _tf:
        _tf.add(ARMS_OUT, arcname="arms")
    shutil.rmtree(ARMS_OUT, ignore_errors=True)
    _n = sum(1 for _p in OUT.rglob("*") if _p.is_file())
R["out_files"] = _n
R["minutes"] = round((time.time() - T_START) / 60, 1)
save()
assert _n < 300, f"/kaggle/working/e24 holds {_n} files (>= 300): Kaggle's output listing would truncate"
print(R["report_md"])
print("E24", VARIANT, "done in", R["minutes"], "min | abort:", ABORT["why"], "| errors:", len(R["errors"]), "| out files:", _n)
'''

# ---------------------------------------------------------------------------------------------------------------------
# build
VARIANTS = {"SMOKE": ("SMOKE", "ALL"), "FULL_A": ("FULL", "A"), "FULL_B": ("FULL", "B")}
NB_NAME = "Q27-XHIGH-E24_SGLANG_SPEED_{}.ipynb"
OWNER = "foysalemonshanto"                     # placeholder account: the user chooses who runs each notebook
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
MODELS = [f"{OWNER}/qwen3-8-27b-int4-redhat/PyTorch/w4a16-g128/1", f"{OWNER}/qwen3-8-27b-dflash2/PyTorch/bf16/1",
          f"{OWNER}/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"]
DATASETS = {"SMOKE": [f"{OWNER}/sglang-0-5-21-cu13-wheelhouse", f"{OWNER}/vllm-0-32-0-cu13-wheelhouse"],
            "FULL_A": [f"{OWNER}/sglang-0-5-21-cu13-wheelhouse", f"{OWNER}/vllm-0-32-0-cu13-wheelhouse"],
            "FULL_B": [f"{OWNER}/sglang-0-5-21-cu13-wheelhouse"]}
CELLS = {"CELL2": CELL2, "CELL3": CELL3, "CELL4": CELL4, "CELL5": CELL5, "CELL6": CELL6}


def _slim(code):
    """The notebook copy of a cell without comments and blank lines (this builder keeps them for readers): ~16 kB under
    the 400 kB commit limit. The cell's AST must not change, so the notebook runs exactly the code reviewed here."""
    toks = [t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type != tokenize.COMMENT]
    out = "\n".join(line.rstrip() for line in tokenize.untokenize(toks).splitlines() if line.strip()) + "\n"
    assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(out)), "comment stripping changed a cell's code"
    return out


built = {}
for variant, (mode, part) in VARIANTS.items():
    c1 = _slim(CELL1.replace("__MODE__", mode).replace("__PART__", part).replace("__BLOBS__", repr(BLOBS)).replace("__SHAS__", repr(SHAS)))
    md = MD0.replace("__VARIANT__", variant).replace("__MODE__", mode).replace("__PART__", part)
    guarded = [("code", "if not RERUN:\n" + textwrap.indent(_slim(c), "    ")) for c in CELLS.values()]
    cells = [("markdown", md), ("code", c1)] + guarded
    for i, (kind, src) in enumerate(cells):
        if kind == "code":
            compile(src, f"<{variant} cell {i}>", "exec")
    nb = {"cells": [{"cell_type": k, "metadata": {}, "source": s.splitlines(True),
                     **({"execution_count": None, "outputs": []} if k == "code" else {})} for k, s in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1)
    out = HERE.parent / NB_NAME.format(variant)
    out.write_text(data, encoding="utf-8", newline="\n")
    nbytes = len(data.encode("utf-8"))
    kb = nbytes / 1000                                   # the strict (decimal) reading of DESIGN's "< 400 kB"
    assert nbytes < 400_000, f"{out.name} is {nbytes} bytes: Kaggle commits above ~407-444 kB never start"
    sha = hashlib.sha256(data.encode("utf-8")).hexdigest()
    built[variant] = {"file": out.name, "sha256": sha, "bytes": nbytes}
    print(out.name, sha[:8], f"{kb:.0f} kB")
    slug = f"q27-xhigh-e24-sglang-speed-{variant.lower().replace('_', '-')}"
    meta = {"id": f"{OWNER}/{slug}", "title": f"Q27 XHIGH E24 SGLang Speed {variant.replace('_', ' ')}", "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "keywords": [], "dataset_sources": DATASETS[variant], "kernel_sources": [],
            "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": MODELS, "docker_image": IMAGE,
            "machine_shape": "NvidiaL4"}
    kdir = HERE / "kernels" / variant
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")

# every planned server argv per variant (both branches of every conditional) for the CPU kit's flag checks
_cat = {v: catalog("<INT4>", "<DFLASH>", "<FP8>", v) for v in VARIANTS}
plan = {"note": "argv after the launch prefix ([<venv>/bin/python, -m, sglang.launch_server] / [<venv>/bin/vllm, serve]); "
                "<INT4> / <DFLASH> = the staged /dev/shm dirs (or the /kaggle/input mounts), <FP8> = the FP8 mount; "
                "conditional arms list both branches; runtime argvs are also recorded in R['argv'] (E24_FORCE_CONDITIONAL=1)",
        "launch_prefix": {"sglang": ["<venv>/bin/python", "-m", "sglang.launch_server"], "vllm": ["<venv>/bin/vllm", "serve"]},
        "base_check": _bc, "builder_sha256": SHAS["notebook_builder"], "notebooks": built,
        "variants": {v: {a: {"engine": _cat[v][a]["engine"], "section": _cat[v][a]["section"], "flags": _cat[v][a]["flags"],
                             "env": _cat[v][a]["env"], "cache": _cat[v][a]["cache"], "patches": list(_cat[v][a]["patches"])}
                         for a in plan_arms(v) if a in _cat[v] and _cat[v][a]["flags"]} for v in VARIANTS}}
for v in VARIANTS:
    _miss = [a for a in plan_arms(v) if a not in _cat[v]]
    assert not _miss, (v, _miss)
(HERE / "plan_argvs_e24.json").write_text(json.dumps(plan, indent=1) + "\n", encoding="utf-8", newline="\n")
print("plan_argvs_e24.json", {v: len(x) for v, x in plan["variants"].items()}, "| modules", {k: v[:8] for k, v in SHAS.items()})
if MISSING:
    print("WARNING: structure-only build, missing modules:", MISSING)
