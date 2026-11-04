"""E25 NB4S STALL (2026-11-01): a small separate GPU notebook with one question: which change removes the ~21 ms in-graph
stall on stage 0 (NB1 R2 B1: the stage-0 FULL graph stops at kernel 172 of 1,036 until the OTHER micro-batch's PP
sampled-token Broadcast Int x8 ends) and speeds up decode: CUDA_DEVICE_MAX_CONNECTIONS=32, Q-pack or Q-defer?

  A0  RC2 / NB1 setup verbatim: merge + background sha256, the Merlin offline install, vLLM #60490 on the installed tree
      (strict pre / post sha, marker, py_compile, origin) + NB1's read-only worker extension, extended by one read-only
      method nb4s_env (the worker's os.environ, pid, rank, device).
  B   NB1 R2 B3 exactly (FP8 E4M3 KV, pin 3,553,580,524 B = 214 blocks, --language-model-only, #60490, MTP K3, seqs 8,
      the FP8 capture list, --profiler-config): read-back, env read-back from every worker by RPC (+ /proc/<pid>/environ of
      the engine processes), probes, sampled, greedy oracle vs RC2's recorded FP8 rows (B1 and B3 were EXACT), warm-up,
      8k@8 steady 90 s x 2, one profiled 8k@8 window (bounded POST /start_profile {"delay_iterations":20,"max_iterations":40}),
      40k@8 steady 90 s.
  S   B + CUDA_DEVICE_MAX_CONNECTIONS=32 (exported after Merlin's build(), which strips MERLIN_* only, so every spawned
      worker and the PLE offload process inherit it before CUDA init).
  QP  B + agent B's nb4_ppack patch rev 2 (pp_utils.py + model_runner.py + gpu_worker.py, strict pre / post sha; a refused
      apply -> QP and QD NOT_INTEGRATED, no boot) + NB4_PPACK=1 (Q-pack: one packed broadcast per step, sent after the
      drafts). The 4 'NB4_PPQ init' lines must agree and the counters must move (readback_label ENGAGED), else the arm is
      not counted. rev 2 (review NB4_PR53948_COMPARISON_2026-11-01.md): F1 shutdown drain, F2 flush inside the profile /
      shutdown RPCs (the /stop_profile hang), F3 draft rows gathered on the main stream, R1 Q-defer read-back by flush
      ratios (ENGAGED_FALLBACK = deferral collapsed to post-at-consume: its own verdict, never counted as engaged).
  QD  the same patched tree + NB4_PPDEFER=1 (Q-defer: the receive posted after the next forward launch); stage-1 send-side
      pp_broadcast kernel durations logged (the main QD risk: the stall moves to stage 1). After QD (or whenever the run
      ends) the patch is reverted and the pre-shas verified.
  Every arm: the same flags (profiler included), the same pin (B's final), the same items and windows in the same order,
  greedy oracle vs B EXACT or NEAR_TIE. Priority when time is short: B > S > QP > QD; an arm starts only with its full
  minimum left (est_min + 5 min); the body cap is clipped to the time left minus 5 min; due() gates every item.
  Analysis after each boot (CPU): agent C's nb4_timeline.py embedded verbatim and sha-checked (analyze() on the boot's
  traces: tick ms, per-stage compute / in-graph stall / NCCL peer wait / transfer / host, graph modes, parity, dependency)
  + nb4s_lib.graph_stalls (per FULL graph: largest in-graph gap, node index, stream, releasing kernel, offset to the
  nearest pp_broadcast end, per stage; stage-1 SendRecv and pp_broadcast send kernels per tick; per-step request counts
  per parity). Steady windows carry the iteration-token histogram split (BALANCED / IMBALANCED / UNRESOLVED): only
  BALANCED windows are compared.
  Verdicts per arm (nb4s_lib.verdict, written in a finally block whatever happens): STALL_FIXED_WIN / STALL_FIXED_NO_SPEED /
  STALL_MOVED / STALL_REDUCED / NO_EFFECT / CORRECTNESS_FAIL / ENGAGED_FALLBACK / NOT_ENGAGED / NOT_INTEGRATED /
  SKIPPED_BUDGET / INCONCLUSIVE_*; the best arm is named (ENGAGED_FALLBACK never is). Q arms: the verdict reads the latest
  read-back (q_readback_final from the whole log after the server stopped > q_readback_end > the mid-arm q_readback).

Klee100 audit lessons built in: one OOM policy (a launch OOM -> one retry 0.25 GiB lower on a fresh port and launch name,
the same rule for every arm; later arms start at B's final pin and a pin mismatch makes that arm INCONCLUSIVE; a body OOM
never retries), OOM / device-fault search in the launch's WHOLE log, try/finally verdict (and patch revert), live records
saved after every item, free memory per card = min(nvidia-smi, the worker's torch free), every request traced in full
(RC2's request log), the never-do list enforced at build time (the arm plan) and at launch (the full final env / argv).

Reuse: the NB1 R2 builder (build_e25_nb1.py) is executed up to its notebook-writing part; its cells are rebuilt and checked
byte for byte against the Kaggle-run NB1 R2 notebook (sha 62670fce). NB4S keeps NB1's CELL 1-5 (marked one-line edits),
NB1's CELL 6 minus the capacity / prefix / routing / PP2-trace functions NB4S never calls (dropped by AST, every dropped name
proven unreferenced), and adds CELL 7 (the arms, the plan) and CELL 8 (report). nb4s_lib.py, nb4_timeline.py (agent C) and
nb4_ppack.py (agent B) are embedded verbatim with their sha256. Nothing outside the NB4S files is written.

Run: python build_e25_nb4s.py -> ../Q27-XHIGH-E25_NB4S_STALL.ipynb + nb4s_code_export.py + the GPU push folder
(kernel-metadata.json + the notebook; nothing is pushed).
"""
import ast
import hashlib
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB1_BUILDER = HERE / "build_e25_nb1.py"
NB1_NB = HERE.parent / "Q27-XHIGH-E25_NB1_CAPACITY_R2.ipynb"
NB1_NB_SHA = "62670fce0e5ee950e1d32027522c5b4feffa55c43b061e4c7394668142b2bd3a"       # NB1 R2, Kaggle-run OK (hosen42)
LIB = HERE / "nb4s_lib.py"
TIMELINE = HERE / "nb4_timeline.py"                                                    # agent C's module, embedded verbatim
PPACK = HERE / "nb4_ppack.py"                                                          # agent B's Q-pack / Q-defer patch module
OUT_NB = HERE.parent / "Q27-XHIGH-E25_NB4S_STALL.ipynb"
EXPORT = HERE / "nb4s_code_export.py"
PUSH = Path(r"D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-nb4s-stall__foysalemonshanto")
SLUG = "q27-xhigh-e25-nb4s-stall"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
MODELS = ([f"foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p{k}/Transformers/default/1" for k in range(1, 6)] +
          ["foysalemonshanto/qwen38-flash-next-fp8ple-overlay/PyTorch/albucino-fp8ple/1",
           "foysalemonshanto/e23-merlin-env/Other/v1/1"])
PUSHED = set()          # shas of NB4S versions that ran on Kaggle (never overwrite one; rebuilds then go to *_REBUILD)
if OUT_NB.exists() and hashlib.sha256(OUT_NB.read_bytes()).hexdigest() in PUSHED:
    OUT_NB = OUT_NB.with_name(OUT_NB.stem + "_REBUILD.ipynb")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def b64z(text):
    import base64
    import zlib
    return base64.b64encode(zlib.compress(text.encode("utf-8"), 9)).decode("ascii")


def once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: x{text.count(old)}"
    return text.replace(old, new)


def other_notebook_shas():
    """sha of every notebook in the Experiment Notebooks folder except the NB4S file (byte-identity check)."""
    return {p.name: sha(p.read_bytes()) for p in sorted(HERE.parent.glob("*.ipynb")) if p.name != OUT_NB.name}


SHAS_BEFORE = other_notebook_shas()

# ---------------------------------------------------------------- NB1 R2 cells, exactly as NB1 composed them
NB1_TEXT = NB1_BUILDER.read_text(encoding="utf-8").replace("\r\n", "\n")
_cut = '\nsubs = {"__LAB__": json.dumps(RC2["LAB"])'
assert NB1_TEXT.count(_cut) == 1
_prefix = NB1_TEXT[:NB1_TEXT.index(_cut)]
assert not [l for l in _prefix.splitlines() if l.startswith(("OUT_NB.write", "(HERE / \"nb1_code_export.py\").write"))], "NB1 prefix must not write"
N1 = {"__file__": str(NB1_BUILDER), "__name__": "nb1_builder_prefix"}
exec(compile(_prefix, str(NB1_BUILDER), "exec"), N1)       # definitions only (it runs RC2's prefix + its own checks)
NB1_SUBS = {"__LAB__": json.dumps(N1["RC2"]["LAB"]), "__PLE__": json.dumps(N1["RC2"]["PLE"]), "__PATCH__": json.dumps(N1["PATCH_TEXT"]),
            "__PPRE__": N1["P_PRE"], "__PPOST__": N1["P_POST"], "__RC2FP8__": json.dumps(N1["RC2_FP8"]),
            "__RC2ORACLE__": json.dumps(N1["RC2_ORACLE"]), "__NB1EXT__": json.dumps(N1["NB1_EXT"])}
NB1_KEYS = ("CELL1", "CELL2", "CELL3", "CELL4", "CELL5", "CELL6", "CELL7", "CELL8")
_nb1_cells = []
for i, k in enumerate(NB1_KEYS):
    c = N1[k]
    for a, b in NB1_SUBS.items():
        c = c.replace(a, b)
    _nb1_cells.append("if not RERUN:\n" + textwrap.indent(c, "    ") if i else c)
assert sha(NB1_NB.read_bytes()) == NB1_NB_SHA, "NB1 R2 notebook is not the Kaggle-run version"
_nb1 = json.loads(NB1_NB.read_text(encoding="utf-8"))
assert ["".join(c["source"]) for c in _nb1["cells"] if c["cell_type"] == "code"] == _nb1_cells, \
    "NB1 cells rebuilt from build_e25_nb1.py differ from the Kaggle-run NB1 R2 notebook"
NB1_CELL_SHA = {k: sha(N1[k].encode()) for k in NB1_KEYS}

# ---------------------------------------------------------------- embedded modules (verbatim, sha-checked at run time)
LIB_SRC = LIB.read_bytes().decode("utf-8")
TL_SRC = TIMELINE.read_bytes().decode("utf-8")
PP_SRC = PPACK.read_bytes().decode("utf-8")
PP_SHA = sha(PP_SRC.encode("utf-8"))
compile(PP_SRC, "nb4_ppack.py", "exec")
_pp_defs = {n.name for n in ast.parse(PP_SRC).body if isinstance(n, ast.FunctionDef)}
assert {"apply_ppack_patch", "revert_ppack_patch", "verify_ppack_patch", "server_env", "readback_label", "parse_log"} <= _pp_defs
assert "\r" not in LIB_SRC, "nb4s_lib.py must be LF"
LIB_SHA, TL_SHA = sha(LIB_SRC.encode("utf-8")), sha(TL_SRC.encode("utf-8"))
compile(LIB_SRC, "nb4s_lib.py", "exec")
compile(TL_SRC, "nb4_timeline.py", "exec")
_tl_defs = {n.name for n in ast.parse(TL_SRC).body if isinstance(n, ast.FunctionDef)}
assert {"analyze", "strip_private", "format_report", "load_traces"} <= _tl_defs, "nb4_timeline.py lacks analyze / strip_private / format_report"

# ---------------------------------------------------------------- the worker extension: NB1's + one read-only method
NB4S_EXT = N1["NB1_EXT"].rstrip("\n") + '''


    def nb4s_env(self, keys=""):
        """E25 NB4S (read-only): this worker's os.environ for the comma-separated `keys` (default CUDA_DEVICE_MAX_CONNECTIONS),
        all NCCL_* / CUDA_* variables, pid, rank and device. CUDA reads CUDA_DEVICE_MAX_CONNECTIONS at context creation, so
        the process env (inherited at spawn, before CUDA init) is what is in effect. Never raises."""
        import os as _os
        out = {"pid": _os.getpid()}
        try:
            ks = [k for k in str(keys or "CUDA_DEVICE_MAX_CONNECTIONS").split(",") if k]
            out["env"] = {k: _os.environ[k] for k in ks if k in _os.environ}
            out["nccl_cuda_env"] = {k: v for k, v in sorted(_os.environ.items()) if k.startswith(("NCCL_", "CUDA_"))}
        except Exception as e:
            out["error"] = repr(e)[:200]
        try:
            import torch.distributed as dist
            out["rank"] = dist.get_rank() if dist.is_initialized() else None
        except Exception as e:
            out["rank_error"] = repr(e)[:120]
        try:
            import torch
            out["device"] = torch.cuda.current_device() if torch.cuda.is_available() else None
        except Exception as e:
            out["device_error"] = repr(e)[:120]
        return out
'''
compile(NB4S_EXT, "<nb1_worker_ext (NB4S)>", "exec")
_ext_ns = {}
exec(NB4S_EXT, _ext_ns)
_probe = _ext_ns["NB1WorkerExt"]().nb4s_env("CUDA_DEVICE_MAX_CONNECTIONS")
assert "env" in _probe and "pid" in _probe, _probe

# ---------------------------------------------------------------- NB1 CELL 1-5 + marked NB4S edits
CELL1 = N1["CELL1"]
CELL1 = once(CELL1, "(RC2 verbatim + E25 NB1 marked edits)", "(RC2 verbatim + E25 NB1 / NB4S marked edits)", "c1 title")
CELL1 = once(CELL1, 'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 240 * 60)     # E25 NB1: 3 cold Merlin boots (~13-17 min each) + ~2.9 h of measurements',
             'BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 95 * 60)      # E25 NB4S: hard stop 95 min (B > S > QP > QD)', "c1 budget")
CELL1 = once(CELL1, 'OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb1"))   # E25 NB1', 'OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb4s"))   # E25 NB4S', "c1 out")
CELL1 = once(CELL1, 'R = {"what": "E25 NB1 FN-S1 Capacity: Flash-Next on Merlin",   # E25 NB1',
             'R = {"what": "E25 NB4S STALL: CUDA_DEVICE_MAX_CONNECTIONS=32 vs the stage-0 in-graph stall (Flash-Next on Merlin)",   # E25 NB4S', "c1 what")
CELL1 = once(CELL1, '        tmp = OUT / "nb1_results.json.tmp"                       # E25 NB1\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "nb1_results.json")                    # E25 NB1',
             '        tmp = OUT / "nb4s_results.json.tmp"                      # E25 NB4S\n        tmp.write_text(json.dumps(R, indent=1, default=str))\n        tmp.replace(OUT / "nb4s_results.json")                   # E25 NB4S', "c1 save")
CELL1 = once(CELL1, 'print("E25 NB1 capacity diagnostic: competition rerun -> fallback submission.json only")',
             'print("E25 NB4S stall diagnostic: competition rerun -> fallback submission.json only")', "c1 rerun")
CELL2 = once(N1["CELL2"], '"env_keys": {k: env[k] for k in sorted(env) if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES"))}}',
             '"env_keys": {k: env[k] for k in sorted(env) if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES", "CUDA_DEVICE_MAX_CONNECTIONS"))}}   # E25 NB4S',
             "c2 env keys")
CELL3, CELL4, CELL5 = N1["CELL3"], N1["CELL4"], N1["CELL5"]

# ---------------------------------------------------------------- NB1 CELL 6 minus the functions NB4S never calls (AST)
DROP = ("commit_mode", "commit_ab", "commit_counter_delta", "pool_blocks_from_usage", "reset_cache", "_hits", "alloc_sequence",
        "reset_succeeded", "fit_alloc", "kv_accounting", "n_fit", "admission_test", "residency_bound", "residency_capacity",
        "_f_valid", "resolve_F", "f_from_log", "residency_probes", "block_layout", "nb1_complete", "nb1_tf_req", "lp_compare",
        "fp8_self_check", "nb1_ladder", "_prefix_counts", "_reuse", "prefix_tests", "_merge", "_clip", "_inter",
        "microbatch_costs", "pp2_trace", "merge_routing", "pp2_profile")
_c6 = N1["CELL6"]
_t6 = ast.parse(_c6)
_lines = _c6.splitlines(True)
_cut_ranges = []
for _n in _t6.body:
    if isinstance(_n, (ast.FunctionDef, ast.ClassDef)) and _n.name in DROP:
        lo = (_n.decorator_list[0].lineno if _n.decorator_list else _n.lineno) - 1
        _cut_ranges.append((lo, _n.end_lineno))
assert len(_cut_ranges) == len(DROP), (len(_cut_ranges), len(DROP))
_keep = [ln for i, ln in enumerate(_lines) if not any(a <= i < b for a, b in _cut_ranges)]
CELL6 = "".join(_keep)
CELL6 = once(CELL6, "# ==================================================================== CELL 6 — E25 NB1: measurement tools (NEW code)",
             "# ==================================================================== CELL 6 — E25 NB1: measurement tools (NB1 R2 verbatim; E25 NB4S "
             "dropped the unused capacity / prefix / routing / PP2-trace functions: " + ", ".join(DROP) + ")", "c6 title")
ast.parse(CELL6)
CELL6_DROPPED = {"names": list(DROP), "nb1_cell6_sha256": NB1_CELL_SHA["CELL6"], "bytes_before": len(_c6), "bytes_after": len(CELL6)}

# ---------------------------------------------------------------- CELL 7: the arms, the boot, the analysis, the plan (NEW code)
CELL7 = r'''# ==================================================================== CELL 7 — E25 NB4S: the arms B / S / QP / QD, the plan (NEW code)
import importlib, base64, zlib
NB4S_LIB_SRC = __LIBSRC__
NB4S_LIB_SHA = "__LIBSHA__"
TL_SHA = "__TLSHA__"                        # agent C's nb4_timeline.py and agent B's nb4_ppack.py: verbatim, zlib + base64
TL_SRC = zlib.decompress(base64.b64decode("__TLB64__")).decode("utf-8")      # (the notebook stays < 400 kB); sha-checked below
PPACK_SHA = "__PPSHA__"
PPACK_SRC = zlib.decompress(base64.b64decode("__PPB64__")).decode("utf-8")
for _src, _sha, _n in ((NB4S_LIB_SRC, NB4S_LIB_SHA, "nb4s_lib"), (TL_SRC, TL_SHA, "nb4_timeline"), (PPACK_SRC, PPACK_SHA, "nb4_ppack")):
    if hashlib.sha256(_src.encode("utf-8")).hexdigest() != _sha:
        raise RuntimeError(f"EMBED_SHA_MISMATCH: {_n}")
    (TMP / f"{_n}.py").write_text(_src, encoding="utf-8")
if str(TMP) not in sys.path:
    sys.path.insert(0, str(TMP))
N = importlib.import_module("nb4s_lib")
TL = importlib.import_module("nb4_timeline")
P = importlib.import_module("nb4_ppack")
NB4S = R.setdefault("nb4s", {})
NB4S.update(question="which change removes the ~21 ms stage-0 in-graph stall and speeds up decode: CUDA_DEVICE_MAX_CONNECTIONS=32 "
                     "(S), Q-pack (QP) or Q-defer (QD)?",
            lib=N.LIB_VERSION, lib_sha256=NB4S_LIB_SHA, timeline_sha256=TL_SHA, ppack_sha256=PPACK_SHA,
            ppack_file_shas={"pp_pre": P.PP_PRE, "pp_post": P.PP_POST, "mr_pre": P.MR_PRE, "mr_post": P.MR_POST,
                             "gw_pre": P.GW_PRE, "gw_post": P.GW_POST},
            cell6_dropped=json.loads(r"""__C6DROP__"""), nb1_r2_notebook_sha256="__NB1SHA__")
B3_PIN = KV_FP8 + (1 << 30)                 # NB1 R2 B3: 3,553,580,524 B (FP8 KV, 214 blocks), the tightest worker kept 191 MiB
if B3_PIN != 3553580524:
    raise RuntimeError(f"B3_PIN {B3_PIN} != 3553580524")
B_EXTRA = EXT_FLAG + ["--language-model-only", "--profiler-config", PROFILER_CFG]   # = NB1 R2 B3 (FP8_EXTRA)
B_ENV = dict(DIAG_ENV)                                                           # = NB1 R2 B3 (no NCCL debug)
ARMS = N.arm_plan(B_EXTRA, B_ENV, B3_PIN, q_env={"QP": P.server_env("Q-pack"), "QD": P.server_env("Q-defer")})
N.check_arm_plan(ARMS)                      # never-do list + "each arm = B + exactly one env key" (raises: no boot)
NB4S["arms"] = {k: {x: a[x] for x in ("name", "port", "env", "change", "expect", "extra", "kv_bytes", "needs_patch", "est_min",
                                      "cap_min", "title")} for k, a in ARMS.items()}
NB4S["order"] = list(N.ORDER)
NB4S["windows_order"] = list(N.WINDOWS)
NB4S["oom_policy"] = ("one policy for every arm: a launch OOM (OOM line anywhere in the launch's full log, no device fault) -> one "
                      "retry 0.25 GiB lower on a fresh port and launch name; every later arm starts at B's final pin; a pin mismatch "
                      "vs B makes that arm INCONCLUSIVE_PIN_MISMATCH; an OOM during the body never retries (BODY_OOM_<item>)")
NB4S["never_do"] = {"env": sorted(N.NEVER_DO_ENV), "argv": list(N.NEVER_DO_ARGS), "argv_values": N.NEVER_DO_ARG_VALUES}
NB4S["budget"] = {"hard_stop_min": BUDGET / 60, "start_reserve_sec": N.START_RESERVE_SEC, "body_reserve_sec": N.BODY_RESERVE_SEC,
                  "priority": "B > S > QP > QD (an arm starts only with est_min x 60 + 300 s left)"}
ARM_OF = {}                                 # launch name (retries included) -> arm
PIN = {"kv_bytes": B3_PIN, "source": "NB1 R2 B3"}
NB4S_READY_SEC = 18 * 60                    # readiness gate per boot (NB1 R2 B3 was ready in 297 s)
CUR = {"item": None}
_HIST = {}


def _tree_py():
    eng = merlin_install()
    return Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"


def _ppack_apply():
    tree, py = _tree_py()
    return P.apply_ppack_patch(tree, py)


def _ppack_revert():
    tree, py = _tree_py()
    rec = P.revert_ppack_patch(tree, py)
    v = P.verify_ppack_patch(tree)              # the pre-shas, verified again after the revert
    if v["state"] != "not_applied":
        raise RuntimeError("REVERT_NOT_PRISTINE: " + json.dumps(v)[:400])
    rec["verify_after"] = v
    return rec


PSEQ = N.PatchSeq(_ppack_apply, _ppack_revert)


def _prom_nb4s(port):
    """e23_lab.prom (same dict, one HTTP read) + the iteration-token histogram buckets with their read time, so a steady
    window's micro-batch split comes from the exact snapshots steady() took at t0 / t1."""
    txt = L.U.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15).read().decode("utf-8", "replace")
    agg, buckets = N.prom_parse(txt)
    h = _HIST.setdefault(port, [])
    h.append((time.perf_counter(), buckets))
    del h[:-600]
    return agg


L.prom = _prom_nb4s


def _hist_at(port, t):
    """The latest histogram snapshot read at or before perf_counter time t (steady() reads m0 then sets t0)."""
    best = None
    for ts, b in _HIST.get(port, []):
        if ts <= t + 0.01:
            best = (ts, b)
    return best if best and t - best[0] < 1.0 else None


_merlin_fn_server_nb1 = merlin_fn_server       # RC2 CELL 2 + NB1's per-boot hooks (env re-exported after build())


def merlin_fn_server(name, kv, port, kv_bytes):
    """E25 NB4S: NB1's Merlin plan, then the never-do gate + every watched key (CUDA_DEVICE_MAX_CONNECTIONS, NB4_PPACK,
    NB4_PPDEFER) at the arm's value on the FULL final env / argv of THIS launch (fail closed: no server is started)."""
    s = _merlin_fn_server_nb1(name, kv, port, kv_bytes)
    arm = ARM_OF.get(name)
    popped = {}
    for w, want in ((arm or {}).get("expect") or {}).items():
        if want is None and w in s.env:              # inherited from the image env, not planned: removed (recorded)
            popped[w] = s.env.pop(w)
    bad = N.launch_check(arm, s.env, s.argv) if arm else [f"launch {name} has no arm"]
    NB4S.setdefault("launch", {})[name] = {"arm": (arm or {}).get("key"), "port": port, "kv_bytes": kv_bytes, "never_do": bad,
                                           "env": {k: s.env[k] for k in sorted(s.env) if k.startswith(("CUDA_", "NCCL_", "NB4_", "VLLM_SERVER_DEV",
                                                                                                        "VLLM_USE_V2", "P2P"))},
                                           "argv_tail": s.argv[-12:], "help": HELP.get(name), "popped_unplanned_env": popped}
    save()
    if bad:
        raise RuntimeError("NEVER_DO_AT_LAUNCH: " + "; ".join(bad))
    return s


def free_min(port):
    """Free MiB per card = min(nvidia-smi, the worker's own torch free memory of the same rank); both recorded."""
    smi = L.gpu_mem() or []
    w = rpc_cache_info(port)
    wf = w.get("free_mib_per_rank") if isinstance(w, dict) else None
    sf = [t_ - u for _, u, t_ in smi] if smi else None
    mins = None
    if sf and wf and len(sf) == len(wf):
        mins = [min(a, b) if isinstance(b, (int, float)) else a for a, b in zip(sf, wf)]
    return {"nvidia_smi": sf, "worker": wf, "min": mins or sf or wf}


def proc_environ():
    """Side record: the watched keys in /proc/<pid>/environ of every engine process (server, workers, PLE)."""
    blobs = {}
    for pid, _ in (engine_procs() or [])[:24]:
        try:
            blobs[pid] = Path(f"/proc/{pid}/environ").read_bytes()
        except Exception:
            blobs[pid] = None
    return blobs


def _live(o, key, val):
    """Every item lands in the record and in the live step record before save() (a hard kill keeps it)."""
    o[key] = val
    rec = LIVE.get("rec")
    if rec is not None:
        rec.setdefault("live", {})[key] = val
    save()
    return val


def q_readback(s, arm):
    """Q arms: the 4 'NB4_PPQ init' lines (every rank must agree: a mismatch hangs NCCL) + moved counters, via
    nb4_ppack.readback_label on the launch's whole log. Other arms: None."""
    if not arm["q_label"]:
        return None
    try:
        lab, det = P.readback_label(s.log() or "", arm["q_label"])
    except Exception as e:
        lab, det = "NOT_INTEGRATED", {"error": f"{type(e).__name__}: {e}"[:300]}
    return {"label": lab, "detail": det}


def hist_window(port, s, band, n):
    """NB1's window() (RC2 levels / lab steady; running == level, prefill done, per-position acceptance, tick / cycle) + the
    window's micro-batch split from the iteration-token histogram at steady()'s own t0 / t1."""
    r = window(port, s, band, n)
    h0, h1 = (_hist_at(port, r["t0"]) if r.get("t0") else None), (_hist_at(port, r["t1"]) if r.get("t1") else None)
    d = N.hist_delta(h0[1], h1[1]) if h0 and h1 else None
    r["parity"] = N.parity_from_hist(d, n, PP, 3)
    return r


def profile_window(port, arm):
    """One profiled 8k@8 window: lab profiled_load (settled decode, then the bounded POST /start_profile
    {"delay_iterations": 20, "max_iterations": 40}, 6 s, /stop_profile); the traces are kept per arm."""
    shutil.rmtree(PROF_DIR, ignore_errors=True)
    PROF_DIR.mkdir(parents=True, exist_ok=True)
    rec = L.profiled_load(port, PROMPTS["8k"], 8, PROF_DIR, secs=6.0, max_new=MAX_NEW["8k"], spec=True, prefill_tok_s=PREFILL["tok_s"])
    rec["profile_request"] = {"delay_iterations": 20, "max_iterations": 40, "secs": 6.0, "level": 8, "band": "8k"}
    if "trace" in rec:
        rec["lab_trace_summary"] = rec.pop("trace")
        rec["lab_trace_note"] = "e23_lab rule: its PIECEWISE verdict is a parser artefact here (graphs research §0.1); nb4_timeline decides"
    g = L.gpu_mem()
    if g:
        rec["gpu_mem_peak_mib"] = [x[1] for x in g]
    kept = TRACES / f"profile_{arm['key'].lower()}"
    shutil.rmtree(kept, ignore_errors=True)
    kept.mkdir(parents=True, exist_ok=True)
    rec["trace_files"] = []
    for f in sorted(PROF_DIR.rglob("*.pt.trace.json*")):
        if f.stat().st_size < 120e6:
            shutil.copy2(f, kept / f.name)
            rec["trace_files"].append([f.name, f.stat().st_size])
    rec["trace_dir"] = str(kept)
    shutil.rmtree(PROF_DIR, ignore_errors=True)
    return rec


WIN_NEED = {"8k@8#1": lambda: WIN + 120 + 8 * BANDS["8k"] / max(500.0, PREFILL["tok_s"]),
            "8k@8#2": lambda: WIN + 120 + 8 * BANDS["8k"] / max(500.0, PREFILL["tok_s"]),
            "profile_8k@8": lambda: 360,
            "40k@8": lambda: WIN + 120 + 8 * BANDS["40k"] / max(500.0, PREFILL["tok_s"])}
RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P"


def arm_body(arm, ref, store):
    """Every arm, the same items in the same order; every item gated by due() on the boot's own deadline."""
    def body(ss):
        s = ss[0]
        p = s.port
        o = {"arm": arm["key"], "items_skipped": []}
        CUR["item"] = "facts"
        o = boot_facts(s, dict(o, census=kv_census(s), probes=L.chat_probes(p), sampled=sampled_check(s)), s.name)
        LIVE["out"] = o
        CUR["item"] = "env_readback"
        _live(o, "env_readback_raw", rpc(p, "nb4s_env", RPC_KEYS))
        _live(o, "env_readback", N.parse_env_readback(o["env_readback_raw"], arm["expect"]))
        _live(o, "proc_environ", N.parse_proc_environ(proc_environ(), arm["expect"]))
        _live(o, "q_readback", q_readback(s, arm))
        _live(o, "free_after_ready_min", free_min(p))
        if short_gate_failed(o):
            o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s),
                     quality_gate="FAIL_SAMPLED" if (o["sampled"] or {}).get("ok") is not True else "FAIL_PROBES")
            return o
        if o["env_readback"].get("definitive"):
            o.update(status="NOT_ENGAGED", early_stop="the workers do not hold the planned env: " + str(o["env_readback"].get("why")))
            return o                          # definitive: the arm's change is not in effect, nothing to measure
        if (o["q_readback"] or {}).get("label") == "NOT_INTEGRATED":
            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="nb4_ppack init lines: " + json.dumps(o["q_readback"])[:300])
            return o
        if due(300):                          # every arm (the 40k probe prompt is variant 7 of the 40k@8 window: same cache state)
            CUR["item"] = "prefill_probe"
            _live(o, "prefill", prefill_probe(p))
        if not due(240):
            o["items_skipped"].append("oracle")
            o["status"] = "SKIPPED_BOOT_BUDGET_ORACLE"
            return o
        CUR["item"] = "oracle"
        _live(o, "oracle", oracle(p, key=store, refs=(ref,)))
        o["oracle_vs"] = (o["oracle"].get("vs_" + ref) or {}).get("status")
        o["quality_gate"] = quality_gate(o, ref)
        if o["quality_gate"] != "PASS":
            o["load_tests_skipped"] = f"oracle gate vs {ref}: {o['quality_gate']}"
            o["status"] = "NO_GO_QUALITY" if o["quality_gate"].startswith("FAIL") else "INCONCLUSIVE_QUALITY"
            o["census_end"] = kv_census(s)
            return o
        if arm["q_label"]:
            _live(o, "q_readback", q_readback(s, arm))         # the counters moved by now (probes + sampled + oracle decode)
            # ENGAGED_FALLBACK is not final here: the 1-stream items so far (probes, sampled) can run empty steps whose
            # 'empty' flushes lower the exec share; the arm goes on and the verdict reads the end / final read-back.
            if (o["q_readback"] or {}).get("label") not in ("ENGAGED", "ENGAGED_FALLBACK"):
                o.update(status=o["q_readback"]["label"], not_integrated=o["q_readback"]["label"] == "NOT_INTEGRATED",
                         early_stop="nb4_ppack read-back: " + json.dumps(o["q_readback"])[:300])
                return o
        if due(150):
            CUR["item"] = "warmup"
            _live(o, "warmup", warmup(p, 8))
        else:
            o["items_skipped"].append("warmup")
        o["steady"] = {}                      # RC2's _windows() gate reads "steady": garbage under load is NO_GO here too
        for wk in N.WINDOWS:
            need = WIN_NEED[wk]()
            if not due(need):
                o["steady"][wk] = {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need), "boot_left_sec": round(boot_left())}
                o["items_skipped"].append(wk)
                continue
            if not s.alive():
                o["steady"][wk] = {"status": "SKIPPED_SERVER_DEAD"}
                continue
            CUR["item"] = wk
            try:
                r = profile_window(p, arm) if wk.startswith("profile") else hist_window(p, s, wk.split("@")[0], 8)
            except Exception as e:
                r = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1200:]}
            o["steady"][wk] = r
            _live(o, "steady", o["steady"])
        CUR["item"] = "end"
        o["census_end"] = kv_census(s)
        o["workers_end"] = rpc_cache_info(p)
        o["free_end_min"] = free_min(p)
        o["q_readback_end"] = q_readback(s, arm)
        o["ppq_stats_last"] = [e for e in P.parse_log(s.log() or "") if e.get("event") == "stats"][-8:] if arm["q_label"] else None
        o["jit_after_warmup"] = jit_kernels(s.log())[-12:]
        return o
    return body


def arm_watched(body, cap_min, name):
    """NB1's watched() with the cap fixed at body START (after the launch): min(cap, left - BODY_RESERVE_SEC); every item
    checks due(); at cap + 300 s the server is killed (hard backstop)."""
    def b(ss):
        cap = N.body_cap_sec(cap_min, left())
        BOOT.update(deadline=time.time() + cap, fired=None)

        def fire():
            BOOT["fired"] = name
            for x in ss:
                try:
                    x.kill()
                except Exception:
                    pass
        tm = threading.Timer(cap + 300, fire)
        tm.daemon = True
        tm.start()
        try:
            o = body(ss)
            LIVE["out"] = o if isinstance(o, dict) else None
        finally:
            tm.cancel()
            BOOT["deadline"] = None
        o["body_cap_sec"] = round(cap)
        if BOOT["fired"]:
            o["boot_watchdog"] = f"FIRED after {cap + 300:.0f} s: server killed"
        return o
    return b


def arm_boot(arm, body):
    """One arm: the boot gate, one Merlin launch, the /dev/shm cleanup; the ONE OOM policy (launch OOM in the launch's
    full log, no device fault -> one retry 0.25 GiB lower on a fresh port and launch name); death labels by item."""
    kvb, port, tries, o = PIN["kv_bytes"], arm["port"], [], {}
    ref = "rc2_fp8" if arm["key"] == "B" else "b"
    while True:
        lname = arm["name"] if not tries else f"{arm['name']}_t{len(tries)}"
        NB1_EXTRA[lname], NB1_ENV[lname], NB1_KV[lname] = list(arm["extra"]), dict(arm["env"]), kvb
        ARM_OF[lname] = arm
        BOOT["tag"] = lname
        gate = boot_gate()
        if gate["ok"] is not True:
            return {"status": "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries, "launch_name": lname, "pin": kvb}
        shm0 = shm_names()
        mon = LoadMonitor(lname)
        mon.start()
        srv, CUR["item"] = None, "launch"
        try:
            srv = merlin_fn_server(lname, "fp8", port, kvb)
            o = with_servers([srv], NB4S_READY_SEC, arm_watched(body, arm["cap_min"], lname))
        finally:
            mon.stop_ev.set()
        try:
            full = srv.log() or ""
        except Exception:
            full = ""
        lines = full.splitlines()
        o.update(kv_bytes=kvb, pin=kvb, launch_name=lname, port=port, boot_gate=gate, shm_cleanup=shm_cleanup(shm0),
                 ready_sec=getattr(srv, "ready_sec", None), load_monitor_last=mon.last, oom_in_log=bool(OOM_ERR.search(full)),
                 xid_in_log=bool(XID.search(full)), oom_lines=[l[:240] for l in lines if OOM_ERR.search(l)][:6],
                 xid_lines=[l[:240] for l in lines if XID.search(l)][:6], died_in=CUR["item"])
        if arm["q_label"] and o.get("q_readback") is None:
            o["q_readback"] = q_readback(srv, arm)        # a launch that failed: the init lines (if any) are still in the log
        if arm["q_label"] and srv is not None:
            o["q_readback_final"] = q_readback(srv, arm)  # the whole log after the server stopped (F1's final stats line)
        oom = o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]
        tries.append({"launch_name": lname, "port": port, "kv_bytes": kvb, "status": o.get("status"), "oom": oom,
                      "device_fault": o["xid_in_log"], "ready_sec": o["ready_sec"]})
        if not oom or o["xid_in_log"] or len(tries) > 1 or left() < NB4S_READY_SEC + arm["cap_min"] * 60:
            break
        kvb -= OOM_STEP
        port += 1
        log("boot OOM:", lname, "-> one retry with KV_BYTES", kvb, "port", port)
    o["tries"] = tries
    if o.get("status") == "SERVER_DIED":
        o["death"] = ("BOOT_CAP_KILL" if o.get("boot_watchdog") else
                      f"BODY_OOM_{o.get('died_in')}" if o["oom_in_log"] and not o["xid_in_log"] else
                      f"DEVICE_FAULT_{o.get('died_in')}" if o["xid_in_log"] else f"SERVER_DIED_IN_{o.get('died_in')}")
        o["status"] = o["death"]
    return gated(o, need=(ref,), expected_rows=4)


def analyze_arm(o, key):
    """After the boot (GPUs idle): nb4s_lib.graph_stalls + nb4_timeline.analyze on the arm's traces; full JSON files kept."""
    d = ((o.get("steady") or {}).get("profile_8k@8") or {}).get("trace_dir")
    if not d or not list(Path(d).glob("*.pt.trace.json*")):
        o["stall"] = {"error": "NO_TRACE_FILE"}
        o["timeline"] = {"error": "NO_TRACE_FILE"}
        return o
    t = time.time()
    try:
        st = N.graph_stalls(d)
        (OUT / f"stall_{key.lower()}.json").write_text(json.dumps(st, indent=1, default=str))
        o["stall"] = {k: v for k, v in st.items() if k != "by_rank"}
        o["stall"]["by_rank"] = {r: {x: y for x, y in v.items() if x != "rows"} for r, v in (st.get("by_rank") or {}).items()}
    except Exception as e:
        o["stall"] = {"error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1200:]}
    try:
        rep = TL.strip_private(TL.analyze(d))
        (OUT / f"timeline_{key.lower()}.json").write_text(json.dumps(rep, indent=1, default=str))
        o["timeline"] = N.summarize_timeline(rep)
        try:
            (OUT / f"timeline_{key.lower()}.md").write_text(TL.format_report(rep))
        except Exception as e:
            o["timeline"]["format_report_error"] = repr(e)[:200]
    except Exception as e:
        o["timeline"] = {"error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1200:]}
    o["analysis_sec"] = round(time.time() - t, 1)
    return o


def run_arm(key):
    arm = ARMS[key]
    o = arm_boot(arm, arm_body(arm, "rc2_fp8" if key == "B" else "b", "b" if key == "B" else None))
    LIVE["out"] = o
    try:
        o["arm"], o["planned"] = key, NB4S["arms"][key]
        o["free_mib_per_card"] = free_per_card(o, o.get("launch_name"))
        if left() > 120:
            analyze_arm(o, key)
    except Exception as e:
        keep_after_error(o, e)
    return o


def _arm_rec(key):
    r = (R["sections"].get(key) or {}).get(ARMS[key]["name"])
    return r if isinstance(r, dict) else None


def arm_clean(o):
    return isinstance(o, dict) and o.get("status") == "DONE" and o.get("correctness") == "CLEAN" and o.get("quality_gate") == "PASS"


def skip_arm(key, why, status="SKIPPED_DEPENDENCY", **kw):
    R["sections"].setdefault(key, {})[ARMS[key]["name"]] = dict({"status": status, "why": why}, **kw)
    save()


def finalize():
    """Always (finally): analysis of any arm that has traces but no analysis yet, then the verdict + nb4s_verdict.json."""
    try:
        recs = {}
        for k in ARMS:
            r = _arm_rec(k)
            if r is not None and "stall" not in r and ((r.get("steady") or {}).get("profile_8k@8") or {}).get("trace_dir"):
                try:
                    analyze_arm(r, k)
                except Exception as e:
                    r["stall"] = {"error": repr(e)[:300]}
            recs[k] = r
        v = N.verdict({k: r for k, r in recs.items() if r is not None})
    except Exception as e:
        v = {"verdict": "ERROR_IN_VERDICT", "why": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1500:]}
    v["arms_status"] = {k: (_arm_rec(k) or {}).get("status") for k in ARMS}
    v["patch"] = NB4S.get("patch")
    v["minutes"] = round((time.time() - T_START) / 60, 1)
    NB4S["verdict"] = v
    try:
        (OUT / "nb4s_verdict.json").write_text(json.dumps(v, indent=1, default=str))
    except Exception:
        pass
    save()
    log("NB4S VERDICT", v.get("verdict"), "|", str(v.get("why"))[:400])
    return v


def nb4s_main(runner=None, pseq=None):
    """install -> #60490 + extension -> sha gate -> B -> S -> QP -> QD (priority order; each needs B clean and its full
    minimum of time). The nb4_ppack patch is applied right before the first Q arm that starts, kept for QD, reverted at
    the end (pre-shas verified); the verdict is written in the finally block whatever happens."""
    runner = runner or run_arm
    pseq = pseq or PSEQ
    try:
        m = R["sections"]["A0"]["merge"]
        if not m.get("ok"):
            NB4S["stop"] = "MERGE_FAILED: no boot started"
            for k in ARMS:
                skip_arm(k, NB4S["stop"])
            return
        step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
        pt = step("A0", "patch_60490", 3, apply_60490)
        if pt.get("status") != "DONE" or not pt.get("ok"):
            for k in ARMS:
                skip_arm(k, f"#60490 patch / worker extension not verified: {pt.get('error') or pt.get('status')}")
            return
        if "rc2_fp8" not in ROWS and not FAKE:
            for k in ARMS:
                skip_arm(k, "RC2's recorded FP8 oracle rows do not match this notebook's prompts (no gate reference)")
            return
        ok, why = sha_gate(min(1500, left() - 900))
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        if not ok:
            for k in ARMS:
                skip_arm(k, "sha gate: " + why, "SKIPPED_SHA_GATE")
            return
        b = step("B", ARMS["B"]["name"], ARMS["B"]["est_min"], lambda: runner("B"))
        if not arm_clean(b):
            for k in N.ORDER[1:]:
                skip_arm(k, f"B not clean (status {b.get('status')}, correctness {b.get('correctness')}, gate {b.get('quality_gate')})")
            return
        PIN.update(kv_bytes=b.get("pin") or PIN["kv_bytes"], source="B's final pin")
        for k in N.ORDER[1:]:
            a = ARMS[k]
            if not N.arm_fits(left(), a):
                skip_arm(k, f"left {left() / 60:.1f} min < its full minimum {a['est_min']} + {N.START_RESERVE_SEC // 60} min "
                            "(priority B > S > QP > QD)", "SKIPPED_BUDGET")
                continue
            ok_p, why_p = pseq.before_arm(a)
            NB4S["patch"] = {"state": pseq.state, "log": pseq.log, "error": pseq.error, "apply": pseq.apply_rec, "revert": pseq.revert_rec}
            if not ok_p:
                skip_arm(k, why_p, "NOT_INTEGRATED" if a["needs_patch"] else "SKIPPED_DEPENDENCY", not_integrated=bool(a["needs_patch"]))
                continue
            step(k, a["name"], a["est_min"], lambda k=k: runner(k))
    finally:
        try:
            pseq.finish()
        except Exception as e:
            pseq.error = f"finish: {type(e).__name__}: {e}"[:300]
        NB4S["patch"] = {"state": pseq.state, "log": pseq.log, "error": pseq.error, "apply": pseq.apply_rec, "revert": pseq.revert_rec}
        finalize()


def _gpu_util():
    while not _gpu_util_stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
            if r.returncode == 0:
                with open(OUT / "nb4s_gpu_util.jsonl", "a") as fh:
                    fh.write(json.dumps({"t": round(time.time() - T_START, 1), "gpus": r.stdout.strip().splitlines()}) + "\n")
        except Exception:
            pass
        _gpu_util_stop.wait(5)


_gpu_util_stop = threading.Event()
if not RERUN and not os.environ.get("RC2_UNIT"):
    if NVSMI_OK:
        threading.Thread(target=_gpu_util, daemon=True).start()
    nb4s_main()
    _gpu_util_stop.set()
    R["host_phase"] = HOST_PHASE
    R["prefill"] = PREFILL
    R["minutes"] = round((time.time() - T_START) / 60, 1)
    save()
save()
'''

# ---------------------------------------------------------------- CELL 8: report + fallback submission
CELL8 = r'''# ==================================================================== CELL 8 — report + fallback submission
if not RERUN and not os.environ.get("RC2_UNIT"):
    write_fallback_submission()
    def _arm_brief(k):
        r = _arm_rec(k) or {}
        w = r.get("steady") or {}
        return {"status": r.get("status"), "correctness": r.get("correctness"), "quality_gate": r.get("quality_gate"),
                "oracle_vs": r.get("oracle_vs"), "pin": r.get("pin"), "tries": r.get("tries"), "ready_sec": r.get("ready_sec"),
                "env_readback": (r.get("env_readback") or {}).get("why"), "proc_environ": (r.get("proc_environ") or {}).get("values"),
                "q_readback": r.get("q_readback"), "q_readback_end": r.get("q_readback_end"),
                "q_readback_final": r.get("q_readback_final"), "free_after_ready_min": (r.get("free_after_ready_min") or {}).get("min"),
                "free_end_min": (r.get("free_end_min") or {}).get("min"),
                "windows": {k2: dict({x: v.get(x) for x in ("status", "tok_s", "tick_ms", "cycle_ms", "mean_acceptance_length",
                                                           "accept_rate_per_position_incl_admission", "running_min", "running_max",
                                                           "preemptions", "running_eq_level")}, parity=(v.get("parity") or {}).get("label"))
                            for k2, v in w.items() if isinstance(v, dict)},
                "stall_stage0": (r.get("stall") or {}).get("stage0"), "stall_stage1": (r.get("stall") or {}).get("stage1"),
                "profile_parity": (r.get("stall") or {}).get("parity"),
                "timeline": {x: (r.get("timeline") or {}).get(x) for x in ("tick_ms_mean", "stages", "tick_stall_ms_median",
                                                                          "tick_share_stall_below_1ms", "graph_modes_steady",
                                                                          "parity_verdict", "dependency_verdict", "per_rank", "error")},
                "why": r.get("why"), "error": r.get("error")}
    R["summary"] = {k: _arm_brief(k) for k in ARMS}
    save()
    _v = NB4S.get("verdict") or {}
    print("BEST ARM (throughput):", _v.get("best_arm"), _v.get("best_verdict"), "| stall-best:", _v.get("stall_best_arm"), _v.get("stall_best_verdict"), "|", _v.get("why"))
    print(json.dumps(_v, default=str)[:5000])
    print(json.dumps(R["summary"], default=str)[:12000])
'''

MD0 = """# E25 NB4S STALL: which change removes the stage-0 in-graph stall?

NB1 R2 B1 (torch profiler, 8k@8): the stage-0 FULL graph (1,036 kernels) stops at kernel 172 (an in-graph TP AllReduce on a
forked stream) for a median ~21.7 ms and resumes ~6 µs after the PP sampled-token `Broadcast Int x8` of the OTHER
micro-batch ends: a false dependency.

| Arm (priority order) | Config = B + exactly one change | Measures |
|---|---|---|
| B | NB1 R2 B3 (FP8 KV, pin 3,553,580,524 B, text-only, #60490, MTP K3, seqs 8) + profiler | env read-back, probes, sampled, oracle vs RC2 FP8, 8k@8 ×2, profiled 8k@8, 40k@8 |
| S | `CUDA_DEVICE_MAX_CONNECTIONS=32` | the same; oracle vs B (EXACT / NEAR_TIE) |
| QP | Q-pack: `nb4_ppack` patch rev 2 + `NB4_PPACK=1` (one packed broadcast per step) | the same + the 4 `NB4_PPQ init` lines (ENGAGED) |
| QD | Q-defer: the same patch + `NB4_PPDEFER=1` (receive posted after the next forward) | the same + stage-1 send-side waits + flush ratios (ENGAGED vs ENGAGED_FALLBACK) |

Per-arm verdicts: `STALL_FIXED_WIN` (stall < 1 ms in ≥ 95 % of stage-0 graphs + 8k@8 gain > max(B spread, 2 %) + oracle OK),
`STALL_FIXED_NO_SPEED`, `STALL_MOVED`, `STALL_REDUCED`, `NO_EFFECT`, `CORRECTNESS_FAIL`, `ENGAGED_FALLBACK` (Q-defer collapsed to
post-at-consume; never counted), `NOT_ENGAGED`, `NOT_INTEGRATED`,
`SKIPPED_BUDGET`, `INCONCLUSIVE_*`; the summary names the best arm. Hard stop 95 min (priority B > S > QP > QD); the patch
is reverted at the end (pre-shas verified). Commit run only; a competition rerun writes the fallback submission.
"""

subs = {**NB1_SUBS, "__NB1EXT__": json.dumps(NB4S_EXT), "__LIBSRC__": json.dumps(LIB_SRC), "__LIBSHA__": LIB_SHA,
        "__TLB64__": b64z(TL_SRC), "__TLSHA__": TL_SHA,
        "__PPB64__": b64z(PP_SRC), "__PPSHA__": PP_SHA, "__C6DROP__": json.dumps(CELL6_DROPPED), "__NB1SHA__": NB1_NB_SHA}
cells_src = []
for i, c in enumerate((CELL1, CELL2, CELL3, CELL4, CELL5, CELL6, CELL7, CELL8)):
    for k, v in subs.items():
        c = c.replace(k, v)
    if i:
        c = "if not RERUN:\n" + textwrap.indent(c, "    ")
    compile(c, "<nb4s cell>", "exec")
    cells_src.append(c)

# every dropped CELL 6 name is unreferenced in the whole notebook (no NameError at run time)
_loaded = set()
for c in cells_src:
    for node in ast.walk(ast.parse(c)):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            _loaded.add(node.id)
_used = sorted(set(DROP) & _loaded)
assert not _used, f"dropped CELL 6 functions still referenced: {_used}"
# the arm plan is valid at build time too (never-do list, one-key diff)
_libns = {"__name__": "nb4s_lib_build"}
exec(compile(LIB_SRC, "nb4s_lib.py", "exec"), _libns)
_ppns = {"__name__": "nb4_ppack_build"}
exec(compile(PP_SRC, "nb4_ppack.py", "exec"), _ppns)
_arms = _libns["arm_plan"](["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt", "--language-model-only", "--profiler-config", "{}"],
                           {"VLLM_SERVER_DEV_MODE": "1"}, 3553580524,
                           q_env={"QP": _ppns["server_env"]("Q-pack"), "QD": _ppns["server_env"]("Q-defer")})
_libns["check_arm_plan"](_arms)
assert _arms["S"]["env"]["CUDA_DEVICE_MAX_CONNECTIONS"] == "32" and _arms["QP"]["env"].get("NB4_PPACK") == "1"
assert _arms["QD"]["env"].get("NB4_PPDEFER") == "1" and "CUDA_DEVICE_MAX_CONNECTIONS" not in _arms["QP"]["env"]

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD0.splitlines(True)}] +
               [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(True)} for c in cells_src],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
data = json.dumps(nb, indent=1).encode("utf-8")
assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
assert len(data) < 400_000, len(data)
OUT_NB.write_bytes(data)
EXPORT.write_bytes("\n\n".join(cells_src).encode("utf-8"))
PUSH.mkdir(parents=True, exist_ok=True)
(PUSH / f"{SLUG}.ipynb").write_bytes(data)
meta = {"id": f"foysalemonshanto/{SLUG}", "title": "Q27 Xhigh E25 NB4S Stall", "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": MODELS, "docker_image": IMAGE, "machine_shape": "NvidiaL4"}
(PUSH / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
SHAS_AFTER = other_notebook_shas()
_changed = sorted(k for k in set(SHAS_BEFORE) | set(SHAS_AFTER) if SHAS_BEFORE.get(k) != SHAS_AFTER.get(k))
facts = {"notebook": OUT_NB.name, "sha256": sha(data), "bytes": len(data), "lib_sha256": LIB_SHA, "timeline_sha256": TL_SHA,
         "ppack_sha256": PP_SHA, "timeline_bytes": len(TL_SRC), "cell6_dropped_bytes": CELL6_DROPPED["bytes_before"] - CELL6_DROPPED["bytes_after"],
         "nb1_r2_sha256": NB1_NB_SHA, "other_notebooks_checked": len(SHAS_BEFORE), "other_notebooks_changed": _changed,
         "cell_bytes": [len(c.encode()) for c in cells_src]}
print(json.dumps(facts, indent=1))
if _changed:
    raise SystemExit(f"OTHER NOTEBOOKS CHANGED DURING THE BUILD (not by this builder?): {_changed}")
