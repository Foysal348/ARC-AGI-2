"""E25 NB4T library (2026-11-01): the next speed levers on top of the Q-defer winner of NB4S (GPU run fa9f35ca, +21.1 % at
8k@8, +19.8 % at 40k@8, oracle EXACT; after Q-defer, stage-1 compute (33.0 ms) binds and stage 0 is 21.6 ms).

Arms (run in this order; each = Q + exactly ONE change):
  Q    the new base: NB4S's B config (NB1 R2 B3: FP8 KV, pin 3,553,580,524 B, --language-model-only, #60490, MTP K3, seqs 8,
       profiler) + the nb4_ppack rev 2 patch + NB4_PPDEFER=1 (Q-defer).
  I8   Q + DRAFT_HEAD=int8 (Merlin launcher knob, tools/merlin/launch.py:319-322: added to --speculative-config as
       "draft_lm_head_quant":"int8", launch.py:441; built by speculator.py:547-603 -> QuantizedDraftLMHead,
       draft_lm_head_quant.py:75-125, which logs "Draft lm_head: INT8 weight-only copy ..."). K stays 3.
  P26  Q + VLLM_PP_LAYER_PARTITION=26,22 (vllm/distributed/utils.py:142-170; Merlin sets 25,23 from the profile layout,
       launch.py:373-374; the arm value is exported after build(), so it wins).
  P27  Q + VLLM_PP_LAYER_PARTITION=27,21 (skipped when P26 OOMs).
  C12  Q + a trailing --max-num-seqs 12 (last wins), measured with 12 streams at 8k@12 / 40k@12 against Q's 8k@8 / 40k@8.
       Measure only: it is never the throughput winner and changes no worker rule.
  Q already carries --language-model-only (NB1 R2 B3 = NB4S B): P26 / P27 change the partition only.

Pure logic, stdlib only (embedded verbatim in the notebook; unit-tested locally and on the Py3.12 CPU kit):
  - the arm plan, the never-do gate (env + argv), the launch check on the final env / argv (watched env, the
    --speculative-config draft head, the last --max-num-seqs, --language-model-only, the pin);
  - the worker env read-back (POST /collective_rpc nb4s_env) and the server-log config read-back (non-default args,
    "Draft lm_head:" lines, "pipeline layers a-b" lines per stage), the capacity facts (pool blocks, max concurrency at
    131,072, per-stage block bytes, per-rank free memory, per-rank load GiB) and a KV demand estimate;
  - the nb4_ppack patch sequencing (applied before Q, kept for every arm, reverted at the end);
  - /metrics parsing + the iteration-token histogram split of a window (recorded, never gating);
  - the per-graph stall parser of the torch-profiler traces and the compact nb4_timeline summary (NB4S, verbatim logic);
  - PARITY-AWARE comparison (the NB4S fix): the parity of every arm is recorded; an arm's stall / timeline comparison with
    Q happens only when both profiled windows carry the same parity label or the same modal split, else PARITY_DIFFERS;
    the base is never rejected for parity; speed windows are valid without a parity condition (parity recorded);
  - the speed comparison, the budget rule, the per-arm verdicts and the throughput winner (8k speed PASS on both windows,
    a valid 40k window, no 40k regression; lever arms I8 / P26 / P27 only).
"""
import gzip
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

LIB_VERSION = "nb4t_lib 1.0 (2026-11-01, arms Q / I8 / P26 / P27 / C12 on the Q-defer base; parity-aware comparison)"
FLAG_CDMC = "CUDA_DEVICE_MAX_CONNECTIONS"
ENV_PACK, ENV_DEFER = "NB4_PPACK", "NB4_PPDEFER"           # = nb4_ppack.ENV_PACK / ENV_DEFER
ENV_PART = "VLLM_PP_LAYER_PARTITION"
KNOB_HEAD = "DRAFT_HEAD"                                    # Merlin launcher knob (input of build(), not a worker env)
MERLIN_PARTITION = "25,23"                                  # flash-next profile layout, set by launch.py:373-374
NUM_LAYERS = 48
WATCHED = (FLAG_CDMC, ENV_PACK, ENV_DEFER, ENV_PART)        # read back on every arm (expected value or absent)
WATCHED_DEFAULT = {ENV_PART: MERLIN_PARTITION}              # set by Merlin when the arm does not set it
ORDER = ("Q", "I8", "P26", "P27", "C12")                    # run order = priority when time is short
LEVER_ARMS = ("I8", "P26", "P27")                           # throughput-winner candidates
MEASURE_ONLY = ("C12",)
Q_LABEL = "Q-defer"                                         # nb4_ppack.server_env / readback_label arm name (every arm)
BASE_SEQS, C12_SEQS = 8, 12
STALL_LT_MS = 1.0
FIXED_SHARE = 0.95
BASE_STALL_MIN_MS = 5.0
SPEED_MIN = 0.02
FULL_MIN_NODES = 500
RELEASE_WINDOW_US = 200.0
START_RESERVE_SEC = 300                                    # an arm starts only with est_min x 60 + 300 s left
BODY_RESERVE_SEC = 300                                     # body cap clipped to left - 300 s (teardown + analysis + verdict)
HARD_STOP_MIN = 120
CLEAN = ("EXACT", "NEAR_TIE")
BLOCK_TOKENS = 4096
MAX_LEN = 131072

# ------------------------------------------------------------------ never-do list (NB4S, unchanged)
NEVER_DO_ENV = {
    "NCCL_GRAPH_MIXING_SUPPORT": lambda v: str(v).strip() == "0",          # unsafe: tp comm mixes eager prefill + graphs
    "NCCL_GRAPH_REGISTER": lambda v: str(v).strip() == "1",                # deadlock risk (SGLang keeps 0)
    "NCCL_PROTO": lambda v: "ll128" in str(v).lower(),                     # forced LL128: corruption risk
    "NCCL_P2P_DISABLE": lambda v: str(v).strip() == "0",                   # P2P on
    "NCCL_P2P_LEVEL": lambda v: True,                                      # P2P on
    "P2P": lambda v: str(v).strip().lower() not in ("off", "0", "false", ""),
    "NCCL_SHM_USE_CUDA_MEMCPY": lambda v: str(v).strip() == "1",           # slower; memcpy hangs in capture
    "NCCL_LAUNCH_ORDER_IMPLICIT": lambda v: True,                          # not a fix (graphs research); never added
    "NCCL_MIN_CTAS": lambda v: True, "NCCL_MAX_CTAS": lambda v: True,      # NCCL CTA tuning: measured negative
    "NCCL_MIN_NCHANNELS": lambda v: True, "NCCL_MAX_NCHANNELS": lambda v: True, "NCCL_NTHREADS": lambda v: True,
    "COUNTERS": lambda v: str(v).strip() == "1",                           # disables the fused QSA writer
}
NEVER_DO_ARGS = ("--no-async-scheduling", "--no-enable-prefix-caching", "--enable-hc-sp")
NEVER_DO_ARG_VALUES = {"--mamba-ssm-cache-dtype": ("bfloat16",)}         # C3 SSM bf16: parked (0 capacity + B2 FAIL)


def never_do_violations(env, argv=()):
    """Every never-do item set in `env` (a dict) or `argv` (a list; the last occurrence of a flag wins)."""
    bad = []
    for k, pred in NEVER_DO_ENV.items():
        if k in (env or {}) and env[k] is not None:
            try:
                hit = pred(env[k])
            except Exception:
                hit = True
            if hit:
                bad.append(f"env {k}={env[k]}")
    for k in (env or {}):
        if k.endswith("_COUNTERS") and str(env[k]).strip() == "1":
            bad.append(f"env {k}={env[k]}")
    a = [str(x) for x in (argv or ())]
    for f in NEVER_DO_ARGS:
        if f in a:
            bad.append(f"argv {f}")
    for f, vals in NEVER_DO_ARG_VALUES.items():
        idx = [i for i, x in enumerate(a) if x == f]
        if idx and idx[-1] + 1 < len(a) and a[idx[-1] + 1] in vals:
            bad.append(f"argv {f} {a[idx[-1] + 1]}")
    return bad


# ------------------------------------------------------------------ argv helpers
def last_value(argv, flag):
    """The value after the LAST occurrence of `flag` in argv (vLLM's argparse: the last one wins), or None."""
    a = [str(x) for x in (argv or ())]
    idx = [i for i, x in enumerate(a) if x == flag]
    return a[idx[-1] + 1] if idx and idx[-1] + 1 < len(a) else None


def spec_from_argv(argv):
    """The effective --speculative-config of an argv (last occurrence) as a dict; {"error": ...} when unparsable."""
    v = last_value(argv, "--speculative-config")
    if v is None:
        return None
    try:
        d = json.loads(v)
        return d if isinstance(d, dict) else {"error": "not a dict"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:120]}


def partition_ok(value, layers=NUM_LAYERS):
    """'n,m' with two positive ints summing to `layers` (vllm/distributed/utils.py:144-152) and stage 0 holding layer 1
    (Qwen4Exp's PLE layer must live on PP rank 0: model_executor/models/config.py:1030-1046)."""
    try:
        p = [int(x) for x in str(value).split(",")]
    except ValueError:
        return False
    return len(p) == 2 and all(x > 0 for x in p) and sum(p) == layers and p[0] >= 2


# ------------------------------------------------------------------ arm plan
def arm_plan(base_extra, base_env, kv_bytes, q_env=None, ports=(8011, 8021, 8031, 8041, 8051)):
    """Q = NB4S B (base_extra / base_env / the B3 pin) + the Q-defer env (q_env = nb4_ppack.server_env("Q-defer"));
    I8 / P26 / P27 / C12 = Q + exactly one change (a build() knob, one env key, or one trailing flag pair). Every arm runs on
    the patched tree with NB4_PPDEFER=1, the same flags (profiler included) and the same pin."""
    q_env = dict(q_env or {ENV_DEFER: "1"})
    qenv = dict(base_env, **q_env)
    change = {"Q": None,
              "I8": {"kind": "build_env", "key": KNOB_HEAD, "value": "int8"},
              "P26": {"kind": "env", "key": ENV_PART, "value": "26,22"},
              "P27": {"kind": "env", "key": ENV_PART, "value": "27,21"},
              "C12": {"kind": "argv", "key": "--max-num-seqs", "value": str(C12_SEQS)}}
    title = {"Q": "NB4S B (FP8 KV, pin 3,553,580,524 B, text-only, #60490, MTP K3, seqs 8, profiler) + Q-defer (nb4_ppack rev 2, "
                  "NB4_PPDEFER=1): the new base",
             "I8": "Q + DRAFT_HEAD=int8 (INT8 weight-only draft LM head, K stays 3)",
             "P26": "Q + VLLM_PP_LAYER_PARTITION=26,22 (one layer from stage 1 to stage 0)",
             "P27": "Q + VLLM_PP_LAYER_PARTITION=27,21 (two layers from stage 1 to stage 0)",
             "C12": "Q + --max-num-seqs 12, measured with 12 streams (8k@12 / 40k@12 vs Q 8k@8 / 40k@8; measure only)"}
    est = {"Q": 22, "I8": 20, "P26": 20, "P27": 20, "C12": 21}       # the full minimum: launch (+ compile) + gates + body + teardown
    cap = {"Q": 16, "I8": 15, "P26": 15, "P27": 15, "C12": 16}
    arms = {}
    for i, k in enumerate(ORDER):
        ch = change[k]
        env, extra, benv = dict(qenv), list(base_extra), {}
        if ch and ch["kind"] == "env":
            env[ch["key"]] = ch["value"]
        elif ch and ch["kind"] == "build_env":
            benv[ch["key"]] = ch["value"]
        elif ch and ch["kind"] == "argv":
            extra += [ch["key"], ch["value"]]
        streams = C12_SEQS if k == "C12" else BASE_SEQS
        expect = {w: env.get(w, WATCHED_DEFAULT.get(w)) for w in WATCHED}
        arms[k] = {"key": k, "name": f"nb4t_{k.lower()}", "port": ports[i], "extra": extra, "env": env, "build_env": benv,
                   "change": ch, "expect": expect,
                   "expect_cfg": {"max_num_seqs": streams, "draft_head": benv.get(KNOB_HEAD), "partition": expect[ENV_PART],
                                  "language_model_only": True, "num_speculative_tokens": 3},
                   "streams": streams, "windows": window_keys(streams), "kv_bytes": int(kv_bytes), "needs_patch": True,
                   "q_label": Q_LABEL, "est_min": est[k], "cap_min": cap[k], "title": title[k],
                   "lever": k in LEVER_ARMS, "measure_only": k in MEASURE_ONLY}
    return arms


def window_keys(n):
    """The same window kinds in the same order in every arm, at the arm's stream count."""
    return (f"8k@{n}#1", f"8k@{n}#2", f"profile_8k@{n}", f"40k@{n}")


def check_arm_plan(arms):
    """Fail closed (raises ValueError): order Q > I8 > P26 > P27 > C12; every arm free of never-do items; Q = the base env +
    NB4_PPDEFER=1 only (no NB4_PPACK, no CUDA_DEVICE_MAX_CONNECTIONS, no partition, no build knob); each other arm differs
    from Q by exactly its ONE change and nothing else (env, build knobs, flags, pin); every arm patched; ports distinct."""
    errs = []
    q = arms.get("Q") or {}
    if tuple(arms) != ORDER:
        errs.append(f"order {tuple(arms)} != {ORDER}")
    if (q.get("env") or {}).get(ENV_DEFER) != "1" or any(w in (q.get("env") or {}) for w in (ENV_PACK, FLAG_CDMC, ENV_PART)):
        errs.append(f"Q env must carry NB4_PPDEFER=1 and none of NB4_PPACK / CUDA_DEVICE_MAX_CONNECTIONS / {ENV_PART}")
    if q.get("build_env"):
        errs.append("Q sets a build knob")
    if "--language-model-only" not in (q.get("extra") or []):
        errs.append("Q lacks --language-model-only (NB4S B carries it)")
    for k, a in arms.items():
        v = never_do_violations(a["env"], a["extra"])
        if v:
            errs.append(f"{k}: never-do {v}")
        if a["kv_bytes"] != q.get("kv_bytes"):
            errs.append(f"{k}: pin differs from Q")
        if not a.get("needs_patch") or a.get("q_label") != Q_LABEL:
            errs.append(f"{k}: every arm runs on the patched tree with Q-defer")
        if k == "Q":
            continue
        ch = a.get("change") or {}
        ed = {x for x in set(a["env"]) | set(q["env"]) if a["env"].get(x) != q["env"].get(x)}
        bd = {x for x in set(a["build_env"]) | set(q["build_env"]) if a["build_env"].get(x) != q["build_env"].get(x)}
        same_extra = a["extra"] == q["extra"]
        if ch.get("kind") == "env":
            ok = ed == {ch["key"]} and a["env"][ch["key"]] == ch["value"] and not bd and same_extra
            if ch["key"] == ENV_PART and not partition_ok(ch["value"]):
                errs.append(f"{k}: partition {ch['value']} invalid")
        elif ch.get("kind") == "build_env":
            ok = bd == {ch["key"]} and a["build_env"][ch["key"]] == ch["value"] and not ed and same_extra
            if ch["key"] == KNOB_HEAD and ch["value"] != "int8":
                errs.append(f"{k}: DRAFT_HEAD must be int8 (int4 hurts acceptance)")
        elif ch.get("kind") == "argv":
            ok = (not ed and not bd and a["extra"][:len(q["extra"])] == q["extra"]
                  and a["extra"][len(q["extra"]):] == [ch["key"], ch["value"]])
        else:
            ok = False
        if not ok:
            errs.append(f"{k}: differs from Q by env {sorted(ed)} build {sorted(bd)} flags-equal {same_extra}; must be exactly {ch}")
        if a["expect"].get(ENV_PART) != (ch["value"] if ch.get("key") == ENV_PART else MERLIN_PARTITION):
            errs.append(f"{k}: expected partition {a['expect'].get(ENV_PART)}")
    if len({a["port"] for a in arms.values()}) != len(arms):
        errs.append("ports not distinct")
    if errs:
        raise ValueError("ARM_PLAN_INVALID: " + "; ".join(errs))
    return True


def launch_check(arm, env, argv, kv_bytes=None):
    """At launch, on the FULL final env / argv of the server process (fail closed): never-do items; every watched env key
    equal to the arm's expected value (absent when expected None; the partition = Merlin's 25,23 unless the arm sets it);
    the effective --speculative-config (MTP K3, probabilistic, draft_lm_head_quant only on I8); the last --max-num-seqs;
    --language-model-only; the last --kv-cache-memory-bytes = this launch's pin."""
    bad = never_do_violations(env, argv)
    for w, want in arm["expect"].items():
        have = (env or {}).get(w)
        if have != want:
            bad.append(f"{arm['key']} launch env {w}={have!r}, expected {want!r}")
    ec = arm["expect_cfg"]
    sp = spec_from_argv(argv)
    if not isinstance(sp, dict) or sp.get("error"):
        bad.append(f"{arm['key']} --speculative-config missing / unparsable: {sp}")
    else:
        if sp.get("method") != "mtp" or sp.get("num_speculative_tokens") != ec["num_speculative_tokens"]:
            bad.append(f"{arm['key']} spec {sp.get('method')} K {sp.get('num_speculative_tokens')}")
        if sp.get("draft_sample_method") != "probabilistic":
            bad.append(f"{arm['key']} draft_sample_method {sp.get('draft_sample_method')}")
        if sp.get("draft_lm_head_quant") != ec["draft_head"]:
            bad.append(f"{arm['key']} draft_lm_head_quant {sp.get('draft_lm_head_quant')!r}, expected {ec['draft_head']!r}")
    seqs = last_value(argv, "--max-num-seqs")
    if seqs != str(ec["max_num_seqs"]):
        bad.append(f"{arm['key']} --max-num-seqs {seqs}, expected {ec['max_num_seqs']}")
    if "--language-model-only" not in [str(x) for x in (argv or ())]:
        bad.append(f"{arm['key']} --language-model-only missing")
    if kv_bytes is not None and last_value(argv, "--kv-cache-memory-bytes") != str(int(kv_bytes)):
        bad.append(f"{arm['key']} --kv-cache-memory-bytes {last_value(argv, '--kv-cache-memory-bytes')}, expected {kv_bytes}")
    return bad


# ------------------------------------------------------------------ worker env read-back (NB4S, unchanged)
def parse_env_readback(rpc, expect, n_workers=4):
    """rpc = {"results": [per-rank dict from nb4s_env]} or {"error": ...}; expect = {key: value or None (absent)}.
    OK only when every one of n_workers distinct ranks answered and each reports exactly the expected values."""
    out = {"expected": dict(expect), "ok": False, "definitive": False}
    if not isinstance(rpc, dict) or rpc.get("error"):
        out["why"] = "rpc error: " + str((rpc or {}).get("error") if isinstance(rpc, dict) else rpc)[:200]
        return out
    res = rpc.get("results")
    if not isinstance(res, list):
        out["why"] = "no results list"
        return out
    rows = []
    for w in res:
        w = w if isinstance(w, dict) else {}
        envd = w.get("env") if isinstance(w.get("env"), dict) else {}
        rows.append({"rank": w.get("rank"), "pid": w.get("pid"), "values": {k: envd.get(k) for k in expect},
                     "error": w.get("error")})
    out["workers"] = rows
    ranks = [r["rank"] for r in rows]
    if len(rows) != n_workers:
        out["why"] = f"{len(rows)} workers answered, expected {n_workers}"
    elif any(r["error"] for r in rows):
        out["why"] = "worker error: " + str([r["error"] for r in rows if r["error"]])[:200]
    elif len(set(ranks)) != n_workers or None in ranks:
        out["why"] = f"ranks {ranks} not {n_workers} distinct"
    elif any(r["values"] != dict(expect) for r in rows):
        out["why"] = f"values {[r['values'] for r in rows]} != {dict(expect)}"
        out["definitive"] = True                      # the workers answered: the env is really not as planned
    else:
        out["ok"] = True
        out["why"] = f"all {n_workers} workers: {dict(expect)}"
    return out


def parse_proc_environ(blobs, expect):
    """Side record: {pid: raw /proc/<pid>/environ bytes or None} of the engine processes -> each process's values."""
    vals = {}
    for pid, raw in (blobs or {}).items():
        if raw is None:
            vals[str(pid)] = "UNREADABLE"
            continue
        d = {}
        for item in (raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)).split("\0"):
            if "=" in item:
                k, v = item.split("=", 1)
                d[k] = v
        vals[str(pid)] = {k: d.get(k) for k in expect}
    readable = [v for v in vals.values() if v != "UNREADABLE"]
    return {"values": vals, "all_expected": bool(readable) and all(v == dict(expect) for v in readable), "readable": len(readable)}


# ------------------------------------------------------------------ server-log config read-back
_WTAG = re.compile(r"\(Worker_PP(\d+)_TP(\d+)")
_HEAD_RX = re.compile(r"Draft lm_head: (INT8|INT4) weight-only copy of the (\d+)x(\d+) shard \(([\d.]+) MiB")
_PLAYERS_RX = re.compile(r"\(pipeline layers (\d+)-(\d+)\)")
_SPEC_RX = re.compile(r"'speculative_config': (\{[^{}]*\})")


def _literal(s):
    import ast
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def log_config(log_text):
    """What the server log says about the knobs NB4T changes: the non-default args (max_num_seqs, language_model_only,
    kv_cache_memory_bytes, speculative_config), every 'Draft lm_head:' build line (mode, shard, MiB, worker), the
    'pipeline layers a-b' range of every stage (QSA static-scale lines, one per worker), per-worker block bytes."""
    t = log_text or ""
    lines = t.splitlines()
    na = next((l for l in lines if "non-default args:" in l), "")
    out = {"non_default_found": bool(na)}
    m = re.search(r"'max_num_seqs': (\d+)", na)
    out["max_num_seqs"] = int(m.group(1)) if m else None
    m = re.search(r"'kv_cache_memory_bytes': (\d+)", na)
    out["kv_cache_memory_bytes"] = int(m.group(1)) if m else None
    m = re.search(r"'language_model_only': (True|False)", na)
    out["language_model_only"] = (m.group(1) == "True") if m else None
    m = _SPEC_RX.search(na)
    out["speculative_config"] = _literal(m.group(1)) if m else None
    heads = []
    for l in lines:
        h = _HEAD_RX.search(l)
        if h:
            w = _WTAG.search(l)
            heads.append({"mode": h.group(1), "n": int(h.group(2)), "k": int(h.group(3)), "mib": float(h.group(4)),
                          "worker": f"PP{w.group(1)}_TP{w.group(2)}" if w else None})
    out["draft_head_lines"] = heads[:8]
    out["draft_head_any"] = any("Draft lm_head:" in l for l in lines)
    ranges = defaultdict(set)
    for l in lines:
        p = _PLAYERS_RX.search(l)
        w = _WTAG.search(l)
        if p and w:
            ranges[int(w.group(1))].add((int(p.group(1)), int(p.group(2))))
    out["stage_layers"] = {str(s): sorted(v) for s, v in sorted(ranges.items())}
    m = re.findall(r"per-worker block bytes(?: unchanged)? \[([\d, ]+)\]", t)
    out["block_bytes_per_stage"] = [int(x) for x in m[-1].split(",")] if m else None
    return out


def config_readback(arm, log_text, kv_bytes=None):
    """The server log must show the arm's knobs (fail closed for a positive verdict). definitive = the log shows a value
    that contradicts the plan (the arm then stops: nothing to measure); a missing line is not definitive."""
    f = log_config(log_text)
    ec = arm["expect_cfg"]
    checks, contra = {}, []
    if f["max_num_seqs"] is not None:
        checks["max_num_seqs"] = f["max_num_seqs"] == ec["max_num_seqs"]
        if not checks["max_num_seqs"]:
            contra.append(f"max_num_seqs {f['max_num_seqs']} != {ec['max_num_seqs']}")
    else:
        checks["max_num_seqs"] = False
    if f["language_model_only"] is not None:
        checks["language_model_only"] = f["language_model_only"] is True
        if not checks["language_model_only"]:
            contra.append("language_model_only False")
    else:
        checks["language_model_only"] = False
    if kv_bytes is not None:
        checks["kv_cache_memory_bytes"] = f["kv_cache_memory_bytes"] == int(kv_bytes)
        if f["kv_cache_memory_bytes"] is not None and not checks["kv_cache_memory_bytes"]:
            contra.append(f"kv_cache_memory_bytes {f['kv_cache_memory_bytes']} != {kv_bytes}")
    sp = f["speculative_config"]
    if isinstance(sp, dict):
        checks["spec_k3_probabilistic"] = sp.get("num_speculative_tokens") == 3 and sp.get("draft_sample_method") == "probabilistic"
        checks["spec_draft_head"] = sp.get("draft_lm_head_quant") == ec["draft_head"]
        if not checks["spec_draft_head"]:
            contra.append(f"spec draft_lm_head_quant {sp.get('draft_lm_head_quant')!r} != {ec['draft_head']!r}")
    else:
        checks["spec_k3_probabilistic"] = checks["spec_draft_head"] = False
    modes = sorted({h["mode"] for h in f["draft_head_lines"]})
    if ec["draft_head"]:
        checks["draft_head_built"] = modes == [ec["draft_head"].upper()]
        if f["draft_head_any"] and not checks["draft_head_built"]:
            contra.append(f"draft head built {modes}, expected {ec['draft_head'].upper()}")
    else:
        checks["draft_head_built"] = not f["draft_head_any"]
        if f["draft_head_any"]:
            contra.append(f"a quantized draft head was built ({modes}) on a BF16-head arm")
    p0 = int(str(ec["partition"]).split(",")[0])
    want = {"0": [(0, p0)], "1": [(p0, NUM_LAYERS)]}
    sl = f["stage_layers"]
    if sl:
        checks["partition"] = {k: [tuple(x) for x in v] for k, v in sl.items()} == want
        if not checks["partition"]:
            contra.append(f"stage layers {sl} != {want}")
    else:
        checks["partition"] = False
    ok = all(checks.values())
    return {"ok": ok, "definitive": bool(contra), "checks": checks, "contradictions": contra,
            "why": "ok" if ok else ("; ".join(contra) if contra else "missing: " + ", ".join(k for k, v in checks.items() if not v)),
            "log": {k: f[k] for k in ("max_num_seqs", "language_model_only", "kv_cache_memory_bytes", "speculative_config",
                                      "draft_head_lines", "stage_layers", "block_bytes_per_stage")}}


# ------------------------------------------------------------------ capacity facts (P26 / P27 read-back; recorded on every arm)
def capacity_facts(rec):
    """Pool blocks (the workers' num_gpu_blocks), the KV token pool and the max concurrency at 131,072 (server log), the
    per-stage block bytes and the bytes the pool really uses per stage (blocks x block bytes vs the pin), free MiB per rank
    after ready / at the end (min of nvidia-smi and the worker), the per-rank model load GiB, and F (fixed blocks per
    request) from the max-concurrency line."""
    rec = rec or {}
    pool = rec.get("pool") or {}
    wk = rec.get("workers") or {}
    blocks = wk.get("num_gpu_blocks") or []
    nb = blocks[0] if len(blocks) == 1 else None
    bb = ((rec.get("config_readback") or {}).get("log") or {}).get("block_bytes_per_stage")
    pin = rec.get("pin")
    mc = pool.get("max_concurrency")
    out = {"num_gpu_blocks": blocks, "kv_tokens": pool.get("kv_tokens"), "max_concurrency_131072": mc,
           "max_concurrency_tokens": pool.get("max_concurrency_tokens"), "block_bytes_per_stage": bb, "pin": pin,
           "free_after_ready_min_mib": (rec.get("free_after_ready_min") or {}).get("min"),
           "free_end_min_mib": (rec.get("free_end_min") or {}).get("min"),
           "rank_load_gib": pool.get("rank_load_gib"), "rank_graph_gib": pool.get("rank_graph_gib")}
    if nb and isinstance(bb, list) and pin:
        out["pool_bytes_per_stage"] = [nb * b for b in bb]
        out["pin_unused_mib_per_stage"] = [round((pin - nb * b) / 2 ** 20, 1) for b in bb]
        out["binding_stage"] = min(range(len(bb)), key=lambda i: pin // bb[i])
    if nb and mc:
        out["F_blocks"] = math.ceil(round(nb / mc - MAX_LEN / BLOCK_TOKENS, 3))
    return out


def kv_demand(n, prompt_tokens, gen_tokens, f_blocks, block=BLOCK_TOKENS):
    """Blocks n streams hold at the end of a window: n x (ceil((prompt + generated) / block) + F) (Merlin's rule)."""
    return int(n) * (math.ceil((prompt_tokens + gen_tokens) / block) + int(f_blocks))


def capacity_predict(pool_blocks, f_blocks, n, prompt_tokens, gen_tokens):
    """A pre-window estimate (recorded, never gating): the KV demand vs the measured pool."""
    if not pool_blocks or f_blocks is None:
        return {"label": "UNKNOWN", "why": "no pool / F"}
    need = kv_demand(n, prompt_tokens, gen_tokens, f_blocks)
    lab = "FITS" if need <= 0.9 * pool_blocks else ("EDGE" if need <= pool_blocks else "OVER_POOL")
    return {"label": lab, "need_blocks": need, "pool_blocks": pool_blocks, "f_blocks": f_blocks, "streams": n,
            "prompt_tokens": prompt_tokens, "gen_tokens_est": gen_tokens}


# ------------------------------------------------------------------ nb4_ppack patch sequencing (NB4S, unchanged)
class PatchSeq:
    """The nb4_ppack patch on the installed tree: applied right before the first arm that needs it (Q: every NB4T arm
    does), kept for every later arm, reverted (pre-shas verified by revert_fn) in finish(). A failed apply -> no arm
    runs (NOT_INTEGRATED). apply_fn / revert_fn raise on any miss."""

    def __init__(self, apply_fn, revert_fn):
        self.apply_fn, self.revert_fn = apply_fn, revert_fn
        self.state, self.log, self.apply_rec, self.revert_rec, self.error = "pristine", [], None, None, None

    def before_arm(self, arm):
        """(ok, why). A patched arm: the tree must be patched (applies once). Other arm: the tree must be pristine."""
        if arm.get("needs_patch"):
            if self.state == "applied":
                return True, "already applied"
            if self.state in ("failed", "revert_failed"):
                return False, f"NOT_INTEGRATED: patch {self.state}: {self.error}"
            try:
                self.apply_rec = self.apply_fn()
                self.state = "applied"
                self.log.append(("apply", arm["key"]))
                return True, "applied"
            except Exception as e:
                self.state, self.error = "failed", f"{type(e).__name__}: {e}"[:500]
                self.log.append(("apply_failed", arm["key"]))
                return False, "NOT_INTEGRATED: " + self.error
        if self.state == "applied":
            ok = self._revert(arm["key"])
            if not ok:
                return False, "tree not restored before a non-patched arm: " + str(self.error)
        return True, "pristine"

    def _revert(self, tag):
        try:
            self.revert_rec = self.revert_fn()
            self.state = "reverted"
            self.log.append(("revert", tag))
            return True
        except Exception as e:
            self.state, self.error = "revert_failed", f"{type(e).__name__}: {e}"[:500]
            self.log.append(("revert_failed", tag))
            return False

    def finish(self):
        if self.state == "applied":
            self._revert("finish")
        return {"state": self.state, "log": self.log, "error": self.error}


# ------------------------------------------------------------------ /metrics: the lab's aggregation + histogram buckets
_PROM_RX = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
ITER_BUCKET = "vllm:iteration_tokens_total_bucket"


def prom_parse(txt):
    """(agg, buckets): agg = exactly e23_lab.prom's dict (sum over labels + '@max'); buckets = {le (float): cumulative
    count} of the iteration-token histogram (summed over label sets)."""
    out, buckets = {}, {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_RX.match(line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        if v != v:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
        out[m.group(1) + "@max"] = max(out.get(m.group(1) + "@max", v), v)
        if m.group(1) == ITER_BUCKET and m.group(2):
            le = re.search(r'le="([^"]+)"', m.group(2))
            if le:
                try:
                    k = float(le.group(1))
                except ValueError:
                    continue
                buckets[k] = buckets.get(k, 0.0) + v
    return out, buckets


def hist_delta(h0, h1):
    """Per-bucket (non-cumulative) step counts between two cumulative snapshots: [[lo, hi, count], ...]."""
    if not h0 or not h1:
        return None
    les = sorted(set(h0) | set(h1))
    cum = [(le, h1.get(le, 0.0) - h0.get(le, 0.0)) for le in les]
    out, prev_le, prev_c = [], 0.0, 0.0
    for le, c in cum:
        out.append([prev_le, le, round(c - prev_c, 1)])
        prev_le, prev_c = le, c
    return out


def parity_from_hist(delta, n, pp=2, k=3):
    """Micro-batch split of a decode window from its iteration-token histogram (recorded, never gating in NB4T):
    balanced = every step schedules ceil(n / pp) x (1 + k) tokens. BALANCED when >= 90 % of the steps fall in the bucket
    that holds that count; IMBALANCED otherwise; UNRESOLVED when that bucket also holds the 2x count or there is no data."""
    if not delta:
        return {"label": "UNRESOLVED", "why": "no histogram"}
    t_bal = math.ceil(n / pp) * (1 + k)
    tot = sum(c for _, _, c in delta if c > 0)
    if tot <= 0:
        return {"label": "UNRESOLVED", "why": "no steps", "balanced_tokens": t_bal}
    hb = next(((lo, hi, c) for lo, hi, c in delta if lo < t_bal <= hi), None)
    if hb is None:
        return {"label": "UNRESOLVED", "why": f"no bucket holds {t_bal}", "balanced_tokens": t_bal}
    if hb[0] < 2 * t_bal <= hb[1]:
        return {"label": "UNRESOLVED", "why": f"bucket ({hb[0]}, {hb[1]}] cannot separate {t_bal} from {2 * t_bal}",
                "balanced_tokens": t_bal}
    share = hb[2] / tot
    return {"label": "BALANCED" if share >= 0.9 else "IMBALANCED", "share_at_balanced_bucket": round(share, 3),
            "balanced_tokens": t_bal, "bucket": [hb[0], hb[1]], "steps": int(tot),
            "dist": [[lo, hi, c] for lo, hi, c in delta if c > 0][:12]}


# ------------------------------------------------------------------ per-graph stall parser (NB4S, unchanged)
def _open(p):
    with (gzip.open if str(p).endswith(".gz") else open)(p, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _rank_of(name, tr, k):
    m_pp, m_tp, m_r = re.search(r"pp(\d+)", name), re.search(r"_tp(\d+)", name), re.search(r"rank[-_]?(\d+)", name)
    di = tr.get("distributedInfo") if isinstance(tr.get("distributedInfo"), dict) else {}
    rank = di.get("rank") if isinstance(di.get("rank"), int) else (int(m_r.group(1)) if m_r else k)
    return rank, (int(m_pp.group(1)) if m_pp else rank // 2), (int(m_tp.group(1)) if m_tp else rank % 2)


def _a(e):
    a = e.get("args")
    return a if isinstance(a, dict) else {}


SCOPE_RX = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def rank_graphs(tr):
    """One rank: graph replays (kernels grouped by the cudaGraphLaunch correlation), NCCL kernels, decode scopes."""
    ev = [e for e in (tr.get("traceEvents") or []) if e.get("ph") == "X" and "ts" in e]
    launches = {}
    for e in ev:
        if str(e.get("cat", "")).lower() in ("cuda_runtime", "cuda_driver") and "GraphLaunch" in str(e.get("name", "")):
            c = _a(e).get("correlation")
            if c is not None:
                launches[c] = e
    groups, nccl = defaultdict(list), []
    for e in ev:
        if str(e.get("cat", "")).lower() not in ("kernel", "gpu_kernel"):
            continue
        a = _a(e)
        c = a.get("correlation")
        if c in launches:
            groups[c].append(e)
        if "nccl" in str(e.get("name", "")).lower():
            nccl.append({"a": float(e["ts"]), "b": float(e["ts"]) + float(e.get("dur", 0.0)), "name": str(e.get("name", "")).split("(")[0],
                         "coll": a.get("Collective name"), "dtype": a.get("dtype"), "nelems": a.get("In msg nelems"),
                         "pg": a.get("Process Group Description"), "group": a.get("Process Group Ranks"),
                         "stream": a.get("stream", e.get("tid")), "in_graph": bool(a.get("graph id"))})
    nccl.sort(key=lambda d: d["b"])
    graphs = []
    for c, ks in groups.items():
        ks.sort(key=lambda e: float(e["ts"]))
        graphs.append({"corr": c, "ks": ks, "n": len(ks), "start": float(ks[0]["ts"]),
                       "end": max(float(k["ts"]) + float(k.get("dur", 0.0)) for k in ks)})
    graphs.sort(key=lambda g: g["start"])
    scopes = sorted((e for e in ev if e.get("cat") == "user_annotation" and SCOPE_RX.match(str(e.get("name", "")))),
                    key=lambda e: float(e["ts"]))
    return graphs, nccl, scopes


def largest_gap(g, nccl):
    """The largest gap inside one graph replay (none of its own kernels running): node index, the stream / kernel after the
    gap, the releasing NCCL kernel and the offset of the gap end to the nearest preceding pp_broadcast kernel end."""
    ks = g["ks"]
    cur = float(ks[0]["ts"]) + float(ks[0].get("dur", 0.0))
    best = (0.0, None, None, None)
    for i in range(1, len(ks)):
        t = float(ks[i]["ts"])
        if t - cur > best[0]:
            best = (t - cur, i, cur, t)
        cur = max(cur, t + float(ks[i].get("dur", 0.0)))
    gap, idx, a, b = best
    out = {"nodes": len(ks), "gap_ms": round(gap / 1000.0, 3), "node_index": idx}
    if idx is None:
        return out
    k = ks[idx]
    out.update(stream=_a(k).get("stream", k.get("tid")), kernel=str(k.get("name", "")).split("(")[0].replace("void ", "")[:60],
               gap_start_ms_in_graph=round((a - g["start"]) / 1000.0, 3))
    if gap * 1e-3 >= STALL_LT_MS:
        cand = [x for x in nccl if b - RELEASE_WINDOW_US <= x["b"] <= b + 5.0]
        if cand:
            r = max(cand, key=lambda x: x["b"])
            out["released_by"] = f"{r['name']} {r['coll']} {r['dtype']}x{r['nelems']} {r['pg']}"
            out["release_lag_us"] = round(b - r["b"], 1)
    ppb = [x["b"] for x in nccl if x["pg"] == "pp_broadcast" and x["b"] <= b + 5.0]
    if ppb:
        out["offset_to_pp_broadcast_end_us"] = round(b - max(ppb), 1)
    return out


def _share_below(vals, lim):
    return round(sum(1 for v in vals if v < lim) / len(vals), 3) if vals else None


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def _stage_stall(rows_by_rank):
    """Aggregate per-graph rows (steady graphs only) of one stage's ranks."""
    gaps, nodes, rel, offs = [], Counter(), Counter(), []
    for rows in rows_by_rank:
        steady = rows[1:-1] if len(rows) > 2 else rows
        for x in steady:
            gaps.append(x["gap_ms"])
            if x["gap_ms"] >= STALL_LT_MS:
                nodes[x["node_index"]] += 1
                rel[x.get("released_by")] += 1
                if x.get("offset_to_pp_broadcast_end_us") is not None:
                    offs.append(x["offset_to_pp_broadcast_end_us"])
    return {"graphs": len(gaps), "stall_ms_median": _med(gaps),
            "stall_ms_mean": round(statistics.mean(gaps), 3) if gaps else None,
            "share_below_1ms": _share_below(gaps, STALL_LT_MS),
            "node_index_mode": nodes.most_common(1)[0][0] if nodes else None, "node_index_hist": dict(nodes.most_common(5)),
            "released_by_mode": rel.most_common(1)[0][0] if rel else None,
            "offset_to_pp_broadcast_end_us_median": _med(offs)}


def graph_stalls(src, full_min=FULL_MIN_NODES):
    """Per-boot stall facts from the traces of the profiled window (a directory or a list of files)."""
    p = Path(src) if not isinstance(src, (list, tuple)) else None
    files = sorted(p.rglob("*.pt.trace.json*")) if p is not None and p.is_dir() else ([p] if p is not None else [Path(x) for x in src])
    if not files:
        return {"error": "NO_TRACE_FILE"}
    ranks = {}
    for k, f in enumerate(files):
        tr = _open(f)
        rank, pp, tp = _rank_of(f.name, tr, k)
        graphs, nccl, scopes = rank_graphs(tr)
        ranks[rank] = {"pp": pp, "tp": tp, "graphs": graphs, "nccl": nccl, "scopes": scopes, "file": f.name}
    stage0 = min(r["pp"] for r in ranks.values())
    out = {"files": [r["file"] for r in ranks.values()], "by_rank": {}}
    rows_stage = defaultdict(list)
    for rk in sorted(ranks):
        r = ranks[rk]
        tg = [g for g in r["graphs"] if g["n"] >= full_min]
        rows = [largest_gap(g, r["nccl"]) for g in tg]
        rows_stage[r["pp"]].append(rows)
        steady = rows[1:-1] if len(rows) > 2 else rows
        gv = [x["gap_ms"] for x in steady]
        rec = {"stage": r["pp"], "target_graphs": len(tg), "graph_nodes": dict(Counter(g["n"] for g in tg).most_common(3)),
               "small_graphs": len(r["graphs"]) - len(tg), "stall_ms_median": _med(gv), "share_below_1ms": _share_below(gv, STALL_LT_MS),
               "steady_graphs": len(steady), "first_graph_gap_ms": rows[0]["gap_ms"] if rows else None,
               "node_index_mode": Counter(x["node_index"] for x in steady if x["gap_ms"] >= STALL_LT_MS).most_common(3),
               "stream_mode": Counter(str(x.get("stream")) for x in steady if x["gap_ms"] >= STALL_LT_MS).most_common(3),
               "kernel_after_gap": Counter(x.get("kernel") for x in steady if x["gap_ms"] >= STALL_LT_MS).most_common(2),
               "released_by": Counter(x.get("released_by") for x in steady if x["gap_ms"] >= STALL_LT_MS).most_common(3),
               "rows": rows[:60]}
        if len(tg) >= 2:
            rec["graph_period_ms_median"] = round(statistics.median([tg[i + 1]["start"] - tg[i]["start"] for i in range(len(tg) - 1)]) / 1000.0, 3)
            lo, hi, nt = tg[0]["start"], tg[-1]["start"], len(tg) - 1
            inwin = [x for x in r["nccl"] if x["b"] > lo and x["a"] < hi]
            sr = sum(min(x["b"], hi) - max(x["a"], lo) for x in inwin if "SendRecv" in x["name"])
            rec["sendrecv_ms_per_tick"] = round(sr / 1000.0 / nt, 3)
            pb = [x["b"] - x["a"] for x in inwin if x["pg"] == "pp_broadcast"]
            rec["pp_broadcast_kernels"] = {"per_tick": round(len(pb) / nt, 2), "ms_per_tick": round(sum(pb) / 1000.0 / nt, 3),
                                           "median_us": round(statistics.median(pb), 1) if pb else None,
                                           "max_ms": round(max(pb) / 1000.0, 3) if pb else None}
        gen = [int(SCOPE_RX.match(str(e.get("name", ""))).group(3)) for e in r["scopes"]]
        rec["requests_by_parity"] = {str(q): dict(Counter(gen[q::2])) for q in (0, 1)}
        out["by_rank"][rk] = rec
    out["stage0"] = _stage_stall(rows_stage[stage0])
    s1 = [v for v in out["by_rank"].values() if v["stage"] != stage0]
    out["stage1"] = dict(_stage_stall([rr for st, lst in rows_stage.items() if st != stage0 for rr in lst]),
                         sendrecv_ms_per_tick_mean=_med([v.get("sendrecv_ms_per_tick") for v in s1]),
                         pp_broadcast_send_ms_per_tick=_med([(v.get("pp_broadcast_kernels") or {}).get("ms_per_tick") for v in s1]),
                         pp_broadcast_send_max_ms=max([(v.get("pp_broadcast_kernels") or {}).get("max_ms") or 0 for v in s1] or [0]))
    ref = out["by_rank"].get(min(rk for rk in ranks if ranks[rk]["pp"] == stage0), {})
    out["parity"] = parity_from_scopes(ref.get("requests_by_parity"))
    return out


def parity_from_scopes(par):
    """{"0": {n_requests: steps}, "1": {...}} -> BALANCED when both parities have requests and their modal counts differ
    by <= 1; IMBALANCED otherwise (e.g. 7 + 1)."""
    if not par:
        return {"label": "UNRESOLVED", "why": "no scopes"}
    mode = {}
    for q in ("0", "1"):
        c = {int(k): v for k, v in (par.get(q) or {}).items()}
        mode[q] = max(c, key=lambda x: c[x]) if c else 0
    lab = "BALANCED" if mode["0"] > 0 and mode["1"] > 0 and abs(mode["0"] - mode["1"]) <= 1 else "IMBALANCED"
    return {"label": lab, "modal_requests": mode, "by_parity": par}


# ------------------------------------------------------------------ nb4_timeline.analyze() -> the per-boot report fields
def _g(d, *ks):
    for k in ks:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def summarize_timeline(rep):
    """Compact per-boot fields from agent C's analyze() output; every field is read defensively (None when missing)."""
    if not isinstance(rep, dict) or rep.get("error"):
        return {"error": (rep or {}).get("error") if isinstance(rep, dict) else "no report"}
    s = rep.get("summary") or {}
    out = {"tick_ms_mean": s.get("tick_ms_mean"), "tick_ms_median": s.get("tick_ms_median"),
           "steady_ticks": s.get("steady_ticks"), "graph_modes_steady": s.get("graph_modes"),
           "graph_modes_by_rank": rep.get("graph_modes"), "clock": _g(rep, "clock", "method"), "stages": {}}
    for st in ("stage0", "stage1"):
        if isinstance(s.get(st), dict):
            out["stages"][st] = {c: _g(s, st, c, "mean_ms") for c in ("compute", "copy", "in_graph_stall", "nccl_peer_wait",
                                                                       "nccl_tp_skew", "nccl_transfer", "host_launch_gap",
                                                                       "host_python_gap", "idle_no_host")}
            hs = [out["stages"][st].get(c) for c in ("host_launch_gap", "host_python_gap", "idle_no_host")]
            out["stages"][st]["host_total"] = round(sum(x for x in hs if x is not None), 3) if any(x is not None for x in hs) else None
    ticks = [t for t in (rep.get("ticks") or []) if isinstance(t, dict)]
    st_t = ticks[1:-1] if len(ticks) > 2 else ticks
    sv = [_g(t, "stall", "in_graph_stall_ms_ref_rank") for t in st_t]
    sv = [x for x in sv if isinstance(x, (int, float))]
    out["tick_stall_ms_median"] = round(statistics.median(sv), 3) if sv else None
    out["tick_share_stall_below_1ms"] = _share_below(sv, STALL_LT_MS)
    by_rank = defaultdict(lambda: defaultdict(list))
    for t in st_t:
        for r, row in (t.get("by_rank") or {}).items():
            for c in ("compute", "copy", "in_graph_stall", "nccl_peer_wait", "nccl_transfer"):
                if isinstance(row.get(c), (int, float)):
                    by_rank[str(r)][c].append(row[c])
            by_rank[str(r)]["tick"].append(t.get("tick_ms"))
    out["per_rank"] = {}
    for r, d in sorted(by_rank.items()):
        m = {c: round(statistics.mean([x for x in v if x is not None]), 3) for c, v in d.items() if [x for x in v if x is not None]}
        if "tick" in m and "compute" in m:
            m["busy"] = round(m["compute"] + m.get("copy", 0.0), 3)
            m["idle"] = round(m["tick"] - m["busy"], 3)
        out["per_rank"][r] = m
    out["stalls"] = rep.get("stalls")
    par = rep.get("parity") or {}
    out["parity_verdict"] = par.get("verdict") if isinstance(par, dict) else None
    out["parity_requests"] = {k: _g(v, "n_hist_by_parity") for k, v in ((par.get("stages") or {}) if isinstance(par, dict) else {}).items()}
    dep = rep.get("dependency") or {}
    out["dependency_verdict"] = dep.get("verdict") if isinstance(dep, dict) else None
    out["ideal_tick"] = s.get("ideal_tick")
    out["pipeline"] = s.get("pipeline")
    out["stage1_idle_ms_mean"] = s.get("stage1_idle_ms_mean")
    return out


def stage_compute(tl):
    """Per-stage compute ms, the binding stage and the tick from a summarize_timeline() dict."""
    st = (tl or {}).get("stages") or {}
    c0, c1 = _g(st, "stage0", "compute"), _g(st, "stage1", "compute")
    out = {"stage0_compute_ms": c0, "stage1_compute_ms": c1, "tick_ms_mean": (tl or {}).get("tick_ms_mean")}
    if isinstance(c0, (int, float)) and isinstance(c1, (int, float)):
        out["binding_stage"] = 0 if c0 >= c1 else 1
        out["stage_gap_ms"] = round(abs(c1 - c0), 3)
    return out


# ------------------------------------------------------------------ parity (the NB4S fix: recorded everywhere, never a base reject)
def parity_of(rec, keys=None):
    """Every parity record of an arm: the profiled window (scope counts per micro-batch parity: label + modal split), the
    timeline's own parity verdict, and the histogram label of every steady window."""
    rec = rec or {}
    sp = (rec.get("stall") or {}).get("parity") or {}
    w = _win(rec)
    ks = keys or [k for k in w if not str(k).startswith("profile")]
    return {"profile_label": sp.get("label"), "modal_split": sp.get("modal_requests"),
            "timeline_verdict": (rec.get("timeline") or {}).get("parity_verdict"),
            "windows": {k: ((w.get(k) or {}).get("parity") or {}).get("label") for k in ks if isinstance(w.get(k), dict)}}


def _split_key(m):
    """A modal split as an order-free key: {"0": 7, "1": 1} == {"0": 1, "1": 7} (which parity is first is arbitrary)."""
    if not isinstance(m, dict) or not m:
        return None
    return tuple(sorted(int(v) for v in m.values()))


def parity_compare(q_rec, x_rec):
    """(status, detail). COMPARED when both profiled windows carry the same parity label (BALANCED / IMBALANCED) or the same
    modal split (order-free); PARITY_DIFFERS otherwise; PARITY_UNRESOLVED when either side has no resolved parity."""
    pq, px = parity_of(q_rec), parity_of(x_rec)
    det = {"q": {"label": pq["profile_label"], "split": pq["modal_split"]}, "x": {"label": px["profile_label"], "split": px["modal_split"]}}
    res = ("BALANCED", "IMBALANCED")
    if pq["profile_label"] not in res or px["profile_label"] not in res:
        return "PARITY_UNRESOLVED", det
    if pq["profile_label"] == px["profile_label"]:
        det["rule"] = "same parity label"
        return "COMPARED", det
    if _split_key(pq["modal_split"]) is not None and _split_key(pq["modal_split"]) == _split_key(px["modal_split"]):
        det["rule"] = "same modal split"
        return "COMPARED", det
    det["rule"] = "label and split differ"
    return "PARITY_DIFFERS", det


def stall_compare(q_rec, x_rec):
    """The stall / per-stage comparison of an arm with Q, made only when parity_compare says COMPARED (else the label).
    class: STALL_STILL_FIXED (>= 95 % of stage-0 graphs < 1 ms, no new stage-1 stall), STALL_MOVED_STAGE1, STALL_RETURNED."""
    st, det = parity_compare(q_rec, x_rec)
    out = {"status": st, "parity": det}
    qs, xs = (q_rec or {}).get("stall") or {}, (x_rec or {}).get("stall") or {}
    if not (xs.get("stage0") or {}).get("graphs") or not (qs.get("stage0") or {}).get("graphs"):
        out["status"] = "NO_TRACE"
        return out
    if st != "COMPARED":
        return out
    q0, x0, q1, x1 = qs["stage0"], xs["stage0"], qs.get("stage1") or {}, xs.get("stage1") or {}
    keys = ("stall_ms_median", "share_below_1ms", "node_index_mode", "released_by_mode")
    out["q_stage0"], out["x_stage0"] = {k: q0.get(k) for k in keys}, {k: x0.get(k) for k in keys}
    out["q_stage1"] = {k: q1.get(k) for k in ("stall_ms_median", "sendrecv_ms_per_tick_mean", "pp_broadcast_send_ms_per_tick")}
    out["x_stage1"] = {k: x1.get(k) for k in ("stall_ms_median", "sendrecv_ms_per_tick_mean", "pp_broadcast_send_ms_per_tick")}
    s1_new = (x1.get("stall_ms_median") or 0.0) >= BASE_STALL_MIN_MS and (q1.get("stall_ms_median") or 0.0) < STALL_LT_MS
    sh = x0.get("share_below_1ms") or 0.0
    out["class"] = "STALL_MOVED_STAGE1" if s1_new else ("STALL_STILL_FIXED" if sh >= FIXED_SHARE else "STALL_RETURNED")
    qc, xc = stage_compute((q_rec or {}).get("timeline")), stage_compute((x_rec or {}).get("timeline"))
    out["q_compute"], out["x_compute"] = qc, xc
    d = {}
    for k in ("stage0_compute_ms", "stage1_compute_ms", "tick_ms_mean"):
        if isinstance(qc.get(k), (int, float)) and isinstance(xc.get(k), (int, float)):
            d[k] = round(xc[k] - qc[k], 3)
    out["compute_delta_ms"] = d
    return out


# ------------------------------------------------------------------ speed
def _win(rec):
    """The arm's windows: rec["steady"] in the notebook (so RC2's garbage-under-load gate sees them), rec["windows"] accepted."""
    w = (rec or {}).get("steady")
    return w if isinstance(w, dict) else ((rec or {}).get("windows") or {})


def window_ok(w):
    """A window counts for speed when STEADY, running == level, prefill done, 0 preemptions and a tok/s value. Its
    micro-batch split is recorded (window parity) and never gates (the NB4S fix: every NB4S window was IMBALANCED)."""
    if not isinstance(w, dict):
        return False, "missing"
    if w.get("status") != "STEADY":
        return False, f"status {w.get('status')}"
    if w.get("running_eq_level") is not True:
        return False, "running != level"
    if (w.get("preemptions") or 0) != 0:
        return False, "preemptions"
    if not isinstance(w.get("tok_s"), (int, float)) or w["tok_s"] <= 0:
        return False, "no tok/s"
    return True, "ok"


def speed_compare(q, x, q_keys, x_keys):
    """q / x: {window key: record}; q_keys / x_keys: the two 8k windows of each arm (paired by position). Gain =
    mean(X) / mean(Q) - 1; Q's spread = |w1 - w2| / mean (both Q windows valid); every X window must be valid.
    PASS when gain > max(spread, SPEED_MIN); REGRESSION when gain < -max(spread, SPEED_MIN); NO_GAIN otherwise."""
    qv = {k: q.get(k) for k in q_keys if window_ok(q.get(k))[0]}
    xv = {k: x.get(k) for k in x_keys if window_ok(x.get(k))[0]}
    out = {"q_keys": list(q_keys), "x_keys": list(x_keys), "q_valid": sorted(qv), "x_valid": sorted(xv),
           "why_invalid": {f"{arm}:{k}": window_ok(d.get(k))[1] for arm, d, ks in (("Q", q, q_keys), ("X", x, x_keys))
                           for k in ks if not window_ok(d.get(k))[0]},
           "window_parity": {"q": [((q.get(k) or {}).get("parity") or {}).get("label") for k in q_keys],
                             "x": [((x.get(k) or {}).get("parity") or {}).get("label") for k in x_keys]}}
    if len(qv) < len(q_keys):
        out.update(status="INCONCLUSIVE_Q_SPREAD_UNKNOWN", why="Q needs both 8k windows valid for its own spread")
        return out
    if len(xv) != len(x_keys):
        out.update(status="INCONCLUSIVE_X_WINDOWS" if xv else "INCONCLUSIVE_NO_X_WINDOW",
                   why=f"all {len(x_keys)} X windows must be valid; valid {sorted(xv)}")
        return out
    qt = [q[k]["tok_s"] for k in q_keys]
    xt = [x[k]["tok_s"] for k in x_keys]
    qm, xm = statistics.mean(qt), statistics.mean(xt)
    spread = abs(qt[0] - qt[1]) / qm if len(qt) == 2 else 0.0
    gain = xm / qm - 1.0
    thr = max(spread, SPEED_MIN)
    out.update(q_tok_s=[round(v, 1) for v in qt], x_tok_s=[round(v, 1) for v in xt], q_mean=round(qm, 1), x_mean=round(xm, 1),
               q_spread=round(spread, 4), gain=round(gain, 4), threshold=round(thr, 4),
               status="PASS" if gain > thr else ("REGRESSION" if gain < -thr else "NO_GAIN"),
               q_tick_ms=[q[k].get("tick_ms") for k in q_keys], x_tick_ms=[x[k].get("tick_ms") for k in x_keys],
               x_accept_len=[x[k].get("mean_acceptance_length") for k in x_keys],
               q_accept_len=[q[k].get("mean_acceptance_length") for k in q_keys])
    return out


def side_window(q, x, q_key, x_key):
    """The 40k window pair: validity + gain (the throughput rule needs both valid and no regression)."""
    oq, ox = window_ok(q.get(q_key)), window_ok(x.get(x_key))
    out = {"q_key": q_key, "x_key": x_key, "q_ok": oq[0], "x_ok": ox[0], "why": [oq[1], ox[1]],
           "parity": [((q.get(q_key) or {}).get("parity") or {}).get("label"), ((x.get(x_key) or {}).get("parity") or {}).get("label")]}
    if oq[0] and ox[0]:
        out["gain"] = round(x[x_key]["tok_s"] / q[q_key]["tok_s"] - 1.0, 4)
        out["tok_s"] = [q[q_key]["tok_s"], x[x_key]["tok_s"]]
        out["tick_ms"] = [q[q_key].get("tick_ms"), x[x_key].get("tick_ms")]
    return out


def window_brief(w):
    """The C12 read-back of one window: tok/s, tick, preemptions, KV usage, running == level, admission, parity."""
    w = w or {}
    return {k: w.get(k) for k in ("status", "level", "tok_s", "tick_ms", "cycle_ms", "running_min", "running_max", "running_eq_level",
                                  "preemptions", "kv_usage_mean", "kv_usage_max", "waiting_max", "mean_acceptance_length",
                                  "tok_s_per_stream", "capacity_predict")} | {
        "parity": (w.get("parity") or {}).get("label"), "admission_sec": (w.get("admission") or {}).get("sec")}


# ------------------------------------------------------------------ budget
def arm_fits(left_sec, arm):
    """An arm starts only with its full minimum left: est_min x 60 + START_RESERVE_SEC."""
    return left_sec >= arm["est_min"] * 60 + START_RESERVE_SEC


def body_cap_sec(cap_min, left_sec):
    """The body cap at body start: the arm's cap, clipped so teardown + analysis + verdict keep BODY_RESERVE_SEC."""
    return max(60.0, min(cap_min * 60.0, left_sec - BODY_RESERVE_SEC))


def launch_oom(rec):
    """True when any launch try of the arm died with an OOM in its log (no device fault), or the body died BODY_OOM_*."""
    rec = rec or {}
    if any(t.get("oom") and not t.get("device_fault") for t in (rec.get("tries") or []) if isinstance(t, dict)):
        return True
    return str(rec.get("status", "")).startswith("BODY_OOM")


def p27_dependency(p26):
    """(run, why): P27 moves one more layer to stage 0 than P26; a P26 OOM (any launch try, or BODY_OOM) skips P27."""
    if launch_oom(p26):
        tries = [{k: t.get(k) for k in ("launch_name", "kv_bytes", "status", "oom")} for t in (p26 or {}).get("tries") or []]
        return False, f"P26 OOM ({(p26 or {}).get('status')}, tries {tries}): P27 puts one more layer on stage 0"
    return True, "P26 did not OOM"


# ------------------------------------------------------------------ verdicts
def latest_q(rec):
    """The latest Q-defer read-back of an arm: q_readback_final (whole log after the server stopped) > q_readback_end > q_readback."""
    return next((rec[k] for k in ("q_readback_final", "q_readback_end", "q_readback") if isinstance((rec or {}).get(k), dict)), None)


def base_check(Q):
    """(ok, label, why): Q must be clean (DONE, gate PASS, env + config as planned, Q-defer ENGAGED). Parity is recorded and
    NEVER a reason to reject the base (NB4S rejected its B for a 7 / 1 split that every arm shared)."""
    if not Q or str(Q.get("status", "")).startswith("SKIPPED") or Q.get("status") is None:
        return False, "INCONCLUSIVE_NOT_RUN", f"Q {Q.get('status') if Q else None}"
    if Q.get("not_integrated") or Q.get("status") == "NOT_INTEGRATED":
        return False, "INCONCLUSIVE_BASE_NOT_INTEGRATED", str(Q.get("why") or (Q.get("q_readback") or {}).get("label"))
    if Q.get("quality_gate") != "PASS":
        return False, "INCONCLUSIVE_BASE_GATE", f"Q gate {Q.get('quality_gate')} / status {Q.get('status')}"
    if Q.get("status") != "DONE":
        return False, "INCONCLUSIVE_BASE_FAILED", f"Q status {Q.get('status')}"
    if not (Q.get("env_readback") or {}).get("ok"):
        return False, "INCONCLUSIVE_BASE_ENV", str((Q.get("env_readback") or {}).get("why"))
    if not (Q.get("config_readback") or {}).get("ok"):
        return False, "INCONCLUSIVE_BASE_CONFIG", str((Q.get("config_readback") or {}).get("why"))
    qr = latest_q(Q)
    if (qr or {}).get("label") != "ENGAGED":
        return False, "INCONCLUSIVE_BASE_NOT_ENGAGED", f"Q-defer read-back {(qr or {}).get('label')}"
    return True, "OK", "Q clean, Q-defer ENGAGED (parity recorded, not gating)"


FAILED_PREFIXES = ("BODY_OOM", "DEVICE_FAULT", "SERVER_DIED", "BOOT_CAP_KILL", "BODY_ERROR", "ERROR", "POSTPROCESS_ERROR",
                   "FAIL_GPU_RELEASE", "PARTIAL_HARD_STOP", "TIMEOUT")


def arm_verdict(Q, X, key, base=None, arms=None):
    """One change arm vs Q. Order: not run / skipped -> launch OOM -> not integrated -> correctness -> Q-defer engagement ->
    env / config read-back -> failed status -> oracle -> base -> speed (8k pair + 40k pair) -> stall / capacity facts."""
    base = base or base_check(Q)
    out = {"arm": key, "measure_only": key in MEASURE_ONLY}
    st = str((X or {}).get("status") or "")
    if not X or not st or st.startswith("SKIPPED"):
        v = ("SKIPPED_P26_OOM" if "P26_OOM" in st else "SKIPPED_BUDGET" if ("BUDGET" in st or not X) else
             "NOT_INTEGRATED" if (X or {}).get("not_integrated") else "SKIPPED_DEPENDENCY")
        out.update(verdict=v, why=(X or {}).get("why") or "not run")
        return out
    if st == "LAUNCH_FAILED" and launch_oom(X):
        out.update(verdict="LAUNCH_OOM", why=f"every launch try OOM: {X.get('tries')}", capacity=capacity_facts(X))
        return out
    if X.get("not_integrated") or st == "NOT_INTEGRATED":
        out.update(verdict="NOT_INTEGRATED", why=X.get("why") or (X.get("q_readback") or {}).get("label"))
        return out
    so = X.get("oracle_vs")
    if so == "FAIL_HIGH_MARGIN" or str(X.get("correctness", "")).startswith("NO_GO") or str(X.get("quality_gate", "")).startswith("FAIL"):
        out.update(verdict="CORRECTNESS_FAIL", why=f"oracle vs Q {so}, correctness {X.get('correctness')}, gate {X.get('quality_gate')}")
        return out
    if st in ("LAUNCH_FAILED", "GPU_NOT_EMPTY"):
        out.update(verdict=st, why=f"tries {X.get('tries')}", capacity=capacity_facts(X))
        return out
    if st.startswith(FAILED_PREFIXES):
        out.update(verdict="INCONCLUSIVE_ARM_FAILED", why=f"arm status {st} (died in {X.get('died_in')})", capacity=capacity_facts(X))
        return out
    qr = latest_q(X)
    if (qr or {}).get("label") != "ENGAGED":
        lab = (qr or {}).get("label")
        out.update(verdict=lab if lab in ("NOT_INTEGRATED", "ENGAGED_FALLBACK") else "NOT_ENGAGED",
                   why=f"Q-defer read-back {lab}: {(qr or {}).get('detail')}")
        return out
    if not (X.get("env_readback") or {}).get("ok"):
        out.update(verdict="NOT_ENGAGED", why="env read-back: " + str((X.get("env_readback") or {}).get("why")))
        return out
    if not (X.get("config_readback") or {}).get("ok"):
        out.update(verdict="NOT_ENGAGED", why="config read-back: " + str((X.get("config_readback") or {}).get("why")))
        return out
    if st != "DONE":
        out.update(verdict="INCONCLUSIVE_ARM_FAILED", why=f"arm status {st}")
        return out
    if so not in CLEAN:
        out.update(verdict="INCONCLUSIVE_ORACLE", why=f"oracle vs Q {so} (status {X.get('status')})")
        return out
    if not base[0]:
        out.update(verdict=base[1], why=base[2])
        return out
    qn = (arms or {}).get("Q", {}).get("streams", BASE_SEQS)
    xn = (arms or {}).get(key, {}).get("streams", C12_SEQS if key == "C12" else BASE_SEQS)
    qk, xk = window_keys(qn), window_keys(xn)
    spd = speed_compare(_win(Q), _win(X), qk[:2], xk[:2])
    side = side_window(_win(Q), _win(X), qk[3], xk[3])
    out["speed"], out["side_40k"] = spd, side
    out["pin"] = {"q": Q.get("pin"), "x": X.get("pin"), "differs": Q.get("pin") != X.get("pin"),
                  "note": "a lower pin only removes KV blocks; windows with 0 preemptions and running == level are comparable"}
    out["stall"] = stall_compare(Q, X)
    out["capacity"] = {"q": capacity_facts(Q), "x": capacity_facts(X)}
    if key in MEASURE_ONLY:
        out["c12"] = {"q": {k: window_brief(_win(Q).get(k)) for k in (qk[0], qk[1], qk[3])},
                      "x": {k: window_brief(_win(X).get(k)) for k in (xk[0], xk[1], xk[3])}}
    s = spd.get("status")
    if s == "PASS":
        if not (side["q_ok"] and side["x_ok"]):
            v = "INCONCLUSIVE_40K"
        elif (side.get("gain") or 0.0) < -SPEED_MIN:
            v = "REGRESSION_40K"
        else:
            v = "WIN"
    elif s in ("REGRESSION", "NO_GAIN"):
        v = s
    else:
        v = s or "INCONCLUSIVE_SPEED"
    out.update(verdict=v, why=f"8k {s} (gain {spd.get('gain')}, threshold {spd.get('threshold')}); 40k {side.get('gain')} "
                              f"(valid {side['q_ok']} / {side['x_ok']}); stall {out['stall'].get('status')} {out['stall'].get('class', '')}")
    return out


def throughput_eligible(v):
    """(ok, why): a lever arm with verdict WIN (8k PASS on both windows, both 40k windows valid, 40k gain >= -SPEED_MIN)."""
    if v.get("measure_only"):
        return False, "measure only (C12)"
    if v.get("verdict") != "WIN":
        return False, f"verdict {v.get('verdict')}"
    spd, side = v.get("speed") or {}, v.get("side_40k") or {}
    if spd.get("status") != "PASS":
        return False, f"8k speed {spd.get('status')}"
    if not (side.get("q_ok") and side.get("x_ok")):
        return False, f"40k window invalid {side.get('why')}"
    if (side.get("gain") or 0.0) < -SPEED_MIN:
        return False, f"40k regression {side.get('gain')}"
    return True, "ok"


def verdict(arms_recs, arms=None):
    """arms_recs: {"Q": rec, "I8": rec, ...} (missing = not run). Per-arm verdicts, the parity of every arm, the
    throughput winner among I8 / P26 / P27 (highest 8k gain among throughput-eligible arms) and the C12 measurement."""
    Q = arms_recs.get("Q") or {}
    base = base_check(Q)
    out = {"lib": LIB_VERSION, "base": {"ok": base[0], "label": base[1], "why": base[2]},
           "criteria": {"speed": f"8k gain > max(Q window-to-window spread, {SPEED_MIN:.0%}) on both 8k windows",
                        "winner": "lever arm (I8 / P26 / P27) with 8k PASS, both 40k windows valid, 40k gain >= "
                                  f"-{SPEED_MIN:.0%}; highest 8k gain wins",
                        "oracle": "vs Q EXACT or NEAR_TIE",
                        "parity": "recorded for every arm; a stall comparison needs the same parity label or modal split, "
                                  "else PARITY_DIFFERS; never a base reject; speed windows are not parity-gated"},
           "parity": {k: parity_of(r) for k, r in arms_recs.items() if isinstance(r, dict)}, "arms": {}}
    for k in ORDER[1:]:
        out["arms"][k] = arm_verdict(Q, arms_recs.get(k), k, base, arms)
    out["throughput_eligible"] = {k: dict(zip(("ok", "why"), throughput_eligible(v))) for k, v in out["arms"].items()}
    elig = [(-(v["speed"]["gain"]), k) for k, v in out["arms"].items() if out["throughput_eligible"][k]["ok"]]
    if elig:
        best = min(elig)
        out["best_arm"], out["best_verdict"] = best[1], "WIN"
        out["best_gain_8k"] = out["arms"][best[1]]["speed"]["gain"]
    else:
        out["best_arm"], out["best_verdict"] = None, ("NO_THROUGHPUT_WINNER" if base[0] else base[1])
    out["verdict"] = f"{out['best_arm']}:WIN" if out["best_arm"] else out["best_verdict"]
    c = out["arms"].get("C12") or {}
    out["c12"] = {"verdict": c.get("verdict"), "gain_8k": (c.get("speed") or {}).get("gain"),
                  "gain_40k": (c.get("side_40k") or {}).get("gain"), "windows": c.get("c12")}
    out["why"] = "; ".join(f"{k} {v['verdict']}" for k, v in out["arms"].items())
    return out
