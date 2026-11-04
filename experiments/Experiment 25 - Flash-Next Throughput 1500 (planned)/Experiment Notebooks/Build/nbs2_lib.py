"""E25 NB-S2 "capacity" library (2026-11-01): capacity levers on the DFlash2 base B = NBDF-K100's D3s (Qwen3.8-Flash-Next
klee100 3bpw-MTP on Merlin v3.0.0 ad42e37, 4 x L4, TP2 x PP2 + EP, FP8 QSA KV, seqs 24, pin 7,933,526,016).

Arms (one change each vs B; this order = the budget priority):
  B    NBDF-K100's D3s verbatim: the dflash_seam env (DFLASH_SEAM=1, DFLASH_DRAFT_SW_BLOCK=1024, mask-row sha), the trailing
       --speculative-config {"method":"dflash", <drafter>, K 3, probabilistic, BF16 drafter KV, FLASH_ATTN} + the trailing
       --compilation-config (multiples of 4 up to 24 x 4 = 96), no DRAFT_HEAD; Q-defer on; the A/A (oracle twice, ladder,
       teacher-forced fresh vs repeat, 8k@16 twice).
  IDX  B + --attention-config {"indexer_kv_dtype":"fp8"} (config/attention.py:88, indexer_qsa.py:154-166: bf16 / fp8 only;
       the JSON form: the --help flag gate checks the literal flag, and `--attention-config.indexer_kv_dtype` is not in it).
  S28  B with --max-num-seqs 28 and captures 4 .. 112 (multiples of 1 + K up to 28 x 4); windows at 28 where residency allows.
  DYN  B with K 4 + num_speculative_tokens_per_batch_size [[1, 8, 4], [9, 24, 3]] (config/speculative.py:493; scheduler.py:
       1779-1783 picks K per step; model_runner.py:2797-2803 truncates the drafts: method-generic) + captures covering both
       tiers (multiples of 5 up to 40 and of 4 from 36 to 96); the V2 runner keeps FULL graphs (config/vllm.py:1069-1076).
  LP   B + --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive
       (config/scheduler.py:79-90, arg_utils.py:1735-1740).
  S32  as S28 at 32 (captures 4 .. 128).
  SPC  B + --enable-mamba-shared-prefix-checkpoint (config/cache.py:198; kv_cache_manager.py:189-198: needs an eagle group
       (dflash = use_eagle, speculative.py:2011; the coordinator flags every group when none is marked,
       kv_cache_coordinator.py:129-135), partial hash hits and no re-prefillable tokens (single-module drafting)).
  Dropped: K4 (MTP-only: N/A_DFLASH), PP4 (the seam's aux relay is PP2-specific: taps [6, 16 | 26, 36, 46] on 2 stages:
       N/A_DFLASH).

Pure logic, stdlib only (+ nbdf_lib / nb4t_lib). Unit-tested locally (nbs2_unit_tests.py) and on the Py3.12 CPU kit.
"""
import copy
import json
import math
import re
import statistics

import nb4t_lib as T
import nbdf_lib as N

LIB_VERSION = "nbs2_lib 1.2 (2026-11-01, DFlash2 base B = NBDF-K100 D3s (closed e2198285); arms B > IDX > S28 > LP > S32 > SPC; CELL 1 arm switches NBS2_ON; DYN moved to NB-S1)"
BASE_KEY = "B"
LEVERS = ("IDX", "S28", "LP", "S32", "SPC")      # coordinator 2026-11-01: DYN moved to NB-S1
DROPPED = {"K4": "N/A_DFLASH: MTP K4 is an MTP-only arm (NB-S2 is DFlash2 only)",
           "PP4": "N/A_DFLASH: dflash_seam's aux-hidden relay is PP2-specific (taps [6, 16 | 26, 36, 46] split over 2 stages, "
                  "nbdf_lib PP0_TAPS / PP1_TAPS from PARTITION 25 / 23, dflash_readback needs PP x TP = 4 aux lines of that split)",
           "DYN": "MOVED_TO_NB_S1: the D4 / D3 dynamic-K schedule runs in NB-S1 (coordinator 2026-11-01); its helpers stay here, unused"}
DROPPED_LABEL = {"K4": "N/A_DFLASH", "PP4": "N/A_DFLASH", "DYN": "MOVED_TO_NB_S1"}
SEQ_ARMS = {"S28": 28, "S32": 32}
IDX_CFG = {"indexer_kv_dtype": "fp8"}
LP_ARGS = ["--max-num-batched-tokens", "2048", "--long-prefill-token-threshold", "1024", "--long-prefill-token-threshold-adaptive"]
SPC_ARGS = ["--enable-mamba-shared-prefix-checkpoint"]
DYN_SCHEDULE = [[1, 8, 4], [9, 24, 3]]
DYN_K = 4
STD_WINDOWS = [["8k", 16, "#1"], ["8k", 16, "#2"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]]
CAPPED_BANDS = ("20k", "40k", "90k", "116k")       # levels capped by THIS boot's resident bound (pool + F)
PORTS = {"B": 8111, "IDX": 8131, "S28": 8151, "DYN": 8171, "LP": 8191, "S32": 8211, "SPC": 8231}
EST = {"B": 50, "IDX": 38, "S28": 40, "DYN": 38, "LP": 41, "S32": 40, "SPC": 42}     # full minimum (min), launch .. teardown
CAP = {"B": 42, "IDX": 31, "S28": 33, "DYN": 31, "LP": 34, "S32": 33, "SPC": 35}     # body cap (min)
PREFIX_TEST_ARMS = ("B", "SPC")
SIBLING_ARMS = ("B", "SPC")
STALL_ARMS = ("B", "LP")
PROFILE_ARMS = ()                                   # NB-S1 profiles; NB-S2 measures capacity only
PRIMARY = {"IDX": "all", "SPC": "all", "S28": "capacity", "S32": "capacity", "DYN": "low_conc", "LP": "stall"}
SPEED_MIN = T.SPEED_MIN
DEADLINE_FAILS = ("LAUNCH_OOM", "NOT_INTEGRATED", "CONCURRENCY_CONTAMINATION", "DFLASH_ZERO_ACCEPT", "SMOKE_FAIL",
                  "CORRECTNESS_FAIL", "CORRECTNESS_FAIL_LONG", "CORRECTNESS_FAIL_TF", "LAUNCH_FAILED", "GPU_NOT_EMPTY",
                  "INCONCLUSIVE_ARM_FAILED", "QDEFER_FAILED", "PREFIX_HIT_ACCEPTANCE_COLLAPSE")


SWITCH_KEYS = (BASE_KEY,) + LEVERS                 # CELL 1's NBS2_ON keys (E25 NB-S2 arm switches)
SWITCH_DEFAULT = {k: True for k in SWITCH_KEYS}


def switch_map(on, keys=SWITCH_KEYS):
    """CELL 1's NBS2_ON -> {arm: bool} for every arm key: absent (None) -> every arm on; a dict -> bool(on[k]), a key missing
    from it -> that arm off. Anything else (not a dict) is an invalid edit: every arm on (recorded by switch_record)."""
    if on is None or not isinstance(on, dict):
        return {k: True for k in keys}
    return {k: bool(on.get(k, False)) for k in keys}


def switch_record(on, keys=SWITCH_KEYS):
    """What the run records about the switches (records only)."""
    return {"NBS2_ON": on if isinstance(on, (dict, type(None))) else repr(on)[:200], "effective": switch_map(on, keys),
            "absent": on is None, "invalid_not_a_dict": on is not None and not isinstance(on, dict),
            "unknown_keys": sorted(str(k) for k in set(on) - set(keys)) if isinstance(on, dict) else [],
            "off": [k for k, v in switch_map(on, keys).items() if not v]}


# ------------------------------------------------------------------ captures
def dflash_captures(seqs, k):
    """Every multiple of 1 + K up to seqs x (1 + K): every decode batch b = 1 .. seqs an exact FULL graph (nbdf_lib rule,
    without its 1024-token cap: 32 x 4 = 128 tokens is far below max_num_batched_tokens)."""
    return [b * (int(k) + 1) for b in range(1, int(seqs) + 1)]


def dense_schedule(schedule, max_seqs, k_max):
    """= Merlin's build_dynamic_sd_schedule_lookup (v1/spec_decode/dynamic/utils.py:77-150, the validation included):
    index b (1 .. max_seqs) -> the K the scheduler uses at b running requests (clamped to num_speculative_tokens)."""
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("schedule must be a non-empty list")
    parsed = []
    for e in schedule:
        if not isinstance(e, (list, tuple)) or len(e) != 3:
            raise ValueError("entry must be (start, end, k)")
        a, b, k = int(e[0]), int(e[1]), int(e[2])
        if a <= 0 or b <= 0 or a > b or k < 0:
            raise ValueError(f"bad entry {e}")
        parsed.append((a, b, k))
    parsed.sort(key=lambda x: x[0])
    prev = 0
    for a, b, _ in parsed:
        if a <= prev:
            raise ValueError("ranges must be non-overlapping")
        prev = b
    if parsed[0][0] != 1:
        raise ValueError("the first range must start at 1")
    dense = [0] * (max_seqs + 1)
    nxt, last = 1, None
    for a, b, k in parsed:
        if a > nxt and last is not None:
            for x in range(nxt, min(a, max_seqs + 1)):
                dense[x] = min(k_max, last)
        for x in range(max(a, nxt), min(b, max_seqs) + 1):
            dense[x] = min(k_max, k)
        nxt = max(nxt, b + 1)
        last = k
        if nxt > max_seqs:
            break
    for x in range(nxt, max_seqs + 1):
        dense[x] = min(k_max, last)
    return dense


def dyn_captures(schedule, seqs, k_max):
    """Token counts covering both tiers exactly: b x (1 + K(b)) for every b = 1 .. seqs (K(b) from the dense schedule)."""
    d = dense_schedule(schedule, seqs, k_max)
    return sorted({b * (1 + d[b]) for b in range(1, seqs + 1)})


def dyn_coverage(captures, schedule, seqs, k_max):
    """Per tier query length q = 1 + K(b): the V2 runner rounds each capture up to a multiple of q and keeps it when
    <= the largest capture (cudagraph_utils.py:332-347); every b must have an exact graph at its own q."""
    d = dense_schedule(schedule, seqs, k_max)
    caps = sorted(set(int(c) for c in captures))
    rows, bad = [], []
    for b in range(1, seqs + 1):
        q = 1 + d[b]
        g = sorted({-(-c // q) * q for c in caps if -(-c // q) * q <= caps[-1]})
        exact = b * q in g
        rows.append({"b": b, "k": d[b], "tokens": b * q, "exact": exact})
        if not exact:
            bad.append(b)
    return {"rows": rows, "uncovered_or_padded_b": bad, "covers_all": not bad, "tiers": sorted({r["k"] for r in rows})}


# ------------------------------------------------------------------ arm plan
def _split_tail(extra):
    """B's extra = base argv + [--speculative-config, spec, --compilation-config, captures] (nbdf_lib.arm_plan)."""
    x = list(extra)
    if len(x) < 4 or x[-4] != "--speculative-config" or x[-2] != "--compilation-config":
        raise ValueError(f"the base arm's trailing argv is not the DFlash pair: {x[-4:]}")
    return x[:-4], x[-4:]


def arm_plan(d3s, seam_env_k4=None, pin=7933526016, seqs=24):
    """B = NBDF-K100's D3s (deep copy, key B); every lever = B + exactly one change. Returns {key: arm} in priority order."""
    if d3s.get("method") != "dflash" or d3s.get("k") != 3 or d3s["env"].get(N.ENV_SW_BLOCK) != str(N.SW_BLOCK):
        raise ValueError("the base must be NBDF-K100's D3s (dflash, K 3, DFLASH_DRAFT_SW_BLOCK=1024)")
    base_extra, tail = _split_tail(d3s["extra"])
    b = copy.deepcopy(d3s)
    b.update(key=BASE_KEY, name="nbs2_b", port=PORTS["B"], parent="D3s", streams=int(seqs), long_aa=False, lever=False,
             title="B: NBDF-K100 D3s (DFlash2 K3, BF16 drafter KV, FLASH_ATTN, DFLASH_DRAFT_SW_BLOCK=1024, Q-defer, seqs 24)",
             windows_fixed=copy.deepcopy(STD_WINDOWS), est_min=EST["B"], cap_min=CAP["B"], optional=False, short_only=False,
             kv_bytes=int(pin), change=None, s2={"role": "base", "prefix_test": True, "sibling": True, "stall": True, "profile": False})
    b["expect_cfg"]["max_num_seqs"] = int(seqs)
    arms = {"B": b}

    def lever(key, title, extra=None, spec=None, captures=None, k=None, streams=None, windows=None, env=None, change=None):
        a = copy.deepcopy(b)
        a.update(key=key, name=f"nbs2_{key.lower()}", port=PORTS[key], lever=True, est_min=EST[key], cap_min=CAP[key],
                 title=f"{key}: B + {title}", change=change)
        if spec is not None:
            a["spec"] = spec
            a["k"] = int(k)
            a["expect_cfg"].update(num_speculative_tokens=int(k))
        if captures is not None:
            a["captures"] = list(captures)
            a["expect_cfg"]["captures"] = list(captures)
        if streams is not None:
            a["streams"] = int(streams)
            a["expect_cfg"]["max_num_seqs"] = int(streams)
        if env:
            a["env"].update(env)
            a["expect"] = {w: a["env"].get(w, N.WATCHED_DEFAULT.get(w)) for w in N.WATCHED}
        a["extra"] = list(extra) if extra is not None else list(b["extra"])
        if windows is not None:
            a["windows_fixed"] = windows
        a["coverage"] = N.coverage(a["captures"], a["streams"], a["k"]) if key != "DYN" else dyn_coverage(
            a["captures"], DYN_SCHEDULE, a["streams"], DYN_K)
        a["s2"] = {"role": "lever", "prefix_test": key in PREFIX_TEST_ARMS, "sibling": key in SIBLING_ARMS,
                   "stall": key in STALL_ARMS, "profile": key in PROFILE_ARMS, "primary": PRIMARY[key]}
        arms[key] = a
        return a

    idx_args = ["--attention-config", json.dumps(IDX_CFG, separators=(",", ":"))]
    lever("IDX", "QSA indexer K cache fp8 (--attention-config {\"indexer_kv_dtype\":\"fp8\"})", extra=list(b["extra"]) + idx_args,
          change={"kind": "argv", "argv": idx_args})
    for key in ("S28",):
        s = SEQ_ARMS[key]
        caps = dflash_captures(s, 3)
        ex = base_extra + [tail[0], tail[1], "--compilation-config", N.compilation_json(caps), "--max-num-seqs", str(s)]
        lever(key, f"--max-num-seqs {s}, captures 4 .. {caps[-1]}", extra=ex, captures=caps, streams=s,
              windows=[["8k", s, "#1"], ["8k", s, "#2"], ["20k", s, ""], ["40k", s, ""], ["90k", 8, ""], ["116k", 6, ""]],
              change={"kind": "seqs", "argv": ["--compilation-config", N.compilation_json(caps), "--max-num-seqs", str(s)]})
    dspec = dict(b["spec"], num_speculative_tokens=DYN_K, num_speculative_tokens_per_batch_size=copy.deepcopy(DYN_SCHEDULE))
    dcaps = dyn_captures(DYN_SCHEDULE, int(seqs), DYN_K)
    denv = None
    if seam_env_k4 is not None and dict(seam_env_k4) != {k_: v for k_, v in b["env"].items() if k_ in N.SEAM_ENV_KEYS}:
        denv = {k_: v for k_, v in dict(seam_env_k4).items() if k_ in N.SEAM_ENV_KEYS}
    if "DYN" in LEVERS:                         # not in NB-S2 (moved to NB-S1)
      lever("DYN", f"DFlash2 D4 at 1-8 running, D3 at 9-24 (K 4 + num_speculative_tokens_per_batch_size {DYN_SCHEDULE}), "
                 f"captures {dcaps[0]} .. {dcaps[-1]} (both tiers exact)",
          extra=base_extra + ["--speculative-config", N.spec_json(dspec), "--compilation-config", N.compilation_json(dcaps)],
          spec=dspec, captures=dcaps, k=DYN_K, env=denv,
          change={"kind": "spec", "argv": ["--speculative-config", N.spec_json(dspec), "--compilation-config", N.compilation_json(dcaps)],
                  "seam_env_k4_differs": denv is not None})
    lever("LP", "--max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive",
          extra=list(b["extra"]) + LP_ARGS, change={"kind": "argv", "argv": list(LP_ARGS)})
    for key in ("S32",):
        s = SEQ_ARMS[key]
        caps = dflash_captures(s, 3)
        ex = base_extra + [tail[0], tail[1], "--compilation-config", N.compilation_json(caps), "--max-num-seqs", str(s)]
        lever(key, f"--max-num-seqs {s}, captures 4 .. {caps[-1]}", extra=ex, captures=caps, streams=s,
              windows=[["8k", s, "#1"], ["8k", s, "#2"], ["20k", s, ""], ["40k", s, ""], ["90k", 8, ""], ["116k", 6, ""]],
              change={"kind": "seqs", "argv": ["--compilation-config", N.compilation_json(caps), "--max-num-seqs", str(s)]})
    lever("SPC", "--enable-mamba-shared-prefix-checkpoint", extra=list(b["extra"]) + SPC_ARGS, change={"kind": "argv", "argv": list(SPC_ARGS)})
    return arms


def check_arm_plan(arms, d3s):
    """Fail closed (ValueError): B == D3s except key / name / port / windows / budget fields; each lever = B + exactly its
    change (env equal, argv = B's argv + the change, or the trailing pair replaced for S28 / S32 / DYN); never-do; ports
    distinct; captures exact for every decode batch (DYN: per tier); pins equal."""
    errs = []
    if tuple(arms) != ("B",) + LEVERS or "B" not in arms:
        raise ValueError(f"NBS2_ARM_PLAN_INVALID: order {tuple(arms)} != {('B',) + LEVERS}")
    b = arms["B"]
    for f in ("env", "extra", "spec", "captures", "k", "method", "kv_bytes", "q_label", "needs_seam", "expect"):
        if b.get(f) != d3s.get(f) and not (f == "kv_bytes"):
            errs.append(f"B.{f} != D3s.{f}")
    base_extra, tail = _split_tail(b["extra"])
    for key, a in arms.items():
        v = T.never_do_violations(a["env"], a["extra"])
        if v:
            errs.append(f"{key}: never-do {v}")
        if a["kv_bytes"] != b["kv_bytes"]:
            errs.append(f"{key}: pin {a['kv_bytes']} != B's {b['kv_bytes']}")
        if a.get("method") != "dflash" or a["env"].get(N.ENV_SEAM) != "1" or a["env"].get(N.ENV_SW_BLOCK) != str(N.SW_BLOCK):
            errs.append(f"{key}: not a DFlash2 SW-1024 arm")
        if not (a.get("coverage") or {}).get("covers_all") or ((a.get("coverage") or {}).get("padded_b") or []):
            errs.append(f"{key}: decode batches without an exact FULL graph {a.get('coverage')}")
        if a.get("parent") != "D3s" or not a.get("needs_patch") or a.get("q_label") != N.Q_LABEL:
            errs.append(f"{key}: parent / Q-defer / patched tree")
        if "--language-model-only" not in a["extra"]:
            errs.append(f"{key}: --language-model-only missing")
        if T.last_value(a["extra"], "--max-num-seqs") != str(a["expect_cfg"]["max_num_seqs"]) or a["streams"] != a["expect_cfg"]["max_num_seqs"]:
            errs.append(f"{key}: --max-num-seqs {T.last_value(a['extra'], '--max-num-seqs')} != {a['expect_cfg']['max_num_seqs']}")
        if N.captures_from_argv(a["extra"]) != a["captures"]:
            errs.append(f"{key}: trailing captures != plan")
        if (T.spec_from_argv(a["extra"]) or {}) != a["spec"]:
            errs.append(f"{key}: trailing spec != plan")
        if key == "B":
            continue
        ch = (a.get("change") or {})
        ed = {x for x in set(a["env"]) | set(b["env"]) if a["env"].get(x) != b["env"].get(x)}
        if ed and not (key == "DYN" and ed <= set(N.SEAM_ENV_KEYS)):
            errs.append(f"{key}: env differs from B by {sorted(ed)}")
        if ch.get("kind") == "argv":
            ok = a["extra"][:len(b["extra"])] == b["extra"] and a["extra"][len(b["extra"]):] == ch.get("argv")
        elif ch.get("kind") == "seqs":
            ok = a["extra"][:len(base_extra) + 2] == base_extra + tail[:2] and a["extra"][len(base_extra) + 2:] == ch.get("argv") \
                and a["spec"] == b["spec"]
        elif ch.get("kind") == "spec":
            ok = a["extra"][:len(base_extra)] == base_extra and a["extra"][len(base_extra):] == ch.get("argv")
            sp = dict(a["spec"])
            ok = ok and sp.pop("num_speculative_tokens_per_batch_size", None) == DYN_SCHEDULE and \
                dict(sp, num_speculative_tokens=3) == b["spec"] and a["k"] == DYN_K
        else:
            ok = False
        if not ok:
            errs.append(f"{key}: argv / spec is not B + exactly its change ({ch.get('kind')})")
        if key in SEQ_ARMS and (a["streams"] != SEQ_ARMS[key] or a["captures"][-1] != SEQ_ARMS[key] * 4):
            errs.append(f"{key}: seqs / captures")
        if key == "DYN":
            try:
                d = dense_schedule(a["spec"]["num_speculative_tokens_per_batch_size"], a["streams"], a["spec"]["num_speculative_tokens"])
            except ValueError as e:
                errs.append(f"DYN schedule invalid: {e}")
            else:
                if d[1:9] != [4] * 8 or d[9:25] != [3] * 16:
                    errs.append(f"DYN dense schedule {d[1:]}")
    if len({a["port"] for a in arms.values()}) != len(arms) or max(a["port"] for a in arms.values()) + 4 > 8355:
        errs.append("ports not distinct / outside Merlin's 8000..8355 scheme")
    if errs:
        raise ValueError("NBS2_ARM_PLAN_INVALID: " + "; ".join(errs))
    return True


def budget_walk(arms, hard_stop_min, setup_min=20.0, minutes=None):
    """The planned walk (recorded + unit-tested): which arms start with est_min + 5 min left, given per-arm run minutes."""
    left = hard_stop_min - setup_min
    walk = []
    for k, a in arms.items():
        fits = left * 60 >= a["est_min"] * 60 + T.START_RESERVE_SEC
        walk.append({"arm": k, "left_min": round(left, 1), "est_min": a["est_min"], "runs": fits})
        if fits:
            left -= (minutes or {}).get(k, a["est_min"])
    return {"walk": walk, "left_end_min": round(left, 1), "runs": [w["arm"] for w in walk if w["runs"]],
            "skipped": [w["arm"] for w in walk if not w["runs"]]}


# ------------------------------------------------------------------ real-failure read-backs (this arm only)
def idx_readback(workers):
    """IDX engaged: the QSA indexer compressed-key cache spec (MLAAttentionSpec, example ...compressed_key_cache) is fp8 on
    every rank that has one (nb1_cache_info kv_cache_specs). (engaged True / False / None = no read-back, recorded)."""
    ranks = (workers or {}).get("ranks") or []
    seen = []
    for w in ranks:
        if not isinstance(w, dict):
            continue
        for s in w.get("kv_cache_specs") or []:
            if isinstance(s, dict) and "compressed_key_cache" in str(s.get("example") or ""):
                seen.append({"rank": w.get("rank"), "type": s.get("type"), "dtype": str(s.get("dtype")),
                             "page_size_bytes": s.get("page_size_bytes")})
    if not seen:
        return {"engaged": None, "why": "no compressed_key_cache spec in the worker read-back (recorded)", "specs": []}
    fp8 = [x for x in seen if "float8" in x["dtype"] or "fp8" in x["dtype"].lower()]
    return {"engaged": len(fp8) == len(seen), "specs": seen[:8],
            "why": "ok" if len(fp8) == len(seen) else f"indexer K cache dtype {sorted({x['dtype'] for x in seen})} != fp8 (silent bf16 path)"}


_DYN_OVERRIDE = "Dynamic speculative decoding changes the target verification length at runtime. Overriding cudagraph_mode"


def dyn_graph_readback(log_text, captures):
    """DYN real failure (vLLM #49548 class): the V2 runner must keep FULL graphs (config/vllm.py:1069-1076). FAIL on the
    PIECEWISE override warning, on target graph lines without any FULL mode, or on a logged cudagraph_mode that is not
    FULL / FULL_AND_PIECEWISE. A missing line is INCONCLUSIVE (recorded)."""
    t = log_text or ""
    gc = N.graph_counts(t)
    modes = sorted({m for v in gc.values() for m in v})
    eng = [l for l in t.splitlines() if "Initializing a V1 LLM engine" in l or "non-default args:" in l]   # the target engine's config lines
    logged = [m for l in eng for m in re.findall(r"cudagraph_mode['\"]?[:=]\s*['\"<]?(?:CUDAGraphMode\.)?([A-Z_]+)", l)]
    out = {"graph_counts": gc, "modes": modes, "logged_mode": sorted(set(logged)), "override_warning": _DYN_OVERRIDE in t}
    if out["override_warning"]:
        return dict(out, status="FAIL", why="the dynamic-SD PIECEWISE override fired (no FULL graphs)")
    bad = [m for m in out["logged_mode"] if m in ("PIECEWISE", "NONE")]
    if bad:
        return dict(out, status="FAIL", why=f"cudagraph_mode {bad}")
    if gc.get("target") and not any(m.startswith("FULL") for m in gc["target"]):
        return dict(out, status="FAIL", why=f"target graphs without FULL: {gc['target']}")
    if not gc:
        return dict(out, status="INCONCLUSIVE", why="no graph-capture lines (recorded)")
    return dict(out, status="PASS", why="FULL graphs kept")


def k_eff(w):
    """Drafted tokens per draft in a window = accept_per_draft / accept_rate (steady(): acc / drafts and acc / draft tokens)."""
    a, r = (w or {}).get("accept_per_draft"), (w or {}).get("accept_rate")
    if isinstance(a, (int, float)) and isinstance(r, (int, float)) and r > 0:
        return round(a / r, 3)
    return None


def dyn_tiers(rec, schedule=None, seqs=24, k_max=DYN_K):
    """Per window: running level, the schedule's K, the measured K_eff (recorded: tier engagement)."""
    d = dense_schedule(schedule or DYN_SCHEDULE, max(seqs, 1), k_max)
    out = {}
    for key, w in (T._win(rec) or {}).items():
        if not isinstance(w, dict) or key.startswith("profile") or not w.get("level"):
            continue
        lv = int(w["level"])
        want = d[min(lv, seqs)]
        ke = k_eff(w)
        out[key] = {"level": lv, "k_schedule": want, "k_eff": ke,
                    "engaged": None if ke is None else abs(ke - want) <= 0.35}
    return out


# ------------------------------------------------------------------ the KV ledger (metrics read; records only)
_METRIC = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{([^}]*)\})?\s+(\S+)", re.M)
LEDGER_KEYS = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting",
               "vllm:num_preemptions_total", "vllm:prompt_tokens_total", "vllm:prefix_cache_hits_total",
               "vllm:prompt_tokens_cached_total")


def parse_metrics(text):
    """The ledger fields of one /metrics read: the plain counters / gauges (summed over engines) + the labelled
    prompt_tokens_by_source{source} and num_requests_waiting_by_reason{reason}."""
    out = {"by_source": {}, "waiting_by_reason": {}}
    for m in _METRIC.finditer(text or ""):
        name, labels, val = m.group(1), m.group(3) or "", m.group(4)
        try:
            v = float(val)
        except ValueError:
            continue
        if v != v:
            continue
        if name in LEDGER_KEYS:
            out[name] = out.get(name, 0.0) + v
        elif name == "vllm:prompt_tokens_by_source_total":
            s = re.search(r'source="([^"]+)"', labels)
            if s:
                out["by_source"][s.group(1)] = out["by_source"].get(s.group(1), 0.0) + v
        elif name == "vllm:num_requests_waiting_by_reason":
            s = re.search(r'reason="([^"]+)"', labels)
            if s:
                out["waiting_by_reason"][s.group(1)] = out["waiting_by_reason"].get(s.group(1), 0.0) + v
    return out


def ledger_summary(samples, t0=None, t1=None, pool=None, level=None, band_tokens=None, F=None, block=4096, gen_tokens=0):
    """samples = [(t, parse_metrics dict)]. Over the window [t0, t1] (else all): KV usage high-water, preemptions, recompute
    tokens (the 'computed' prompt tokens added inside a steady window: no new request is admitted there, so they are
    re-prefill after a preemption), waiting by reason (max); the block ledger (blocks in use = usage x pool; free = pool -
    in use; running-required ESTIMATE = level x (ceil((prompt + generated) / block) + F); reclaimable = NOT_READABLE in
    Merlin ad42e37 (no metric))."""
    sel = [(t, m) for t, m in samples if (t0 is None or t >= t0 - 0.01) and (t1 is None or t <= t1 + 2.5)]
    if not sel:
        return {"samples": 0}
    us = [m.get("vllm:kv_cache_usage_perc") for _, m in sel if isinstance(m.get("vllm:kv_cache_usage_perc"), (int, float))]
    first, last = sel[0][1], sel[-1][1]

    def d(key, sub=None):
        a = (first.get(sub) or {}).get(key) if sub else first.get(key)
        b = (last.get(sub) or {}).get(key) if sub else last.get(key)
        return round(b - a, 1) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
    wr = {}
    for _, m in sel:
        for r, v in (m.get("waiting_by_reason") or {}).items():
            wr[r] = max(wr.get(r, 0.0), v)
    out = {"samples": len(sel), "kv_usage_max": max(us) if us else None, "kv_usage_mean": round(statistics.mean(us), 4) if us else None,
           "preemptions": d("vllm:num_preemptions_total"), "recompute_tokens": d("computed", "by_source"),
           "local_cache_hit_tokens": d("local_cache_hit", "by_source"), "prompt_tokens": d("vllm:prompt_tokens_total"),
           "waiting_by_reason_max": wr,
           "running_max": max((m.get("vllm:num_requests_running") or 0) for _, m in sel)}
    if isinstance(pool, (int, float)) and pool > 0 and us:
        out["blocks_in_use_max"] = int(round(max(us) * pool))
        out["free_blocks_min"] = int(round(pool - max(us) * pool))
    if level and band_tokens and isinstance(F, (int, float)):
        per = math.ceil((band_tokens + (gen_tokens or 0) / max(1, level)) / block) + int(F)
        out["running_required_blocks_est"] = int(level * per)
        out["running_required_note"] = "ESTIMATE: level x (ceil((prompt + generated per stream) / block) + F)"
    out["reclaimable_blocks"] = "NOT_READABLE (no metric in Merlin ad42e37)"
    return out


# ------------------------------------------------------------------ the prefill-stall probe (LP) and the sibling probe (SPC)
def stall_stats(series, t_inject, t_done):
    """series = {stream: [(t, n_tokens)]} polled every ~20 ms. Gap = time between two token-count increases of one decoding
    stream. Baseline = gaps ending before t_inject, injection = gaps ending in [t_inject, t_done]. p50 / p95 / max in ms."""
    base, inj = [], []
    for pts in (series or {}).values():
        last_t, last_n = None, None
        for t, n in pts:
            if last_n is not None and n > last_n:
                gap = (t - last_t) * 1000.0
                (base if t < t_inject else inj if t <= t_done else []).append(gap)
            if last_n is None or n > last_n:
                last_t, last_n = t, n

    def q(xs, p):
        if not xs:
            return None
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(math.ceil(p * len(xs))) - 1)], 1)
    return {"baseline": {"gaps": len(base), "p50_ms": q(base, 0.5), "p95_ms": q(base, 0.95), "max_ms": round(max(base), 1) if base else None},
            "injection": {"gaps": len(inj), "p50_ms": q(inj, 0.5), "p95_ms": q(inj, 0.95), "max_ms": round(max(inj), 1) if inj else None},
            "p95_ratio": (round(q(inj, 0.95) / q(base, 0.95), 3) if base and inj and q(base, 0.95) else None)}


def sibling_label(rec, block):
    """SPC activation read-back (recorded): prefix-hit tokens of the third sibling C (= P + Z after P + X and P + Y).
    Without the junction checkpoint the hit stops at the last whole block of P (Mamba align state); with it, near len(P)."""
    p_len = (rec or {}).get("prefix_tokens")
    hc = (rec or {}).get("hits", {}).get("C")
    if not isinstance(p_len, int) or not isinstance(hc, (int, float)) or not isinstance(block, int) or block <= 0:
        return {"label": "INCONCLUSIVE_SIBLING", "why": "missing numbers"}
    base = (p_len // block) * block
    lab = "SPC_JUNCTION_HIT" if hc >= base + 128 else ("BLOCK_BOUNDARY_ONLY" if hc >= base - 64 else "BELOW_BLOCK_BOUNDARY")
    return {"label": lab, "prefix_tokens": p_len, "block": block, "block_boundary_hit": base, "c_hits": hc}


# ------------------------------------------------------------------ verdict
def _valid(w):
    return T.window_ok(w)[0]


def band_compare(B, X, band, thr_floor=None):
    """Per band: B's and X's valid windows (any level), mean tok/s, gain, threshold max(B's 8k A/A spread, 2 %, floor)."""
    bw = {k: v for k, v in T._win(B).items() if isinstance(v, dict) and k.split("@")[0] == band and _valid(v)}
    xw = {k: v for k, v in T._win(X).items() if isinstance(v, dict) and k.split("@")[0] == band and _valid(v)}
    out = {"band": band, "b_windows": sorted(bw), "x_windows": sorted(xw)}
    if not bw or not xw:
        out.update(status="INCONCLUSIVE", why=f"valid windows B {len(bw)} / X {len(xw)}")
        return out
    bt, xt = [v["tok_s"] for v in bw.values()], [v["tok_s"] for v in xw.values()]
    bm, xm = statistics.mean(bt), statistics.mean(xt)
    aa = aa_spread(B)
    thr = max(aa or 0.0, SPEED_MIN, float((thr_floor or {}).get(band) or 0.0))
    gain = xm / bm - 1.0
    out.update(b_tok_s=[round(x, 1) for x in bt], x_tok_s=[round(x, 1) for x in xt], b_mean=round(bm, 1), x_mean=round(xm, 1),
               b_level=sorted({v.get("level") for v in bw.values()}), x_level=sorted({v.get("level") for v in xw.values()}),
               gain=round(gain, 4), threshold=round(thr, 4),
               b_al=[v.get("mean_acceptance_length") for v in bw.values()], x_al=[v.get("mean_acceptance_length") for v in xw.values()],
               status="PASS" if gain > thr else ("REGRESSION" if gain < -thr else "NO_REGRESSION"))
    out["levels_differ"] = out["b_level"] != out["x_level"]
    return out


def aa_spread(B):
    w = T._win(B)
    v = [w.get(k) for k in ("8k@16#1", "8k@16#2")]
    if all(_valid(x) for x in v):
        a, b = v[0]["tok_s"], v[1]["tok_s"]
        return round(abs(a - b) / ((a + b) / 2), 4)
    return None


BANDS = ("8k", "20k", "40k", "90k", "116k")


def primary_verdict(kind, bands, X=None, B=None, stall_ratio_max=0.9):
    """The lever's own question, after N.arm_verdict reached its speed section:
      capacity (S28 / S32): 20k at the arm's level vs B's 20k@24 PASS (8k / 40k recorded), no long-band REGRESSION;
      low_conc (DYN): 90k@8 or 116k@6 PASS (the K4 tier), no REGRESSION at 8k / 20k / 40k (the K3 tier);
      stall (LP): the injection p95 stall ratio X / B <= 0.9, no band REGRESSION;
      all (IDX / SPC): any band PASS and none REGRESSION."""
    reg = [b for b, c in bands.items() if c.get("status") == "REGRESSION"]
    ok = {b: c.get("status") for b, c in bands.items()}
    if kind == "capacity":
        if reg:
            return "REGRESSION", f"regression at {reg}"
        st = ok.get("20k")
        return ("WIN" if st == "PASS" else "NO_GAIN" if st == "NO_REGRESSION" else "INCONCLUSIVE_SPEED"), f"20k {st}; bands {ok}"
    if kind == "low_conc":
        hi = [b for b in ("8k", "20k", "40k") if ok.get(b) == "REGRESSION"]
        if hi:
            return "REGRESSION", f"K3-tier regression at {hi}"
        lo = [b for b in ("90k", "116k") if ok.get(b) == "PASS"]
        if any(ok.get(b) == "REGRESSION" for b in ("90k", "116k")):
            return "REGRESSION", f"K4-tier regression {ok}"
        if lo:
            return "WIN", f"K4 tier PASS at {lo}; bands {ok}"
        if all(ok.get(b) in (None, "INCONCLUSIVE") for b in ("90k", "116k")):
            return "INCONCLUSIVE_SPEED", f"no valid 90k / 116k pair: {ok}"
        return "NO_GAIN", f"bands {ok}"
    if kind == "stall":
        if reg:
            return "REGRESSION", f"regression at {reg}"
        xs = ((X or {}).get("stall_probe") or {}).get("stats") or {}
        bs = ((B or {}).get("stall_probe") or {}).get("stats") or {}
        xp, bp = (xs.get("injection") or {}).get("p95_ms"), (bs.get("injection") or {}).get("p95_ms")
        if not isinstance(xp, (int, float)) or not isinstance(bp, (int, float)) or bp <= 0:
            return "INCONCLUSIVE_STALL", f"injection p95 X {xp} / B {bp}"
        r = xp / bp
        return ("WIN" if r <= stall_ratio_max else "NO_GAIN"), f"injection p95 stall {xp} ms vs B {bp} ms (ratio {r:.3f}); bands {ok}"
    if reg:
        return "REGRESSION", f"regression at {reg}"
    win = [b for b, s in ok.items() if s == "PASS"]
    if win:
        return "WIN", f"PASS at {win}; bands {ok}"
    if all(s in (None, "INCONCLUSIVE") for s in ok.values()):
        return "INCONCLUSIVE_SPEED", f"bands {ok}"
    return "NO_GAIN", f"bands {ok}"


def lever_verdict(B, X, key, base=None, arms=None):
    """nbdf_lib.arm_verdict (every real-failure gate, the TF envelope, TF_REQUIRED, the A/A-relative oracle) with B as its
    'Q'; when it reached its speed section, the lever's primary question decides (TF_NOT_VERIFIED -> INCONCLUSIVE_TF)."""
    v = N.arm_verdict(B, X, key, base, arms)
    if "speed" not in v:
        return v
    bands = {b: band_compare(B, X, b, N.LONG_THR_FLOOR) for b in BANDS}
    v["bands"] = bands
    v["nbdf_style_verdict"] = v.get("speed_verdict") or v.get("verdict")
    kind = PRIMARY.get(key, "all")
    pv, why = primary_verdict(kind, bands, X, B)
    v["primary"] = {"kind": kind, "verdict": pv, "why": why}
    if pv == "WIN" and v.get("tf_verification_pending"):
        v["speed_verdict"], pv = pv, "INCONCLUSIVE_TF"
    if pv == "WIN" and (X or {}).get("admission_deadlock"):
        v["ranked_verdict"], pv = pv, "ADMISSION_DEADLOCK"
    v.update(verdict=pv, why=why + (" | TF not verified" if v.get("tf_verification_pending") else ""))
    return v


def verdict(recs, arms):
    """Every lever vs B; the throughput winner among WIN (highest primary gain); the arm table; dropped arms recorded."""
    B = recs.get("B") or {}
    base = N.base_usable(B)
    out = {"lib": LIB_VERSION, "base": {"ok": base[0], "label": base[1], "why": base[2], "role": "recorded; levers run anyway"},
           "dropped": dict(DROPPED), "arms": {}, "aa_spread_8k": aa_spread(B),
           "criteria": {"real_failures": "crash / garbage / request errors / zero acceptance / NaN / a real fallback (IDX not fp8, "
                                         "DYN graphs not FULL, wrong spec) / missing init / TF outside B's A/A envelope",
                        "tf_required": "no WIN without TF PASS at 2k / 8k / 40k vs B's A/A (TF_REQUIRED)",
                        "primary": PRIMARY}}
    for k in LEVERS:
        out["arms"][k] = lever_verdict(B, recs.get(k), k, base, arms)
        if k == "DYN" and isinstance(recs.get(k), dict):
            out["arms"][k]["tiers"] = dyn_tiers(recs[k])
    for k, why in DROPPED.items():
        out["arms"][k] = {"arm": k, "verdict": DROPPED_LABEL.get(k, "N/A_DFLASH"), "why": why}
    elig = []
    for k in LEVERS:
        v = out["arms"][k]
        if v.get("verdict") == "WIN":
            g = [c.get("gain") for c in (v.get("bands") or {}).values() if c.get("status") == "PASS" and isinstance(c.get("gain"), (int, float))]
            elig.append((-(max(g) if g else 0.0), k))
    out["winners"] = [k for _, k in sorted(elig)]
    out["best_arm"] = out["winners"][0] if out["winners"] else None
    out["best_verdict"] = "WIN" if out["best_arm"] else "NO_THROUGHPUT_WINNER"
    out["best_gain_8k"] = None
    out["verdict"] = f"{out['best_arm']}:WIN" if out["best_arm"] else out["best_verdict"]
    out["table"] = N.arm_table({k: r for k, r in recs.items() if isinstance(r, dict)})
    out["why"] = "; ".join(f"{k} {v.get('verdict')}" for k, v in out["arms"].items())
    return out
