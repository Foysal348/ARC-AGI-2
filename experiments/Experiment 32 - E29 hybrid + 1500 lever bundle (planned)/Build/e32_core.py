"""E32 core (2026-11-02): the 1500 lever bundle on the E29_DFLASH_D3S engine. Pure logic, stdlib only.

The builder (build_e32_levers.py) writes e32_lib.py = a verbatim cut of E25 nbd1_lib.py (the NB-D1 per-lever argv / env /
captures / read-back helpers, sha-pinned) + this file. The notebook imports e32_lib from the E32 payload dataset; e32_nbx.py
(the notebook glue, executed into the notebook namespace) calls it. Nothing here touches the notebook state.

Levers (one plain literal line per lever at the top of CELL 1, E32_<LEVER> = value; OFF = the E29_DFLASH_D3S engine):
  SEQS  E32_SEQS = 24 | 28 | 32      --max-num-seqs + the DFlash captures of the D tries (the S28 / S32 pair = one switch)
  SPEC  E32_SPEC = 'K3' | 'D4s' | 'DYN'   DFlash2 K3 (E29) | K4 | K4 + num_speculative_tokens_per_batch_size (one switch)
  LP    E32_LP   = bool              --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive
  SPC   E32_SPC  = bool              --enable-mamba-shared-prefix-checkpoint
  IDX   E32_IDX  = bool              --attention-config {"indexer_kv_dtype":"fp8"}
  G1    E32_G1   = bool              VLLM_GDN_DECODE_FASTPATH=1
  G2    E32_G2   = bool              VLLM_GDN_SHARED_ALIGN_INDEX=1
A bad value = that lever OFF (recorded as invalid), never a stop. The levers act on the DFlash tries only; every MTP try
(the E29 fallback) is the E29 config exactly.

Drop rule (real failures only: crash / OOM / a launch the server rejects / the drafter not engaged / a zero-accept canary /
the DYN fallback lines): ONE lever per failed try, the same D rung again (pick):
  1. SPEC = 'DYN' and a DYN fallback line in the log ('Overriding cudagraph_mode' / 'Disabling
     num_speculative_tokens_per_batch_size', NB-S1) -> SPEC
  2. a lever whose own flag / env text is in an error line (Error / Traceback / error: / FAIL_ / unrecognized arguments) -> it
  3. E29_DFLASH_CONFIG_MISMATCH: a failing seqs check -> SEQS, a failing spec check -> SPEC; the capacity gate
     NO_GO_INSUFFICIENT_KV -> SEQS
  4. memory (an OOM line that is not a device fault, or the F2 probe's E29_DFLASH_LOW_FREE): MEMORY_ORDER; no memory lever
     left -> None (E29's rule: the next D rung)
  5. anything else (not ready, launch failed, probes, warm-up, death before the phases, a device fault): GENERAL_ORDER
With no E32 lever on, pick returns None and E29's own rules run unchanged (D ladder, MTP K3 fallback, I8, E28 ladder).
"""
import copy
import hashlib
import json
import re

LIB_VERSION = "e32_lib 1.0 (2026-11-02)"
LEVERS = ("SEQS", "SPEC", "LP", "SPC", "IDX", "G1", "G2")
SWITCH_VAR = {k: "E32_" + k for k in LEVERS}
OFF = {"SEQS": 24, "SPEC": "K3", "LP": False, "SPC": False, "IDX": False, "G1": False, "G2": False}
CHOICES = {"SEQS": (24, 28, 32), "SPEC": ("K3", "D4s", "DYN")}
DEFAULT = {"SEQS": 28, "SPEC": "DYN", "LP": True, "SPC": True, "IDX": True, "G1": True, "G2": True}
GENERAL_ORDER = ("SPEC", "SPC", "IDX", "G1", "G2", "LP", "SEQS")   # newest / riskiest code path first
MEMORY_ORDER = ("LP", "SEQS", "SPEC", "SPC")                       # the levers that add activation / graph / state memory
MEMORY_STATUS = ("E29_DFLASH_LOW_FREE",)                          # the F2 probe: a worker below 512 MiB free
NAMED = {"SPEC": ("num_speculative_tokens", "cudagraph_capture_sizes"),
         "SPC": ("shared_prefix_checkpoint", "shared-prefix-checkpoint"),
         "IDX": ("indexer_kv_dtype", "attention_config", "attention-config"),
         "G1": ("gdn_decode_fastpath",), "G2": ("gdn_shared_align",),
         "LP": ("long_prefill_token_threshold", "long-prefill-token-threshold", "max_num_batched_tokens", "max-num-batched-tokens"),
         "SEQS": ("max_num_seqs", "max-num-seqs", "cudagraph_capture_sizes")}
ERR_LINE = re.compile(r"Error|Traceback|error:|FAIL_|unrecognized arguments|invalid choice")
CODE = {"LP": "l", "SPC": "p", "IDX": "i", "G1": "f", "G2": "a"}
SPEED_BANDS = {"8k": 8192, "20k": 20000, "40k": 40000, "90k": 90000, "116k": 116000}
SPEED_DEFAULT = {"on": True, "windows": [["8k", 16], ["20k", 24], ["90k", 8], ["116k", 6]], "cap_min": 15}
SPEED_REQUIRED = ("90k", "116k")                                  # user rule: the 90k and ~116k bands always run
FP_ENV_PREFIX = ("VLLM_", "DFLASH_", "NCCL_", "NB4_", "DRAFT_", "KV_BYTES", "P2P", "SCALES", "MERLIN_", "PYTORCH_CUDA_ALLOC",
                 "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
MERLIN_COMMIT_PIN = "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5"     # the Merlin v3.0.0 tree the #60490 patch / seam pins expect
_MISSING = object()


# ------------------------------------------------------------------ switches
def _valid(k, v):
    if k in CHOICES:
        return type(v) is type(OFF[k]) and v in CHOICES[k]
    return type(v) is bool


def parse(sw, harness_json=None):
    """The CELL 1 switch values {lever: value} -> {"cfg": {lever: value}, "on": [...], "invalid": {lever: why}}. A missing /
    bad value = that lever OFF (recorded); never raises. harness_json (kits only) overrides keys."""
    sw = dict(sw or {})
    over = {}
    if harness_json:
        try:
            over = {k: v for k, v in json.loads(harness_json).items() if k in LEVERS}
        except Exception as e:
            over = {}
            sw["_harness_error"] = repr(e)[:200]
    sw.update(over)
    cfg, invalid, given = {}, {}, {}
    for k in LEVERS:
        v = sw.get(k, _MISSING)
        given[k] = "<missing>" if v is _MISSING else v
        if v is not _MISSING and _valid(k, v):
            cfg[k] = v
        else:
            cfg[k] = OFF[k]
            invalid[k] = ("missing" if v is _MISSING else f"bad value {v!r} (allowed: {CHOICES.get(k, (True, False))})")[:200]
    return {"cfg": cfg, "on": [k for k in LEVERS if cfg[k] != OFF[k]], "invalid": invalid, "given": given,
            "harness_override": sorted(over), "lib": LIB_VERSION}


def is_on(cur, k):
    return cur.get(k, OFF[k]) != OFF[k]


def on_list(cur):
    return [k for k in LEVERS if is_on(cur, k)]


# ------------------------------------------------------------------ argv / env / the DFlash plan of one try
def lever_argv(cur):
    """The argv of the on argv-levers, in a fixed order (before the DFlash tail; the tail comes last: last wins)."""
    out = []
    if cur.get("LP"):
        out += list(LP_ARGS)
    if cur.get("SPC"):
        out += list(SPC_ARGS)
    if cur.get("IDX"):
        out += list(IDX_ARGS)
    return out


def lever_env(cur):
    return {GDN_ENV[k]: "1" for k in ("G1", "G2") if cur.get(k)}


def dyn_schedule(seqs):
    """NB-D1's DYN_SCHEDULE [[1, 8, 4], [9, 24, 3]]; with more seqs the last range ends at seqs (the same dense table as
    Merlin's lookup, which carries the last K to max_num_seqs)."""
    s = copy.deepcopy(DYN_SCHEDULE)
    s[-1][1] = max(int(s[-1][1]), int(seqs))
    return s


def df_plan(base, cur, spec_json, compilation_json, df_spec_argv):
    """The DFlash part of a D try for the current lever values. base = E29's e29_df_setup result {"spec", "k", "captures",
    "tail"} (24 seqs, K3). SEQS / SPEC OFF -> the base objects exactly (the E29 tail). The json / argv builders are the
    notebook's own (nbdf_lib spec_json / compilation_json, e29_lib df_spec_argv)."""
    seqs, sv = int(cur.get("SEQS", OFF["SEQS"])), cur.get("SPEC", OFF["SPEC"])
    if seqs == OFF["SEQS"] and sv == OFF["SPEC"]:
        return {"k": int(base["k"]), "spec": copy.deepcopy(base["spec"]), "captures": list(base["captures"]),
                "tail": list(base["tail"]), "seqs": seqs, "spec_value": sv, "base": True}
    spec = copy.deepcopy(base["spec"])
    if sv == "K3":
        k, caps = int(base["k"]), dflash_captures(seqs, int(base["k"]))
    elif sv == "D4s":
        k = K4
        spec = dict(spec, num_speculative_tokens=K4)
        caps = dflash_captures(seqs, K4)
    else:
        k = K4
        spec = dict(spec, num_speculative_tokens=K4)
        spec["num_speculative_tokens_per_batch_size"] = dyn_schedule(seqs)
        caps = dyn_captures(seqs)
    return {"k": k, "spec": spec, "captures": caps, "tail": list(df_spec_argv(spec_json(spec), compilation_json(caps))),
            "seqs": seqs, "spec_value": sv, "base": False}


def plan_coverage(plan):
    """Every decode batch b = 1 .. seqs has an exact capture at its own query length 1 + K(b) (record + a build assert)."""
    seqs, caps = int(plan["seqs"]), set(int(c) for c in plan["captures"])
    if plan.get("spec_value") == "DYN":
        d = dense_schedule(plan["spec"]["num_speculative_tokens_per_batch_size"], seqs, K4)
        ks = [d[b] for b in range(1, seqs + 1)]
    else:
        ks = [int(plan["k"])] * seqs
    bad = [b for b, k in zip(range(1, seqs + 1), ks) if b * (1 + k) not in caps]
    return {"covers_all": not bad, "uncovered_b": bad, "tiers": sorted(set(ks)), "max_capture": max(caps) if caps else None}


def tag(cur):
    """A short boot-try name suffix of the on levers ("" = none): _e32 + s8 / s2 (seqs 28 / 32) + d / y (D4s / DYN) + l p i f a."""
    s = ""
    if is_on(cur, "SEQS"):
        s += "s" + str(int(cur["SEQS"]))[-1]
    if is_on(cur, "SPEC"):
        s += {"D4s": "d", "DYN": "y"}[cur["SPEC"]]
    s += "".join(CODE[k] for k in ("LP", "SPC", "IDX", "G1", "G2") if cur.get(k))
    return "_e32" + s if s else ""


# ------------------------------------------------------------------ the drop rule
def dyn_fail_lines(log_text):
    return [l.strip()[:200] for l in (log_text or "")[-400000:].splitlines() if any(x in l for x in DYN_FAIL_RX)][:4]


def pick(cur, status, log_text="", oom=False, fault=False, checks=None):
    """(lever, class) to drop after a real failure of a D try with E32 levers on, or (None, class) = E29's rule. oom / fault =
    the notebook's e29_lib is_oom / is_device_fault on the same log."""
    on = on_list(cur)
    if not on:
        return None, "NO_E32_LEVER_ON"
    t = (log_text or "")[-400000:]
    if cur.get("SPEC") == "DYN" and dyn_fail_lines(t):
        return "SPEC", "DYN_FALLBACK_LINE"
    err = "\n".join(l for l in t.splitlines() if ERR_LINE.search(l)).lower()
    for k in GENERAL_ORDER:
        if k in on and any(w in err for w in NAMED[k]):
            return k, "NAMED_IN_ERROR_LINE"
    if str(status) == "E29_DFLASH_CONFIG_MISMATCH" and isinstance(checks, dict):
        bad = [str(a).lower() for a, b in checks.items() if b is not True]
        if "SEQS" in on and any("seq" in a for a in bad):
            return "SEQS", "CONFIG_MISMATCH_SEQS"
        if "SPEC" in on and any("spec" in a for a in bad):
            return "SPEC", "CONFIG_MISMATCH_SPEC"
    if str(status) == "NO_GO_INSUFFICIENT_KV" and "SEQS" in on:     # the capacity gate C-8 (no worker fits): back to 24 seqs
        return "SEQS", "CAPACITY_GATE"
    if (oom and not fault) or str(status) in MEMORY_STATUS:
        k = next((x for x in MEMORY_ORDER if x in on), None)
        return (k, "MEMORY") if k else (None, "MEMORY_NO_MEMORY_LEVER")
    return next(x for x in GENERAL_ORDER if x in on), ("DEVICE_FAULT" if fault else "GENERAL")


def drop(cur, lever):
    """The lever values after dropping one lever (that lever back to OFF)."""
    out = dict(cur)
    out[lever] = OFF[lever]
    return out


# ------------------------------------------------------------------ the in-run relaunch argv
def strip_last(seq, sub):
    """seq without the LAST exact contiguous occurrence of sub (unchanged when sub is empty or absent)."""
    seq, n = list(seq or []), len(sub or [])
    for i in range(len(seq) - n, -1, -1):
        if n and seq[i:i + n] == list(sub):
            return seq[:i] + seq[i + n:]
    return seq


def set_last_flag(seq, flag, value):
    seq = list(seq)
    for i in range(len(seq) - 2, -1, -1):
        if seq[i] == flag:
            seq[i + 1] = str(value)
            return seq
    return seq


def rebuild_extra(extra, old_argv, old_tail, new_seqs, new_argv, new_tail):
    """A booted D try's NB5 extra (base + --max-num-seqs S + E32 argv + DFlash tail) for a new lever set: the old tail and
    the old E32 argv removed (exact), --max-num-seqs set, the new E32 argv + tail appended (= what nb5_boot builds)."""
    x = strip_last(strip_last(extra, old_tail), old_argv)
    return set_last_flag(x, "--max-num-seqs", int(new_seqs)) + list(new_argv) + list(new_tail)


# ------------------------------------------------------------------ the speed pass (record only)
def speed_cfg(raw):
    """The CELL 1 E32_SPEED switch -> {"on", "windows", "cap_min"}. 'on' not a bool -> off; a bad window list (or one
    without the 90k and 116k bands, user rule) -> the default windows; recorded, never raises."""
    c = dict(SPEED_DEFAULT, windows=copy.deepcopy(SPEED_DEFAULT["windows"]))
    why = []
    r = raw if isinstance(raw, dict) else {}
    if not isinstance(raw, dict):
        why.append(f"not a dict: {raw!r}"[:120])
    if "on" in r:
        if type(r["on"]) is bool:
            c["on"] = r["on"]
        else:
            c["on"] = False
            why.append(f"on {r['on']!r} -> off")
    if "windows" in r:
        w = r["windows"]
        ok = (isinstance(w, list) and w and all(isinstance(x, (list, tuple)) and len(x) == 2 and x[0] in SPEED_BANDS
                                              and type(x[1]) is int and 1 <= x[1] <= 64 for x in w)
              and all(any(x[0] == b for x in w) for b in SPEED_REQUIRED))
        if ok:
            c["windows"] = [[str(x[0]), int(x[1])] for x in w]
        else:
            why.append(f"windows {w!r} -> default (90k and 116k required)"[:200])
    if "cap_min" in r:
        if type(r["cap_min"]) is int and 1 <= r["cap_min"] <= 30:
            c["cap_min"] = r["cap_min"]
        else:
            why.append(f"cap_min {r['cap_min']!r} -> 15")
    c["invalid"] = why or None
    return c


def speed_need_sec(band, n, prefill_tok_s, win=90.0):
    """= NB-D1's win_need: the window + 120 s + the prompt tokens at max(500, the measured prefill rate)."""
    return float(win) + 120.0 + int(n) * SPEED_BANDS[band] / max(500.0, float(prefill_tok_s or 0.0))


def speed_record(band, n, r, al_c=None):
    r = r or {}
    o = {k: r.get(k) for k in ("status", "tok_s", "running_mean", "running_max", "mean_acceptance_length", "accept_rate",
                               "preemptions", "kv_usage_max", "step_ms", "window_sec", "gen_tokens", "error")}
    o.update(band=band, level=int(n), al_counters=(al_c or {}).get("al"), admission_sec=(r.get("admission") or {}).get("sec"),
             prefill_tok_s=(r.get("admission") or {}).get("prefill_tok_s"), garbage=(r.get("garbage") or {}).get("garbage"))
    return o


def speed_line(minutes, key, o):
    f = lambda v, nd=1: f"{v:.{nd}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
    return (f"[LIVE speed {f(minutes)} min] {key} tok/s {f(o.get('tok_s'))} | AL {f(o.get('mean_acceptance_length'), 2)} | "
            f"running {o.get('running_max')} | preempt {f(o.get('preemptions'), 0)} | status {o.get('status')}"
            + (f" | GARBAGE {o.get('garbage')}" if o.get("garbage") else ""))


# ------------------------------------------------------------------ the engine fingerprint (record only)
def fingerprint(argv, env, kv_bytes, merlin, levers):
    """argv + the engine env keys + the KV pin + the Merlin tree facts + every lever's state; sha256 over the canonical JSON
    (the short sha in the LIVE start line compares lever results across notebook revisions)."""
    e = {k: str(v) for k, v in sorted((env or {}).items()) if str(k).startswith(FP_ENV_PREFIX)}
    fp = {"argv": [str(x) for x in (argv or [])], "env": e, "kv_bytes": int(kv_bytes or 0), "merlin": merlin or {},
          "levers": levers or {}}
    h = hashlib.sha256(json.dumps(fp, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return dict(fp, sha256=h, short=h[:12])


def lever_states(parsed, cur, drops, active):
    """{lever: {"switch", "state", "why"}}: on / off (switch) / invalid (-> off) / dropped (+ why) / n/a (no DFlash)."""
    out, dmap = {}, {d["lever"]: d for d in drops or []}
    cfg = (parsed or {}).get("cfg", {})
    for k in LEVERS:
        sw = (parsed or {}).get("given", {}).get(k)
        if k in (parsed or {}).get("invalid", {}):
            st, why = "invalid->off", parsed["invalid"][k]
        elif k in dmap:
            st, why = "dropped", dmap[k].get("why")
        elif cfg.get(k, OFF[k]) == OFF[k]:
            st, why = "off", None
        elif not active:
            st, why = "n/a", "no DFlash try (E29's MTP K3 fallback runs without levers)"
        elif is_on(cur or {}, k):
            st, why = "on", None
        else:
            st, why = "off", None
        out[k] = {"switch": sw, "state": st, "why": why}
    return out


def short_states(states):
    return {k: f"{v['switch']} {v['state']}" for k, v in (states or {}).items()}


# ------------------------------------------------------------------ the read-back of one booted D try
def readback(cur, log_text, crb_log=None, worker_env=None, cache_info=None):
    """Record (every lever's own read-back); real_failure only for the DYN fallback lines (NB-S1's real-failure rule).
    crb_log = config_readback()["log"]; worker_env = [{"env": {...}}] (/proc environ of the 4 workers); cache_info =
    nb1_cache_info RPC results (the per-rank kv_cache_specs)."""
    t = log_text or ""
    lg = crb_log if isinstance(crb_log, dict) else {}
    rec = {"levers_on": on_list(cur)}
    nd = bool(_nondefault(t))
    rec["nondefault_args_line"] = nd
    rec["LP"] = {"want": [2048, 1024, True] if cur.get("LP") else [1024, 0, False],
                 "logged": [logged_int(t, "max_num_batched_tokens"), logged_int(t, "long_prefill_token_threshold"),
                            logged_bool(t, "long_prefill_token_threshold_adaptive")]}
    rec["SPC"] = {"want": bool(cur.get("SPC")), "logged": logged_bool(t, "enable_mamba_shared_prefix_checkpoint")}
    ir = idx_readback({"ranks": [r for r in ((cache_info or {}).get("results") or []) if isinstance(r, dict)]})
    rec["IDX"] = {"want": bool(cur.get("IDX")), "worker_specs_fp8": ir.get("engaged"), "specs": (ir.get("specs") or [])[:4],
                  "logged_fp8": logged_indexer_fp8(t)}
    envs = [((w or {}).get("env") or {}) for w in (worker_env or [])]
    for k in ("G1", "G2"):
        rec[k] = {"want": "1" if cur.get(k) else None, "workers": [e.get(GDN_ENV[k]) for e in envs]}
    sp = lg.get("speculative_config") if isinstance(lg.get("speculative_config"), dict) else None
    rec["SPEC"] = {"want": cur.get("SPEC"), "logged_k": (sp or {}).get("num_speculative_tokens"),
                   "logged_schedule": (sp or {}).get("num_speculative_tokens_per_batch_size"), "dyn_fallback_lines": dyn_fail_lines(t)}
    rec["SEQS"] = {"want": int(cur.get("SEQS", OFF["SEQS"])), "logged": lg.get("max_num_seqs")}
    applied = {}
    for k in LEVERS:
        r = rec[k]
        if k == "LP":
            applied[k] = None if not nd else (r["logged"] == r["want"] or (not cur.get("LP") and r["logged"][0] in (None, 1024)
                                                                            and r["logged"][1] in (None, 0) and r["logged"][2] is not True))
        elif k == "SPC":
            applied[k] = None if not nd else (bool(r["logged"]) == r["want"])
        elif k == "IDX":
            v = r["worker_specs_fp8"] if r["worker_specs_fp8"] is not None else (r["logged_fp8"] if nd else None)
            applied[k] = None if v is None else (bool(v) == r["want"])
        elif k in ("G1", "G2"):
            applied[k] = None if len(r["workers"]) < 4 else all(v == r["want"] for v in r["workers"])
        elif k == "SPEC":
            applied[k] = None if r["logged_k"] is None else (int(r["logged_k"]) == (K4 if cur.get("SPEC") in ("D4s", "DYN") else 3)
                                                              and (bool(r["logged_schedule"]) == (cur.get("SPEC") == "DYN")))
        else:
            applied[k] = None if r["logged"] is None else int(r["logged"]) == r["want"]
    rec["applied"] = applied                      # True / False / None (unreadable): record only
    rec["real_failure"] = "E32_DYN_FALLBACK" if (cur.get("SPEC") == "DYN" and rec["SPEC"]["dyn_fallback_lines"]) else None
    return rec


def summary_text(s):
    s = s or {}
    sp = s.get("speed") or {}
    return "\n".join([
        f"  levers: {json.dumps(s.get('levers_short'))}",
        f"  drops: {json.dumps([[d.get('lever'), d.get('class'), d.get('status'), d.get('try')] for d in s.get('drops') or []])}",
        f"  fingerprint {s.get('fingerprint_short')} | seqs {s.get('seqs')} | K {s.get('k')} | kv {s.get('kv_bytes')} | pool {s.get('pool_blocks')}"
        f" | F {s.get('f_eff')} | workers {s.get('workers')}",
        "  speed pass: " + "; ".join(f"{k} {v.get('tok_s')} tok/s AL {v.get('mean_acceptance_length')} pre {v.get('preemptions')} "
                                      f"[{v.get('status')}]" for k, v in (sp.get("windows") or {}).items()),
        f"  run: tok/s window {s.get('tok_s_window')} | per 15 min {s.get('tok_s_per_15min')} | AL {s.get('al')} | preemptions "
        f"{s.get('preemptions')}"])
