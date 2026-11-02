"""E25 NB-D1 library (2026-11-02): 1500+ speed levers on the DFlash2 PRODUCTION base B = the E29_DFLASH config (what
production runs): NBDF-K100's D3s (DFlash2 K3, DFLASH_DRAFT_SW_BLOCK=1024, BF16 drafter KV, FLASH_ATTN, dflash_seam 341d859b
with GRD, Q-defer, seqs 24, pin 7,933,526,016) + DFLASH_DRAFT_EMBED_HOST=1 from the FIRST try + the QSA prefill-logits
workspace VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 / VLLM_E2_QSA_LOGITS_GRANULE_MB=64 (NB1_ENV: applied after Merlin's build(),
which sets 256 / 256 at launch.py:419-420; read back in all 4 workers). Klee100 3bpw-MTP target on Merlin v3.0.0 ad42e37,
Kaggle 4 x L4, TP2 x PP2 + EP.

Arms (each = B + exactly ONE change; this order = the run order inside a copy; no cross-arm skip):
  Q256 the QSA workspace at Merlin's default 256 / 256 (explicit NB1_ENV values; the control for B's 128 / 64)
  D4s  DFlash2 K4 (trailing --speculative-config K 4, captures every multiple of 5 up to 24 x 5 = 120)
  DYN  K4 + num_speculative_tokens_per_batch_size [[1, 8, 4], [9, 24, 3]] (NB-S1's DYN: captures b x 5 for b <= 8, b x 4
       for 9-24; real failure = the 'Overriding cudagraph_mode' / 'Disabling num_speculative_tokens_per_batch_size' line)
  G1   VLLM_GDN_DECODE_FASTPATH=1          G2   VLLM_GDN_SHARED_ALIGN_INDEX=1
  IDX  --attention-config {"indexer_kv_dtype":"fp8"} (speed only; on the D-SW layout 0 extra KV blocks: kit R9 540 = 540)
  LP   --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive
  S28  --max-num-seqs 28 (captures every multiple of 4 up to 112)     S32  --max-num-seqs 32 (captures up to 128)
  SPC  --enable-mamba-shared-prefix-checkpoint (+ the sibling activation read-back, recorded)
  PROF no boot of its own: two 30-s torch.profiler windows 8k@16 / 20k@24 inside B's boot (kernel-fusion sizing)
  BAA  no boot of its own: B's A/A pair (8k@16 #2) + 40k@16 (off: B runs the lever window set only = "B light")
LIGHT design (user: runs must not be huge): every arm = boot + the smoke hard gate (no ladder) + the read-back of its one
change (a real failure -> NOT_INTEGRATED for that arm only) + its windows (8k@16, 20k@24, 90k@8, 116k@6 for every lever:
90k and ~116k always); no TF, no oracle, no A/A on a lever. Hard-only gates; every label is recorded only.

Pure logic, stdlib only; nbdf_lib (N) and nb4t_lib (T) are handed in as module objects (never imported or edited here).
"""
import copy
import json
import math
import re
import statistics

LIB_VERSION = "nbd1-lib 2026-11-02 r1 (DFlash2 production base B = E29_DFLASH D3s + EMBED_HOST + QSA 128/64; LIGHT levers)"
BASE_KEY = "B"
LEVERS = ("Q256", "D4s", "DYN", "G1", "G2", "IDX", "LP", "S28", "S32", "SPC")
ARM_KEYS = (BASE_KEY,) + LEVERS
ITEM_KEYS = ("BAA", "PROF")                         # inside B's boot (no boot of their own)
SWITCH_KEYS = ("B", "BAA", "PROF") + LEVERS
SWITCH_DEFAULT = {k: True for k in SWITCH_KEYS}     # the kit notebook: every arm on
COPIES = {"a": dict({k: False for k in SWITCH_KEYS}, B=True, BAA=True, PROF=True, Q256=True, D4s=True, DYN=True),
          "b": dict({k: False for k in SWITCH_KEYS}, B=True, G1=True, G2=True, IDX=True, LP=True),
          "c": dict({k: False for k in SWITCH_KEYS}, B=True, S28=True, S32=True, SPC=True)}
OWNERS = {"a": "foysalemonshanto", "b": "hosen42", "c": "foysalemonshanto"}
PORTS = {"B": 8011, "Q256": 8031, "D4s": 8051, "DYN": 8071, "G1": 8091, "G2": 8111, "IDX": 8131, "LP": 8151, "S28": 8171,
         "S32": 8191, "SPC": 8211}
N_WORKERS, TP = 4, 2
PIN = 7933526016
SEQS = 24
ENV_EH = "DFLASH_DRAFT_EMBED_HOST"
QSA_KEYS = ("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "VLLM_E2_QSA_LOGITS_GRANULE_MB")   # Merlin launch.py:419-420 sets both 256
QSA_B = {QSA_KEYS[0]: "128", QSA_KEYS[1]: "64"}                                       # E29_DFLASH rev 2c (production)
QSA_MERLIN = {QSA_KEYS[0]: "256", QSA_KEYS[1]: "256"}                                 # Q256: Merlin's default, explicit
GDN_ENV = {"G1": "VLLM_GDN_DECODE_FASTPATH", "G2": "VLLM_GDN_SHARED_ALIGN_INDEX"}
D1_ENV_KEYS = QSA_KEYS + (GDN_ENV["G1"], GDN_ENV["G2"])                              # + N.WATCHED in every arm's expect
IDX_ARGS = ["--attention-config", '{"indexer_kv_dtype":"fp8"}']
LP_ARGS = ["--max-num-batched-tokens", "2048", "--long-prefill-token-threshold", "1024", "--long-prefill-token-threshold-adaptive"]
SPC_ARGS = ["--enable-mamba-shared-prefix-checkpoint"]
SEQ_ARMS = {"S28": 28, "S32": 32}
K4 = 4
DYN_SCHEDULE = [[1, 8, 4], [9, 24, 3]]
DYN_FAIL_RX = ("Overriding cudagraph_mode", "Disabling num_speculative_tokens_per_batch_size")   # = nbs1_lib.DYN_FAIL_RX
DYN_SOURCE_CHECK = (   # = nbs1_lib.DYN_SOURCE_CHECK (NB-S1, verified on Merlin ad42e37)
    "Merlin ad42e37 config/speculative.py:493 num_speculative_tokens_per_batch_size = [(start, end, K)] inclusive ranges, "
    "clamped to num_speculative_tokens (v1/spec_decode/dynamic/utils.py); the scheduler (v1/core/sched/scheduler.py:348) "
    "picks K per batch size. config/vllm.py:1068-1085 overrides FULL cudagraphs to PIECEWISE unless the V2 model runner "
    "is used (NBDF runs the V2 GPU runner: v1/worker/gpu); v1/worker/gpu/cudagraph_utils.py:283-300 then captures the "
    "decode graphs of every tier. -> num_speculative_tokens 4 + [[1, 8, 4], [9, 24, 3]], captures = every b x 5 for b <= 8 "
    "and every b x 4 for b 9-24. Per-batch K reaches the drafter: v1/worker/gpu/model_runner.py:2786-2803 passes "
    "num_speculative_tokens=<this batch's K> to speculator.propose() and sets draft_tokens[:, K:] = -1 when K < the max; "
    "DFlashSpeculator.propose (spec_decode/dflash/speculator.py:343-369, the DFlash2 base) accepts the argument and drafts "
    "its full fixed block (num_query_per_req = 1 + num_speculative_steps, :62), so the drafter cost stays K4 at every batch "
    "size and only the target's verify length shrinks at 9-24 running. Real-failure read-back: an 'Overriding "
    "cudagraph_mode' or 'Disabling num_speculative_tokens_per_batch_size' log line.")
PROF_WINDOWS = (("8k", 16), ("20k", 24))
PROF_SECS = 30.0
B_WINDOWS_FULL = [["8k", 16, "#1"], ["8k", 16, "#2"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]]
LEVER_WINDOWS = [["8k", 16, "#1"], ["20k", 24, ""], ["90k", 8, ""], ["116k", 6, ""]]
CAPPED_BANDS = ("20k", "40k", "90k", "116k")        # levels capped by THIS boot's resident bound (never skipped: min(level, bound))
B_REF = {"source": "E25 NB-S1 full (D3s + EMBED_HOST, QSA 256/256), D:/ARC AGI 2/_scratch/e25_nbs1_full_out/",
         "tok_s": {"8k@16#1": 630.8, "8k@16#2": 609.8, "20k@24": 763.0, "40k@16": 598.7, "90k@8": 383.0, "116k@6": 324.0},
         "al": 3.16, "preemptions": 0}
TITLES = {"B": "B = E29_DFLASH production: NBDF-K100 D3s (DFlash2 K3, SW-1024, BF16 drafter KV, FLASH_ATTN, seam 341d859b + GRD, "
               "Q-defer, seqs 24, pin 7,933,526,016) + DFLASH_DRAFT_EMBED_HOST=1 (first try) + QSA 128 / 64",
          "Q256": "B with the QSA workspace at Merlin's default 256 / 256", "D4s": "B with DFlash2 K4 (captures 5 .. 120)",
          "DYN": "B with K4 + num_speculative_tokens_per_batch_size [[1, 8, 4], [9, 24, 3]]",
          "G1": "B + VLLM_GDN_DECODE_FASTPATH=1", "G2": "B + VLLM_GDN_SHARED_ALIGN_INDEX=1",
          "IDX": "B + --attention-config {\"indexer_kv_dtype\":\"fp8\"}",
          "LP": "B + --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive",
          "S28": "B with --max-num-seqs 28 (captures 4 .. 112)", "S32": "B with --max-num-seqs 32 (captures 4 .. 128)",
          "SPC": "B + --enable-mamba-shared-prefix-checkpoint"}
KIT_KEYS = ("nbd1_arm_plan_ok", "nbd1_launch_path_all_arms_ok", "nbd1_help_flag_gate_real_ok", "nbd1_argparse_one_change_per_arm",
            "nbd1_kv_probes_ok", "nbd1_qsa_env_final_ok", "nbd1_captures_argv_packviews_ok", "nbd1_switch_default_all_on")

# ------------------------------------------------------------------ the time model (minutes)
# GPU data: E25 NB-S1 full 2026-11-01/02 (DFlash2 D3s + EMBED_HOST, this exact boot path): first boot ready 714 s, warm boots
# 252 / 282 s; window spacing 8k@16 125 s, 20k@24 217 s, 40k@16 351 s, 90k@8 273 s, 116k@6 272 s. NBDF-K100 / NB-M1: setup
# 5.6, items (facts 1.5, smoke 1.3, prefill probe 0.5, warmup 1.0, end 1.2), the two 30-s profiles 7.3 + their analysis.
# A changed compile key (spec K, captures, attention / scheduler / cache config) recompiles: "recompile".
SETUP_MIN = 6.0
READY_MIN = {"first": 12.0, "recompile": 7.0, "warm": 4.8, "warm_capture": 5.5}
READY_CLASS = {"B": "first", "Q256": "recompile", "D4s": "recompile", "DYN": "recompile", "G1": "warm", "G2": "warm", "IDX": "recompile",
               "LP": "recompile", "S28": "warm_capture", "S32": "warm_capture", "SPC": "recompile"}
ITEM_MIN = {"facts": 1.5, "smoke": 1.3, "prefill_probe": 0.5, "warmup": 1.0, "end": 1.2, "prof": 7.3, "sibling": 1.0}
WINDOW_MIN = {"8k@16": 2.1, "20k@24": 3.65, "40k@16": 5.85, "90k@8": 4.6, "116k@6": 4.55, "20k@28": 4.3, "20k@32": 4.8}
POST_MIN = {"prof": 3.8, "arm": 0.8}
BAND_TOK = {"8k": 8192, "20k": 20000, "40k": 40000, "90k": 90000, "116k": 116000}
WIN_SEC, ADMIT_TOK_S, LONG_RESET_SEC = 90.0, 3000.0, 60.0   # body: WIN, K100 prefill_admit_floor_tok_s, the long reset's +60 s
START_RESERVE_MIN = 5.0          # nb4t_lib START_RESERVE_SEC: an arm starts only with est_min + 5 min left
COPY_RUNTIME_MAX = 150           # build-time assertion: each copy's planned runtime <= this (user: ~120-150 min per copy)
BUDGET_MIN = 160                 # one hard stop for every copy: max over the copies of (runtime, full-cap end) + 5, rounded up to 5
DEADMAN_MIN = BUDGET_MIN + 5


def _c(x):
    return int(math.ceil(round(x, 6)))


def _wkey(w):
    return f"{w[0]}@{int(w[1])}"


# ------------------------------------------------------------------ switches (CELL 1's NBD1_ON)
def switch_map(on, keys=SWITCH_KEYS):
    """NBD1_ON -> {key: bool}: absent (None) or not a dict -> every key on (recorded as invalid); a dict -> bool(on[k]),
    a missing key -> off."""
    if on is None or not isinstance(on, dict):
        return {k: True for k in keys}
    return {k: bool(on.get(k, False)) for k in keys}


def switch_record(on, keys=SWITCH_KEYS):
    m = switch_map(on, keys)
    return {"NBD1_ON": on if isinstance(on, (dict, type(None))) else repr(on)[:200], "effective": m, "absent": on is None,
            "invalid_not_a_dict": on is not None and not isinstance(on, dict),
            "unknown_keys": sorted(str(k) for k in set(on) - set(keys)) if isinstance(on, dict) else [],
            "off": [k for k, v in m.items() if not v]}


# ------------------------------------------------------------------ captures
def dflash_captures(seqs, k):
    """Every multiple of 1 + K up to seqs x (1 + K): every decode batch an exact FULL graph (nbdf_lib rule)."""
    return [b * (int(k) + 1) for b in range(1, int(seqs) + 1)]


def dense_schedule(schedule, max_seqs, k_max):
    """= Merlin's build_dynamic_sd_schedule_lookup (v1/spec_decode/dynamic/utils.py:77-150, the validation included;
    nbs2_lib.dense_schedule): index b (1 .. max_seqs) -> the K the scheduler uses at b running requests."""
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


def dyn_captures(seqs=SEQS):
    """b x (1 + K(b)): b x 5 for b <= 8 and b x 4 for b 9-24 (= nbs1_lib.dyn_captures)."""
    d = dense_schedule(DYN_SCHEDULE, int(seqs), K4)
    return sorted({b * (1 + d[b]) for b in range(1, int(seqs) + 1)})


def dyn_coverage(captures, seqs=SEQS):
    """Every b = 1 .. seqs an exact graph at its own tier's query length 1 + K(b) (cudagraph_utils.py:332-347 rounding)."""
    d = dense_schedule(DYN_SCHEDULE, int(seqs), K4)
    caps = sorted(set(int(c) for c in captures))
    bad = []
    for b in range(1, int(seqs) + 1):
        q = 1 + d[b]
        g = {-(-c // q) * q for c in caps if -(-c // q) * q <= caps[-1]}
        if b * q not in g:
            bad.append(b)
    return {"covers_all": not bad, "uncovered_or_padded_b": bad, "padded_b": [], "uncovered_b": bad, "tiers": sorted(set(d[1:]))}


def _split_tail(extra):
    """D3s's extra = base argv + [--speculative-config, spec, --compilation-config, captures] (nbdf_lib.arm_plan)."""
    x = list(extra)
    if len(x) < 4 or x[-4] != "--speculative-config" or x[-2] != "--compilation-config":
        raise ValueError(f"the base arm's trailing argv is not the DFlash pair: {x[-4:]}")
    return x[:-4], x[-4:]


# ------------------------------------------------------------------ items / windows / time
B_ITEMS = {"ladder": False, "prefill_probe": True, "oracle": False, "long_oracle": False, "prefix": False, "sibling": False, "tf": False}
LEVER_ITEMS = {"ladder": False, "prefill_probe": False, "oracle": False, "long_oracle": False, "prefix": False, "sibling": False,
               "tf": False}


def item_on(arm, item):
    """A body item of this arm (NB-D1 light plans; an item missing from the plan is on = the NBDF-K100 default)."""
    return bool((arm.get("d1_items") or {}).get(item, True))


def b_windows(switch):
    return copy.deepcopy(B_WINDOWS_FULL if switch_map(switch).get("BAA") else LEVER_WINDOWS)


def lever_windows(key):
    if key in SEQ_ARMS:
        s = SEQ_ARMS[key]
        return [["20k", s, ""], ["90k", 8, ""], ["116k", 6, ""]]
    return copy.deepcopy(LEVER_WINDOWS)


def prof_plan(arm, switch):
    """[(band, level)] of the PROF windows: B's boot only, when PROF is on."""
    return [tuple(x) for x in PROF_WINDOWS] if arm.get("key") == BASE_KEY and switch_map(switch).get("PROF") else []


def win_need_sec(band, n, prof=False):
    """= the body's win_need() at the admission rate min(prefill, 3000 tok/s): a gated item starts only with this left."""
    if prof:
        return 360 + 10 * max(0, n - 8) + n * BAND_TOK[band] / ADMIT_TOK_S
    return WIN_SEC + 120 + n * BAND_TOK[band] / ADMIT_TOK_S


def body_plan(items):
    """[(name, min, need_sec | None)] in body order -> (body_min, cap_min, rows): the cap holds every gated item's due()
    need from its planned start, + 1 min (NB-M1L's rule)."""
    t, need, rows = 0.0, 0.0, []
    for nm, m, nd in items:
        if nd is not None:
            need = max(need, t + nd / 60.0)
        rows.append([nm, m, None if nd is None else round(nd / 60.0, 2), round(t, 2)])
        t += m
    return round(t, 2), _c(max(t, need) + 1.0), rows


def body_items(key, windows, prof_windows, items):
    it = [("facts", ITEM_MIN["facts"], None), ("smoke", ITEM_MIN["smoke"], 240)]
    if items.get("prefill_probe"):
        it.append(("prefill_probe", ITEM_MIN["prefill_probe"], 300))
    if items.get("sibling"):
        it.append(("sibling", ITEM_MIN["sibling"], 240))
    it.append(("warmup", ITEM_MIN["warmup"], 150))
    for b, n, tag in windows:
        nd = win_need_sec(b, int(n)) + (LONG_RESET_SEC if b in ("90k", "116k") else 0.0)
        it.append((f"{b}@{int(n)}{tag}", WINDOW_MIN[f"{b}@{int(n)}"], nd))
    for b, n in prof_windows:
        it.append((f"profile_{b}@{n}", ITEM_MIN["prof"] / len(PROF_WINDOWS), win_need_sec(b, n, prof=True)))
    it.append(("end", ITEM_MIN["end"], None))
    return it


def arm_time(key, windows, prof_windows, items):
    """(est_min, cap_min, model) of one arm: ready (its class) + the body + post."""
    body, cap, rows = body_plan(body_items(key, windows, prof_windows, items))
    rc = READY_CLASS[key]
    post = POST_MIN["prof"] if prof_windows else POST_MIN["arm"]
    return _c(READY_MIN[rc] + body + post), cap, {"ready": READY_MIN[rc], "ready_class": rc, "body": body, "post": post, "items": rows}


# ------------------------------------------------------------------ the arm plan
def expect_of(N, a):
    """Every watched env key (nbdf_lib WATCHED: the seam keys incl. EMBED_HOST) + the NB-D1 keys (QSA, GDN): value or None."""
    e = {w: a["env"].get(w, N.WATCHED_DEFAULT.get(w)) for w in N.WATCHED}
    for k in D1_ENV_KEYS:
        e[k] = a["env"].get(k)
    return e


def arm_plan(N, d3s, seam_env_k4=None, switch=None, pin=PIN, seqs=SEQS):
    """d3s = NBDF-K100's D3s (nbdf_lib.arm_plan after configure(SEQS=24, ...)). Returns {key: arm} in ARM_KEYS order: B =
    a deep copy + EMBED_HOST + QSA 128 / 64 (the E29_DFLASH config); every lever = a deep copy of B + exactly one change."""
    if d3s.get("method") != "dflash" or d3s.get("k") != 3 or d3s["env"].get(N.ENV_SW_BLOCK) != str(N.SW_BLOCK):
        raise ValueError("the base must be NBDF-K100's D3s (dflash, K 3, DFLASH_DRAFT_SW_BLOCK=1024)")
    sw = switch_map(switch)
    base_extra, tail = _split_tail(d3s["extra"])
    b = copy.deepcopy(d3s)
    b["env"].update({ENV_EH: "1"}, **QSA_B)
    bw, bp = b_windows(sw), prof_plan({"key": BASE_KEY}, sw)
    est, cap, tm = arm_time(BASE_KEY, bw, bp, B_ITEMS)
    b.update(key=BASE_KEY, name="nbd1_b", port=PORTS[BASE_KEY], parent="D3s", streams=int(seqs), long_aa=False, lever=False,
             title=TITLES[BASE_KEY], windows_fixed=bw, prof_windows=[list(x) for x in bp], est_min=est, cap_min=cap, time_model=tm,
             kv_bytes=int(pin), optional=False, short_only=False, iso=False, d1_change=None, d1_items=dict(B_ITEMS),
             d1_eh_first=True, base_change={"env": dict({ENV_EH: "1"}, **QSA_B)})
    b["expect_cfg"]["max_num_seqs"] = int(seqs)
    b["expect"] = expect_of(N, b)
    arms = {BASE_KEY: b}
    seam_k4 = {k_: v for k_, v in dict(seam_env_k4 or {}).items() if k_ in N.SEAM_ENV_KEYS}
    seam_b = {k_: v for k_, v in d3s["env"].items() if k_ in N.SEAM_ENV_KEYS}
    for key in LEVERS:
        a = copy.deepcopy(b)
        env_add, kind, argv = {}, None, []
        if key == "Q256":
            env_add, kind = dict(QSA_MERLIN), "env"
        elif key in GDN_ENV:
            env_add, kind = {GDN_ENV[key]: "1"}, "env"
        elif key in ("IDX", "LP", "SPC"):
            argv = list({"IDX": IDX_ARGS, "LP": LP_ARGS, "SPC": SPC_ARGS}[key])
            a["extra"], kind = list(b["extra"]) + argv, "argv"
        elif key in SEQ_ARMS:
            s = SEQ_ARMS[key]
            caps = dflash_captures(s, 3)
            argv = ["--compilation-config", N.compilation_json(caps), "--max-num-seqs", str(s)]
            a["extra"], kind = base_extra + tail[:2] + argv, "seqs"
            a["captures"], a["streams"] = caps, s
            a["expect_cfg"].update(max_num_seqs=s, captures=caps)
        elif key in ("D4s", "DYN"):
            sp = dict(b["spec"], num_speculative_tokens=K4)
            caps = dflash_captures(seqs, K4)
            if key == "DYN":
                sp["num_speculative_tokens_per_batch_size"] = copy.deepcopy(DYN_SCHEDULE)
                caps = dyn_captures(seqs)
            argv = ["--speculative-config", N.spec_json(sp), "--compilation-config", N.compilation_json(caps)]
            a["extra"], kind = base_extra + argv, "spec"
            a["spec"], a["k"], a["captures"] = sp, K4, caps
            a["expect_cfg"].update(num_speculative_tokens=K4, captures=caps)
            if seam_k4 and seam_k4 != seam_b:           # NBDF-K100's D4s seam env (normally = D3s's): seam keys only
                env_add = {k_: v for k_, v in seam_k4.items() if seam_b.get(k_) != v}
            ok_r, cap_r = N.qsa_ring(K4)
            a["ring_needed"], a["ring_capacity_expected"] = not ok_r, cap_r
        a["env"].update(env_add)
        a["coverage"] = dyn_coverage(a["captures"], a["streams"]) if key == "DYN" else N.coverage(a["captures"], a["streams"], a["k"])
        wins = lever_windows(key)
        items = dict(LEVER_ITEMS, sibling=key == "SPC")
        est, cap, tm = arm_time(key, wins, [], items)
        a.update(key=key, name=f"nbd1_{key.lower()}", port=PORTS[key], title=TITLES[key], lever=True, windows_fixed=wins,
                 prof_windows=[], est_min=est, cap_min=cap, time_model=tm, d1_items=items, base_change=None,
                 d1_change={"kind": kind, "env": env_add, "argv": argv})
        a["expect"] = expect_of(N, a)
        arms[key] = a
    if b.get("coverage") is None or not b["coverage"].get("covers_all"):
        b["coverage"] = N.coverage(b["captures"], b["streams"], b["k"])
    return arms


def check_arm_plan(N, T, arms, d3s):
    """Fail closed (ValueError): the order; B = D3s + EMBED_HOST + QSA 128 / 64 exactly (argv, spec, captures, pin, Q-defer
    identical); every arm a DFlash2 SW-1024 arm with EMBED_HOST from the first try, Q-defer on the patched tree, no
    never-do item, exact FULL graphs for every decode batch (DYN per tier), the pin, the windows incl. 90k + 116k; each
    lever = B + exactly its one change (env equal / argv = B's + the change / the trailing pair replaced)."""
    errs = []
    if tuple(arms) != ARM_KEYS:
        raise ValueError(f"NBD1_ARM_PLAN_INVALID: order {tuple(arms)} != {ARM_KEYS}")
    b = arms[BASE_KEY]
    base_extra, tail = _split_tail(d3s["extra"])
    for f in ("extra", "spec", "captures", "k", "method", "q_label", "needs_patch", "kv_bytes"):
        if b.get(f) != d3s.get(f):
            errs.append(f"B.{f} != D3s.{f}")
    ed = {x: b["env"].get(x) for x in set(b["env"]) | set(d3s["env"]) if b["env"].get(x) != d3s["env"].get(x)}
    if ed != dict({ENV_EH: "1"}, **QSA_B):
        errs.append(f"B env differs from D3s by {ed} (want EMBED_HOST + QSA 128 / 64 only)")
    if b["kv_bytes"] != PIN or b["streams"] != SEQS or not b.get("d1_eh_first"):
        errs.append("B pin / seqs / EMBED_HOST first")
    names, ports = set(), set()
    for key, a in arms.items():
        names.add(a["name"])
        ports.add(a["port"])
        v = T.never_do_violations(a["env"], a["extra"])
        if v:
            errs.append(f"{key}: never-do {v}")
        if a.get("method") != "dflash" or a["env"].get(N.ENV_SEAM) != "1" or a["env"].get(N.ENV_SW_BLOCK) != str(N.SW_BLOCK):
            errs.append(f"{key}: not a DFlash2 SW-1024 arm")
        if a["env"].get(ENV_EH) != "1" or not a.get("d1_eh_first"):
            errs.append(f"{key}: DFLASH_DRAFT_EMBED_HOST=1 not from the first try")
        if "DRAFT_HEAD" in a["env"] or (a.get("spec") or {}).get("draft_lm_head_quant"):
            errs.append(f"{key}: a draft head on a DFlash2 arm")
        if a.get("parent") != "D3s" or not a.get("needs_patch") or a.get("q_label") != N.Q_LABEL or a["env"].get(T.ENV_DEFER) != "1":
            errs.append(f"{key}: parent / Q-defer / patched tree")
        if "--language-model-only" not in a["extra"]:
            errs.append(f"{key}: --language-model-only missing")
        if a["kv_bytes"] != b["kv_bytes"]:
            errs.append(f"{key}: pin {a['kv_bytes']} != B's")
        if T.last_value(a["extra"], "--max-num-seqs") != str(a["expect_cfg"]["max_num_seqs"]) or a["streams"] != a["expect_cfg"]["max_num_seqs"]:
            errs.append(f"{key}: --max-num-seqs {T.last_value(a['extra'], '--max-num-seqs')} != {a['expect_cfg']['max_num_seqs']}")
        if N.captures_from_argv(a["extra"]) != a["captures"] or a["expect_cfg"].get("captures") != a["captures"]:
            errs.append(f"{key}: trailing captures != plan")
        if (T.spec_from_argv(a["extra"]) or {}) != a["spec"]:
            errs.append(f"{key}: trailing spec != plan")
        cov = a.get("coverage") or {}
        if not cov.get("covers_all") or cov.get("padded_b"):
            errs.append(f"{key}: decode batches without an exact FULL graph {cov}")
        if set(a["expect"]) != set(N.WATCHED) | set(D1_ENV_KEYS) or a["expect"] != expect_of(N, a):
            errs.append(f"{key}: expect != the env (WATCHED + the NB-D1 keys)")
        if not any(w[0] == "90k" for w in a["windows_fixed"]) or not any(w[0] == "116k" for w in a["windows_fixed"]):
            errs.append(f"{key}: the window set must include 90k and 116k (user rule)")
        if key == BASE_KEY:
            continue
        ch = a.get("d1_change") or {}
        diff = {x: a["env"].get(x) for x in set(a["env"]) | set(b["env"]) if a["env"].get(x) != b["env"].get(x)}
        kind = ch.get("kind")
        if kind == "env":
            ok = diff == ch.get("env") and bool(diff) and a["extra"] == b["extra"] and a["spec"] == b["spec"] and a["captures"] == b["captures"]
            if key == "Q256":
                ok = ok and diff == QSA_MERLIN
            elif key in GDN_ENV:
                ok = ok and diff == {GDN_ENV[key]: "1"}
        elif kind == "argv":
            ok = not diff and a["extra"] == b["extra"] + list(ch.get("argv") or []) and a["spec"] == b["spec"] and bool(ch.get("argv"))
        elif kind == "seqs":
            s = SEQ_ARMS.get(key)
            ok = (not diff and a["extra"] == base_extra + tail[:2] + list(ch.get("argv") or []) and a["spec"] == b["spec"]
                  and s is not None and a["streams"] == s and a["captures"] == dflash_captures(s, 3))
        elif kind == "spec":
            sp = dict(a["spec"])
            sched = sp.pop("num_speculative_tokens_per_batch_size", None)
            ok = (set(diff) <= set(N.SEAM_ENV_KEYS) and a["extra"] == base_extra + list(ch.get("argv") or [])
                  and dict(sp, num_speculative_tokens=3) == b["spec"] and a["k"] == K4 and sp.get("num_speculative_tokens") == K4
                  and ((sched == DYN_SCHEDULE and a["captures"] == dyn_captures(a["streams"])) if key == "DYN"
                       else (sched is None and a["captures"] == dflash_captures(a["streams"], K4))))
            if key == "DYN":
                d = dense_schedule(DYN_SCHEDULE, a["streams"], K4)
                ok = ok and d[1:9] == [4] * 8 and d[9:25] == [3] * 16
        else:
            ok = False
        if not ok:
            errs.append(f"{key}: not B + exactly its one change ({kind}; env diff {diff})")
        if a.get("windows_fixed") != lever_windows(key):
            errs.append(f"{key}: windows {a.get('windows_fixed')} != the light lever set")
        if any(item_on(a, x) for x in ("oracle", "tf", "long_oracle", "prefix", "ladder")):
            errs.append(f"{key}: an oracle / TF / A/A item on a lever (light design)")
    if len(names) != len(arms) or len(ports) != len(arms) or max(ports) + 8 > 8355 or min(ports) < 8000:
        errs.append("names / ports not distinct or outside Merlin's 8000..8355 scheme (+ retry offsets)")
    if errs:
        raise ValueError("NBD1_ARM_PLAN_INVALID: " + "; ".join(errs))
    return True


# ------------------------------------------------------------------ the D pin ladder (EMBED_HOST from the first try)
def retry_plan(d_retry, pin=PIN, eh_first=True):
    """NBDF-K100's d_retry = [[pin, {EMBED_HOST}], [7,665,090,560, {EH}], ...]: with EMBED_HOST already on the first try, the
    same-pin EH rung is not a retry (it would repeat the first try) -> the 0.25-GiB rungs only (= e29_lib.df_tries)."""
    plan = [list(x) for x in (d_retry or [])]
    if eh_first:
        plan = [x for x in plan if int(x[0]) != int(pin)]
    return plan


# ------------------------------------------------------------------ read-backs (real failures, this arm only)
def env_rows(env_raw):
    """[(rank, {key: value})] of the nb4s_env RPC (the requested RPC_KEYS; an absent key is missing from the dict)."""
    res = (env_raw or {}).get("results") if isinstance(env_raw, dict) else None
    out = []
    for w in res if isinstance(res, list) else []:
        w = w if isinstance(w, dict) else {}
        out.append((w.get("rank"), w.get("env") if isinstance(w.get("env"), dict) else {}))
    return out


def _nondefault(log_text):
    return next((l for l in (log_text or "").splitlines() if "non-default args:" in l), "")


def logged_int(log_text, key):
    m = re.search(r"'" + re.escape(key) + r"': (\d+)", _nondefault(log_text))
    return int(m.group(1)) if m else None


def logged_bool(log_text, key):
    m = re.search(r"'" + re.escape(key) + r"': (True|False)", _nondefault(log_text))
    return (m.group(1) == "True") if m else None


def logged_indexer_fp8(log_text):
    lines = [l for l in (log_text or "").splitlines() if "non-default args:" in l or "Initializing a V1 LLM engine" in l]
    return any(re.search(r"indexer_kv_dtype['\"]?\s*[:=]\s*['\"]fp8", l) for l in lines)


def granule_logged(log_text):
    return sorted({int(m) for m in re.findall(r"QSA prefill logits backing: granule (\d+) MiB", log_text or "")})


def idx_readback(workers):
    """The QSA indexer compressed-key cache spec (nb1_cache_info kv_cache_specs) dtype per rank (= nbs2_lib.idx_readback)."""
    seen = []
    for w in ((workers or {}).get("ranks") or []):
        if not isinstance(w, dict):
            continue
        for s in w.get("kv_cache_specs") or []:
            if isinstance(s, dict) and "compressed_key_cache" in str(s.get("example") or ""):
                seen.append({"rank": w.get("rank"), "dtype": str(s.get("dtype")), "page_size_bytes": s.get("page_size_bytes")})
    if not seen:
        return {"engaged": None, "specs": []}
    fp8 = [x for x in seen if "float8" in x["dtype"] or "fp8" in x["dtype"].lower()]
    return {"engaged": len(fp8) == len(seen), "specs": seen[:8]}


def qsa_readback(arm, env_raw, log_text=""):
    """Record: every worker's QSA env == the arm's planned values (B / levers 128 / 64, Q256 256 / 256), Merlin's logged
    granule at the first prefill (qsa_indexer.py:797). label QSA_VERIFIED / QSA_ENV_MISMATCH / UNVERIFIED_QSA."""
    want = {k: (arm.get("env") or {}).get(k) for k in QSA_KEYS}
    rows = env_rows(env_raw)
    got = [{k: e.get(k) for k in QSA_KEYS} for _, e in rows]
    ok = len(got) == N_WORKERS and all(g == want for g in got)
    return {"want": want, "workers": len(got), "env": got[:N_WORKERS], "granule_logged_mib": granule_logged(log_text),
            "label": "QSA_VERIFIED" if ok else ("UNVERIFIED_QSA" if len(got) < N_WORKERS else "QSA_ENV_MISMATCH")}


def readback(arm, log_text, env_raw, crb, workers, drb):
    """The arm's planned NB-D1 settings read back. A real failure (-> NOT_INTEGRATED, this arm only) = the arm's OWN one change
    read back with another value (Q256's QSA 256 / 256, G1 / G2's switch, IDX's fp8 indexer cache, LP's logged flags, SPC's
    flag, S28 / S32's seqs, D4s / DYN's K / schedule, DYN's fallback lines) or another lever's change present (the arm is then
    not B + one change). B's base settings (QSA 128 / 64 on every arm but Q256, EMBED_HOST) read back differently are
    recorded only (base_mismatch: the arm measures on, = E29_DFLASH's record-only QSA read-back); a missing / unreadable
    read-back is recorded only. crb = nbdf_lib.config_readback, drb = dflash_rb (the seam label)."""
    k, why, unread, base_mm = arm["key"], [], [], []
    t = log_text or ""
    rec = {"arm": k}
    rows = env_rows(env_raw)
    ranks = [r for r, _ in rows]
    full = len(rows) == N_WORKERS and len(set(ranks)) == N_WORKERS and None not in ranks
    for key in D1_ENV_KEYS:                       # QSA / GDN env in all 4 workers (planned value) / absent everywhere
        vals = [e.get(key) for _, e in rows]
        rec[key] = vals
        want = (arm.get("env") or {}).get(key)
        if not full:
            unread.append(key)
            continue
        if want is not None and any(v != want for v in vals):
            (why if (key not in QSA_KEYS or k == "Q256") else base_mm).append(f"{key} in the workers {vals}, expected {want!r} on {N_WORKERS}")
        elif want is None and any(v is not None for v in vals):
            why.append(f"{key} set in a worker of {k}: {vals}")
    rec["qsa"] = qsa_readback(arm, env_raw, t)
    chk = (drb or {}).get("checks") or {}
    rec["embed_host_as_planned"] = chk.get("embed_host_as_planned")
    if chk.get("embed_host_as_planned") is False:
        base_mm.append("DFLASH_DRAFT_EMBED_HOST=1 not applied on the drafter ranks (dflash read-back embed_host_as_planned False)")
    elif chk.get("embed_host_as_planned") is None:
        unread.append("embed_host_as_planned")
    ir = idx_readback(workers)
    rec["idx"], rec["idx_logged_fp8"] = ir, logged_indexer_fp8(t)
    nd = bool(_nondefault(t))
    rec["nondefault_args_line"] = nd
    if k == "IDX":
        if ir["engaged"] is False or (ir["engaged"] is None and nd and not rec["idx_logged_fp8"]):
            why.append(f"indexer_kv_dtype fp8 not applied (worker specs {ir.get('specs')[:2]}, logged {rec['idx_logged_fp8']})")
        elif ir["engaged"] is None and not nd:
            unread.append("indexer_kv_dtype")
    elif ir["engaged"] is True or rec["idx_logged_fp8"]:
        why.append(f"the fp8 indexer cache on {k}")
    log_ = (crb or {}).get("log") or {}
    seqs = log_.get("max_num_seqs")
    rec["logged_max_num_seqs"] = seqs
    if seqs is not None and seqs != arm["expect_cfg"]["max_num_seqs"]:
        why.append(f"logged max_num_seqs {seqs} != {arm['expect_cfg']['max_num_seqs']}")
    mbt, lpt, lpa = (logged_int(t, "max_num_batched_tokens"), logged_int(t, "long_prefill_token_threshold"),
                     logged_bool(t, "long_prefill_token_threshold_adaptive"))
    spc = logged_bool(t, "enable_mamba_shared_prefix_checkpoint")
    rec.update(logged_max_num_batched_tokens=mbt, logged_long_prefill_token_threshold=lpt, logged_long_prefill_adaptive=lpa,
               logged_shared_prefix_checkpoint=spc)
    if not nd:
        unread.append("non-default args line (LP / SPC / batched tokens)")
    elif k == "LP":
        if (mbt, lpt, lpa) != (2048, 1024, True):
            why.append(f"logged batched tokens / long-prefill threshold / adaptive {(mbt, lpt, lpa)} != (2048, 1024, True)")
    elif mbt not in (None, 1024) or lpt not in (None, 0) or lpa is True:
        why.append(f"long-prefill / batched-token change on {k}: {(mbt, lpt, lpa)}")
    if nd and k == "SPC" and spc is not True:
        why.append(f"logged enable_mamba_shared_prefix_checkpoint {spc} on SPC")
    elif nd and k != "SPC" and spc is True:
        why.append(f"the shared-prefix checkpoint on {k}")
    sp = log_.get("speculative_config") if isinstance(log_.get("speculative_config"), dict) else None
    rec["logged_spec"] = sp
    if sp is not None and int(sp.get("num_speculative_tokens") or 0) != int(arm["k"]):
        why.append(f"logged num_speculative_tokens {sp.get('num_speculative_tokens')} != {arm['k']}")
    if k == "DYN" and sp is not None and not sp.get("num_speculative_tokens_per_batch_size"):
        why.append("the dynamic schedule is not in the logged speculative_config")
    if k != "DYN" and sp is not None and sp.get("num_speculative_tokens_per_batch_size"):
        why.append(f"a dynamic schedule on {k}")
    hits = [l.strip()[:200] for l in t.splitlines() if any(x in l for x in DYN_FAIL_RX)]
    rec["dyn_fallback_lines"] = hits[:4]
    if hits:
        why.append(f"dynamic speculative decoding fell back (cudagraph FULL lost / schedule disabled): {hits[:2]}")
    rec["unreadable"] = unread
    rec["base_mismatch"] = base_mm
    rec["why"] = "; ".join(why) or None
    rec["real_failure"] = bool(why)
    rec["ok"] = not why
    return rec


def sibling_label(rec, block):
    """SPC activation read-back (recorded; = nbs2_lib.sibling_label): prefix-hit tokens of the third sibling C (= P + Z after
    P + X and P + Y). Without the junction checkpoint the hit stops at the last whole block of P; with it, near len(P)."""
    p_len = (rec or {}).get("prefix_tokens")
    hc = (rec or {}).get("hits", {}).get("C")
    if not isinstance(p_len, int) or not isinstance(hc, (int, float)) or not isinstance(block, int) or block <= 0:
        return {"label": "INCONCLUSIVE_SIBLING", "why": "missing numbers"}
    base = (p_len // block) * block
    lab = "SPC_JUNCTION_HIT" if hc >= base + 128 else ("BLOCK_BOUNDARY_ONLY" if hc >= base - 64 else "BELOW_BLOCK_BOUNDARY")
    return {"label": lab, "prefix_tokens": p_len, "block": block, "block_boundary_hit": base, "c_hits": hc}


# ------------------------------------------------------------------ the KV ledger (metrics read; records only; = nbs2_lib)
_METRIC = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{([^}]*)\})?\s+(\S+)", re.M)
LEDGER_KEYS = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting",
               "vllm:num_preemptions_total", "vllm:prompt_tokens_total", "vllm:prefix_cache_hits_total",
               "vllm:prompt_tokens_cached_total")


def parse_metrics(text):
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
           "waiting_by_reason_max": wr, "running_max": max((m.get("vllm:num_requests_running") or 0) for _, m in sel)}
    if isinstance(pool, (int, float)) and pool > 0 and us:
        out["blocks_in_use_max"] = int(round(max(us) * pool))
        out["free_blocks_min"] = int(round(pool - max(us) * pool))
    if level and band_tokens and isinstance(F, (int, float)):
        per = math.ceil((band_tokens + (gen_tokens or 0) / max(1, level)) / block) + int(F)
        out["running_required_blocks_est"] = int(level * per)
    return out


# ------------------------------------------------------------------ the [LIVE] line (print only)
def _band(wkey):
    return str(wkey).split("@")[0]


def b_window_for(b_rec, wkey):
    """B's window for a lever window key: the same key, else the first valid same-band window (S28's 20k@28 -> B's 20k@24)."""
    st = (b_rec or {}).get("steady") or {}
    w = st.get(wkey)
    if isinstance(w, dict) and isinstance(w.get("tok_s"), (int, float)):
        return wkey, w
    for kk, v in st.items():
        if isinstance(v, dict) and not str(kk).startswith("profile") and _band(kk) == _band(wkey) and isinstance(v.get("tok_s"), (int, float)):
            return kk, v
    return None, None


def live_line(minutes, arm_key, wkey, w, b_rec=None):
    """[LIVE <min>] <arm> <window> tok/s <x> | AL <y> | running <n> | preempt <p> | vs B <+z %> (print only). B itself (and a
    lever without a B window in this run) is compared with the NB-S1 full reference (vs B(ref))."""
    w = w or {}
    x = w.get("tok_s")

    def f(v, nd=1):
        return f"{v:.{nd}f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
    run = w.get("running_max") if w.get("running_max") is not None else w.get("level")
    run = int(run) if isinstance(run, (int, float)) else run
    vs = "n/a"
    if isinstance(x, (int, float)):
        ref_key, ref = (None, None) if arm_key == BASE_KEY else b_window_for(b_rec, wkey)
        if ref is not None and ref.get("tok_s"):
            vs = f"{100.0 * (x / ref['tok_s'] - 1.0):+.1f} %"
        else:
            r_ = B_REF["tok_s"].get(wkey) or next((v for kk, v in B_REF["tok_s"].items() if _band(kk) == _band(wkey)), None)
            vs = f"(ref) {100.0 * (x / r_ - 1.0):+.1f} %" if r_ else "n/a"
    st = "" if w.get("status") in (None, "STEADY") else f" [{w.get('status')}]"
    return (f"[LIVE {f(minutes)}] {arm_key} {wkey} tok/s {f(x)} | AL {f(w.get('mean_acceptance_length'), 2)} | running {run} | "
            f"preempt {f(w.get('preemptions'), 0)} | vs B {vs}{st}")


# ------------------------------------------------------------------ budget / copies
def copy_walk(arms, switch, budget_min=BUDGET_MIN):
    """The planned walk of one copy: setup, then every switched-on arm in order (it starts with est_min + 5 min left)."""
    sw = switch_map(switch)
    left = budget_min - SETUP_MIN
    walk, runtime = [], SETUP_MIN
    for k, a in arms.items():
        if not sw.get(k, True):
            continue
        fits = left >= a["est_min"] + START_RESERVE_MIN
        walk.append({"arm": k, "est_min": a["est_min"], "cap_min": a["cap_min"], "left_min": round(left, 1), "runs": fits})
        if fits:
            left -= a["est_min"]
            runtime += a["est_min"]
    return {"walk": walk, "runtime_min": round(runtime, 1), "left_end_min": round(left, 1), "runs": [w["arm"] for w in walk if w["runs"]],
            "skipped": [w["arm"] for w in walk if not w["runs"]], "all_run": all(w["runs"] for w in walk), "budget_min": budget_min}


def copy_need_min(arms, switch):
    """The latest planned end of a copy with every arm's FULL body cap (start + ready + cap + post)."""
    sw = switch_map(switch)
    t, end = SETUP_MIN, SETUP_MIN
    for k, a in arms.items():
        if not sw.get(k, True):
            continue
        tm = a["time_model"]
        end = max(end, t + tm["ready"] + a["cap_min"] + tm["post"])
        t += a["est_min"]
    return round(end, 1)


# ------------------------------------------------------------------ PROF / records (record only)
def prof_record(b_rec, switch):
    """PROF's record from B's boot: DONE when both 30-s profiled windows kept trace files, else NOT_INTEGRATED / NOT_RUN."""
    sw = switch_map(switch)
    if not sw.get("PROF"):
        return {"status": "SKIPPED_SWITCH_OFF", "why": "switched off in CELL 1 (NBD1_ON)"}
    if not sw.get(BASE_KEY):
        return {"status": "SKIPPED_DEPENDENCY", "why": "PROF runs inside B's boot; B is switched off"}
    st = (b_rec or {}).get("steady") or {}
    keys = [f"profile_{b}@{n}" for b, n in PROF_WINDOWS]
    wins = {kk: st.get(kk) for kk in keys}
    if not any(isinstance(w, dict) for w in wins.values()):
        return {"status": "NOT_RUN", "why": f"B status {(b_rec or {}).get('status')}: no PROF window ran", "windows": {}}
    got = {kk: [f[0] for f in ((w or {}).get("trace_files") or [])] for kk, w in wins.items()}
    ok = all(got[kk] for kk in keys)
    return {"status": "DONE" if ok else "NOT_INTEGRATED", "not_integrated": not ok, "secs": PROF_SECS,
            "windows": {kk: {"status": (w or {}).get("status"), "trace_files": got[kk][:8], "trace_dir": (w or {}).get("trace_dir"),
                             "profile_request": (w or {}).get("profile_request")} for kk, w in wins.items()},
            "graph_spans": (b_rec or {}).get("graph_spans"), "timeline": (b_rec or {}).get("timeline"),
            "stage_compute": (b_rec or {}).get("stage_compute"),
            "why": "ok" if ok else f"PROF windows without trace files: {[kk for kk in keys if not got[kk]]}"}


def _valid(T, w):
    try:
        return bool(T.window_ok(w)[0])
    except Exception:
        return False


def band_table(T, B, X, thr_floor=None, speed_min=0.02):
    """Per band: B's and X's valid windows (any level), mean tok/s, gain, the threshold max(B's 8k A/A spread, 2 %, the long
    floor). Recorded only (labels never stop an arm)."""
    wb, wx = T._win(B or {}) or {}, T._win(X or {}) or {}
    aa = None
    a8 = [wb.get(k) for k in ("8k@16#1", "8k@16#2")]
    if all(isinstance(w, dict) and _valid(T, w) for w in a8):
        aa = abs(a8[0]["tok_s"] - a8[1]["tok_s"]) / ((a8[0]["tok_s"] + a8[1]["tok_s"]) / 2)
    out = {"b_aa_spread_8k": None if aa is None else round(aa, 4), "bands": {}}
    for band in ("8k", "20k", "40k", "90k", "116k"):
        bv = [v["tok_s"] for kk, v in wb.items() if isinstance(v, dict) and _band(kk) == band and _valid(T, v)]
        xv = [v["tok_s"] for kk, v in wx.items() if isinstance(v, dict) and _band(kk) == band and _valid(T, v)]
        r = {"b_tok_s": [round(x, 1) for x in bv], "x_tok_s": [round(x, 1) for x in xv]}
        if bv and xv:
            bm, xm = sum(bv) / len(bv), sum(xv) / len(xv)
            thr = max(aa or 0.0, speed_min, float((thr_floor or {}).get(band) or 0.0))
            g = xm / bm - 1.0
            r.update(gain=round(g, 4), threshold=round(thr, 4), label="GAIN" if g > thr else ("REGRESSION" if g < -thr else "WITHIN_NOISE"))
        else:
            r["label"] = "INCONCLUSIVE"
        out["bands"][band] = r
    return out


def arm_records(T, rec, key, b_rec=None, thr_floor=None):
    rec = rec or {}
    return {"arm": key, "status": rec.get("status"), "why": rec.get("why") or rec.get("early_stop"),
            "windows": {k: {f: (w or {}).get(f) for f in ("status", "tok_s", "tick_ms", "mean_acceptance_length", "preemptions",
                                                          "running_eq_level", "level")}
                        for k, w in (rec.get("steady") or {}).items() if isinstance(w, dict)},
            "pool_blocks": (rec.get("workers") or {}).get("num_gpu_blocks"), "residents": rec.get("k100_residents"),
            "pin": rec.get("pin"), "tries": rec.get("tries"), "smoke_gate": rec.get("smoke_gate"),
            "d1_readback": {k: (rec.get("d1_readback") or {}).get(k) for k in ("ok", "why", "unreadable")},
            "qsa": ((rec.get("d1_readback") or {}).get("qsa") or {}).get("label"),
            "qsa_end": ((rec.get("d1_records") or {}).get("qsa_end") or {}).get("label"),
            "bands_vs_b": band_table(T, b_rec, rec, thr_floor) if key != BASE_KEY and b_rec else None}


def verdict(T, recs, arms, switch, thr_floor=None):
    """Record only (user rule: labels never stop anything; read the raw numbers): every arm's raw windows and its bands vs B."""
    b = recs.get(BASE_KEY)
    out = {"lib": LIB_VERSION, "verdict": "RECORDED_RAW_NUMBERS", "switch": switch_map(switch), "b_ref": B_REF,
           "arms": {k: arm_records(T, recs.get(k), k, b, thr_floor) for k in arms},
           "prof": prof_record(b, switch)}
    out["table"] = {k: {wk: (w.get("tok_s"), w.get("mean_acceptance_length"), w.get("preemptions")) for wk, w in v["windows"].items()}
                    for k, v in out["arms"].items()}
    out["gains"] = {k: {bd: (c.get("gain"), c.get("label")) for bd, c in ((v.get("bands_vs_b") or {}).get("bands") or {}).items()}
                    for k, v in out["arms"].items() if v.get("bands_vs_b")}
    out["why"] = "; ".join(f"{k} {v.get('status')}" for k, v in out["arms"].items())
    return out


# ------------------------------------------------------------------ the NB-D1 real-kit gate (build time)
def gate_from_report(base_gate, rep, closed_sha, keys=KIT_KEYS):
    """The notebook's KV_KIT_GATE: NBDF-K100's passed gate (its real kit validated the base boot path, e2198285) AND the
    NB-D1 real kit on THIS closed notebook: provenance (the kit's notebook_sha256 == the closed build), ALL_OK, every NB-D1
    key True. Any miss keeps the gate closed (every arm SKIPPED_KV_KIT_GATE at run time)."""
    g = copy.deepcopy(base_gate or {})
    why = []
    if not (base_gate or {}).get("passed"):
        why.append("the embedded NBDF-K100 real-kit gate did not pass")
    v = (rep or {}).get("verdict") or {}
    chk = v.get("checks") or {}
    nbs = (rep or {}).get("notebook_sha256")
    if not nbs or nbs != closed_sha:
        why.append(f"STALE_NOTEBOOK_SHA: the kit ran {nbs}, the closed NB-D1 build is {closed_sha}")
    if v.get("ALL_OK") is not True:
        why.append(f"real kit not ALL_OK: {sorted(k for k, x in chk.items() if x is not True)[:12]}")
    miss = [k for k in keys if chk.get(k) is not True]
    if miss:
        why.append(f"NB-D1 real-kit checks not True: {miss}")
    g.update(passed=not why, why=why or ["NBDF-K100 gate passed + the NB-D1 real kit ALL OK on the closed build, provenance bound"],
             nbd1_kit_checks={k: chk.get(k) for k in keys}, nbd1_provenance={"kit_notebook_sha256": nbs, "closed_build_sha256": closed_sha})
    return g


def closed_gate(base_gate, why="NB-D1 closed build: the NB-D1 real CPU kit has not passed (build with --kv-gate-report)"):
    g = copy.deepcopy(base_gate or {})
    g.update(passed=False, why=[why], nbd1_kit_checks={k: None for k in KIT_KEYS}, nbd1_provenance=None)
    return g


def summary_json(x):
    return json.dumps(x, default=str)
