"""E25 NB-M1 library (2026-11-02): MTP lever arms on the PROVEN production base Q = NBDF-K100's Q arm verbatim = E28 K100
production (klee100 3bpw-MTP on Merlin v3.0.0 ad42e37, TP2 x PP2, MTP K3 probabilistic, Q-defer, I8 int8 draft head,
seqs 24, pin 7,933,526,016, its Q pin ladder, A/A, windows 8k@16 x 2 / 20k@24 / 40k@16 / 90k@8 / 116k@6, TF, oracle,
profiles). Pure logic, stdlib only; nbdf_lib (N) and nb4t_lib (T) are read through the module objects handed in.

Arms (each = Q + exactly ONE change, run in this order; no cross-arm skip):
  V65  VLLM_DRAFT_VOCAB_FILE (+ VLLM_DRAFT_VOCAB_SHA256) = the 65,536-token ARC table (draft_head.py / mtp.py:525-600)
  V32  the same with the 32,768-token table
  I4   draft_lm_head_quant int4 (DRAFT_HEAD=int4 into Merlin's build(): launch.py:319-322; speculative.py:610-622)
  IDX  --attention-config {"indexer_kv_dtype":"fp8"} (JSON form: the dotted form fails the --help gate)
  S28  --max-num-seqs 28 + captures = Merlin's capture_sizes_for(28, 3) (Q's rule extended)
  S32  --max-num-seqs 32 + capture_sizes_for(32, 3)
  LP   --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive
  G1   VLLM_GDN_DECODE_FASTPATH=1          G2   VLLM_GDN_SHARED_ALIGN_INDEX=1
  K4   MTP K4 (trailing --speculative-config) + captures = every multiple of 5 up to 24 x 5
  DYN  K4 + num_speculative_tokens_per_batch_size [[1, 8, 4], [9, 24, 3]] + captures covering both tiers
  PROF no change: 30-s torch.profiler windows at 8k@16 and 20k@24 inside Q's own boot (NB-S1's profile machinery)
The draft head of a lever = Q's (int8 while Q_I8 is on, else BF16), I4's = int4. Vocab: Merlin quantizes the REDUCED
head (QuantizedDraftLMHead wraps the N/TP-row ParallelLMHead; N/TP % 64 == 0), so a vocab arm runs vocab + int8 = Q + one
change (draft_head.py, speculator.py:547-603, mtp.py:548-566 / 697-752). Real failures (that arm only, NOT_INTEGRATED):
the arm's own change not read back (env in all 4 workers, the REDUCED DRAFT VOCAB lines, the draft head lines, the indexer
cache dtype, the logged seqs / batched tokens / long-prefill threshold / spec K / dynamic schedule) or present on another
arm."""
import copy
import hashlib
import json
import math
import re

LIB_VERSION = "nbm1-lib 2026-11-02 r1 (MTP levers on Q = NBDF-K100 Q / E28 K100)"
LEVERS = ("V65", "V32", "I4", "IDX", "S28", "S32", "LP", "G1", "G2", "K4", "DYN")
ARM_KEYS = ("Q",) + LEVERS
SWITCH_KEYS = ARM_KEYS + ("PROF",)
SWITCH_DEFAULT = {k: True for k in SWITCH_KEYS}
COPIES = {"a": dict({k: False for k in SWITCH_KEYS}, Q=True, V65=True, V32=True, I4=True),
          "b": dict({k: False for k in SWITCH_KEYS}, Q=True, IDX=True, S28=True, S32=True, LP=True),
          "c": dict({k: False for k in SWITCH_KEYS}, Q=True, G1=True, G2=True, K4=True, DYN=True, PROF=True)}
OWNERS = {"a": "foysalemonshanto", "b": "esfersami50", "c": "foysalemonshanto"}
PORTS = {"Q": 8011, "V65": 8101, "V32": 8111, "I4": 8121, "IDX": 8131, "S28": 8141, "S32": 8151, "LP": 8161,
         "G1": 8171, "G2": 8181, "K4": 8191, "DYN": 8201}
GDN_ENV = {"G1": "VLLM_GDN_DECODE_FASTPATH", "G2": "VLLM_GDN_SHARED_ALIGN_INDEX"}
ENV_VOCAB, ENV_VOCAB_SHA = "VLLM_DRAFT_VOCAB_FILE", "VLLM_DRAFT_VOCAB_SHA256"     # draft_head.py:32-33 (Merlin's names)
M1_ENV_KEYS = (GDN_ENV["G1"], GDN_ENV["G2"], ENV_VOCAB, ENV_VOCAB_SHA)
VOCAB_DS = "foysalemonshanto/q27-e29-final-payload"
VOCAB_TABLES = {"V65": ("draft_vocab_k100_arc_n65536.pt", "a915fa6d5ad711817976de5c988a3950f43e7b2fbc058b04325e75c8501cf022", 65536),
                "V32": ("draft_vocab_k100_arc_n32768.pt", "a164d1a71d27dab600c014673c34accfdeb8695141a63be8aa4bfb4026695248", 32768)}
VOCAB_SIZE, TP, N_WORKERS = 248320, 2, 4
IDX_ARGS = ["--attention-config", '{"indexer_kv_dtype":"fp8"}']
LP_ARGS = ["--max-num-batched-tokens", "2048", "--long-prefill-token-threshold", "1024", "--long-prefill-token-threshold-adaptive"]
SEQ_ARMS = {"S28": 28, "S32": 32}
K4 = 4
DYN_SCHEDULE = [[1, 8, 4], [9, 24, 3]]
DYN_FAIL_RX = ("Overriding cudagraph_mode", "Disabling num_speculative_tokens_per_batch_size")
CAPTURE_LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128]   # Merlin registry.py:53 _LADDER
PROF_WINDOWS = (("8k", 16), ("20k", 24))
PROF_SECS = 30.0
TITLES = {"Q": "Q = NBDF-K100 Q verbatim = E28 K100 production (MTP K3, I8, Q-defer, seqs 24, pin 7,933,526,016), A/A",
          "V65": "Q + VLLM_DRAFT_VOCAB_FILE 65,536-token ARC table (+ its sha256), reduced head under Q's int8 quant",
          "V32": "Q + VLLM_DRAFT_VOCAB_FILE 32,768-token ARC table (+ its sha256), reduced head under Q's int8 quant",
          "I4": "Q + draft_lm_head_quant int4 (DRAFT_HEAD=int4) instead of int8",
          "IDX": "Q + --attention-config {\"indexer_kv_dtype\":\"fp8\"}",
          "S28": "Q + --max-num-seqs 28 (captures capture_sizes_for(28, 3))",
          "S32": "Q + --max-num-seqs 32 (captures capture_sizes_for(32, 3))",
          "LP": "Q + --max-num-batched-tokens 2048 --long-prefill-token-threshold 1024 --long-prefill-token-threshold-adaptive",
          "G1": "Q + VLLM_GDN_DECODE_FASTPATH=1", "G2": "Q + VLLM_GDN_SHARED_ALIGN_INDEX=1",
          "K4": "Q + MTP K4 (captures every multiple of 5 up to 120)",
          "DYN": "Q + MTP K4 at <= 8 running, K3 at 9-24 (num_speculative_tokens_per_batch_size)"}

# ------------------------------------------------------------------ the time model (minutes; GPU data 2026-11-01)
# NBDF-K100 GPU (q27-xhigh-e27-nbdf-k100): setup 5.6, Q ready 543 s, Q body hit its 41-min cap after the profiles (TF not
# run), Q post 6.4; window spacing 8k@16 110 s, 20k@24 215 s, 40k@16 183 s, 90k@8 275 s, 116k@6 272 s. NB-S1 GPU v1: first
# boot ready 732 s, later boots 285 / 255 s. A changed compile key (vocab table, attention / scheduler config, spec K)
# recompiles: "recompile" in between.
SETUP_MIN = 6.0
READY_MIN = {"first": 11.0, "recompile": 7.0, "warm": 4.8, "warm_capture": 5.5}
ITEM_MIN = {"facts": 1.5, "smoke_ladder": 3.0, "smoke": 1.3, "prefill_probe": 0.5, "oracle": 1.5, "oracle_aa": 1.5,
            "long_oracle": 2.5, "prefix": 4.0, "warmup": 1.0, "profiles_q": 7.0, "tf_q": 3.0, "tf": 1.5, "end": 1.2,
            "prof": 7.3}
WINDOW_MIN = {"8k@16": 1.85, "20k@24": 3.6, "40k@16": 3.05, "90k@8": 4.6, "116k@6": 4.55, "20k@28": 4.2, "20k@32": 4.6}
POST_MIN = {"q": 3.8, "lever": 0.8}
Q_WINDOWS = [["8k", 16, "#1"], ["8k", 16, "#2"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]]
LEVER_ITEMS_OFF = {"ladder": False, "prefill_probe": False, "long_oracle": False, "prefix": False, "profiles": False, "tf": False}
PLAN = {   # per lever: ready class, window set (A/A-free; 90k + 116k always), the TF (target-numerics levers only)
    "V65": ("recompile", [["8k", 16, "#1"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]], False),
    "V32": ("recompile", [["8k", 16, "#1"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]], False),
    "I4": ("recompile", [["8k", 16, "#1"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]], False),
    "IDX": ("recompile", [["8k", 16, "#1"], ["90k", 8, ""], ["116k", 6, ""]], True),
    "S28": ("warm_capture", [["20k", 28, ""], ["90k", 8, ""], ["116k", 6, ""]], False),   # 20k at the arm's seqs (Q: 20k@24)
    "S32": ("warm_capture", [["20k", 32, ""], ["90k", 8, ""], ["116k", 6, ""]], False),
    "LP": ("recompile", [["8k", 16, "#1"], ["90k", 8, ""], ["116k", 6, ""]], True),
    "G1": ("warm", [["8k", 16, "#1"], ["90k", 8, ""], ["116k", 6, ""]], False),
    "G2": ("warm", [["8k", 16, "#1"], ["90k", 8, ""], ["116k", 6, ""]], False),
    "K4": ("recompile", [["8k", 16, "#1"], ["90k", 8, ""], ["116k", 6, ""]], False),
    "DYN": ("recompile", [["8k", 16, "#1"], ["90k", 8, ""], ["116k", 6, ""]], False)}
BUDGET_MIN = 180            # the hard stop of every copy (the copies differ only in NBM1_ON): max over the copies' walks
DEADMAN_MIN = BUDGET_MIN + 5
COPY_RUNTIME_MAX = 175      # build-time assertion: each copy's planned runtime (setup + its arms) <= this (C: Q + PROF = 69)
START_RESERVE_MIN = 5.0     # nb4t_lib START_RESERVE_SEC: an arm starts only with est_min + 5 min left


def _wkey(w):
    return f"{w[0]}@{int(w[1])}"


def q_time(prof_on):
    """(est_min, cap_min, breakdown) of Q (+ PROF's two 30-s profiles inside its boot when on)."""
    body = {k: ITEM_MIN[k] for k in ("facts", "smoke_ladder", "prefill_probe", "oracle", "oracle_aa", "long_oracle", "prefix",
                                       "warmup", "profiles_q", "tf_q", "end")}
    body["windows"] = round(sum(WINDOW_MIN[_wkey(w)] for w in Q_WINDOWS), 2)
    if prof_on:
        body["prof"] = ITEM_MIN["prof"]
    b = sum(body.values())
    est = READY_MIN["first"] + b + POST_MIN["q"]
    return int(math.ceil(est)), int(math.ceil(b + 1.0)), {"ready": READY_MIN["first"], "body": body, "post": POST_MIN["q"]}


def lever_time(key):
    rc, wins, tf = PLAN[key]
    body = {k: ITEM_MIN[k] for k in ("facts", "smoke", "oracle", "warmup", "end")}
    body["windows"] = round(sum(WINDOW_MIN[_wkey(w)] for w in wins), 2)
    if tf:
        body["tf"] = ITEM_MIN["tf"]
    b = sum(body.values())
    est = READY_MIN[rc] + b + POST_MIN["lever"]
    return int(math.ceil(est)), int(math.ceil(b + 1.0)), {"ready": READY_MIN[rc], "ready_class": rc, "body": body, "post": POST_MIN["lever"]}


# ------------------------------------------------------------------ switches (CELL 1's NBM1_ON)
def switch_map(on, keys=SWITCH_KEYS):
    """NBM1_ON -> {key: bool}: absent (None) or not a dict -> every arm on; a dict -> bool(on[k]), a missing key -> off."""
    if on is None or not isinstance(on, dict):
        return {k: True for k in keys}
    return {k: bool(on.get(k, False)) for k in keys}


def switch_record(on, keys=SWITCH_KEYS):
    m = switch_map(on, keys)
    return {"NBM1_ON": on if isinstance(on, (dict, type(None))) else repr(on)[:200], "effective": m, "absent": on is None,
            "invalid_not_a_dict": on is not None and not isinstance(on, dict),
            "unknown_keys": sorted(str(k) for k in set(on) - set(keys)) if isinstance(on, dict) else [],
            "off": [k for k, v in m.items() if not v]}


# ------------------------------------------------------------------ captures / argv helpers
def capture_sizes_for(seqs, k=3):
    """= Merlin tools/merlin/registry.py:83-89 (the shipped ladder, the top appended)."""
    top = int(seqs) * (int(k) + 1)
    s = [x for x in CAPTURE_LADDER if x <= top]
    if top not in s:
        s.append(top)
    return s


def k4_captures(seqs=24):
    return [b * (K4 + 1) for b in range(1, int(seqs) + 1)]


def dense_schedule(schedule, max_seqs, k_max):
    """= Merlin v1/spec_decode/dynamic/utils.py build_dynamic_sd_schedule_lookup (nbs2_lib.dense_schedule)."""
    parsed = sorted((int(a), int(b), int(k)) for a, b, k in schedule)
    dense, nxt, last = [0] * (max_seqs + 1), 1, None
    for a, b, k in parsed:
        if a > nxt and last is not None:
            for x in range(nxt, min(a, max_seqs + 1)):
                dense[x] = min(k_max, last)
        for x in range(max(a, nxt), min(b, max_seqs) + 1):
            dense[x] = min(k_max, k)
        nxt, last = max(nxt, b + 1), k
        if nxt > max_seqs:
            break
    for x in range(nxt, max_seqs + 1):
        dense[x] = min(k_max, last)
    return dense


def dyn_captures(seqs=24):
    d = dense_schedule(DYN_SCHEDULE, int(seqs), K4)
    return sorted({b * (1 + d[b]) for b in range(1, int(seqs) + 1)})


def dyn_coverage(captures, seqs=24):
    """Every b = 1 .. seqs an exact graph at its own tier's query length 1 + K(b) (cudagraph_utils.py rounding)."""
    d = dense_schedule(DYN_SCHEDULE, int(seqs), K4)
    caps = sorted(set(int(c) for c in captures))
    bad = []
    for b in range(1, int(seqs) + 1):
        q = 1 + d[b]
        g = {-(-c // q) * q for c in caps if -(-c // q) * q <= caps[-1]}
        if b * q not in g:
            bad.append(b)
    return {"covers_all": not bad, "uncovered_or_padded_b": bad, "padded_b": [], "uncovered_b": bad,
            "tiers": sorted(set(d[1:]))}


def set_last(extra, flag, value):
    """The LAST occurrence of `flag` gets `value` (vLLM's argparse: last wins)."""
    out = list(extra)
    idx = [i for i, x in enumerate(out) if x == flag]
    if not idx or idx[-1] + 1 >= len(out):
        raise ValueError(f"{flag} not in the base argv")
    out[idx[-1] + 1] = value
    return out


def spec_with_head(spec, head):
    sp = dict(spec)
    sp.pop("draft_lm_head_quant", None)
    if head:
        sp["draft_lm_head_quant"] = head
    return sp


# ------------------------------------------------------------------ vocab tables (the E29 payload dataset)
def vocab_candidates(inp, env):
    import os.path as _p
    d = (env or {}).get("NBM1_VOCAB_DIR")
    out = [_p.join(str(inp), "datasets", VOCAB_DS), _p.join(str(inp), VOCAB_DS.split("/")[1])]
    return out + ([d] if d else [])


def resolve_vocab(inp, env, reader=None):
    """{arm: {"file", "path", "sha256", "n", "found", "why"}}: the first candidate dir whose file has the pinned sha256."""
    import os.path as _p

    def rd(p):
        with open(p, "rb") as fh:
            return fh.read()
    reader = reader or rd
    out = {}
    for k, (name, sh, n) in VOCAB_TABLES.items():
        rec = {"file": name, "sha256": sh, "n": n, "path": None, "found": False, "tried": []}
        for d in vocab_candidates(inp, env):
            p = _p.join(d, name)
            try:
                b = reader(p)
            except Exception as e:
                rec["tried"].append([p, type(e).__name__])
                continue
            got = hashlib.sha256(b).hexdigest()
            rec["tried"].append([p, got[:12]])
            if got == sh:
                rec.update(path=p, found=True)
                break
        rec["why"] = "ok" if rec["found"] else f"VOCAB_TABLE_MISSING: {name} (sha {sh[:12]}) not under {vocab_candidates(inp, env)} (dataset {VOCAB_DS})"
        out[k] = rec
    return out


# ------------------------------------------------------------------ the arm plan
def m1_arms(N, aq, switch=None, vocab=None):
    """aq = NBDF-K100's Q arm (nbdf_lib.arm_plan + configure(SEQS=24, ...) + the K100 Q tail). Returns {key: arm}: Q =
    the same arm (budget fields and the M1 env keys in its expect only), every lever = a deep copy + exactly one change."""
    sw = switch_map(switch) if not isinstance(switch, dict) or set(switch) != set(SWITCH_KEYS) else dict(switch)
    vocab = vocab or {}

    def expect_of(a):
        e = {w: a["env"].get(w, N.WATCHED_DEFAULT.get(w)) for w in N.WATCHED}
        for k in M1_ENV_KEYS:
            e[k] = a["env"].get(k)
        return e

    q = copy.deepcopy(aq)
    est, cap, tb = q_time(sw.get("PROF", True) and sw.get("Q", True))
    q.update(key="Q", name="nbm1_q", title=TITLES["Q"], est_min=est, cap_min=cap, time_model=tb, m1_lever=False,
             m1_change=None, m1_items=None, m1_head="q")
    q["expect"] = expect_of(q)
    arms = {"Q": q}
    for k in LEVERS:
        a = copy.deepcopy(q)
        env_add, argv_ops, head = {}, [], None
        if k in GDN_ENV:
            env_add[GDN_ENV[k]] = "1"
        elif k in VOCAB_TABLES:
            v = vocab.get(k) or {}
            env_add[ENV_VOCAB] = v.get("path") or f"MISSING:{VOCAB_TABLES[k][0]}"
            env_add[ENV_VOCAB_SHA] = VOCAB_TABLES[k][1]
        elif k == "I4":
            head = "int4"
        elif k == "IDX":
            argv_ops = [["append", list(IDX_ARGS)]]
        elif k == "LP":
            argv_ops = [["append", list(LP_ARGS)]]
        elif k in SEQ_ARMS:
            s = SEQ_ARMS[k]
            caps = capture_sizes_for(s, 3)
            argv_ops = [["set", "--max-num-seqs", str(s)], ["set", "--compilation-config", N.compilation_json(caps)]]
            a["captures"], a["streams"] = caps, s
            a["expect_cfg"].update(max_num_seqs=s, captures=caps)
            a["coverage"] = N.coverage(caps, s, 3)
        elif k in ("K4", "DYN"):
            sp = {"method": "mtp", "num_speculative_tokens": K4, "draft_sample_method": "probabilistic"}
            caps = k4_captures(24)
            if k == "DYN":
                sp["num_speculative_tokens_per_batch_size"] = copy.deepcopy(DYN_SCHEDULE)
                caps = dyn_captures(24)
            a["spec"], a["k"], a["captures"] = sp, K4, caps
            argv_ops = [["append", ["--speculative-config", N.spec_json(spec_with_head(sp, "int8"))]],
                        ["set", "--compilation-config", N.compilation_json(caps)]]
            a["expect_cfg"].update(num_speculative_tokens=K4, captures=caps)
            a["coverage"] = N.coverage(caps, 24, K4) if k == "K4" else dyn_coverage(caps, 24)
            ok_r, cap_r = N.qsa_ring(K4)
            a["ring_needed"], a["ring_capacity_expected"] = not ok_r, cap_r
            a["m1_trailing_spec"] = True
        a["env"].update(env_add)
        a["extra"] = apply_argv_ops(q["extra"], argv_ops)
        a["expect"] = expect_of(a)
        e_, c_, tb_ = lever_time(k)
        a.update(key=k, name=f"nbm1_{k.lower()}", port=PORTS[k], title=TITLES[k], long_aa=False, lever=True, m1_lever=True,
                 m1_change={"env": env_add, "argv": argv_ops, "head": head}, m1_head=head or "q",
                 m1_items=dict(LEVER_ITEMS_OFF, tf=PLAN[k][2]), m1_no_ladder=True, windows_fixed=copy.deepcopy(PLAN[k][1]),
                 est_min=e_, cap_min=c_, time_model=tb_, optional=False, short_only=False, iso=False)
        if k in VOCAB_TABLES:
            a["m1_vocab"] = {"file": VOCAB_TABLES[k][0], "sha256": VOCAB_TABLES[k][1], "n": VOCAB_TABLES[k][2],
                             "path": (vocab.get(k) or {}).get("path"), "found": bool((vocab.get(k) or {}).get("found"))}
        arms[k] = a
    return arms


def apply_argv_ops(extra, ops):
    x = list(extra)
    for op in ops or []:
        if op[0] == "append":
            x = x + list(op[1])
        elif op[0] == "set":
            x = set_last(x, op[1], op[2])
        else:
            raise ValueError(f"unknown argv op {op}")
    return x


def apply_head(N, arm, q_i8_on):
    """At boot (arm_boot's first lines): the lever's draft head = int4 (I4) or Q's (int8 while Q_I8 is on, else BF16).
    K4 / DYN carry a trailing spec: its draft_lm_head_quant follows the head. Returns the head."""
    head = "int4" if arm.get("m1_head") == "int4" else ("int8" if q_i8_on else None)
    arm["draft_head"] = head
    arm["expect_cfg"]["draft_head"] = head
    if arm.get("m1_trailing_spec"):
        js = N.spec_json(spec_with_head(arm["spec"], head))
        x = list(arm["extra"])
        idx = [i for i, v in enumerate(x) if v == "--speculative-config"]
        x[idx[-1] + 1] = js
        arm["extra"] = x
        for op in (arm.get("m1_change") or {}).get("argv") or []:     # the plan record follows (check_arms stays exact)
            if op[0] == "append" and op[1][:1] == ["--speculative-config"]:
                op[1][1] = js
    return head


def install(N, arms):
    """Rebind nbdf_lib's arm-set constants to the NB-M1 arms (every arm MTP: no D / SW / short / probe arm); the verdict
    then compares every lever with Q. Idempotent. Returns the new ORDER."""
    N.ORDER, N.D_ARMS = tuple(arms), tuple(k for k in arms if k != "Q")
    N.OPTIONAL = N.SHORT_ARMS = N.CONV_TORCH_ARMS = N.SHORT_EXIT_ARMS = N.ADMISSION_PROBE_ARMS = N.SW_BLOCK_ARMS = ()
    N.KV_RUN_OF = {}
    N.K_OF.update({k: a["k"] for k, a in arms.items()})
    return N.ORDER


def check_arms(N, T, arms, aq=None):
    """Raises unless: Q first and the listed order; unique names / ports (Merlin's 8000..8355 scheme); every arm MTP with
    Q-defer on the patched tree, no never-do item, coverage of every decode batch; Q = aq (when given) except the budget /
    name / expect fields; every lever = Q's env + exactly its planned env change, Q's argv with exactly its argv ops, Q's
    spec / head except the planned K / head change."""
    if tuple(arms) != ARM_KEYS:
        raise ValueError(f"NB-M1 arms {tuple(arms)} != {ARM_KEYS}")
    q = arms["Q"]
    if len({a["name"] for a in arms.values()}) != len(arms) or len({a["port"] for a in arms.values()}) != len(arms):
        raise ValueError("NB-M1: duplicate arm name / port")
    if max(a["port"] for a in arms.values()) + 8 > 8355:
        raise ValueError("NB-M1: ports outside Merlin's 8000..8355 scheme (with retry / iso offsets)")
    if aq is not None:
        for f in ("env", "extra", "spec", "captures", "k", "method", "kv_bytes", "q_label", "needs_patch", "expect_cfg", "port"):
            qa, qb = q.get(f), aq.get(f)
            if f == "expect_cfg":                # arm_boot sets draft_head at boot (Q_I8): not a plan difference
                qa, qb = ({x: y for x, y in (v or {}).items() if x != "draft_head"} for v in (qa, qb))
            if qa != qb:
                raise ValueError(f"NB-M1 Q.{f} differs from NBDF-K100's Q")
    for k, a in arms.items():
        if a.get("method") != "mtp" or (a.get("spec") or {}).get("method") != "mtp":
            raise ValueError(f"NB-M1 {k}: not an MTP arm")
        bad = T.never_do_violations(a["env"], a["extra"])
        if bad:
            raise ValueError(f"NB-M1 {k}: never-do {bad}")
        if not a.get("needs_patch") or a.get("q_label") != N.Q_LABEL or a["env"].get(T.ENV_DEFER) != "1":
            raise ValueError(f"NB-M1 {k}: every arm runs Q-defer on the patched tree")
        if any(x.startswith("DFLASH_") for x in a["env"]):
            raise ValueError(f"NB-M1 {k}: a DFLASH_* variable on an MTP arm")
        cov = a.get("coverage") or {}
        if not cov.get("covers_all"):
            raise ValueError(f"NB-M1 {k}: decode batches without a FULL graph {cov}")
        if N.captures_from_argv(a["extra"]) != a["captures"] or T.last_value(a["extra"], "--max-num-seqs") != str(a["expect_cfg"]["max_num_seqs"]):
            raise ValueError(f"NB-M1 {k}: argv captures / seqs != plan")
        if set(a["expect"]) != set(N.WATCHED) | set(M1_ENV_KEYS):
            raise ValueError(f"NB-M1 {k}: expect keys")
        if k == "Q":
            if any(q["env"].get(x) for x in M1_ENV_KEYS) or "--speculative-config" in q["extra"]:
                raise ValueError("NB-M1 Q: an M1 change on the base")
            continue
        ch = a["m1_change"]
        diff = {e: a["env"].get(e) for e in set(a["env"]) | set(q["env"]) if a["env"].get(e) != q["env"].get(e)}
        if diff != ch["env"]:
            raise ValueError(f"NB-M1 {k}: env diff {diff} != the planned change {ch['env']}")
        if a["extra"] != apply_argv_ops(q["extra"], ch["argv"]):
            raise ValueError(f"NB-M1 {k}: argv differs from Q beyond the planned change")
        n_changes = bool(ch["env"]) + bool(ch["head"]) + (1 if ch["argv"] else 0)
        if n_changes != 1:
            raise ValueError(f"NB-M1 {k}: {n_changes} change kinds (exactly one)")
        if k in ("K4", "DYN"):
            want = {"method": "mtp", "num_speculative_tokens": K4, "draft_sample_method": "probabilistic"}
            if k == "DYN":
                want["num_speculative_tokens_per_batch_size"] = DYN_SCHEDULE
                d = dense_schedule(DYN_SCHEDULE, 24, K4)
                if d[1:9] != [4] * 8 or d[9:25] != [3] * 16:
                    raise ValueError(f"NB-M1 DYN dense schedule {d[1:]}")
            if a["spec"] != want or a["k"] != K4:
                raise ValueError(f"NB-M1 {k}: spec {a['spec']}")
        elif a["spec"] != q["spec"] or a["k"] != q["k"]:
            raise ValueError(f"NB-M1 {k}: spec / K differ from Q")
        if (ch["head"] is not None) != (k == "I4") or (k == "I4" and ch["head"] != "int4"):
            raise ValueError(f"NB-M1 {k}: draft head change")
        if not a.get("windows_fixed") or not any(w[0] == "90k" for w in a["windows_fixed"]) or not any(w[0] == "116k" for w in a["windows_fixed"]):
            raise ValueError(f"NB-M1 {k}: the window set must include 90k and 116k (user rule)")
    return True


def pre_gate(arm, kv_gate):
    """Before a lever's boot: None = launch, else (status, why). The real CPU kit gate (closed build: every lever
    SKIPPED_KV_KIT_GATE, Q still runs); a vocab arm needs its table found with the pinned sha."""
    if not (kv_gate or {}).get("passed"):
        return "SKIPPED_KV_KIT_GATE", "the NB-M1 real CPU kit gate did not pass: " + str((kv_gate or {}).get("why"))[:400]
    v = arm.get("m1_vocab")
    if v is not None and not v.get("found"):
        return "NOT_INTEGRATED", f"VOCAB_TABLE_MISSING: {v.get('file')} (sha {str(v.get('sha256'))[:12]}) not found in {VOCAB_DS}"
    return None


def item_on(arm, item):
    """A body item of this arm (Q: every item; a lever: LEVER_ITEMS_OFF + its plan's TF)."""
    return bool((arm.get("m1_items") or {}).get(item, True))


# ------------------------------------------------------------------ read-backs (real failures)
_VOCAB_RX = re.compile(r"REDUCED DRAFT VOCAB ACTIVE: tp_rank=(\d+) N=(\d+) of (\d+) \(file (\S+), table")
_WTAG = re.compile(r"\(Worker_PP(\d+)_TP(\d+)")


def env_rows(env_raw):
    """[(rank, {key: value})] of the nb4s_env RPC (every requested key, RPC_KEYS)."""
    res = (env_raw or {}).get("results") if isinstance(env_raw, dict) else None
    out = []
    for w in res if isinstance(res, list) else []:
        w = w if isinstance(w, dict) else {}
        out.append((w.get("rank"), w.get("env") if isinstance(w.get("env"), dict) else {}))
    return out


def env_values(env_raw, key):
    return [e.get(key) for _, e in env_rows(env_raw)]


def vocab_lines(log_text):
    out = []
    for l in (log_text or "").splitlines():
        m = _VOCAB_RX.search(l)
        if m:
            w = _WTAG.search(l)
            out.append({"tp_rank": int(m.group(1)), "n": int(m.group(2)), "of": int(m.group(3)), "file": m.group(4),
                        "worker": f"PP{w.group(1)}_TP{w.group(2)}" if w else None})
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
    t = (log_text or "")
    lines = [l for l in t.splitlines() if "non-default args:" in l or "Initializing a V1 LLM engine" in l]
    return any(re.search(r"indexer_kv_dtype['\"]?\s*[:=]\s*['\"]fp8", l) for l in lines)


def idx_readback(workers):
    """The QSA indexer compressed-key cache spec (nb1_cache_info kv_cache_specs) dtype per rank (nbs2_lib.idx_readback)."""
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


def readback(arm, log_text, env_raw, crb, workers, i8rb):
    """The lever's own change read back (real failure -> NOT_INTEGRATED, this arm only) and every OTHER lever's change
    absent. crb = nbdf_lib.config_readback, i8rb = the E28 I8 read-back with want = the arm's head."""
    k, why = arm["key"], []
    ch = arm.get("m1_change") or {"env": {}, "argv": [], "head": None}
    t = log_text or ""
    rec = {"arm": k}
    for key in M1_ENV_KEYS:                      # the env in all 4 workers (planned value) / absent everywhere
        vals = env_values(env_raw, key)
        rec[key] = vals
        want = (arm.get("env") or {}).get(key)
        if want is not None:
            if len(vals) != N_WORKERS or any(v != want for v in vals):
                why.append(f"{key} in the workers {vals}, expected {want!r} on {N_WORKERS}")
        elif any(v is not None for v in vals):
            why.append(f"{key} set in a worker of {k}: {vals}")
    vl = vocab_lines(t)
    rec["vocab_lines"] = vl[:4]
    if k in VOCAB_TABLES:
        n = VOCAB_TABLES[k][2]
        good = [x for x in vl if x["n"] == n and x["of"] == VOCAB_SIZE and x["file"] == arm["env"].get(ENV_VOCAB)]
        if len({x["tp_rank"] for x in good}) != TP:
            why.append(f"REDUCED DRAFT VOCAB ACTIVE lines with N={n} and the planned file on {TP} ranks: {vl[:4]}")
    elif vl:
        why.append(f"a reduced draft vocab on {k}: {vl[:2]}")
    lab = (i8rb or {}).get("label")
    rec["head_label"], rec["head"] = lab, arm.get("draft_head")
    want_lab = "I8_ENGAGED" if arm.get("draft_head") else "BF16_HEAD"
    if lab != want_lab:
        why.append(f"draft head read-back {lab} (want {want_lab} for head {arm.get('draft_head')!r})")
    heads = (i8rb or {}).get("draft_head_lines") or []
    if arm.get("draft_head") and len(heads) < TP:
        why.append(f"{len(heads)} draft head build lines (< {TP} PP1 ranks)")
    if k in VOCAB_TABLES and arm.get("draft_head") and heads and any(h.get("n") != VOCAB_TABLES[k][2] // TP for h in heads):
        why.append(f"the quantized head is not the reduced one: {heads[:2]}")
    ir = idx_readback(workers)
    rec["idx"] = ir
    rec["idx_logged_fp8"] = logged_indexer_fp8(t)
    if k == "IDX":
        if ir["engaged"] is False or (ir["engaged"] is None and not rec["idx_logged_fp8"]):
            why.append(f"indexer_kv_dtype fp8 not applied (worker specs {ir.get('specs')[:2]}, logged {rec['idx_logged_fp8']})")
    elif ir["engaged"] is True or rec["idx_logged_fp8"]:
        why.append("the fp8 indexer cache on a non-IDX arm")
    log_ = (crb or {}).get("log") or {}
    seqs = log_.get("max_num_seqs")
    rec["logged_max_num_seqs"] = seqs
    if seqs is not None and seqs != arm["expect_cfg"]["max_num_seqs"]:
        why.append(f"logged max_num_seqs {seqs} != {arm['expect_cfg']['max_num_seqs']}")
    if k in SEQ_ARMS and seqs != SEQ_ARMS[k]:
        why.append(f"logged max_num_seqs {seqs} != {SEQ_ARMS[k]}")
    mbt, lpt, lpa = (logged_int(t, "max_num_batched_tokens"), logged_int(t, "long_prefill_token_threshold"),
                     logged_bool(t, "long_prefill_token_threshold_adaptive"))
    rec.update(logged_max_num_batched_tokens=mbt, logged_long_prefill_token_threshold=lpt, logged_long_prefill_adaptive=lpa)
    if k == "LP":
        if (mbt, lpt, lpa) != (2048, 1024, True):
            why.append(f"logged batched tokens / long-prefill threshold / adaptive {(mbt, lpt, lpa)} != (2048, 1024, True)")
    elif mbt not in (None, 1024) or lpt not in (None, 0):
        why.append(f"long-prefill / batched-token change on {k}: {(mbt, lpt, lpa)}")
    sp = log_.get("speculative_config") if isinstance(log_.get("speculative_config"), dict) else None
    rec["logged_spec"] = sp
    want_k = arm["k"]
    if sp is not None and int(sp.get("num_speculative_tokens") or 0) != want_k:
        why.append(f"logged num_speculative_tokens {sp.get('num_speculative_tokens')} != {want_k}")
    if k in ("K4", "DYN") and sp is None:
        why.append("no logged speculative_config (K 4 not read back)")
    if k == "DYN" and sp is not None and sp.get("num_speculative_tokens_per_batch_size") in (None, []):
        why.append("the dynamic schedule is not in the logged speculative_config")
    if k != "DYN" and sp is not None and sp.get("num_speculative_tokens_per_batch_size"):
        why.append(f"a dynamic schedule on {k}")
    hits = [l.strip()[:200] for l in t.splitlines() if any(x in l for x in DYN_FAIL_RX)]
    rec["dyn_fallback_lines"] = hits[:4]
    if hits:
        why.append(f"dynamic speculative decoding fell back (FULL graphs lost / schedule disabled): {hits[:2]}")
    rec["why"] = "; ".join(why) or None
    rec["real_failure"] = bool(why)
    rec["ok"] = not why
    return rec


# ------------------------------------------------------------------ PROF (inside Q's boot)
def prof_record(q_rec, switch):
    """PROF's own record from Q's boot: DONE when both 30-s profiled windows kept trace files, else NOT_INTEGRATED."""
    sw = switch_map(switch) if not isinstance(switch, dict) else switch
    if not sw.get("PROF", True):
        return {"status": "SKIPPED_SWITCH_OFF", "why": "switched off in CELL 1 (NBM1_ON)"}
    if not sw.get("Q", True):
        return {"status": "SKIPPED_DEPENDENCY", "why": "PROF runs inside Q's boot; Q is switched off"}
    st = (q_rec or {}).get("steady") or {}
    keys = [f"profile_prof_{b}@{n}" for b, n in PROF_WINDOWS]
    wins = {kk: st.get(kk) for kk in keys}
    if not any(isinstance(w, dict) for w in wins.values()):
        return {"status": "NOT_RUN", "why": f"Q status {(q_rec or {}).get('status')}: no PROF window ran", "windows": {}}
    got = {kk: [f[0] for f in ((w or {}).get("trace_files") or [])] for kk, w in wins.items()}
    ok = all(got[kk] for kk in keys)
    return {"status": "DONE" if ok else "NOT_INTEGRATED", "not_integrated": not ok, "secs": PROF_SECS,
            "windows": {kk: {"status": (w or {}).get("status"), "trace_files": got[kk][:8], "trace_dir": (w or {}).get("trace_dir"),
                             "profile_request": (w or {}).get("profile_request")} for kk, w in wins.items()},
            "why": "ok" if ok else f"PROF windows without trace files: {[kk for kk in keys if not got[kk]]}"}


# ------------------------------------------------------------------ budget / copies
def copy_estimate(arms, switch):
    """The planned walk of one copy (setup, then every switched-on arm in order with est_min + 5 min left)."""
    sw = switch_map(switch)
    left = BUDGET_MIN - SETUP_MIN
    walk, runtime = [], SETUP_MIN
    for k, a in arms.items():
        if not sw.get(k, True):
            continue
        est = q_time(sw.get("PROF") and sw.get("Q"))[0] if k == "Q" else a["est_min"]
        fits = left >= est + START_RESERVE_MIN
        walk.append({"arm": k, "est_min": est, "left_min": round(left, 1), "runs": fits})
        if fits:
            left -= est
            runtime += est
    return {"walk": walk, "runtime_min": round(runtime, 1), "left_end_min": round(left, 1),
            "all_run": all(w["runs"] for w in walk), "budget_min": BUDGET_MIN}


# ------------------------------------------------------------------ records (record only)
def _valid(T, w):
    try:
        return bool(T.window_ok(w)[0])
    except Exception:
        return False


def band_table(T, Q, X, thr_floor=None, speed_min=0.02):
    """Per band: Q's and X's valid windows (any level), mean tok/s, gain, the threshold max(Q's 8k A/A spread, 2 %, the
    long floor). Recorded only (labels / verdicts never stop an arm)."""
    wq, wx = T._win(Q or {}) or {}, T._win(X or {}) or {}
    aa = None
    a8 = [wq.get(k) for k in ("8k@16#1", "8k@16#2")]
    if all(isinstance(w, dict) and _valid(T, w) for w in a8):
        aa = abs(a8[0]["tok_s"] - a8[1]["tok_s"]) / ((a8[0]["tok_s"] + a8[1]["tok_s"]) / 2)
    out = {"q_aa_spread_8k": None if aa is None else round(aa, 4), "bands": {}}
    for band in ("8k", "20k", "40k", "90k", "116k"):
        qv = [v["tok_s"] for kk, v in wq.items() if isinstance(v, dict) and kk.split("@")[0] == band and _valid(T, v)]
        xv = [v["tok_s"] for kk, v in wx.items() if isinstance(v, dict) and kk.split("@")[0] == band and _valid(T, v)]
        r = {"q_tok_s": [round(x, 1) for x in qv], "x_tok_s": [round(x, 1) for x in xv]}
        if qv and xv:
            qm, xm = sum(qv) / len(qv), sum(xv) / len(xv)
            thr = max(aa or 0.0, speed_min, float((thr_floor or {}).get(band) or 0.0))
            g = xm / qm - 1.0
            r.update(gain=round(g, 4), threshold=round(thr, 4), label="GAIN" if g > thr else ("REGRESSION" if g < -thr else "WITHIN_NOISE"))
        else:
            r["label"] = "INCONCLUSIVE"
        out["bands"][band] = r
    return out


def arm_records(T, rec, key, q_rec=None, thr_floor=None):
    rec = rec or {}
    acc = rec.get("oracle_acceptance") or {}
    return {"arm": key, "status": rec.get("status"), "al": acc.get("al"), "per_pos": acc.get("per_pos"),
            "windows": {k: {f: (w or {}).get(f) for f in ("status", "tok_s", "tick_ms", "mean_acceptance_length", "preemptions",
                                                          "running_eq_level", "level")}
                        for k, w in (rec.get("steady") or {}).items() if isinstance(w, dict)},
            "pool_blocks": (rec.get("workers") or {}).get("num_gpu_blocks"), "residents": rec.get("k100_residents"),
            "pin": rec.get("pin"), "tries": rec.get("tries"), "m1_readback": {k: (rec.get("m1_readback") or {}).get(k) for k in ("ok", "why")},
            "i8": (rec.get("i8_readback") or {}).get("label"), "tf_gate": (rec.get("tf_gate") or {}).get("status"),
            "oracle_vs": rec.get("oracle_vs"), "bands_vs_q": band_table(T, q_rec, rec, thr_floor) if key != "Q" and q_rec else None}
