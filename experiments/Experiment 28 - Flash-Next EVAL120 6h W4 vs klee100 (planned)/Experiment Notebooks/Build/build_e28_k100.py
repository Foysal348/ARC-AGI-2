"""E28 K100 EVAL120 6 h (lead 2026-11-01, user decisions in ../../README.md): the Flash-Next klee100 3bpw-MTP family with every
tested speed lever, a 6-hour EVAL120 commit (+ the hidden rerun), built as a thin wrapper over the K100Q builder (Exp 25
Build/build_e26_k100_quality.py, imported, never written; it imports build_e25_nb6.py -> build_e25_nb5.py). The K100Q cells are
rebuilt and asserted byte for byte against the K100Q notebook that ran VALID on GPU today (sha 4506a71f, esfersami50, E26
Results/E26_K100Q_GPU_v1_2026-11-01.md); only the marked "E28 K100" edits below are applied.

Kept from K100Q (proven): the klee100 boot (Models, overlay merge, fixkeys, GPU_METHOD_MAP_PASS gate, static QSA scales), FP8
KV, MTP K3, --language-model-only, Q-defer (nb4_ppack rev 2, read-back recorded), the K3 pin ladder (7,933,526,016 B first ->
pool 479, F 6, 21 workers), MERLIN_SHARE_DRAFT_WORKSPACE=1, #60490, the relaunch with the booted config, the E15B watchdog, the
commit deadman (k100q_deadman), CELL 0 = E18 R2 CELL 0 verbatim, the E15B writer, rerun = every hidden test task with the
identity fallback, the full saved responses (Duck transcripts + nb5_requests.jsonl of every request, the Duck bundle).

E28 K100 edits (marked "# E28 K100" in the cells; the scheduler module's own edits are marked "# E28 SCHED"):
  K1  (CELL 1)   deadman at HARD_STOP + 5 min = 415 min (24,900 s); default commit profile E28EVAL120
  K2  (CELL 1)   E28 constants (user 2026-11-01: T1 104,000 = K100Q's, B2 120,000; the server max-model-len stays K100Q's
                 131,072, B2's Duck window = the normal 130,560; the 6-h window, I8 env) + B2 <= the 120k rule; E28_LEVERS; the profiles: E28EVAL120 (source eval, all 120 tasks,
                 415-min budget, hard margin 5 -> HARD_STOP 410 min = 6 h 50 min, one phase Q_E28_K100_EVAL120 with the fixed
                 21,600-s window) replaces K100QUALITY (no 30-task list, no quality-only phase); RERUN_PROD = NB6's (every
                 hidden task, 12 h - 3 min, E18 ladder) + the klee100 boot levers + the same E28 levers
  K3b (CELL 1)   the Duck request timeout 7,200 s with the scheduler (T1); B2 requests 9,000 s (E28_B2_REQUEST_TIMEOUT=9000,
                 user 2026-11-01)
  K4  (CELL 1)   the W4 / 27B reference tables are not embedded (README: no W4 reference on 120 tasks; size)
  K5  (CELL 1)   ONE sha-pinned lzma + base85 literal with e28k100_lib.py + e28_sched_patch.py -> E28L, E28S; the E28 config
                 record
  K6  (CELL 3)   the A / B / C scheduler: E28S.apply_e28_sched(E12_DIR) after the payload manifest check, verify, then
                 E28S.server_env() into the notebook env (commit: any failure = NO_GO_E28_SCHED, CELL 0's fallback stays;
                 rerun: recorded loudly, reverted, the original scheduler runs: continuing is never worse)
  K7  (CELL 3)   the scheduler's NOTEBOOK_EDITS N1 / N2 / N4, verbatim (the attempt's own projected peak, capped by K100Q's
                 MAX_LEN 131,072; the C-7 check; one non-blocking slot)
  K8  (engine)   (none: MAX_LEN, E21_VLLM_MAX_MODEL_LEN and the capacity plan stay K100Q's 131,072)
  K9  (engine)   admission = 'slots' as in the proven K100Q run (lead 2026-11-01: no worker may sit idle); WORKERS 24 and
                 --max-num-seqs 24 on the TOP rung only (pin 7,933,526,016 B, pool 479; user 2026-11-01: K3_ONLY v2 460 ->
                 562 tok/s at 20k from 16 to 24 streams); every lower rung keeps K100Q's R2 value (18 / 14 / 9 / 6); the
                 same-pin I8 -> BF16 retry keeps its rung's value; the K100Q captures (max 84) cover 24 x (1 + 3) / 2 = 48
                 tokens per PP2 micro-batch (asserted); the
                 engine schedules KV, a preemption is a recompute: counted, and flagged per attempt in the report); the
                 E28 capacity note (T1 / B2 blocks, what a projected gate would admit) is recorded only
  K10 (engine)   I8: DRAFT_HEAD=int8 into Merlin's build() on the boot tries while I8 is on (NB4T's env); a try that fails
                 with it (launch failure / OOM / config mismatch before any phase) is retried at the SAME pin with the BF16
                 head (I8_UNSUPPORTED_ON_K100, a lever drop; the pin and the 21 workers stay); the I8 read-back (NB4T) per
                 try, and (audit 2026-11-01) a REQUIRED I8_ENGAGED on a real server booted with the INT8 head: anything
                 else = I8_READBACK_UNVERIFIED -> the server stops -> the same pin, BF16 head (kind READBACK_UNVERIFIED);
                 e28_i8_static (the Merlin conditions on this checkpoint before any boot)
  K11 (CELL 6/7) the scheduler's N3 (no ORDER2 under E28_SCHED=1) + the profile's plain_order (no ORDER2 even without it)
  K12 (engine)   nb5_qab_phase_record: P.get("task_list") or the phase's TASK_IDS (EVAL120 has no fixed list)
  K13 (CELL 8)   the fixed 6-h solve window; the I8 static check before the first boot
  K14 (CELL 9)   the E28 run report (e28_k100_report.json: score, tok/s, preemptions, prefix hit, the per-attempt table,
                 the B2 attempts, I8 / Q-defer / scheduler / boot facts) instead of the K100Q quality report; in the bundle
  K15 (markdown) the E28 header

usage: python build_e28_k100.py [--interim]   (--interim: build without e28_sched_patch.py, the scheduler integration point empty)
       -> ../Q27-XHIGH-E28_K100_EVAL120_6H.ipynb + e28k100_code_export.py + e28k100_build_facts.json + the GPU push folder
          D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e28-k100-eval120-6h__esfersami50/ (nothing is pushed)
"""
import hashlib
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
sys.path.insert(0, str(E25B))
sys.path.insert(0, str(HERE))


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


_REF27 = E25B / "k100q_ref27b.json"            # build_e26_k100_quality.py rewrites it at import (the same bytes, checked)
_REF27_SHA0 = sha(_REF27.read_bytes())
import build_e26_k100_quality as KQ   # noqa: E402  (module import only: its build() is never called here)
assert sha(_REF27.read_bytes()) == _REF27_SHA0, "the K100Q builder import changed k100q_ref27b.json"
N6, B = KQ.N6, KQ.B

K100Q_SHA = "4506a71f957267bfeba5b32271d654f4ed3e8d5a95d3b65042aff74aef997f45"    # GPU run VALID 2026-11-01 (esfersami50)
K100Q_NB = KQ.OUT_NB
BASE_CELLS = KQ.cells()
assert sha(N6.notebook_bytes(BASE_CELLS)[0]) == K100Q_SHA, "build_e26_k100_quality.py no longer reproduces K100Q 4506a71f"
assert sha(K100Q_NB.read_bytes()) == K100Q_SHA, "the K100Q notebook on disk changed"
assert [k for k, _ in BASE_CELLS] == ["markdown"] + ["code"] * 8

OUT_NB = HERE.parent / "Q27-XHIGH-E28_K100_EVAL120_6H.ipynb"
EXPORT = HERE / "e28k100_code_export.py"
FACTS = HERE / "e28k100_build_facts.json"
LIB = HERE / "e28k100_lib.py"
SCHED = HERE / "e28_sched_patch.py"
SCHED_API = HERE / "E28_SCHED_API.md"
SLUG = "q27-xhigh-e28-k100-eval120-6h"
OWNER = "esfersami50"                          # the klee100 inputs are shared with sami's account (K100Q ran there)
PUSH = Path(r"D:/ARC AGI 2/_scratch/push_e25") / f"{SLUG}__{OWNER}"
DEFAULT_PROFILE = "E28EVAL120"
PUSHED: set = set()   # sha256 of every E28 K100 notebook that ran on Kaggle (never overwrite one: rebuild -> *_REBUILD)
INTERIM = "--interim" in sys.argv[1:]
DUCK_SRC = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/src/duck_b/arc2_duck.py")

# ------------------------------------------------------------------ the numbers (user decisions 2026-11-01, README sections 1-4)
T1 = 104000                         # A / B1 / C attempts = K100Q's (user 2026-11-01); reserve 4,096 -> longest request 99,904
B2_TOKENS = 120000                  # B2 (cap-cut) re-runs only (user 2026-11-01)
SERVER_MAX_LEN = 131072             # the proven K100Q --max-model-len (unchanged: no 262k)
T1_CTX = 131072                     # = MAX_LEN (K100Q)
B2_WINDOW = SERVER_MAX_LEN - 512    # 130,560 = the normal Duck window
B2_TIMEOUT = 9000                   # E28_B2_REQUEST_TIMEOUT (user 2026-11-01); T1 requests 7,200 s
WORKERS = 24                        # user 2026-11-01: workers / --max-num-seqs 24 on the TOP rung only; lower rungs = K100Q's R2
PP2_MICROBATCH_TOKENS = -(-WORKERS * (1 + 3) // 2)   # 48: 24 seqs x (1 + K3) over the 2 PP micro-batches
RESERVE = 4096
WINDOW_SEC = 6 * 3600               # the fixed 6-h solve window
BUDGET_MIN, HARD_MARGIN_MIN = 415, 5          # HARD_STOP = 410 min = 6 h 50 min; deadman = 415 min = HARD_STOP + 5 (K100Q rule)
DEADMAN_SEC = BUDGET_MIN * 60
I8_ENV = {"DRAFT_HEAD": "int8"}     # NB4T arm I8 (build_e25_nb4t.py NB4T_BUILD_ENV / nb4t_lib.arm_plan: build_env DRAFT_HEAD=int8)
assert B2_TOKENS <= 120000 and B2_TOKENS - RESERVE <= B2_WINDOW and B2_WINDOW + 512 == SERVER_MAX_LEN == 131072 and T1 == 104000
assert PP2_MICROBATCH_TOKENS == 48 and PP2_MICROBATCH_TOKENS in KQ.K100_CAPS and max(KQ.K100_CAPS) >= PP2_MICROBATCH_TOKENS, KQ.K100_CAPS
assert BUDGET_MIN - HARD_MARGIN_MIN == 410 and DEADMAN_SEC == (BUDGET_MIN - HARD_MARGIN_MIN) * 60 + 5 * 60

# NB4T's I8 env, cut from its builder / lib (never retyped)
_nb4t_lib = (E25B / "nb4t_lib.py").read_text(encoding="utf-8")
assert '"I8": {"kind": "build_env", "key": KNOB_HEAD, "value": "int8"}' in _nb4t_lib and 'KNOB_HEAD = "DRAFT_HEAD"' in _nb4t_lib
assert '_arms["I8"]["build_env"] == {"DRAFT_HEAD": "int8"}' in (E25B / "build_e25_nb4t.py").read_text(encoding="utf-8")

# ------------------------------------------------------------------ the new code: e28k100_lib.py (+ e28_sched_patch.py)
LIB_TEXT = LIB.read_bytes().decode("utf-8")
assert "\r" not in LIB_TEXT
compile(LIB_TEXT, "e28k100_lib.py", "exec")
import e28k100_lib as E28L   # noqa: E402
for _rx in ("_HEAD_RX", "_SPEC_RX", "_WTAG"):    # the I8 read-back regexes = NB4T's, verbatim
    _line = next(l for l in LIB_TEXT.splitlines() if l.startswith(_rx + " = "))
    assert _line.split("#")[0].rstrip() in _nb4t_lib, _rx
assert E28L.I8_ENV == I8_ENV

SCHED_TEXT = None
if SCHED.is_file():
    SCHED_TEXT = SCHED.read_bytes().decode("utf-8")
    assert "\r" not in SCHED_TEXT
    import e28_sched_patch as E28S   # noqa: E402
    assert (E28S.T1, E28S.B2_TOKENS, E28S.MAX_MODEL_LEN, E28S.B2_WINDOW) == (T1, B2_TOKENS, SERVER_MAX_LEN, B2_WINDOW)
    _se = E28S.server_env()
    assert {k: _se.get(k) for k in ("E28_SCHED", "E28_T1", "E28_B2_TOKENS", "E28_B2_WINDOW")} == {
        "E28_SCHED": "1", "E28_T1": str(T1), "E28_B2_TOKENS": str(B2_TOKENS), "E28_B2_WINDOW": str(B2_WINDOW)}, _se
    assert _se.get("E28_B2_REQUEST_TIMEOUT") in (None, str(B2_TIMEOUT)), _se   # the notebook sets 9,000 either way
    SCHED_EDIT_LABELS = [lab.split()[0] for lab, _, _ in E28S.NOTEBOOK_EDITS]
    assert SCHED_EDIT_LABELS[:3] == ["N1", "N2", "N3"] and set(SCHED_EDIT_LABELS) <= {"N1", "N2", "N3", "N4"}, SCHED_EDIT_LABELS
    _duck = DUCK_SRC.read_bytes()
    assert sha(_duck) == E28S.DUCK_PRE == "b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128"
    _pt = E28S.apply_e28_sched(_duck.decode("utf-8"))   # build-time: the patch applies to the payload's arc2_duck.py
    assert _pt["ok"] and sha(_pt["text"].encode("utf-8")) == E28S.DUCK_POST
    compile(_pt["text"], "arc2_duck.py (E28 SCHED)", "exec")
    assert E28S.revert_e28_sched(_pt["text"])["text"] == _duck.decode("utf-8")
    SCHED_REC = {"file": str(SCHED), "sha256": sha(SCHED_TEXT), "e28_sched_sha": E28S.E28_SCHED_SHA,
                 "api": str(SCHED_API), "api_sha256": sha(SCHED_API.read_bytes()) if SCHED_API.is_file() else None}
else:
    assert INTERIM, "e28_sched_patch.py missing (use --interim for a build without the scheduler)"
    E28S = None
    SCHED_REC = {"file": None, "interim": True}
MODULES = {"e28k100_lib.py": LIB_TEXT}
if SCHED_TEXT is not None:
    MODULES["e28_sched_patch.py"] = SCHED_TEXT
MODULES_JSON = json.dumps(MODULES, sort_keys=True, separators=(",", ":"))

# ------------------------------------------------------------------ build facts: the 6-h plan, the capacity, the tokens
_K3V2 = KQ.K3V2 or {}
POOL1, F1 = KQ.EXPECTED[0]["pool_usable_est"], KQ.EXPECTED[0]["f_est"]
assert POOL1 == 479 and F1 == 6 and KQ.EXPECTED[0]["workers_est"] == 21 and KQ.K100_PINS[0] == 7933526016
CAPACITY = E28L.capacity_note(POOL1, 4096, F1, 1.0, T1, B2_TOKENS, T1_CTX, SERVER_MAX_LEN, WORKERS)
assert CAPACITY["t1_blocks"] == 36 and CAPACITY["b2_blocks"] == 38 and CAPACITY["t1_streams"] == 13, CAPACITY
CAPACITY["k100q_rule_projected_120k"] = 479 // (math.ceil(120000 / 4096) + 6)        # 13 (the K100Q record's "13 projected")
HARD_STOP_PLAN = {"setup_min": 15, "boot_min": 13, "probes_warmup_min": 2, "window_min": WINDOW_SEC / 60,
                  "stop_before_hard_min": 3, "hard_margin_min": HARD_MARGIN_MIN}
HARD_STOP_PLAN["sum_min"] = round(sum(HARD_STOP_PLAN.values()), 1)
assert HARD_STOP_PLAN["sum_min"] <= BUDGET_MIN
_LATEST_END = BUDGET_MIN - HARD_MARGIN_MIN - 3 - 1     # the solve end cap (HARD_STOP - 180 s) - NB5's 60-s room check
WINDOW_BY_FAILED_TRIES = {}
for _n in (0, 1, 2, 3):                                 # a failed try (an I8 drop or a rung) costs ~13 min (K100Q ready 777 s)
    _st = 15 + 13 + 2 + 13 * _n
    _w = min(WINDOW_SEC / 60, _LATEST_END - _st)
    WINDOW_BY_FAILED_TRIES[f"{_n}_failed"] = {"solve_start_min": _st, "window_min": round(_w, 1),
                                              "short_by_min": round(WINDOW_SEC / 60 - _w, 1), "solve_end_min": round(_st + _w, 1)}
HARD_STOP_PLAN.update(hard_stop_min=BUDGET_MIN - HARD_MARGIN_MIN, deadman_min=DEADMAN_SEC / 60, latest_solve_end_min=_LATEST_END,
                      window_by_failed_tries=WINDOW_BY_FAILED_TRIES)
assert WINDOW_BY_FAILED_TRIES["1_failed"]["short_by_min"] == 0
TOKEN_PLAN = {"k100q_tokens_per_attempt": round(2543542 / 60), "k100q_window_tok_s": 385.0,
              "phase_a_120_attempts_m": round(120 * 2543542 / 60 / 1e6, 2)}
TOKEN_PLAN["phase_a_hours_at_385"] = round(TOKEN_PLAN["phase_a_120_attempts_m"] * 1e6 / 385 / 3600, 2)
TOKEN_PLAN["note"] = ("K100Q (30 tasks x 2 attempts, 104k) generated 2,543,542 tokens at a 385 tok/s window average with 16.5 mean "
                      "running (slots admission, kept here); phase A of 120 tasks alone needs about the first 4 h")


def packed_lzma(text, label):
    return KQ.packed_lzma(text, label)


# ------------------------------------------------------------------ E28 K100 edits
def once(text, old, new, label):
    return B.once(text, old, new, "E28 K100 " + label)


E28_DEFS = r'''# ---- E28 K100 (user 2026-11-01, Exp 28 README): EVAL120 6 h with every tested speed lever on klee100. T1 = 104,000 per
# attempt (= K100Q, reserve 4,096) for A / B1 / C; B2 (cap-cut re-runs) = 120,000 inside K100Q's 131,072 max-model-len and
# the normal Duck window 130,560 (the 120k rule below holds for B2 too); I8 = DRAFT_HEAD=int8 (NB4T's
# env, an input of Merlin's build()); the A / B / C scheduler (e28_sched_patch.py, E28_SCHED=1); no ORDER2; workers 21 (the
# R2 rule) with K100Q's 'slots' admission (the engine schedules KV; preemptions = recompute, counted per attempt)
E28_T1 = __T1__                    # A / B1 / C attempts
E28_B2 = __B2__                    # B2 (cap-cut) re-runs only
E28_B2_WINDOW = 131072 - 512             # = NB5_WINDOW (K100Q's max-model-len 131,072 is unchanged)
E28_B2_TIMEOUT = __B2TIMEOUT__             # E28_B2_REQUEST_TIMEOUT (user 2026-11-01)
E28_WINDOW_SEC = __WINDOW__               # the fixed 6-h solve window
E28_I8_ENV = __I8ENV__     # NB4T arm I8 (Merlin launch.py:319-322 -> "draft_lm_head_quant":"int8")
E28_I8 = {"want": False, "on": False, "static": None, "dropped": None, "readback": None}   # set in CELL 8 / per boot try
assert E28_B2 <= 120000 and E28_B2 - 4096 <= E28_B2_WINDOW and E28_T1 == 104000   # the 120k rule, B2 included
E28_LEVERS = dict(K100_BOOT_LEVERS, nb5_gate=1.0, nb5_attempt={"b0": E28_T1, "b1": E28_T1},
                  nb5_env={"E12_B_SKIP_B1_IF_ACCEPTED": "0", "E12_B_B1_ACC_MODE": "0", "E12_B_EXTRA_ATTEMPTS": "0"},   # NB6 / E15B SUBMIT
                  admit_mode="slots", e28_workers_rule=True, f_rule="measured", pin_ladder=True, patch_60490=True,
                  lever_env={"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, no_pure_decode=True, continue_after_failed_boot=True,
                  i8=True, plain_order=True, e28_sched=True,
                  top_rung_seqs=__WORKERS__)   # user 2026-11-01: 24 workers / --max-num-seqs 24 on the TOP rung only
E28_PP2_MICROBATCH_TOKENS = -(-E28_LEVERS["top_rung_seqs"] * (1 + 3) // 2)   # 48 tokens per PP2 micro-batch (FULL graphs)
assert max(K100_CAPS) >= E28_PP2_MICROBATCH_TOKENS and E28_PP2_MICROBATCH_TOKENS in K100_CAPS, (K100_CAPS, E28_PP2_MICROBATCH_TOKENS)
'''
E28_PROFILE = r'''    # E28 K100: EVAL120 (all 120 eval tasks, plain order), 415-min budget, hard margin 5 (HARD_STOP 410 = 6 h 50 min, the
    # deadman at 415), ONE phase with the fixed 6-h window (shortened, never skipped, when the boot is late; the K100Q rule)
    "E28EVAL120": dict(_NB5_T, time_budget_sec=__BUDGET__ * 60, hard_margin_sec=__HARDMARGIN__ * 60, source="eval", n_tasks=120,
                       quality_ab=True, equal_windows=True, fixed_window_sec=E28_WINDOW_SEC,
                       nb5_phases=[["Q_E28_K100_EVAL120", "all", None, "on", "A", "fp8"]], **E28_LEVERS),
}
PROFILE_TABLE["RERUN_PROD"] = dict(PROFILE_TABLE["RERUN_PROD"], **E28_LEVERS)   # E28 K100: the hidden rerun = NB6's (every task, 12 h - 3 min, E18 ladder) + klee100 + E28
'''


def cell1(c):
    c = once(c, "# ---- E26 K100 (user 2026-11-01): the commit's ABSOLUTE 150-min DEADMAN, armed first",
             "# ---- E26 K100 (user 2026-11-01): the commit's ABSOLUTE DEADMAN (E28 K100: HARD_STOP 410 min + 5 = 415 min), armed first",
             "K1 deadman comment")
    c = once(c, "K100_DEADMAN_SEC = 9000\n", f"K100_DEADMAN_SEC = {DEADMAN_SEC}   # E28 K100: HARD_STOP (410 min) + 5 min\n", "K1 deadman")
    c = once(c, 'NB5_DEFAULT_PROFILE = "K100QUALITY"\n', f'NB5_DEFAULT_PROFILE = "{DEFAULT_PROFILE}"   # E28 K100\n', "K1 profile")
    c = once(c, 'K100_PROFILES = ("K100QUALITY", "RERUN_PROD")\n', f'K100_PROFILES = ("{DEFAULT_PROFILE}", "RERUN_PROD")   # E28 K100\n', "K2 guard")
    defs = (E28_DEFS.replace("__T1__", str(T1)).replace("__B2__", str(B2_TOKENS)).replace("__B2TIMEOUT__", str(B2_TIMEOUT)).replace("__WORKERS__", str(WORKERS))
           .replace("__WINDOW__", str(WINDOW_SEC)).replace("__I8ENV__", json.dumps(I8_ENV)))
    c = once(c, "PROFILE_TABLE = {\n", defs + "PROFILE_TABLE = {\n", "K2 defs")
    i = c.index("    # E26 K100: NB5QUALITY_R2's FP8 phase on klee100 (the same row:")
    j = c.index("PROFILE_TABLE[\"RERUN_PROD\"] = dict(PROFILE_TABLE[\"RERUN_PROD\"], **K100_BOOT_LEVERS)   # E26 K100: the hidden rerun boots klee100\n")
    j = c.index("\n", j) + 1
    old = c[i:j]
    assert old.count('"K100QUALITY": dict(') == 1 and old.count("}\n") == 1, old[:200]
    c = c[:i] + E28_PROFILE.replace("__BUDGET__", str(BUDGET_MIN)).replace("__HARDMARGIN__", str(HARD_MARGIN_MIN)) + c[j:]
    # K2b (size, the 400-kB commit limit): the NB5 / NB6 profile rows this notebook can never run are cut (the guard admits
    # E28EVAL120 / RERUN_PROD only; RERUN_ALL stays: the hidden_equiv branch names it)
    for a_, b_, gone in (('    "NB5EVAL120": dict(', '    "RERUN_ALL": dict(', "NB5EVAL120 / NB5EVAL30 / NB5EVAL30MTP / NB5QUALITY / NB5SMOKEDIAG / NB5BUDGET"),
                         ('    # spec R2 (lead 2026-10-31): 3 h hard runtime', '    # NB6 E2: the hidden rerun (every hidden test task', "NB5QUALITY_R2"),
                         ('    # NB6 E2: the commit check before Submit', '    # E28 K100: EVAL120 (all 120 eval tasks', "NB6COMMIT")):
        assert c.count(a_) == 1 and c.count(b_) == 1, (a_, c.count(a_), c.count(b_))
        i_, j_ = c.index(a_), c.index(b_)
        assert i_ < j_ and c[i_:j_].count('": dict(') == len(gone.split(" / ")), (gone, c[i_:j_].count('": dict('))
        c = c[:i_] + f"    # E28 K100 (size): the unused rows {gone} are cut (the guard admits E28EVAL120 / RERUN_PROD only)\n" + c[j_:]
    c = once(c, 'E12_DUCK_REQUEST_TIMEOUT = int(os.environ.get("E12_DUCK_REQUEST_TIMEOUT", 3600 if _E12_REV4 else 0)) or None   # E15B #47\n',
             'E12_DUCK_REQUEST_TIMEOUT = int(os.environ.get("E12_DUCK_REQUEST_TIMEOUT", (7200 if P.get("e28_sched") else 3600) if _E12_REV4 else 0)) or None   # E15B #47\n'
             '# E28 K100 (lead 2026-11-01): the T1 request timeout 7,200 s (a 99,904-token request at ~24 tok/s per stream ~70 min);\n'
             "# B2 requests 9,000 s (E28_B2_REQUEST_TIMEOUT=9000, set with the scheduler env in CELL 3)\n",
             "K3b request timeout")
    ref = next(l for l in c.splitlines() if l.startswith("K100Q_REF = json.loads("))
    ref27 = next(l for l in c.splitlines() if l.startswith("K100Q_REF27 = json.loads("))
    c = once(c, ref + "\n" + ref27 + "\n",
             "K100Q_REF = K100Q_REF27 = None   # E28 K100: the W4 / 27B reference tables are not used (no W4 reference on 120 tasks)\n",
             "K4 refs")
    mods = ("# ---- E28 K100: the new code, ONE sha-pinned lzma + base85 literal: e28k100_lib.py (I8 read-back / static check, the\n"
            "# projected-peak note, the run report) + e28_sched_patch.py (the A / B / C scheduler, Build/E28_SCHED_API.md)\n"
            "E28_SRC = json.loads(" + packed_lzma(MODULES_JSON, "e28_k100_modules") + ")\n"
            "for _n, _s in E28_SRC.items():\n"
            "    (TMP / _n).write_text(_s, encoding=\"utf-8\")\n"
            "import e28k100_lib as E28L\n"
            "E28S, E28S_ERROR = None, None\n"
            "if \"e28_sched_patch.py\" in E28_SRC:\n"
            "    try:\n"
            "        import e28_sched_patch as E28S\n"
            "    except Exception as _e:\n"
            "        E28S_ERROR = f\"{type(_e).__name__}: {_e}\"[:400]\n"
            "        print(\"!!!!!!!!!! E28 SCHED MODULE NOT LOADED:\", E28S_ERROR, flush=True)\n"
            "else:\n"
            "    E28S_ERROR = \"interim build: e28_sched_patch.py not embedded\"\n")
    c = once(c, "PPK, PPK_ERROR = None, None   # E26 K100 Q-defer:", mods + "PPK, PPK_ERROR = None, None   # E26 K100 Q-defer:", "K5 modules")
    c = once(c, 'print("NB5:", json.dumps(CONFIG["nb5"], default=str), flush=True)\n',
             'CONFIG["nb5"]["e28"] = {"t1": E28_T1, "b2": E28_B2, "server_max_len": 131072, "b2_timeout": E28_B2_TIMEOUT,\n'
             '                        "b2_window": E28_B2_WINDOW, "window_sec": E28_WINDOW_SEC, "i8_env": E28_I8_ENV, "levers": E28_LEVERS,\n'
             '                        "modules": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in E28_SRC.items()},\n'
             '                        "sched_sha": getattr(E28S, "E28_SCHED_SHA", None), "sched_error": E28S_ERROR,\n'
             '                        "lib": E28L.LIB_VERSION}   # E28 K100\n'
             'print("NB5:", json.dumps(CONFIG["nb5"], default=str), flush=True)\n', "K5 config")
    return c


E28_SCHED_APPLY = r'''# ---- E28 K100: the A / B / C scheduler (e28_sched_patch.py, Build/E28_SCHED_API.md section 2) on arc2_duck.py: after the
# manifest check (the ORIGINAL payload bytes), before the import; the module checks the pre sha, writes atomically, verifies
# the post sha + marker + py_compile. Commit: any failure = NO_GO_E28_SCHED (raised: no solve, CELL 0's fallback stays);
# rerun: recorded loudly, reverted to the original file, E28_SCHED unset (the original scheduler runs; never worse than none)
E28_SCHED_REC = {"want": bool(P.get("e28_sched")), "applied": False}
if E28_SCHED_REC["want"]:
    try:
        if E28S is None:
            raise RuntimeError("E28_SCHED_MODULE_MISSING: " + str(E28S_ERROR))
        E28_SCHED_REC.update({k: v for k, v in E28S.apply_e28_sched(E12_DIR).items() if k != "text"})
        _ok, _why = E28S.verify_e28_sched(E12_DIR)
        E28_SCHED_REC.update(verify_ok=_ok, verify_why=_why, state=E28S.tree_state(E12_DIR))
        if not _ok:
            raise RuntimeError("E28_SCHED_VERIFY_FAILED: " + str(_why))
        _e28env = dict(E28S.server_env())
        if E11_HARNESS:              # local / CPU-kit harness only: the fake run's short attempts (NB5_ATTEMPT_B0 / E28H_B2_TOKENS)
            _e28env.update(E28_T1=str(NB5_ATTEMPT["b0"]), E28_B2_TOKENS=os.environ.get("E28H_B2_TOKENS", _e28env["E28_B2_TOKENS"]))
        _e28env["E28_B2_REQUEST_TIMEOUT"] = str(E28_B2_TIMEOUT)   # E28 K100 (user 2026-11-01): B2 requests 9,000 s; T1 7,200 s
        if int(_e28env["E28_T1"]) != int(NB5_ATTEMPT["b0"]):
            E28_SCHED_REC["warning"] = f"E28_T1 {_e28env['E28_T1']} != NB5_ATTEMPT b0 {NB5_ATTEMPT['b0']}"
        os.environ.update(_e28env)
        E28_SCHED_REC.update(applied=True, env=_e28env)
        print("E28 SCHED: applied", json.dumps({k: E28_SCHED_REC.get(k) for k in ("applied", "before", "after", "sha_after", "verify_why")}), flush=True)
    except Exception as _e:
        E28_SCHED_REC.update(applied=False, error=f"{type(_e).__name__}: {_e}"[:800])
        for _k in (getattr(E28S, "ENV_NAMES", None) or ("E28_SCHED",)):
            os.environ.pop(_k, None)
        try:
            E28_SCHED_REC["revert"] = {k: v for k, v in E28S.revert_e28_sched(E12_DIR).items() if k != "text"}
        except Exception as _e2:
            E28_SCHED_REC["revert_error"] = f"{type(_e2).__name__}: {_e2}"[:400]
        print("!!!!!!!!!! NO_GO_E28_SCHED:", E28_SCHED_REC["error"], "-> " + ("RERUN: the ORIGINAL scheduler runs" if RERUN
              else "commit: no solve (CELL 0's fallback)"), flush=True)
        if not RERUN:
            raise RuntimeError("NO_GO_E28_SCHED: " + E28_SCHED_REC["error"])
CONFIG["e28_sched"] = E28_SCHED_REC
'''


def cell3(c):
    c = once(c, 'assert not _bad, ("E12 package hash mismatch", _bad)\n',
             'assert not _bad, ("E12 package hash mismatch", _bad)\n' + E28_SCHED_APPLY, "K6 scheduler apply")
    if E28S is not None:
        for lab, old, new in E28S.NOTEBOOK_EDITS:         # N1 / N2 / N4 live in CELL 3; N3 in CELL 6/7
            if lab.split()[0] in ("N1", "N2", "N4"):
                c = B.once(c, old, new, "E28 SCHED " + lab)
    else:
        c = once(c, "def _nb5_peak_len(self):\n", "# E28 K100 SCHED INTEGRATION POINT (interim build): NOTEBOOK_EDITS N1 / N2 go here\n"
                 "def _nb5_peak_len(self):\n", "K7 interim marker")
    return c


E28_ENGINE = r'''


# ==================================================================== E28 K100: I8 static check (NEW, e28k100_lib)
def e28_i8_static():
    """E28 K100 I8 (README section 1; measured on W4 only, NB4T): Merlin's conditions for the INT8 draft head
    (speculator._maybe_quantize_draft_lm_head, QuantizedDraftLMHead) on THIS checkpoint, read before any boot from the
    merged klee100 dir and the installed Merlin tree (E28L.i8_static_check). Never raises (an error = INCONCLUSIVE)."""
    try:
        cfg = json.loads((CKPT / "config.json").read_text(encoding="utf-8"))
        qf = CKPT / "quantization_config.json"
        qc = json.loads(qf.read_text(encoding="utf-8")) if qf.is_file() else (cfg.get("quantization_config") or {})
        tree = None if FAKE else ((MERLIN_ENG or {}).get("kv") or {}).get("MERLIN_TREE")
        src = {}
        for rel in ("vllm/models/qwen4_exp/nvidia/mtp.py", "vllm/models/qwen4_exp/nvidia/model.py",
                    "vllm/v1/worker/gpu/spec_decode/speculator.py"):
            if tree and (Path(tree) / rel).is_file():
                src[rel] = (Path(tree) / rel).read_text(encoding="utf-8", errors="replace")
        return dict(E28L.i8_static_check(cfg, qc, src), files=sorted(src), fake=bool(FAKE))
    except Exception as e:
        return {"verdict": "INCONCLUSIVE", "why": [f"{type(e).__name__}: {e}"[:300]]}


def e28_i8_gate(trec):
    """E28 K100 I8 (audit 2026-11-01, HIGH): a boot with DRAFT_HEAD=int8 is trusted only on the read-back label I8_ENGAGED
    (argv + log spec int8 and the 'Draft lm_head: INT8 weight-only copy' build lines, NB4T's evidence). Anything else on a
    real server (I8_SPEC_ONLY, I8_CONTRADICTION, a BF16 head) -> status I8_READBACK_UNVERIFIED + trec["i8_failure"]; the
    caller returns (with_servers stops this server) and nb5_boot retries the SAME pin with the BF16 head
    (I8_UNSUPPORTED_ON_K100, kind READBACK_UNVERIFIED). The fake server keeps its own label (FAKE_SERVER). True = stop."""
    rb = trec.get("i8_readback") or {}
    if FAKE or (NB5_BOOT_LEVERS or {}).get("DRAFT_HEAD") != "int8" or rb.get("label") == "I8_ENGAGED":
        return False
    trec["status"] = "I8_READBACK_UNVERIFIED"
    trec["i8_failure"] = {"label": rb.get("label"),
                          "why": rb.get("contradictions") or [k for k, v in (rb.get("checks") or {}).items() if v is not True]}
    log("!!!!!!!!!! E28 K100 I8_READBACK_UNVERIFIED", json.dumps(trec["i8_failure"], default=str)[:600],
        "-> this server stops; the same pin with the BF16 draft head")
    return True
'''


def engine(c):
    c = once(c, '    if NB5_ADMIT_MODE == "slots":                  # R2 (user decision): workers = max_num_seqs = the 64k-context rule\n',
             '    if NB5_ADMIT_MODE == "slots" or P.get("e28_workers_rule"):   # R2 (user decision): workers = max_num_seqs = the 64k-context rule\n'
             '        # E28 K100: the R2 rule (21 workers) also when a profile is not "slots" (E28 runs "slots", as K100Q)\n',
             "K9 workers rule")
    c = once(c, '        _w = min(_wr, int(_seq)) if _seq else _wr\n',
             '        _w = min(_wr, int(_seq)) if _seq else _wr\n'
             '        if P.get("top_rung_seqs") and _seq and int(_seq) == int(P["top_rung_seqs"]):   # E28 K100 (user 2026-11-01): the top\n'
             '            _w = int(_seq)                   # rung launched 24 -> 24 workers (lower rungs: K100Q\'s R2 value, the min above)\n',
             "K9b workers fixed")
    c = once(c, '"captures_cover_workers": NB5L.captures_cover(NB5_CAPS16 if P.get("caps16") else (K100_CAPS if P.get("k100_caps") else [32]), _w)}   # E26 K100',
             '"captures_cover_workers": NB5L.captures_cover(NB5_CAPS16 if P.get("caps16") else (K100_CAPS if P.get("k100_caps") else [32]), _w),   # E26 K100\n'
             '                              "captures_cover_pp2_microbatch": max(K100_CAPS) >= -(-_w * 4 // 2), "top_rung_seqs": P.get("top_rung_seqs"),\n'
             '                              "rung_workers": _w}   # E28 K100: this rung\'s workers',
             "K9b pp2 coverage record")
    c = once(c, '        seqs = NB5L.seqs_for(NB5_BOOT_KV, tr["gib"]) if NB5_SEQS_RULE == "table" else (K100_SEQS_TABLE[round(float(tr["gib"]), 2)] if NB5_SEQS_RULE == "k100" else None)   # E26 K100\n',
             '        seqs = NB5L.seqs_for(NB5_BOOT_KV, tr["gib"]) if NB5_SEQS_RULE == "table" else (K100_SEQS_TABLE[round(float(tr["gib"]), 2)] if NB5_SEQS_RULE == "k100" else None)   # E26 K100\n'
             '        if P.get("top_rung_seqs") and abs(float(tr["gib"]) - K100_PIN_STEPS_GIB[0]) < 1e-9:   # E28 K100 (user 2026-11-01):\n'
             '            seqs = int(P["top_rung_seqs"])            # 24 on the TOP rung only (an I8 -> BF16 retry there keeps 24)\n',
             "K9c top-rung seqs")
    c = once(c, '    if caps.get("max_num_seqs") is None:\n        plan["warning"] = "max_num_seqs not found in the server log: workers = pool streams"\n',
             '    if caps.get("max_num_seqs") is None:\n        plan["warning"] = "max_num_seqs not found in the server log: workers = pool streams"\n'
             '    plan["e28"] = E28L.capacity_note(pool, block, plan["f_eff"], NB5_GATE, E28_T1, E28_B2, MAX_LEN, MAX_LEN,\n'
             '                                     plan["workers"])   # E28 K100: T1 / B2 blocks (recorded; "slots" admits by workers)\n',
             "K9 capacity note")
    # K10: I8 on the boot tries (DRAFT_HEAD into build() via the boot levers), the drop rule, the read-back
    c = once(c, "        NB5_BOOT_LEVERS = dict(levers)\n",
             "        NB5_BOOT_LEVERS = dict(levers, **(E28_I8_ENV if E28_I8[\"on\"] else {}))   # E28 K100 I8: DRAFT_HEAD=int8 into build()\n",
             "K10 levers")
    c = once(c, '+ ("" if levers or not NB5_LEVER_ENV else "_nolever"))\n',
             '+ ("" if levers or not NB5_LEVER_ENV else "_nolever")\n'
             '                + ("_i8" if E28_I8["on"] else ""))   # E28 K100\n', "K10 name")
    c = once(c, '        trec = {"name": name, "pin_gib": tr["gib"], "kv_bytes": tr["kv_bytes"], "max_num_seqs_launch": seqs, "levers": dict(levers)}\n',
             '        trec = {"name": name, "pin_gib": tr["gib"], "kv_bytes": tr["kv_bytes"], "max_num_seqs_launch": seqs, "levers": dict(levers),\n'
             '                "e28_i8": bool(E28_I8["on"])}   # E28 K100\n', "K10 trec")
    c = once(c, '        fail = NB5L.boot_failure(trec.get("status"), trec.pop("log_text_for_ladder", ""), levers, trec.get("phases_started", False))\n',
             '        _e28_log = trec.pop("log_text_for_ladder", "")   # E28 K100: kept for the I8 drop record\n'
             '        fail = NB5L.boot_failure(trec.get("status"), _e28_log, levers, trec.get("phases_started", False))\n',
             "K10 log")
    c = once(c, '        if not fail["failed"] or not P.get("pin_ladder") or ABORT.get("why") or (P.get("e18_ladder") and len(brec["tries"]) >= 3):\n',
             '        if (E28_I8["on"] and not ABORT.get("why") and not trec.get("phases_started") and not (P.get("e18_ladder") and len(brec["tries"]) >= 3)\n'
             '                and (fail["failed"] or str(trec.get("status")) in ("INCONCLUSIVE_CONFIG_MISMATCH", "I8_READBACK_UNVERIFIED"))):\n'
             '            # E28 K100 I8 (user 2026-11-01): a try that failed with the INT8 draft head, or booted without a verified INT8\n'
             '            # head (audit 2026-11-01: I8_READBACK_UNVERIFIED) -> the SAME pin with the BF16 head (I8_UNSUPPORTED_ON_K100, a\n'
             '            # lever drop: the proven pin and the 21 workers stay)\n'
             '            _e28_dr = (dict(label=E28L.I8_DROP, kind="READBACK_UNVERIFIED", status=trec.get("status"), readback=trec.get("i8_failure"))\n'
             '                       if str(trec.get("status")) == "I8_READBACK_UNVERIFIED" else E28L.i8_failure_reason(trec.get("status"), _e28_log))\n'
             '            E28_I8.update(on=False, dropped=dict(_e28_dr, try_name=name, pin_gib=tr["gib"]))\n'
             '            brec.setdefault("levers_dropped", []).append({"keys": ["DRAFT_HEAD"], "why": E28L.I8_DROP + ": " + E28_I8["dropped"]["kind"],\n'
             '                                                          "try": name})\n'
             '            log("!!!!!!!!!! E28 K100", E28L.I8_DROP, json.dumps(E28_I8["dropped"], default=str)[:600], "-> the same pin, BF16 draft head")\n'
             '            continue\n'
             '        if not fail["failed"] or not P.get("pin_ladder") or ABORT.get("why") or (P.get("e18_ladder") and len(brec["tries"]) >= 3):\n',
             "K10 drop rule")
    c = once(c, '        trec["config_readback"] = NB5L.config_readback(s.log(), s.argv, NB5_BOOT_KV, mtp_on)    # C-11\n',
             '        trec["config_readback"] = NB5L.config_readback(s.log(), s.argv, NB5_BOOT_KV, mtp_on)    # C-11\n'
             '        trec["i8_readback"] = ({"label": "FAKE_SERVER", "want": NB5_BOOT_LEVERS.get("DRAFT_HEAD")} if FAKE else\n'
             '                               E28L.i8_readback(s.log(), s.argv, NB5_BOOT_LEVERS.get("DRAFT_HEAD")))   # E28 K100 I8 (NB4T, recorded)\n'
             '        E28_I8["readback"] = trec["i8_readback"].get("label")\n'
             '        if e28_i8_gate(trec):                                # E28 K100 I8 (audit 2026-11-01): INT8 requested -> I8_ENGAGED required\n'
             '            return {}                                        # this server stops (with_servers); the ladder retries the same pin, BF16\n',
             "K10 readback + gate")
    # K12: EVAL120 has no fixed task list
    c = once(c, '    tasks = NB5L.qab_tasks(E12_RECORDS, SOLUTIONS, rep.get("eval_unit_records"), NB5_QAB_GROUP_OF, P["task_list"],\n',
             '    tasks = NB5L.qab_tasks(E12_RECORDS, SOLUTIONS, rep.get("eval_unit_records"), NB5_QAB_GROUP_OF, P.get("task_list") or list(TASK_IDS),   # E28 K100\n',
             "K12")
    anchor = '    return K100Q.finish_table(tasks, window, phase_dir, window_start_epoch, task_solve, P.get("task_list"))\n'
    c = once(c, anchor, anchor + E28_ENGINE, "K10 static check def")
    return c


CELL7_HEAD = "def e21_cell7():\n"
CELL7_PACK = r'''# ---- E28 K100 (size: the 400-kB commit limit): e21_cell7() = E15B CELL 7 (R3K2) as K100Q runs it, UNCHANGED, as ONE
# sha-pinned lzma + base85 literal executed into this namespace (E27 NBDF's technique; the def is the same object either
# way). The readable text is in Build/e28k100_code_export.py.
exec(compile(__PACKED__, "<E28 K100: e21_cell7 = E15B CELL 7>", "exec"), globals())
'''


def pack_cell7(c):
    """The e21_cell7 def (from 'def e21_cell7():' to the end of the cell, untouched by any edit) -> CELL7_PACK."""
    i = c.index(CELL7_HEAD)
    assert c.count(CELL7_HEAD) == 1
    body = c[i:]
    assert body == BASE_CELLS[6][1][BASE_CELLS[6][1].index(CELL7_HEAD):], "e21_cell7 was edited: pack only the K100Q text"
    return c[:i] + CELL7_PACK.replace("__PACKED__", packed_lzma(body, "e21_cell7")), {"sha256": sha(body), "bytes": len(body.encode())}


def cell6(c):
    if E28S is not None:
        n3 = next(e for e in E28S.NOTEBOOK_EDITS if e[0].split()[0] == "N3")
        c = B.once(c, n3[1], n3[2], "E28 SCHED " + n3[0])
        c = once(c, '    E12_ORDER = (list(TASK_IDS) if os.environ.get("E28_SCHED", "").strip() == "1"   # E28 SCHED N3: no ORDER2\n',
                 '    E12_ORDER = (list(TASK_IDS) if os.environ.get("E28_SCHED", "").strip() == "1" or P.get("plain_order")   # E28 SCHED N3: no ORDER2\n',
                 "K11 plain order")
    else:
        c = once(c, "    E12_ORDER = sorted(TASK_IDS, key=lambda t: (-_e12_order2_score(t), _e12_total_cells(t), _E12_INDEX[t]))\n",
                 "    E12_ORDER = (list(TASK_IDS) if P.get(\"plain_order\")   # E28 K100: no ORDER2 (interim: the scheduler's N3 goes here)\n"
                 "                 else sorted(TASK_IDS, key=lambda t: (-_e12_order2_score(t), _e12_total_cells(t), _E12_INDEX[t])))\n",
                 "K11 plain order (interim)")
    return c


E28_I8_SETUP = r'''    E28_I8.update(want=bool(P.get("i8")), on=bool(P.get("i8")))   # E28 K100 I8: DRAFT_HEAD=int8 from the first boot try
    if E28_I8["want"]:
        E28_I8["static"] = e28_i8_static()
        if E28_I8["static"].get("verdict") == "UNSUPPORTED_STATIC":
            E28_I8.update(on=False, dropped={"label": E28L.I8_DROP, "kind": "STATIC", "why": E28_I8["static"].get("why")})
            log("!!!!!!!!!! E28 K100", E28L.I8_DROP, "(static):", json.dumps(E28_I8["static"], default=str)[:600], "-> BF16 draft head")
    NB5_REPORT["e28_i8"] = E28_I8
    NB5_REPORT["e28_sched"] = E28_SCHED_REC
    log("E28 K100 I8", json.dumps({k: E28_I8.get(k) for k in ("want", "on", "dropped")}, default=str)[:600], "| static",
        (E28_I8.get("static") or {}).get("verdict"), "| scheduler applied", E28_SCHED_REC.get("applied"))
'''


def runner(c):
    c = once(c, '                      "source": "NB5 QUALITY Q_FP8_K3 window (planned 4049.8 s, actual 4049.5 s)"})\n',
             '                      "source": "E28 K100: the fixed 6-h EVAL120 window (README section 1)"})\n', "K13 window label")
    c = once(c, "    k100_setup()   # E26 K100: OVERLAY_MERGE_K6 gate, worker extension, static scales sidecar (any miss: no boot)\n",
             "    k100_setup()   # E26 K100: OVERLAY_MERGE_K6 gate, worker extension, static scales sidecar (any miss: no boot)\n"
             + E28_I8_SETUP, "K13 I8 setup")
    return c


E28_REPORT = r'''# ---- E28 K100: the run report (e28k100_lib.build_e28_report): score, tok/s, preemptions, prefix hit, the per-attempt table
# (A / B1 / B2 / C), the B2 (200k) attempts with their context reached, I8 / Q-defer / scheduler / boot facts
E28_OUT = {"status": None}
try:
    _kn = P["nb5_phases"][0][0]   # E28 K100: the commit phase (Q_E28_K100_EVAL120) / the rerun's EVAL
    _kp = (NB5_REPORT.get("phases") or {}).get(_kn) or {}
    _kd = BASE_OUT / f"phase_{_kn}"
    _kpc = ((NB5_REPORT.get("quality_ab") or {}).get("per_config") or {}).get(_kn) or {}
    _kb = NB5_BOOTS[-1] if NB5_BOOTS else {}
    _to_ep = lambda s_: (time.time() - (time.perf_counter() - (NOTEBOOK_T0 + float(s_)))) if s_ is not None else None
    _kw = dict(_kpc.get("window") or {})
    _kw.update(start_epoch=_to_ep(_kw.get("start_sec")), end_epoch=_to_ep(_kw.get("end_sec")))
    _rr = _load_json(_kd / "e12_run_report.json") if (_kd / "e12_run_report.json").exists() else {}
    _ga = _load_json(_kd / "e12_generated_attempts.json") if (_kd / "e12_generated_attempts.json").exists() else {}
    _qrb = ((_kb.get("tries") or [{}])[-1] or {}).get("qdefer_readback") or {}
    _qd = {"applied": K100_QD.get("applied"), "status": K100_QD.get("status"), "error": K100_QD.get("error"),
           "readback_label": _qrb.get("label"), "exec_share_min": _qrb.get("exec_share_min"), "launch_env": K100_QD.get("launch_env")}
    _sch = {"rec": E28_SCHED_REC, "e28": ((_rr.get("e12_run") or {}).get("rev6_scheduler") or {}).get("e28")}
    _ksr = _kd / "nb5_sampler.jsonl"   # E28 K100: the 15-s sampler (preemption counter) -> the per-attempt preemption flag
    _ksamp = [json.loads(l_) for l_ in _ksr.read_text(encoding="utf-8").splitlines() if l_.strip()] if _ksr.exists() else None
    E28_OUT = E28L.build_e28_report(phase=_kp, per_config=_kpc, run_report=_rr, generated_attempts=_ga, records=E12_RECORDS,
                                    solutions=SOLUTIONS, task_ids=list(NB5_ALL_IDS), boot=_kb, i8=E28_I8, qdefer=_qd, sched=_sch,
                                    window=_kw, budgets={"T1": E28_T1, "A": E28_T1, "B1": E28_T1, "C": E28_T1, "B2": E28_B2},
                                    phase_of=lambda r_: (r_.get("e28") or {}).get("phase"), rerun=RERUN, max_model_len=MAX_LEN,
                                    sampler=_ksamp)
    if not RERUN and _kp:
        try:
            E28_OUT["finish"] = k100_finish(_kpc.get("tasks"), _kpc.get("window"), _kd, _kw.get("start_epoch"), _kp.get("task_solve"))
        except Exception as _e:
            E28_OUT["finish_error"] = repr(_e)[:300]
        E28_OUT["flags"] = K100Q.review_flags(P, NB5_PIN, _kp, _kpc)
        E28_OUT["table_md"] = E28L.markdown(E28_OUT)
except BaseException:
    E28_OUT.update(status="ERROR", error=traceback.format_exc()[-2500:])
try:
    write_json_atomic(BASE_OUT / "e28_k100_report.json", E28_OUT)
    print("E28 K100 REPORT:", json.dumps({k: E28_OUT.get(k) for k in ("status", "score", "engine", "attempt_summary", "b2", "flags",
                                                                         "boot", "boot_tries", "errors")}, default=str)[:12000], flush=True)
    print("E28 K100 I8 / Q-defer / scheduler:", json.dumps({"i8": E28_OUT.get("i8"), "qdefer": E28_OUT.get("qdefer"),
                                                            "sched_applied": E28_SCHED_REC.get("applied")}, default=str)[:3000], flush=True)
    print(E28_OUT.get("table_md") or "E28 K100: no per-attempt table", flush=True)
except BaseException:
    print("E28 K100 report write error:", traceback.format_exc()[-1500:], flush=True)
'''


def report(c):
    c = once(c, '            print("deployment_comparison: not applicable (single phase Q_K100_FP8_K3; k100_quality_report.json compares it with "\n'
                '                  "NB5 QUALITY\'s FP8 phase)", flush=True)\n',
             '            print("deployment_comparison: not applicable (E28 K100: one phase; e28_k100_report.json is its report)", flush=True)\n',
             "K14 deployment line")
    c = once(c, "for t in (P.get(\"task_list\") or NB5_QUALITY_TASKS)}}",
             "for t in (P.get(\"task_list\") or NB5_ALL_IDS)}}   # E28 K100", "K14 paired list")
    c = once(c, KQ.K100_REPORT, E28_REPORT, "K14 report")
    c = once(c, '        for fp in [BASE_OUT / "nb5_report.json", BASE_OUT / "k100_quality_report.json", SUBMISSION_PATH,',
             '        for fp in [BASE_OUT / "nb5_report.json", BASE_OUT / "e28_k100_report.json", SUBMISSION_PATH,   # E28 K100\n'
             '                  ', "K14 bundle")
    return c


MD0 = f"""# E28 K100 EVAL120 6 h · klee100 3bpw-MTP, every tested speed lever · __PROFILE__

- **Question:** 120 eval tasks in a fixed **6-h** solve window on klee100 (W3 g128 / W2 g64 routed experts via humming, W4 MTP,
  FP8 PLE graft, fixkeys) with every tested speed lever. Paired with the W4 notebook (two accounts at once); the winner goes
  into Exp 27 (DFlash2).
- **Base = K100Q** (sha 4506a71f, GPU VALID 2026-11-01: 23 / 30 tasks, 385 tok/s window average, 0 preemptions, prefix hit
  85.4 %): klee100 boot + gates, FP8 KV + static QSA scales, MTP K3, Q-defer (nb4_ppack rev 2), pin 7,933,526,016 B -> pool
  479, F 6 (K100Q: 21 workers; E28: 24), MERLIN_SHARE_DRAFT_WORKSPACE=1, #60490, the relaunch, the watchdog, the commit deadman.
- **E28 levers:** **I8** (DRAFT_HEAD=int8; a failed boot with it -> the same pin with the BF16 head, I8_UNSUPPORTED_ON_K100);
  thinking **T1 = {T1:,}** per attempt (reserve 4,096), **B2 = {B2_TOKENS:,}** for cap-cut re-runs (user 2026-11-01),
  K100Q's **--max-model-len {SERVER_MAX_LEN:,}**; the **A / B / C scheduler** (E28_SCHED=1: attempt 1 of every task, then the B1 / B2
  re-runs, then attempt 2), **no ORDER2**, best train -> attempt_1; K100Q's 'slots' admission with **24 workers / --max-num-seqs 24 on the top rung** (lower rungs: K100Q's 18 / 14 / 9 / 6;
  preemptions counted, flagged per attempt).
- **Budget:** {BUDGET_MIN} min, HARD_STOP {BUDGET_MIN - HARD_MARGIN_MIN} min (6 h 50 min), deadman at {DEADMAN_SEC // 60} min
  (commit only). **Dual mode (proven):** commit = EVAL120 scored; rerun = every hidden task, the identity fallback, E15B's
  writer, the same levers, 12 h. Report: `q27x/e28_k100_report.json` + `q27x/nb5_report.json`; full saved responses in the
  Duck bundle.
"""


def cells():
    cl = list(BASE_CELLS)
    out = [("markdown", MD0.replace("__PROFILE__", DEFAULT_PROFILE)), cl[1], ("code", cell1(cl[2][1])), ("code", cell3(cl[3][1])),
           cl[4], ("code", engine(cl[5][1])), ("code", cell6(cl[6][1])), ("code", runner(cl[7][1])), ("code", report(cl[8][1]))]
    assert out[1][1] == KQ.E18R2_CELL0, "CELL 0 must stay E18 R2's (dual mode + fallback submission)"
    return out


def kernel_metadata():
    return {"id": f"{OWNER}/{SLUG}", "title": "Q27 Xhigh E28 K100 Eval120 6h", "code_file": f"{SLUG}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": KQ.DATASETS, "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": KQ.MODELS, "docker_image": KQ.DOCKER, "machine_shape": "NvidiaL4"}


def notebook_cells(cl):
    """The cells as written into the notebook: CELL 6/7's e21_cell7 def packed (CELL7_PACK); everything else as built."""
    out = list(cl)
    c6, rec = pack_cell7(cl[6][1])
    out[6] = ("code", c6)
    return out, rec


def notebook_bytes(cl):
    """N6.notebook_bytes (B.norm + compile of every code cell, the K100Q JSON) re-dumped as compact JSON (E27 NBDF's
    technique: separators (",", ":"), the same cells and sources)."""
    data, n = N6.notebook_bytes(cl)
    nb = json.loads(data.decode("utf-8"))
    return (json.dumps(nb, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8"), n, "compact"


def build():
    cl = cells()
    B.undefined_names_check(cl)                      # on the readable cells (the packed e21_cell7 is the same text)
    ncl, pack7 = notebook_cells(cl)
    data, n, fmt = notebook_bytes(ncl)
    assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
    assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", len(data))
    out = OUT_NB
    if out.exists() and sha(out.read_bytes()) in PUSHED:
        out = out.with_name(out.stem + "_REBUILD.ipynb")
        assert not (out.exists() and sha(out.read_bytes()) in PUSHED), out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / f"{SLUG}.ipynb").write_bytes(data)
    (PUSH / "kernel-metadata.json").write_bytes((json.dumps(kernel_metadata(), indent=2) + "\n").encode("utf-8"))
    EXPORT.write_bytes("\n\n".join(B.norm(t) for k, t in cl if k == "code").encode("utf-8"))
    facts = {"notebook": str(out), "notebook_sha256": sha(data), "bytes": len(data), "cells": n, "json_format": fmt, "e21_cell7_packed": pack7,
             "export_note": "the export holds the READABLE cells (e21_cell7 unpacked); the notebook's CELL 6/7 carries it packed",
             "interim": E28S is None, "base": {"k100q_sha256": K100Q_SHA, "k100q_notebook": str(K100Q_NB),
                                               "builder": str(E25B / "build_e26_k100_quality.py"),
                                               "builder_sha256": sha((E25B / "build_e26_k100_quality.py").read_bytes())},
             "default_profile": DEFAULT_PROFILE, "phase": "Q_E28_K100_EVAL120",
             "modules": {nm: sha(t) for nm, t in MODULES.items()}, "modules_json_sha256": sha(MODULES_JSON), "sched": SCHED_REC,
             "lib_version": E28L.LIB_VERSION,
             "t1": T1, "b2": B2_TOKENS, "server_max_len": SERVER_MAX_LEN, "b2_window": B2_WINDOW, "b2_timeout": B2_TIMEOUT, "workers": WORKERS, "pp2_microbatch_tokens": PP2_MICROBATCH_TOKENS, "reserve": RESERVE,
             "window_sec": WINDOW_SEC, "budget_min": BUDGET_MIN, "hard_margin_min": HARD_MARGIN_MIN, "deadman_sec": DEADMAN_SEC,
             "hard_stop_plan": HARD_STOP_PLAN, "token_plan": TOKEN_PLAN, "capacity": CAPACITY, "i8_env": I8_ENV,
             "pins": KQ.K100_PINS, "pin_steps_gib": KQ.K100_GIB, "seqs_table": {str(k): v for k, v in KQ.K100_SEQS_TABLE.items()},
             "captures": KQ.K100_CAPS, "expected": KQ.EXPECTED, "k3_v2_gpu": _K3V2,
             "payload": {"file": KQ.PAYLOAD_NAME, "sha256": KQ.PAYLOAD_SHA, "bytes": len(KQ.PAYLOAD_BYTES),
                         "entries": {k: v[0] for k, v in sorted(KQ.PAYLOAD.items())}, "dataset": KQ.DATASETS[0], "staged": str(KQ.DS_STAGE)},
             "dataset_files_read": {"k100load_lab.py": KQ.LAB_SHA, "qsa_kv_scales_262k.json": KQ.SCALES_SHA, KQ.PAYLOAD_NAME: KQ.PAYLOAD_SHA,
                                    "nb4_ppack.py": KQ.PPACK_SHA, "k100q_report.py": sha(KQ.REPORT_LIB_TEXT.encode()) if KQ.REPORT_LIB_TEXT else None},
             "qdefer": {"sha256": KQ.PPACK_SHA, "shas": KQ.K100_QD_SHAS},
             "embedded": {"nb5_lib.py (in the payload file)": sha(B.NB5_LIB.encode()),
                          "k100q_report.py": sha(KQ.REPORT_LIB_TEXT.encode()) if KQ.REPORT_LIB_TEXT else None},
             "nb6_sha256": KQ.NB6_SHA, "e18r2_sha256": KQ.E18R2_SHA, "k3only_sha256": KQ.K3ONLY_SHA,
             "kernel_metadata": kernel_metadata(), "push": str(PUSH), "export": str(EXPORT),
             "push_order": ["0. the tf-ref dataset version that K100Q read (ds_e25_nb0_tf_ref_k100q: k100q_payload_emb.json, "
                            "k100q_report.py, nb4_ppack.py) must be the mounted version (K100Q ran on it; nothing new to push)",
                            "1. the CPU kits (esfersami50)", "2. the GPU kernel (user go only)"],
             "decisions": {
                 "workers": "24 workers and --max-num-seqs 24 on the TOP rung (7,933,526,016 B, pool 479) of both profiles (user "
                            "2026-11-01; K3_ONLY v2 460 -> 562 tok/s at 20k from 16 to 24 streams; the extra ~0.17 GiB at 24 seqs fits PP1's "
                            "free memory, the pin stays); lower rungs keep K100Q's R2 value 18 / 14 / 9 / 6; captures = K100Q's "
                            "(max 84) cover 48 tokens per PP2 micro-batch (24 x 4 / 2); NB5's whole-batch rule (24 x 4 = 96) is recorded "
                            "as captures_cover_workers False and is not the PP2 bound",
                 "admission": "'slots' as the proven K100Q run (mean 16.5 running, max 21, 0 preemptions at 104k): lead 2026-11-01, the user's no-idle rule; 21 workers, the engine schedules KV; a preemption is a recompute (counted, and flagged per attempt from the 15-s sampler); the scheduler's N1 (the attempt's C-9 peak: needed for B2), N2 (C-7 above-pool check; the slot itself = 1) and N4 (a non-blocking slot) are kept: inert or required in slots",
                 "peaks": "a T1 attempt's projected peak = min(131,072, 8,192 + 104,000 + 8,192) = 120,384 (36 blocks at F 6); "
                          "B2 = min(131,072, 136,384) = 131,072 (38 blocks): K100Q's MAX_LEN, unchanged",
                 "i8_drop": "a boot try that fails with the INT8 head (launch failure / OOM before any phase / C-11 config mismatch) "
                            "is retried at the SAME pin with the BF16 head (I8_UNSUPPORTED_ON_K100); so is a real boot whose read-back is not "
                            "I8_ENGAGED (I8_READBACK_UNVERIFIED, kind READBACK_UNVERIFIED); a static UNSUPPORTED verdict drops "
                            "I8 before the first boot; the pin ladder only moves after the BF16-head try fails",
                 "sched_failure": "commit: NO_GO_E28_SCHED (raised in CELL 3: no solve, the identity fallback); rerun: reverted, the "
                                  "original scheduler (never worse than no answers)",
                 "attempt_env": "NB6 / E15B SUBMIT's attempt env (no fixed LOCAL_ANALYZER_SEED: E28_SCHED_API section 4)"}}
    FACTS.write_bytes((json.dumps(facts, indent=1, default=str) + "\n").encode("utf-8"))
    print("wrote", out.name, "sha256", sha(data), "|", len(data), "bytes |", n, "cells |", fmt,
          "| INTERIM (no scheduler)" if E28S is None else "| scheduler " + SCHED_REC["sha256"][:12])
    return out


if __name__ == "__main__":
    build()
