"""E28 W4 EVAL120 6 h (lead 2026-11-01, user decisions in ../../README.md): the Flash-Next W4 family with every tested speed
lever, built as a thin wrapper over the NB6Q SUBMIT builder (Exp 25 build_e25_nb6q.py, which imports build_e25_nb6.py ->
build_e25_nb5.py; none of them is written). NB6Q's cells are rebuilt and asserted byte for byte against the NB6Q notebook on
disk (sha 499a4319, CPU kits ALL OK); only the marked "E28 W4" edits below are applied. Kept from NB6Q (proven): the RC2 W4
boot, FP8 KV + static scales, MTP K3, nb4_ppack rev 2 Q-defer (NB4_PPDEFER=1), #60490, MERLIN_SHARE_DRAFT_WORKSPACE=1, the
pin ladder + E18 R2 relaunch / readiness / fast kill, the E15B watchdog, CELL 0 = E18 R2 CELL 0 verbatim, the E15B writer,
rerun = every hidden test task with the identity fallback.

E28 W4 edits (marked "# E28 W4 Wn" in the cells):
  W1  (CELL 1) default commit profile = E28EVAL120 (the hidden rerun stays RERUN_PROD)
  W2  (CELL 1) E28_LEVERS + the profiles: E28EVAL120 = source eval, all 120 tasks, 412-min budget (_NB5_T fracs / margins:
               HARD_STOP 410 min = 6 h 50 min), a fixed 6-h solve window (equal_windows + fixed_window_sec, the K100Q
               mechanism: shortened, never skipped, when the boot is late); RERUN_PROD = NB6's 12 h - 3 min, every test task,
               the same E28 levers. Levers: T1 = 104,000 per attempt (reserve 4,096; user 2026-11-01), C12 (launch --max-num-seqs 12 on every
               rung, workers 12 regardless of the pool rule), I8 (DRAFT_HEAD=int8 into Merlin's build(), NB4T), pin ladder
               starting at the I8 pin (RC2 + 0.75 GiB = 3,285,145,068 B; then 0.5 / 0 GiB, E18's <= 3 attempts),
               --language-model-only (NB4T Q), NB6Q's server --max-model-len 131072, B2 = 120,000 (9,000-s request timeout; T1 7,200 s), slots,
               no ORDER2, the A/B/C scheduler
  W3  (CELL 1) NB5_EXTRA_ARGS + the profile's e28_extra_args (--language-model-only); E28 constants (build env, B2 budget) and
               the B2 assert (<= 120,000; every attempt keeps the 130,560 window)
  W4  (removed with the user's 131,072 server, 2026-11-01: E21_VLLM_MAX_MODEL_LEN stays NB6Q's 131072)
  W5  (CELL 1) the commit DEADMAN at HARD_STOP + 5 min (= E26 K100Q's deadman code, names changed; commit only)
  W6  (engine) merlin_fn_server (--max-model-len stays NB6Q's MAX_LEN 131072): E28_BUILD_ENV into build() only
               (= NB4T CELL 2's edit), recorded in the plan; the fake gets it too (kit read-back path)
  W7  (engine) nb5_capacity: workers = P["workers_fixed"] (12) ONLY on the C12 rung (the I8 start pin + its same-pin BF16
               retry); every lower rung keeps NB6Q's slots rule min(rule, launch seqs); recorded in plan["e28_workers"]
  W8  (engine) nb5_boot: the first rung = P["pin_start_gib"] (0.75) unless #60490 failed (then the RC2 pin, NB5's rule)
  W9  (engine) nb5_boot: launch --max-num-seqs = P["seqs_fixed"] (12) ONLY when the rung's kv bytes = the I8 start pin
               (P["seqs_fixed_kv_bytes"] = 3,285,145,068; the same-pin BF16 retry too); lower rungs = NB6Q's SEQS_TABLE
               (coordinator 2026-11-01: 12 on the ~149-block RC2 pin would be a preemption storm); E28_RUNG + trec["e28_rung"]
  W10 (engine) nb5_boot_try: e28_readback after the warm-up and at the end (I8 label, seqs / max-model-len / text-only, pool /
               F / workers; record only)
  W20 / W21 / W22 (engine) the I8 boot GATE (lead 2026-11-01, K100 audits): after ready, DRAFT_HEAD=int8 must read back ENGAGED
               (NB4T's read-back), else I8_READBACK_UNVERIFIED: the server stops, the SAME pin is retried with the BF16 head
               (own launch name, recorded in brec["i8_fallback"]); only a failed BF16 retry steps the pin ladder down
  W23 (engine) the mid-run relaunch = K100Q's proven fix D (build_e26_k100_quality.py K8b, NB5_ACTIVE_BOOT, ported
               verbatim): e21_relaunch boots the try whose server came up (kv bytes, launch extra = seqs, levers, NB4_PPDEFER)
               instead of NB6Q's RC2 pin with the boot's seqs; + E28: that try's I8 state (E28_BUILD_ENV, DRAFT_HEAD only if it
               used it); NB5_REPORT["relaunch_boot"] records pin / seqs / head / c12 / workers (coordinator 2026-11-01)
  W11 (CELL 6/7) no ORDER2 when P["no_order2"]: plain task order (TASK_IDS)
  W12 (CELL 1 end) ONE packed literal (json -> lzma -> base85, sha256 checked) with e28w4_lib.py [+ e28_sched_patch.py]
  W13 (CELL 8) the fixed solve window (K100Q K9 snippet) before the boots
  W14 (CELL 9) the E28 summary in nb5_report.json + one print
  W15 (markdown) the E28 header
  SCHED (integration point) e28_sched_patch.py via Build/E28_SCHED_API.md (see sched_edits())

usage: python build_e28_w4.py  ->  ../Q27-XHIGH-E28_W4_EVAL120_6H.ipynb + e28w4_code_export.py + e28w4_build_facts.json +
       the GPU push folder D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e28-w4-eval120-6h__foysalemonshanto/ (nothing is pushed)
"""
import ast
import base64
import hashlib
import json
import lzma
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
sys.path.insert(0, str(E25B))
import build_e25_nb6q as NQ   # noqa: E402  (module import only: build_e25_nb6q.build() is never called here)
sys.path.insert(0, str(HERE))
import e28w4_unpack as U      # noqa: E402  (the review view: shared with e28w4_unit_tests.py)

B, N6 = NQ.B, NQ.N6
NB6Q_SHA = "499a431904fdaf47831ef14529b7b956c45d9e374c6c427b3b3e4698fa8a5c7d"     # Q27-XHIGH-E25_NB6Q_SUBMIT.ipynb (kits ALL OK)
NB6Q_NB = B.NBDIR / "Q27-XHIGH-E25_NB6Q_SUBMIT.ipynb"
K100Q_BUILDER = E25B / "build_e26_k100_quality.py"
NB4T_LIB = E25B / "nb4t_lib.py"
OUT = HERE.parent / "Q27-XHIGH-E28_W4_EVAL120_6H.ipynb"
EXPORT = HERE / "e28w4_code_export.py"
FACTS = HERE / "e28w4_build_facts.json"
DIFF = HERE / "e28w4_review_diff_vs_nb6q.diff"
LIB = HERE / "e28w4_lib.py"
SCHED = HERE / "e28_sched_patch.py"
SCHED_API = HERE / "E28_SCHED_API.md"
PUSH_ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
PUSHQ = PUSH_ROOT / "q27-xhigh-e25-nb6q-submit__foysalemonshanto"
SLUG = "q27-xhigh-e28-w4-eval120-6h"
PUSH = PUSH_ROOT / f"{SLUG}__foysalemonshanto"
PUSH_SAMI = PUSH_ROOT / f"{SLUG}__esfersami50"     # the GPU runs go on esfersami50 (user decision 2026-11-01); same notebook + sources
PUSHED: set = set()   # sha256 of every E28 W4 notebook that ran on Kaggle (never overwrite one: rebuild -> *_REBUILD)

# the numbers (user decisions 2026-11-01, README sections 1 / 3 / 4; NB4T GPU run 675b1b9d)
T1 = 104000                     # user 2026-11-01 (NB6Q's proven value): attempt 1, B1 re-runs, Phase-C attempts (reserve 4,096)
B2_TOKENS = 120000              # user 2026-11-01: B2 (cap-cut) re-runs only
SERVER_MAX_LEN = 131072         # user 2026-11-01: the proven server --max-model-len (NB6Q's MAX_LEN, unchanged)
B2_WINDOW = SERVER_MAX_LEN - 512   # = NB5_WINDOW (130,560): B2 keeps the normal Duck window
B2_TIMEOUT = 9000               # user 2026-11-01: E28_B2_REQUEST_TIMEOUT; T1 = 7,200 s (E12_DUCK_REQUEST_TIMEOUT)
C12 = 12                        # NB4T arm C12: a trailing --max-num-seqs 12; workers 12 (user decision) -- on the I8 rung only
I8_PIN_GIB = 0.75               # NB4T arm I8: OOM at 3,553,580,524 (RC2 + 1.0 GiB), booted at 3,285,145,068 (RC2 + 0.75 GiB)
RC2_FP8_PIN = 2479838700
assert RC2_FP8_PIN + int(round(I8_PIN_GIB * 2 ** 30)) == 3285145068
C12_KV_BYTES = RC2_FP8_PIN + int(round(I8_PIN_GIB * 2 ** 30))   # the only rung with C12 (coordinator 2026-11-01)
BUDGET_MIN, HARD_MARGIN_MIN, WINDOW_SEC = 412, 2, 6 * 3600     # HARD_STOP 410 min = 6 h 50 min; deadman 415 min; solve 6 h
DEADMAN_AFTER_HARD_SEC = 5 * 60


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def once(text, old, new, label):
    assert text.count(old) == 1, (label, text.count(old))
    return text.replace(old, new)


def b85x(text):
    """lzma (preset 9 extreme) + base85 (E27 NBDF / NB6Q Q1): no quote / backslash in the alphabet."""
    return base64.b85encode(lzma.compress(text.encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


def lf(p):
    return p.read_bytes().decode("utf-8").replace("\r\n", "\n")


# ------------------------------------------------------------------ provenance (fail the build on any drift)
NQ_CELLS = NQ.cells()
assert sha(N6.notebook_bytes(NQ_CELLS)[0]) == NB6Q_SHA, "build_e25_nb6q.py no longer reproduces NB6Q SUBMIT 499a4319"
assert sha(NB6Q_NB.read_bytes()) == NB6Q_SHA, "NB6Q SUBMIT notebook changed on disk"
# NB4T's I8 read-back regexes (e28w4_lib.py copies them verbatim)
_nb4t = lf(NB4T_LIB)
LIB_SRC = lf(LIB)
for _rx in ('_WTAG = re.compile(r"\\(Worker_PP(\\d+)_TP(\\d+)")',
            '_HEAD_RX = re.compile(r"Draft lm_head: (INT8|INT4) weight-only copy of the (\\d+)x(\\d+) shard \\(([\\d.]+) MiB")',
            '''_SPEC_RX = re.compile(r"'speculative_config': (\\{[^{}]*\\})")'''):
    assert _rx in _nb4t, ("nb4t_lib regex changed", _rx)
    assert ("_E28" + _rx.replace("re.compile", "_e28_re.compile")) in LIB_SRC, ("e28w4_lib.py: not NB4T's regex", _rx)
compile(LIB_SRC, "e28w4_lib.py", "exec")
assert {"e28_i8_parse", "e28_i8_gate", "e28_readback", "e28_summary", "e28_sched_report", "_e28_literal", "_e28_block", "_d", "_l",
        "preempt_events", "flag_preemptions"} == {
    n.name for n in ast.parse(LIB_SRC).body if isinstance(n, ast.FunctionDef)}

# ------------------------------------------------------------------ W5: the E26 K100Q deadman, cut by ast, names changed
_k100_src = lf(K100Q_BUILDER)
K100_DEADMAN_CODE = next(ast.literal_eval(n.value) for n in ast.parse(_k100_src).body if isinstance(n, ast.Assign)
                         and getattr(n.targets[0], "id", None) == "K100_DEADMAN_CODE")
K100_DEADMAN_SHA = sha(K100_DEADMAN_CODE)
_dm = K100_DEADMAN_CODE
_dm_head_old = _dm[:_dm.index("K100_DEADMAN_SEC = __K100DEADMAN__\n")]
assert _dm_head_old.startswith("# ---- E26 K100 (user 2026-11-01): the commit's ABSOLUTE 150-min DEADMAN") and _dm_head_old.count("\n") == 4
_dm = _dm.replace(_dm_head_old, (
    "# ---- E28 W4 W5 (lead 2026-11-01): the commit's ABSOLUTE DEADMAN at HARD_STOP + 5 min = E26 K100Q's deadman code (cut from\n"
    "# build_e26_k100_quality.py by ast, sha-pinned at build time; only the names, the marker file and the fire time differ).\n"
    "# A daemon thread sleeps until NOTEBOOK_T0 + E28_DEADMAN_SEC, then: one marker line (stdout fd + stderr fd +\n"
    "# /kaggle/working/e28w4_deadman.txt), SIGKILL every child (psutil tree; else os.killpg on the servers' groups + pkill -9 -f\n"
    "# vllm), os._exit(0). No lock, no main thread, no cell. The hidden rerun (RERUN_PROD) has no deadman (= K100Q).\n"))
for _o, _n, _c in (("K100_DEADMAN_SEC = __K100DEADMAN__\n",
                    "E28_DEADMAN_SEC = (HARD_STOP - NOTEBOOK_T0) + __E28DEADMANAFTER__   # E28 W4 W5: HARD_STOP + 5 min\n", 1),
                   ("K100_DEADMAN", "E28_DEADMAN", None), ("k100_deadman", "e28_deadman", None),
                   ("@@K100Q_DEADMAN_FIRED@@", "@@E28W4_DEADMAN_FIRED@@", 1), ("k100q_deadman.txt", "e28w4_deadman.txt", 1),
                   ('"K100 DEADMAN armed: fires at +', '"E28 W4 DEADMAN armed: fires at +', 1),
                   ('name="k100q_deadman"', 'name="e28w4_deadman"', 1)):
    assert _o in _dm and (_c is None or _dm.count(_o) == _c), ("deadman rename", _o, _dm.count(_o))
    _dm = _dm.replace(_o, _n)
_dm_code = "\n".join(l for l in _dm.splitlines() if not l.lstrip().startswith("#"))
assert "K100" not in _dm_code and "k100" not in _dm_code, "deadman: a K100 name is left in the code"
E28_DEADMAN_CODE = _dm.replace("__E28DEADMANAFTER__", str(DEADMAN_AFTER_HARD_SEC))

# ------------------------------------------------------------------ the scheduler module (integration point)
SCHED_STATE = {"api": SCHED_API.is_file(), "module": SCHED.is_file()}
SCHED_SRC = lf(SCHED) if SCHED.is_file() else None
SCHED_SHA = sha(SCHED_SRC) if SCHED_SRC is not None else None
SCHED_API_SHA = sha(SCHED_API.read_bytes()) if SCHED_API.is_file() else None

# ------------------------------------------------------------------ CELL 1 edits
_W1_OLD = 'NB5_DEFAULT_PROFILE = "NB6COMMIT"\n'
_W1_NEW = 'NB5_DEFAULT_PROFILE = "E28EVAL120"   # E28 W4 W1: the commit profile (the hidden rerun stays RERUN_PROD)\n'
_W2_OLD = ('    "NB6COMMIT": dict(_NB5_T, time_budget_sec=50 * 60, source="eval", task_list=NB6_COMMIT_TASKS, **NB6_LEVERS),\n'
           '}\n')
_W2_NEW = _W2_OLD + r'''# ---- E28 W4 W2 (lead 2026-11-01, user decisions: Exp 28 README sections 1 / 3 / 4): every tested speed lever on NB6Q's base.
# T1 = 104,000 per attempt (user 2026-11-01, NB6Q's proven value: attempt 1, B1 re-runs, Phase-C attempts; reserve 4,096 ->
# the longest request 99,904); B2 (cap-cut) re-runs = 120,000 (user 2026-11-01); the server keeps NB6Q's proven --max-model-len
# 131,072 and every attempt (B2 too) the 130,560 Duck window.
# C12 (NB4T arm C12: a trailing --max-num-seqs 12; CAPS16 covers 12 x (1 + K3) = 48): launch seqs 12 and workers 12 ONLY on
# the I8 start pin (seqs_fixed_kv_bytes = 3,285,145,068 B) and its same-pin BF16 retry; every lower rung (0.5 / 0 GiB: smaller
# pools, the RC2 pin ~149 blocks) keeps NB6Q's proven rule: SEQS_TABLE seqs (seqs_rule 'table', inherited) and workers =
# min(the slots rule, launch seqs) (coordinator 2026-11-01). Admission = NB6Q's 'slots' (lead 2026-11-01, the user's no-idle
# rule, = K100Q): one slot per attempt, the engine schedules KV (a preemption = a recompute: counted at run level and per attempt);
# C-7 (an attempt whose own projected peak exceeds the pool never starts) stays.
# I8 (NB4T arm I8): DRAFT_HEAD=int8 into
# Merlin's build() only; it OOMed at RC2 + 1.0 GiB and booted at RC2 + 0.75 GiB = 3,285,145,068 B -> the ladder starts there
# (then 0.5 / 0 GiB; E18 R2's <= 3 attempts). --language-model-only = NB4T's Q flags. No ORDER2 (user 2026-11-01): plain task
# order; the A/B/C scheduler (e28_sched_patch.py, E28_SCHED=1) for the commit AND the hidden rerun. Request timeout (Duck's
# E12_DUCK_REQUEST_TIMEOUT, E15B #47 = 3,600 s): 7,200 s for A / B1 / C (lead 2026-11-01: a ~100k-token request at ~25 tok/s
# per stream takes ~67 min); B2 = 9,000 s (user 2026-11-01, E28_B2_REQUEST_TIMEOUT).
E28_LEVERS = dict(NB6_LEVERS, nb5_attempt={"b0": __T1__, "b1": __T1__}, seqs_fixed=__C12__, workers_fixed=__C12__,
                  seqs_fixed_kv_bytes=__C12KV__,
                  nb5_env=dict(NB6_LEVERS["nb5_env"], E12_DUCK_REQUEST_TIMEOUT="7200"),
                  pin_start_gib=__I8PIN__, build_env={"DRAFT_HEAD": "int8"}, e28_extra_args=["--language-model-only"],
                  b2_attempt=__B2__, b2_request_timeout=__B2TIMEOUT__, no_order2=True, e28_sched=True)
# E28EVAL120: source eval, all 120 tasks; NB5's long-commit timing (_NB5_T: admit / drain fractions, 2-min hard margin) with a
# 412-min budget -> HARD_STOP 410 min (6 h 50 min), the solve end 407 min, the deadman 415 min; a FIXED 6-h solve window
# (K100Q's mechanism: it starts after the boot + preflight and is shortened, never skipped, when the boot is late)
PROFILE_TABLE["E28EVAL120"] = dict(_NB5_T, time_budget_sec=__BUDGETMIN__ * 60, hard_margin_sec=__HARDMIN__ * 60, source="eval",
                                   n_tasks=120, equal_windows=True, fixed_window_sec=__WINDOW__, **E28_LEVERS)
# the hidden rerun: NB6's proven 12 h - 3 min budget, every test task, one EVAL phase, the same E28 levers
PROFILE_TABLE["RERUN_PROD"] = dict(_NB5_T, time_budget_sec=12.0 * 3600 - 3 * 60, source="test", n_tasks=None, **E28_LEVERS)
'''
for _k, _v in (("__T1__", str(T1)), ("__C12__", str(C12)), ("__I8PIN__", repr(I8_PIN_GIB)), ("__B2TIMEOUT__", str(B2_TIMEOUT)), ("__C12KV__", str(C12_KV_BYTES)),
               ("__B2__", str(B2_TOKENS)), ("__BUDGETMIN__", str(BUDGET_MIN)), ("__HARDMIN__", str(HARD_MARGIN_MIN)),
               ("__WINDOW__", str(WINDOW_SEC))):
    _W2_NEW = once(_W2_NEW, _k, _v, "W2 " + _k) if _W2_NEW.count(_k) == 1 else _W2_NEW.replace(_k, _v)
assert not re.findall(r"__[A-Z0-9]+__", _W2_NEW)
_W3_OLD = ('        {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": NB5_CAPS16}, separators=(",", ":"))]\n')
_W3_NEW = _W3_OLD + ('NB5_EXTRA_ARGS = NB5_EXTRA_ARGS + list(P.get("e28_extra_args") or [])   # E28 W4 W3: --language-model-only (NB4T Q)\n')
_W3B_OLD = "assert NB5_CAP <= 131072 - 512 and max(NB5_ATTEMPT.values()) <= 120000, NB5_ATTEMPT\n"
_W3B_NEW = _W3B_OLD + r'''# ---- E28 W4 W3: the E28 constants. Every attempt keeps the asserts above (<= 120,000 per attempt, the 130,560 window) and the
# server NB6Q's --max-model-len 131,072; the B2 (cap-cut) re-run = 120,000 (user 2026-11-01; its projected peak is capped at
# 131,072 by NB5's own C-9 rule, as every NB6Q attempt). E28_BUILD_ENV = Merlin launcher knobs read by build() only (I8:
# DRAFT_HEAD=int8, NB4T CELL 2).
E28_BUILD_ENV = dict(P.get("build_env") or {})
E28_B2_ATTEMPT = int(os.environ.get("E28_B2_ATTEMPT", P.get("b2_attempt") or 0)) if E11_HARNESS else int(P.get("b2_attempt") or 0)
if E28_B2_ATTEMPT:
    assert E11_HARNESS or E28_B2_ATTEMPT == 120000, E28_B2_ATTEMPT                    # B2: the user's 120k (<= 120,000)
    assert E11_HARNESS or E28_B2_ATTEMPT <= 120000, E28_B2_ATTEMPT
assert not P.get("seqs_fixed") or NB5L_CAPTURES_COVER(NB5_CAPS16 if P.get("caps16") else [32], int(P["seqs_fixed"])), P.get("seqs_fixed")
assert not P.get("seqs_fixed") or (P.get("seqs_fixed_kv_bytes") == NB5_KV_BYTES["fp8"] + int(round(float(P["pin_start_gib"]) * 2 ** 30))
                                   and P.get("seqs_rule") == "table"), (P.get("seqs_fixed_kv_bytes"), P.get("seqs_rule"))
E28_RUNG = {}   # E28 W4 W9: the current boot try's rung (kv bytes, gib, c12, seqs, workers); nb5_boot sets it, nb5_capacity reads it
'''
_W3C_OLD = 'CONFIG = {"PROFILE": PROFILE, "RERUN": RERUN, "profile_row": P}\n'
_W3C_NEW = _W3C_OLD + r'''CONFIG["e28"] = {"build_env": E28_BUILD_ENV, "b2_attempt": E28_B2_ATTEMPT, "b2_request_timeout": P.get("b2_request_timeout"),   # E28 W4 W3
                 "t1": NB5_ATTEMPT, "seqs_fixed": P.get("seqs_fixed"), "workers_fixed": P.get("workers_fixed"),
                 "seqs_fixed_kv_bytes": P.get("seqs_fixed_kv_bytes"), "seqs_rule_lower_rungs": P.get("seqs_rule"),
                 "pin_start_gib": P.get("pin_start_gib"), "extra_args": P.get("e28_extra_args"), "no_order2": bool(P.get("no_order2")),
                 "sched": bool(P.get("e28_sched")), "fixed_window_sec": P.get("fixed_window_sec")}
'''
# captures_cover is NB5L's (loaded later in this cell): the same one-line rule, inline (NB2 CAPS16 rule: max >= seqs x (1 + K))
_W3B_NEW = _W3B_NEW.replace("NB5L_CAPTURES_COVER(NB5_CAPS16 if P.get(\"caps16\") else [32], int(P[\"seqs_fixed\"]))",
                            "max(NB5_CAPS16 if P.get(\"caps16\") else [32]) >= int(P[\"seqs_fixed\"]) * (3 + 1)")
_W4_OLD = "E21_VLLM_MAX_MODEL_LEN = 131072\n"
_W4_NEW = "E21_VLLM_MAX_MODEL_LEN = E28_SERVER_MAX_LEN   # E28 W4 W4: the server's real max-model-len (E21's 0.32 finish_reason clamp)\n"
_W5_OLD = "assert NOTEBOOK_T0 < ADMIT_DEADLINE < DRAIN_DEADLINE < HARD_STOP < VERIFY_DEADLINE < NOTEBOOK_T0 + TOTAL_BUDGET_SEC\n"
_W5_NEW = _W5_OLD + E28_DEADMAN_CODE

# ------------------------------------------------------------------ engine edits
_W6A_OLD = '    extra = ["--served-model-name", "q27", "--max-model-len", str(MAX_LEN), "--port", str(port),\n'
_W6A_NEW = ('    extra = ["--served-model-name", "q27", "--max-model-len", str(E28_SERVER_MAX_LEN), "--port", str(port),   # E28 W4 W6\n')
_W6B_OLD = ('                              "8" if kv == "fp8" else "3", "--port", str(port), "--max-model-len", str(MAX_LEN),\n')
_W6B_NEW = ('                              "8" if kv == "fp8" else "3", "--port", str(port), "--max-model-len", str(E28_SERVER_MAX_LEN),   # E28 W4 W6\n')
_W6C_OLD = '                       dict(os.environ, **NB5_BASE_ENV, **NB5_BOOT_LEVERS), port, LOGS)\n'
_W6C_NEW = '                       dict(os.environ, **NB5_BASE_ENV, **NB5_BOOT_LEVERS, **E28_BUILD_ENV), port, LOGS)   # E28 W4 W6: FAKE only\n'
_W6D_OLD = "    base.update(NB5_BOOT_LEVERS)                       # NB5 R2: this boot try's lever env (lever b)\n"
_W6D_NEW = _W6D_OLD + ("    base.update(E28_BUILD_ENV)                         # E28 W4 W6: I8 DRAFT_HEAD=int8 into build() only (= NB4T CELL 2)\n")
_W6E_OLD = ('"lever_env": {k: env.get(k) for k in sorted({**NB5_BASE_ENV, **NB5_BOOT_LEVERS})}}   # NB5Q-ENV-1')
_W6E_NEW = ('"lever_env": {k: env.get(k) for k in sorted({**NB5_BASE_ENV, **NB5_BOOT_LEVERS})},\n'
            '                                                       "build_env": dict(E28_BUILD_ENV)}   # NB5Q-ENV-1 / E28 W4 W6')
_W7_OLD = '    if caps.get("max_num_seqs") is None:\n        plan["warning"] = "max_num_seqs not found in the server log: workers = pool streams"\n'
_W7_NEW = ('    if P.get("workers_fixed"):                       # E28 W4 W7: workers 12 ONLY on the C12 rung (the I8 start pin + its\n'
           '        _e28c12 = bool(E28_RUNG.get("c12"))           # same-pin BF16 retry); lower rungs keep NB6Q\'s slots rule above\n'
           '        plan["e28_workers"] = {"rung_kv_bytes": E28_RUNG.get("kv_bytes"), "rung_gib": E28_RUNG.get("gib"), "c12_rung": _e28c12,\n'
           '                               "launch_seqs": E28_RUNG.get("seqs"), "rule_workers": plan["workers"],\n'
           '                               "workers_fixed": int(P["workers_fixed"]) if _e28c12 else None}\n'
           '        if _e28c12 and plan["workers"] >= 1:          # C-8 NO_GO kept (0 workers stays 0)\n'
           '            plan["workers"] = int(P["workers_fixed"])\n'
           '        plan["e28_workers"]["workers"] = E28_RUNG["workers"] = plan["workers"]\n') + _W7_OLD
_W8_OLD = '        start = NB5_PIN["start_gib"]\n'
_W8_NEW = _W8_OLD + ('        if start is None and P.get("pin_start_gib") is not None:   # E28 W4 W8: the I8 pin first (NB4T: RC2 + 0.75 GiB);\n'
                     '            start = (float(P["pin_start_gib"]) if (FAKE or not P.get("patch_60490")      # #60490 not verified -> the RC2\n'
                     '                                                   or (NB5_PIN.get("patch_60490") or {}).get("ok")) else 0.0)   # pin (NB5)\n')
_W9_OLD = '        seqs = NB5L.seqs_for(NB5_BOOT_KV, tr["gib"]) if NB5_SEQS_RULE == "table" else None\n'
_W9_NEW = ('        globals()["E28_RUNG"] = {"kv_bytes": int(tr["kv_bytes"]), "gib": tr["gib"],   # E28 W4 W9 (a new dict per try):\n'
           '                                 "c12": bool(P.get("seqs_fixed"))      # C12 (seqs 12, workers 12) ONLY on the I8 start pin\n'
           '                                 and int(tr["kv_bytes"]) == int(P.get("seqs_fixed_kv_bytes") or -1)}   # (+ its BF16 retry)\n'
           '        seqs = (int(P["seqs_fixed"]) if E28_RUNG["c12"] else   # = NB6Q\'s SEQS_TABLE + slots rule (coordinator 2026-11-01)\n'
           '                NB5L.seqs_for(NB5_BOOT_KV, tr["gib"]) if NB5_SEQS_RULE == "table" else None)\n'
           '        E28_RUNG["seqs"] = seqs\n')
_W9B_OLD = '        trec = {"name": name, "pin_gib": tr["gib"], "kv_bytes": tr["kv_bytes"], "max_num_seqs_launch": seqs, "levers": dict(levers)}\n'
_W9B_NEW = _W9B_OLD + '        trec["e28_rung"] = E28_RUNG                   # E28 W4 W9: this try\'s rung (+ its workers, set by nb5_capacity W7)\n'
# W23 (engine): K100Q fix D (build_e26_k100_quality.py K8b, verbatim) + the E28 I8 state. NB6Q's e21_relaunch booted
# NB5_KV_BYTES[kv] (the RC2 pin, ~149 blocks) with the boot's launch extra (E28: seqs 12) and the boot's workers.
_W23A_OLD = 'NB5_PIN = {"start_gib": None, "levers": None, "patch_60490": None}'
_W23A_NEW = ('NB5_ACTIVE_BOOT = {}   # E26 K100 (E28 W4 W23): the booted try (kv_bytes, extra = seqs + captures, levers, name); e21_relaunch re-uses it\n'
             'NB5_PIN = {"start_gib": None, "levers": None, "patch_60490": None}')
_W23B_OLD = "        E11_SERVER, SGL_URL = s, s.url\n        trec[\"ready_sec\"] = round(time.perf_counter() - NOTEBOOK_T0)\n"
_W23B_NEW = (_W23B_OLD +
             "        NB5_ACTIVE_BOOT.clear()                               # E26 K100: this try booted: its exact config for a relaunch\n"
             "        NB5_ACTIVE_BOOT.update(name=name, kv=NB5_BOOT_KV, kv_bytes=int(kv_bytes), extra=list(NB5_BOOT_EXTRA),\n"
             "                               levers=dict(NB5_BOOT_LEVERS), max_num_seqs=trec.get(\"max_num_seqs_launch\"),\n"
             "                               pin_gib=trec.get(\"pin_gib\"), env_keep={k_: v_ for k_, v_ in (getattr(s, \"env\", None) or {}).items()\n"
             "                                                                       if k_ == \"NB4_PPDEFER\"})\n"
             "        NB5_ACTIVE_BOOT.update(build_env=dict(E28_BUILD_ENV), rung=trec.get(\"e28_rung\"))   # E28 W4 W23: + the I8 state\n")
_W23C_OLD = "        srv = merlin_fn_server(old.name, NB5_BOOT_KV, old.port, NB5_KV_BYTES[NB5_BOOT_KV])\n"
_W23C_NEW = ("        global NB5_BOOT_EXTRA, NB5_BOOT_LEVERS\n"
             "        _ab = dict(NB5_ACTIVE_BOOT)                       # E26 K100: the exact booted config (kv bytes, seqs, captures, levers)\n"
             "        if _ab:\n"
             "            NB5_BOOT_EXTRA, NB5_BOOT_LEVERS = list(_ab[\"extra\"]), dict(_ab[\"levers\"])\n"
             "            E28_BUILD_ENV.clear()                         # E28 W4 W23: the booted try's I8 state (DRAFT_HEAD only if it used it)\n"
             "            E28_BUILD_ENV.update(_ab.get(\"build_env\") or {})\n"
             "        _kvb = int(_ab[\"kv_bytes\"]) if _ab else NB5_KV_BYTES[NB5_BOOT_KV]\n"
             "        srv = merlin_fn_server(old.name, _ab.get(\"kv\") or NB5_BOOT_KV, old.port, _kvb)\n"
             "        srv.env.update(_ab.get(\"env_keep\") or {})           # E26 K100 Q-defer: the relaunch keeps NB4_PPDEFER\n"
             "        NB5_REPORT.setdefault(\"relaunch_boot\", []).append({\"kv_bytes\": _kvb, \"max_num_seqs\": _ab.get(\"max_num_seqs\"),\n"
             "                                                       \"pin_gib\": _ab.get(\"pin_gib\"), \"from\": _ab.get(\"name\"),\n"
             "                                                       \"draft_head\": E28_BUILD_ENV.get(\"DRAFT_HEAD\"),   # E28 W4 W23\n"
             "                                                       \"c12\": (_ab.get(\"rung\") or {}).get(\"c12\"),\n"
             "                                                       \"workers\": (globals().get(\"NB5_CAPACITY\") or {}).get(\"workers\")})\n")
# W21 (engine, nb5_boot_try body): the I8 gate right after the C-11 read-back, before the probes (lead 2026-11-01)
_W21_OLD = '        ok, pr = nb5_probes(s)\n        trec["probes"] = pr\n'
_W21_NEW = ('        if (E28_BUILD_ENV.get("DRAFT_HEAD") == "int8"                     # E28 W4 W21: the I8 gate (NB4T read-back ENGAGED)\n'
            '                and (not FAKE or os.environ.get("E28_I8_GATE_IN_FAKE") == "1")):   # (FAKE: the kit e2e only)\n'
            '            _i8ok, trec["i8_gate"] = e28_i8_gate(s.log() or "", "int8")\n'
            '            if not _i8ok:                                             # stop this server cleanly (with_servers kills it);\n'
            '                trec["status"] = "I8_READBACK_UNVERIFIED"             # nb5_boot retries the SAME pin with the BF16 head\n'
            '                log("!!!!!!!!!! E28 W4 I8 gate:", json.dumps(trec["i8_gate"]), "-> I8_READBACK_UNVERIFIED: the same pin, BF16 head")\n'
            '                return {}\n') + _W21_OLD
# W22 (engine, nb5_boot): I8_READBACK_UNVERIFIED -> drop DRAFT_HEAD, the same rung again (only a failed BF16 retry steps down)
_W22_OLD = '        if not fail["failed"] or not P.get("pin_ladder") or ABORT.get("why") or (P.get("e18_ladder") and len(brec["tries"]) >= 3):\n'
_W22_NEW = ('        if (trec.get("status") == "I8_READBACK_UNVERIFIED" and E28_BUILD_ENV.get("DRAFT_HEAD") and not ABORT.get("why")\n'
            '                and not (P.get("e18_ladder") and len(brec["tries"]) >= 3)):   # E28 W4 W22: the same pin, BF16 head\n'
            '            brec.setdefault("i8_fallback", []).append({"try": name, "pin_gib": tr["gib"], "kv_bytes": tr["kv_bytes"],\n'
            '                                                       "seqs": seqs, "i8_gate": trec.get("i8_gate")})\n'
            '            E28_BUILD_ENV.pop("DRAFT_HEAD", None)                        # every later launch (relaunch too) = BF16 head\n'
            '            log("E28 W4: I8 unverified at pin", tr["gib"], "-> the same pin with the BF16 draft head")\n'
            '            continue                                                     # the same rung (i unchanged)\n') + _W22_OLD
# W20 (engine, nb5_boot): a BF16-head retry gets its own launch name (its own server log; the I8 try's log is kept)
_W20_OLD = ('        name = (tag if not P.get("pin_ladder") else f"{tag}_pin{int(round(tr[\'gib\'] * 100))}" + ("" if levers or not NB5_LEVER_ENV else "_nolever"))\n')
_W20_NEW = _W20_OLD + ('        if (P.get("build_env") or {}).get("DRAFT_HEAD") and not E28_BUILD_ENV.get("DRAFT_HEAD"):   # E28 W4 W20\n'
                       '            name += "_bf16head"\n')
_W10A_OLD = '        nb6q_readback(trec, s, "after_warmup")   # NB6Q Q2: Q-defer read-back after the warm-up (record only)\n'
_W10A_NEW = _W10A_OLD + '        e28_readback(trec, s, "after_warmup")    # E28 W4 W10: I8 / C12 / 131072 / text-only + pool / F (record only)\n'
_W10B_OLD = '    nb6q_readback(trec, srv, "final")           # NB6Q Q3: Q-defer read-back, whole log after the server stopped (record only)\n'
_W10B_NEW = _W10B_OLD + '    e28_readback(trec, srv, "final")            # E28 W4 W10: the same on the whole log (record only)\n'
# ------------------------------------------------------------------ CELL 6/7
_W11_OLD = "    E12_ORDER = sorted(TASK_IDS, key=lambda t: (-_e12_order2_score(t), _e12_total_cells(t), _E12_INDEX[t]))\n"
_W11_NEW = ("    E12_ORDER = (list(TASK_IDS) if P.get(\"no_order2\") else      # E28 W4 W11: no ORDER2 (user 2026-11-01): plain task order\n"
            "                 sorted(TASK_IDS, key=lambda t: (-_e12_order2_score(t), _e12_total_cells(t), _E12_INDEX[t])))\n")
# ------------------------------------------------------------------ CELL 8
_W13_OLD = "    _groups = []\n"
_W13_NEW = ("    if P.get(\"fixed_window_sec\"):    # E28 W4 W13 (= E26 K100Q K9): the fixed solve window (one phase; shortened, never skipped)\n"
            "        NB5_EQ.update(window_sec=float(P[\"fixed_window_sec\"]), plan={\"window_sec\": float(P[\"fixed_window_sec\"]), \"fixed\": True,\n"
            "                      \"source\": \"E28 W4: the 6-h EVAL120 solve window (user 2026-11-01)\"})\n"
            "    NB5_REPORT[\"e28_sched\"] = globals().get(\"E28_SCHED_REC\")   # E28 W4 W13: the scheduler patch state (CELL 3)\n"
            "    _groups = []\n")
# ------------------------------------------------------------------ CELL 9
_W14_OLD = '    nb5_write_report()\n    print("NB5 REPORT:", json.dumps({"throughput": NB5_REPORT["throughput"], "phases": {n: {k: p.get(k) for k in (\n'
_W14_NEW = ('    try:                                       # E28 W4 W14: the E28 block (I8 / C12 / 131072 read-back, pool / F, scheduler)\n'
            '        NB5_REPORT["e28_summary"] = e28_summary()\n'
            '        print("E28 W4 SUMMARY:", json.dumps(NB5_REPORT["e28_summary"], default=str)[:4000], flush=True)\n'
            '        NB5_REPORT["e28_sched_report"] = e28_sched_report(BASE_OUT / "e28_sched_report.json", BASE_OUT / "phase_EVAL" / "nb5_sampler.jsonl")\n'
            '        print("E28 W4 SCHED REPORT:", json.dumps(NB5_REPORT["e28_sched_report"], default=str)[:6000], flush=True)\n'
            '    except Exception as _e:\n'
            '        print("E28 W4 summary error:", repr(_e)[:300], flush=True)\n') + _W14_OLD
# ------------------------------------------------------------------ markdown
_W15_OLD = "# E25 NB6Q SUBMIT · FN-Duck production submission + Q-defer · NB6COMMIT\n"
_W15_NEW = ("# E28 W4 EVAL120 6 h · Flash-Next W4 + every tested speed lever · E28EVAL120\n\n"
            "- **E28 W4 = NB6Q SUBMIT (sha 499a4319) + the E28 levers** (user decisions 2026-11-01, Exp 28 README): T1 = 104,000 per\n"
            "  attempt, B2 (cap-cut) re-runs 120,000, server `--max-model-len 131072` (NB6Q); C12 (`--max-num-seqs 12`, 12 workers) on\n"
            "  the I8 start pin and its same-pin BF16 retry only (lower rungs: NB6Q's seqs table + slots worker rule); I8\n"
            "  (`DRAFT_HEAD=int8`, pin ladder from RC2 + 0.75 GiB = 3,285,145,068 B, an I8 boot gate with a same-pin BF16-head retry);\n"
            "  `--language-model-only`; slots admission; request timeouts 7,200 s (T1) / 9,000 s (B2); no ORDER2; the A/B/C\n"
            "  scheduler (`E28_SCHED=1`) in the commit and the hidden rerun. Commit = E28EVAL120: all 120 eval tasks, a fixed 6-h solve\n"
            "  window, HARD_STOP 6 h 50 min, deadman 6 h 55 min. Hidden rerun = RERUN_PROD (every test task, 12 h, identity fallback).\n"
            "\n# (base) E25 NB6Q SUBMIT · FN-Duck production submission + Q-defer\n")


# ------------------------------------------------------------------ size (the Kaggle commit limit ~407 kB: keep < 400 kB)
_ZB64 = re.compile(r'__import__\("zlib"\)\.decompress\(__import__\("base64"\)\.b64decode\("([A-Za-z0-9+/=]+)"\)\)')


def w16_repack(t, label, expect):
    """W16: NB5's packed() literals (zlib + base64; nb5_lib.py in CELL 1, the e23 lab in the engine) re-encoded as lzma +
    base85. The decoded text is byte-identical (NB5's own sha256 check in the literal is unchanged and still runs)."""
    ms = list(_ZB64.finditer(t))
    assert len(ms) == expect, (label, len(ms))
    for m in reversed(ms):
        import zlib
        raw = zlib.decompress(base64.b64decode(m.group(1))).decode("utf-8")
        new = '__import__("lzma").decompress(__import__("base64").b85decode("' + b85x(raw) + '"))'
        assert lzma.decompress(base64.b85decode(b85x(raw))).decode("utf-8") == raw
        t = t[:m.start()] + new + t[m.end():]
    return t


def w17_pack_cell7(t):
    """W17: e21_cell7 (NB6Q's CELL 6/7 report function, E15B CELL 7 as wrapped by E21; no E28 edit inside) packed verbatim
    (lzma + base85, sha256 checked) and defined at the same place by exec into the notebook namespace (E27 NBDF technique)."""
    tree = ast.parse(t)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "e21_cell7"]
    assert len(fn) == 1 and fn[0] is tree.body[-1], "e21_cell7 must be the last statement of CELL 6/7"
    lines = t.splitlines(True)
    start = sum(len(l) for l in lines[:fn[0].lineno - 1])
    seg = t[start:]
    assert seg.startswith("def e21_cell7():\n")
    loader = ('# ---- E28 W4 W17 (the Kaggle 400-kB commit limit): e21_cell7 = NB6Q\'s text verbatim (E15B CELL 7 as wrapped by E21),\n'
              '# packed (lzma + base85, sha256 checked) and defined here by exec into the notebook namespace (E27 NBDF technique)\n'
              'E28_C7_SHA = "' + sha(seg) + '"\n'
              '_e28_c7 = __import__("lzma").decompress(__import__("base64").b85decode("' + b85x(seg) + '")).decode("utf-8")\n'
              'if hashlib.sha256(_e28_c7.encode("utf-8")).hexdigest() != E28_C7_SHA:\n'
              '    raise RuntimeError("E28_C7_EMBED_SHA_MISMATCH: e21_cell7")\n'
              'exec(compile(_e28_c7, "e21_cell7 (NB6Q verbatim, packed)", "exec"), globals())\n')
    return t[:start] + loader, seg


_ENG_A = "\n# ---- RC2 CELL 1 input block, wrapped\n"
_ENG_B = "# ==================================================================== NB5 glue (new)"


def w18_pack_engine(t, ref):
    """W18: NB6Q's engine-cell text from 'RC2 CELL 1 input block' to the NB5 glue (nb5_rc2_inputs, nb5_rc2_prompts, the E23
    step runner, RC2's checks, the E21 donors; no E28 edit inside, asserted = NB6Q's bytes) packed verbatim and exec'd at
    the same place into the notebook namespace (E27 NBDF technique)."""
    a, b = t.index(_ENG_A) + 1, t.index(_ENG_B)
    seg = t[a:b]
    ra, rb = ref.index(_ENG_A) + 1, ref.index(_ENG_B)
    assert seg == ref[ra:rb], "W18: the packed engine region must be NB6Q's text, byte for byte"
    loader = ('# ---- E28 W4 W18 (the Kaggle 400-kB commit limit): NB6Q\'s engine-cell text from "RC2 CELL 1 input block" to the NB5\n'
              '# glue (nb5_rc2_inputs, nb5_rc2_prompts, the E23 step runner, RC2\'s checks, the E21 donors), verbatim, packed (lzma +\n'
              '# base85, sha256 checked) and exec\'d here into the notebook namespace (E27 NBDF technique). No E28 edit inside.\n'
              'E28_ENG_SHA = "' + sha(seg) + '"\n'
              '_e28_eng = __import__("lzma").decompress(__import__("base64").b85decode("' + b85x(seg) + '")).decode("utf-8")\n'
              'if hashlib.sha256(_e28_eng.encode("utf-8")).hexdigest() != E28_ENG_SHA:\n'
              '    raise RuntimeError("E28_ENG_EMBED_SHA_MISMATCH: engine donors")\n'
              'exec(compile(_e28_eng, "engine donors (NB6Q verbatim, packed)", "exec"), globals())\n\n\n')
    return t[:a] + loader + t[b:], seg


# ------------------------------------------------------------------ the scheduler module (Build/E28_SCHED_API.md)
DUCK_LOCAL = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/src/duck_b/arc2_duck.py")
SCHED_MOD = None
SCHED_WIRED = False
if SCHED_SRC is not None:
    import types as _types
    SCHED_MOD = _types.ModuleType("e28_sched_patch_build")
    exec(compile(SCHED_SRC, "e28_sched_patch.py", "exec"), SCHED_MOD.__dict__)
    _duck = DUCK_LOCAL.read_bytes()
    assert sha(_duck) == SCHED_MOD.DUCK_PRE == "b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128", "arc2_duck.py pre"
    _rec = SCHED_MOD.apply_e28_sched(_duck.decode("utf-8"))
    assert _rec["ok"] and _rec["applied"] == "applied" and sha(_rec["text"]) == SCHED_MOD.DUCK_POST, "sched apply (text)"
    assert SCHED_MOD.revert_e28_sched(_rec["text"])["text"].encode("utf-8") == _duck, "sched revert (text)"
    compile(_rec["text"], "arc2_duck.py (E28 SCHED)", "exec")
    assert SCHED_MOD.server_env() == {"E28_SCHED": "1", "E28_T1": str(T1), "E28_B2_TOKENS": str(B2_TOKENS),
                                      "E28_B2_WINDOW": str(B2_WINDOW)}, SCHED_MOD.server_env()
    SCHED_EDIT_LABELS = [e[0].split()[0] for e in SCHED_MOD.NOTEBOOK_EDITS]
    assert SCHED_EDIT_LABELS[:3] == ["N1", "N2", "N3"] and set(SCHED_EDIT_LABELS) <= {"N1", "N2", "N3", "N4"}, SCHED_EDIT_LABELS
    assert {"apply_e28_sched", "verify_e28_sched", "revert_e28_sched", "tree_state", "server_env"} <= set(vars(SCHED_MOD))
    SCHED_PATCHED_SHA = sha(_rec["text"])
    SCHED_WIRED = True

_SCHED_CELL3_OLD = 'assert not _bad, ("E12 package hash mismatch", _bad)\n'
_SCHED_CELL3_NEW = _SCHED_CELL3_OLD + r'''# ---- E28 W4 SCHED (Build/E28_SCHED_API.md section 2): the A -> B -> C scheduler patch on the extracted arc2_duck.py, right
# after the manifest check (it verified the ORIGINAL payload) and before `import arc2_duck`: e28_sched_patch.apply_e28_sched
# (pre sha, atomic write, post sha + marker + py_compile), then E28_SCHED=1 / E28_T1 / E28_B2_TOKENS / E28_B2_WINDOW in THIS
# process (Duck reads them at call time; nothing goes to the server env). Any failure: arc2_duck.py back to the payload's
# original bytes (manifest sha re-checked), a loud log line, E28_SCHED unset; then the COMMIT stops here (NO_GO_E28_SCHED, before
# any boot: a commit without A / B / C would not measure it; = the K100 notebook) while the hidden RERUN falls back to NB6Q's
# scheduler (two attempts per task, plain order: W11) so the submission is still written. Recorded in E28_SCHED_REC /
# nb5_report.json. B2 request timeout: E28_B2_REQUEST_TIMEOUT = the profile's 9,000 s (user 2026-11-01). The B2 window = NB5_WINDOW.
# A profile without e28_sched touches nothing.
E28_SCHED_REC = {"state": "NOT_REQUESTED", "requested": bool(P.get("e28_sched"))}
for _k in ("E28_SCHED", "E28_T1", "E28_B2_TOKENS", "E28_B2_WINDOW", "E28_B2_REQUEST_TIMEOUT"):
    os.environ.pop(_k, None)                     # only this notebook's own decision below turns the scheduler on
if P.get("e28_sched"):
    try:
        if "e28_sched_patch.py" not in (globals().get("E28_SRC") or {}):
            raise RuntimeError("E28_SCHED_NOT_EMBEDDED: " + str(globals().get("E28_LOAD_ERROR")))
        E28S = __import__("types").ModuleType("e28_sched_patch")
        exec(compile(E28_SRC["e28_sched_patch.py"], "e28_sched_patch.py", "exec"), E28S.__dict__)
        E28_SCHED_REC.update(E28S.apply_e28_sched(E12_DIR))
        _ok, _why = E28S.verify_e28_sched(E12_DIR)
        if not _ok:
            raise RuntimeError("E28_SCHED_NOT_VERIFIED: " + _why)
        _e28_env = dict(E28S.server_env(), E28_T1=str(NB5_ATTEMPT["b0"]), E28_B2_TOKENS=str(E28_B2_ATTEMPT),
                        E28_B2_WINDOW=str(NB5_WINDOW))
        if P.get("b2_request_timeout"):
            _e28_env["E28_B2_REQUEST_TIMEOUT"] = str(int(P["b2_request_timeout"]))
        if not E11_HARNESS and {k: v for k, v in _e28_env.items() if k != "E28_B2_REQUEST_TIMEOUT"} != E28S.server_env():
            raise RuntimeError("E28_SCHED_ENV_MISMATCH: " + json.dumps({"notebook": _e28_env, "module": E28S.server_env()}))
        os.environ.update(_e28_env)
        E28_SCHED_REC.update(state="APPLIED", verify=_why, env=_e28_env, sched_sha=E28S.E28_SCHED_SHA,
                             module_sha256=E28_MOD_SHAS.get("e28_sched_patch.py"))
        print("E28 W4 SCHED: A -> B -> C scheduler APPLIED |", _why, "| env", json.dumps(_e28_env), flush=True)
    except Exception as _e:
        E28_SCHED_REC.update(state="FAILED", error=f"{type(_e).__name__}: {_e}"[:800])
        for _k in ("E28_SCHED", "E28_T1", "E28_B2_TOKENS", "E28_B2_WINDOW", "E28_B2_REQUEST_TIMEOUT"):
            os.environ.pop(_k, None)
        try:                                     # the payload's original arc2_duck.py, manifest sha re-checked
            with zipfile.ZipFile(_io.BytesIO(_E12_ZIP_BYTES)) as _z:
                _z.extract("arc2_duck.py", E12_DIR)
            for _pyc in (E12_DIR / "__pycache__").glob("arc2_duck.*.pyc"):
                _pyc.unlink()
            E28_SCHED_REC["restored"] = hashlib.sha256((E12_DIR / "arc2_duck.py").read_bytes()).hexdigest() == _E12_MANIFEST["arc2_duck.py"]
        except Exception as _e2:
            E28_SCHED_REC["restored"] = False
            E28_SCHED_REC["restore_error"] = repr(_e2)[:300]
        assert E28_SCHED_REC["restored"] is True, ("E28 SCHED: arc2_duck.py could not be restored", E28_SCHED_REC)
        if not RERUN:                            # the commit: no run without the scheduler (lead 2026-11-01, = the K100 notebook)
            print("!!!!!!!!!! E28 W4 SCHED FAILED: " + E28_SCHED_REC["error"] + " -> NO_GO_E28_SCHED (commit stops before any boot; "
                  "CELL 0's fallback submission stays) !!!!!!!!!!", flush=True)
            raise RuntimeError("NO_GO_E28_SCHED: " + E28_SCHED_REC["error"])
        print("!!!!!!!!!! E28 W4 SCHED FAILED: " + E28_SCHED_REC["error"] + " -> hidden rerun: NB6Q's scheduler (original "
              "arc2_duck.py restored) !!!!!!!!!!", flush=True)
'''
# W19: N1 / N2 cap the B2 attempt at the SERVER's max-model-len (262,144); A / B1 / C keep MAX_LEN = 131,072 (NB6Q's values)
_W19A_OLD = "    return int(min(MAX_LEN, pk)) if pk else MAX_LEN\n"
_W19A_NEW = ('    return (int(min(E28_SERVER_MAX_LEN if (_e28p and _e28p.get("phase") == "B2") else MAX_LEN, pk)) if pk   # E28 W4 W19:\n'
             '            else MAX_LEN)                                     # B2 = the server\'s 262,144; A / B1 / C keep 131,072\n')
_W19B_OLD = '    need = (int(NB5L.cdiv(min(MAX_LEN, NB5_PROMPT_ALLOW + int(_e28p["budget"]) + NB5_TOOL_ALLOW), NB5_BLOCKS["block"])\n'
_W19B_NEW = ('    need = (int(NB5L.cdiv(min(E28_SERVER_MAX_LEN if _e28p.get("phase") == "B2" else MAX_LEN,   # E28 W4 W19: B2 cap\n'
             '                              NB5_PROMPT_ALLOW + int(_e28p["budget"]) + NB5_TOOL_ALLOW), NB5_BLOCKS["block"])\n')
_W11B_OLD = '    E12_ORDER = (list(TASK_IDS) if os.environ.get("E28_SCHED", "").strip() == "1"   # E28 SCHED N3: no ORDER2\n'
_W11B_NEW = ('    E12_ORDER = (list(TASK_IDS) if (P.get("no_order2") or os.environ.get("E28_SCHED", "").strip() == "1")   # E28 SCHED N3 +\n'
             '                 # E28 W4 W11: no ORDER2 (user 2026-11-01), also when the scheduler did not load: plain task order\n')


def sched_edits(cl):
    """SCHED integration (Build/E28_SCHED_API.md): CELL 3 = the module's N1 + N2 + N4 and the apply block after the
    manifest check; CELL 6/7 = the module's N3 (+ W11: the profile's no_order2 also counts); else NB6Q + W11 only."""
    if not SCHED_WIRED:
        cl[6][1] = once(cl[6][1], _W11_OLD, _W11_NEW, "E28 W4 W11")
        return cl
    t3 = cl[3][1]
    _c3 = [x for x in SCHED_EDIT_LABELS if x != "N3"]          # N1 / N2 (+ N4 when the module offers it) are CELL 3 text
    t3, done = SCHED_MOD.apply_notebook_edits(t3, labels=_c3)
    assert len(done) == len(_c3), done
    # (W19, the 262,144 B2 caps, removed with the user's 131,072 server of 2026-11-01: N1 / N2 / N4 cap every attempt at MAX_LEN)
    t3 = once(t3, _SCHED_CELL3_OLD, _SCHED_CELL3_NEW, "E28 W4 SCHED apply")
    assert t3.index("E28 W4 SCHED (Build/E28_SCHED_API.md") < t3.index("import arc2_duck as E12A")
    cl[3][1] = t3
    t6, done = SCHED_MOD.apply_notebook_edits(cl[6][1], labels=["N3"])
    assert len(done) == 1, done
    cl[6][1] = once(t6, _W11B_OLD, _W11B_NEW, "E28 W4 W11 (on N3)")
    return cl


def pack():
    mods = {"e28w4_lib.py": LIB_SRC}
    if SCHED_WIRED:
        mods["e28_sched_patch.py"] = SCHED_SRC
    blob = json.dumps(mods, sort_keys=True)
    b85 = b85x(blob)
    assert json.loads(lzma.decompress(base64.b85decode(b85)).decode("utf-8")) == mods
    assert '"' not in b85 and "\\" not in b85
    return mods, b85


def w12(mods, b85):
    shas = {k: sha(v) for k, v in mods.items()}
    return (r'''


# ==================================================================== E28 W4 W12: the E28 modules (packed, sha256 checked)
# At the end of CELL 1: CELL 3 applies the scheduler patch before `import arc2_duck`; the engine cell calls e28_readback.
# e28w4_lib.py (I8 / C12 / 131072 / text-only read-back + boot pool / F, record only)__SCHEDNOTE__, packed verbatim
# (json -> lzma -> base85, the NB6Q Q1 pattern) and exec'd into this namespace. A load failure leaves NB6Q's behaviour and
# is logged; it never stops the run.
E28_MOD_SHAS = __SHAS__
try:
    E28_SRC = json.loads(__import__("lzma").decompress(__import__("base64").b85decode(
        "__B85__")).decode("utf-8"))
    for _e28n, _e28h in E28_MOD_SHAS.items():
        if hashlib.sha256(E28_SRC[_e28n].encode("utf-8")).hexdigest() != _e28h:
            raise RuntimeError("E28_EMBED_SHA_MISMATCH: " + _e28n)
    exec(compile(E28_SRC["e28w4_lib.py"], "e28w4_lib.py", "exec"), globals())
except Exception as _e28_e:
    E28_LOAD_ERROR = repr(_e28_e)[:600]
    print("!!!!!!!!!! E28 W4 LOAD FAILED: " + E28_LOAD_ERROR + " -> no E28 read-back !!!!!!!!!!", flush=True)

    def e28_readback(trec, s, stage):
        trec.setdefault("e28_readback", {})[stage] = {"error": "E28_LOAD_FAILED"}

    def e28_summary():
        return {"error": "E28_LOAD_FAILED", "why": E28_LOAD_ERROR}

    def e28_sched_report(path=None, sampler_path=None):
        return {"error": "E28_LOAD_FAILED", "why": E28_LOAD_ERROR}

    def e28_i8_gate(log_text, want="int8"):
        return False, {"label": "E28_LOAD_FAILED", "verdict": "I8_READBACK_UNVERIFIED"}
'''.replace("__SHAS__", json.dumps(shas, sort_keys=True)).replace("__B85__", b85)
            .replace("__SCHEDNOTE__", " + e28_sched_patch.py (the A/B/C scheduler)" if "e28_sched_patch.py" in mods else ""))


def cells():
    cl = [list(c) for c in NQ_CELLS]
    assert cl[0][0] == "markdown" and "CELL 1 — NB5 configuration" in cl[2][1] and "CELL 5 — NB5 engine" in cl[5][1]
    assert "CELL 6/7" in cl[6][1] and "CELL 8 — NB5: boots and phases" in cl[7][1] and "CELL 9 — NB5 report" in cl[8][1]
    cl[0][1] = once(cl[0][1], _W15_OLD, _W15_NEW, "W15 markdown")
    t = cl[2][1]
    for o, n, lab in ((_W1_OLD, _W1_NEW, "W1"), (_W2_OLD, _W2_NEW, "W2"), (_W3_OLD, _W3_NEW, "W3"), (_W3B_OLD, _W3B_NEW, "W3b"),
                      (_W3C_OLD, _W3C_NEW, "W3c"), (_W5_OLD, _W5_NEW, "W5")):
        t = once(t, o, n, "E28 W4 " + lab)
    cl[2][1] = t
    t = cl[5][1]
    for o, n, lab in ((_W6C_OLD, _W6C_NEW, "W6c"), (_W6D_OLD, _W6D_NEW, "W6d"),
                      (_W6E_OLD, _W6E_NEW, "W6e"), (_W7_OLD, _W7_NEW, "W7"), (_W8_OLD, _W8_NEW, "W8"), (_W9_OLD, _W9_NEW, "W9"), (_W9B_OLD, _W9B_NEW, "W9b"),
                      (_W10A_OLD, _W10A_NEW, "W10a"), (_W10B_OLD, _W10B_NEW, "W10b"), (_W21_OLD, _W21_NEW, "W21"),
                      (_W22_OLD, _W22_NEW, "W22"), (_W20_OLD, _W20_NEW, "W20"),
                      (_W23A_OLD, _W23A_NEW, "W23a"), (_W23B_OLD, _W23B_NEW, "W23b"), (_W23C_OLD, _W23C_NEW, "W23c")):
        t = once(t, o, n, "E28 W4 " + lab)
    cl[5][1] = t
    mods, b85 = pack()
    cl[2][1] = cl[2][1].rstrip("\n") + w12(mods, b85)        # W12 at the END of CELL 1: CELL 3 needs the scheduler module
    cl[7][1] = once(cl[7][1], _W13_OLD, _W13_NEW, "E28 W4 W13")
    cl[8][1] = once(cl[8][1], _W14_OLD, _W14_NEW, "E28 W4 W14")
    cl = sched_edits(cl)
    C7["pre_pack"] = [tuple(c) for c in cl]                  # the review view (before the size-only W16 / W17 / W18 packings)
    cl[2][1] = w16_repack(cl[2][1], "W16 CELL 1 nb5_lib", 1)
    cl[5][1] = w16_repack(cl[5][1], "W16 engine e23 lab", 1)
    cl[6][1], C7["seg"] = w17_pack_cell7(cl[6][1])
    cl[5][1], C7["eng"] = w18_pack_engine(cl[5][1], NQ_CELLS[5][1])
    return [tuple(c) for c in cl], mods


C7 = {"seg": None, "eng": None}


def notebook_bytes(cl):
    """NB6's notebook JSON (N6.notebook_bytes, indent 0); compact separators only if that is >= 400 kB (E27 NBDF)."""
    data, n = N6.notebook_bytes(cl)
    if len(data) < 400_000:
        return data, n, "indent0"
    nb = json.loads(data.decode("utf-8"))
    return (json.dumps(nb, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8"), n, "compact"


def kernel_metadata():
    m = json.loads((PUSHQ / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert m["id"] == "foysalemonshanto/q27-xhigh-e25-nb6q-submit" and m["enable_gpu"] is True and m["machine_shape"] == "NvidiaL4"
    m.update(id=f"foysalemonshanto/{SLUG}", title="Q27 Xhigh E28 W4 Eval120 6h", code_file=f"{SLUG}.ipynb")
    return m


def diff_text(cl):
    """The review diff (e28w4_unpack.review_diff): NB6Q -> the E28 W4 cells with the size-only packings undone and every
    packed module literal normalized; the unit tests recompute it from the two notebooks and compare."""
    up, n = U.unpack_cells(cl)
    assert n == {"W17": 1, "W18": 1}, n
    assert [(k, U.normalize(t)) for k, t in up] == [(k, U.normalize(t)) for k, t in C7["pre_pack"]], "unpack != the pre-pack cells"
    return U.review_diff(NQ_CELLS, up)


def build():
    cl, mods = cells()
    # pyflakes over the notebook namespace in execution order: the packed engine region right after the engine cell, the packed
    # e21_cell7 after CELL 6/7, e28w4_lib.py with CELL 1 (the scheduler module runs in its own module namespace: compiled only)
    _chk = []
    for i, c in enumerate(cl):
        _chk.append(c)
        if i == 2:
            _chk.append(("code", mods["e28w4_lib.py"]))
        if i == 5:
            _chk.append(("code", C7["eng"]))
        if i == 6:
            _chk.append(("code", C7["seg"]))
    B.undefined_names_check(_chk)
    if "e28_sched_patch.py" in mods:
        compile(mods["e28_sched_patch.py"], "e28_sched_patch.py", "exec")   # packed code runs in the namespace
    data, n, fmt = notebook_bytes(cl)
    assert b"\r" not in data.replace(b"\\r", b""), "CR in the notebook"
    assert len(data) < 400_000, ("Kaggle commit notebook size limit (keep < 400 kB)", len(data))
    out = OUT
    if out.exists() and sha(out.read_bytes()) in PUSHED:
        out = out.with_name(out.stem + "_REBUILD.ipynb")
        assert not (out.exists() and sha(out.read_bytes()) in PUSHED), out
    out.write_bytes(data)
    EXPORT.write_bytes("\n\n".join(B.norm(t) for k, t in cl if k == "code").encode("utf-8"))
    PUSH.mkdir(parents=True, exist_ok=True)
    (PUSH / f"{SLUG}.ipynb").write_bytes(data)
    meta = kernel_metadata()
    (PUSH / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    PUSH_SAMI.mkdir(parents=True, exist_ok=True)
    (PUSH_SAMI / f"{SLUG}.ipynb").write_bytes(data)
    meta_s = dict(meta, id=f"esfersami50/{SLUG}")
    (PUSH_SAMI / "kernel-metadata.json").write_bytes((json.dumps(meta_s, indent=2) + "\n").encode("utf-8"))
    dt = diff_text(cl)
    diff_cells = [i for i, (x, y) in enumerate(zip(NQ_CELLS, cl)) if x != y]
    facts = {"notebook": str(out), "sha256": sha(data), "bytes": len(data), "json_format": fmt, "cells": n,
             "nb6q_sha256": NB6Q_SHA, "nb6q_bytes": NB6Q_NB.stat().st_size, "changed_cells_vs_nb6q": diff_cells,
             "modules": {k: {"sha256": sha(v), "bytes_lf": len(v.encode("utf-8"))} for k, v in mods.items()},
             "k100q_deadman_code_sha256": K100_DEADMAN_SHA, "sched": dict(SCHED_STATE, wired=SCHED_WIRED, patched_arc2_duck_sha256=globals().get("SCHED_PATCHED_SHA"),
                                                                         module_sha256=SCHED_SHA, api_sha256=SCHED_API_SHA),
             "numbers": {"t1": T1, "b2": B2_TOKENS, "server_max_model_len": SERVER_MAX_LEN, "b2_window": B2_WINDOW, "b2_timeout": B2_TIMEOUT, "c12": C12, "i8_pin_gib": I8_PIN_GIB,
                         "i8_pin_bytes": RC2_FP8_PIN + int(round(I8_PIN_GIB * 2 ** 30)), "budget_min": BUDGET_MIN,
                         "hard_margin_min": HARD_MARGIN_MIN, "hard_stop_min": BUDGET_MIN - HARD_MARGIN_MIN,
                         "deadman_min": BUDGET_MIN - HARD_MARGIN_MIN + DEADMAN_AFTER_HARD_SEC / 60, "window_sec": WINDOW_SEC},
             "diff_lines": {"added": sum(1 for l in dt.splitlines() if l.startswith("+") and not l.startswith("+++")),
                            "removed": sum(1 for l in dt.splitlines() if l.startswith("-") and not l.startswith("---"))},
             "export": str(EXPORT), "export_sha256": sha(EXPORT.read_bytes()), "review_diff": str(DIFF), "review_diff_sha256": sha(dt), "push": str(PUSH), "push_esfersami50": str(PUSH_SAMI), "kernel_metadata": meta,
             "kernel_metadata_esfersami50": meta_s}
    FACTS.write_bytes((json.dumps(facts, indent=1) + "\n").encode("utf-8"))
    DIFF.write_bytes(dt.encode("utf-8"))
    _h = U.hunks(dt)
    assert _h and all(any("E28" in ln for ln in h) for h in _h), [h for h in _h if not any("E28" in ln for ln in h)][:3]
    assert sha(NB6Q_NB.read_bytes()) == NB6Q_SHA, "NB6Q notebook changed during the build"
    print("wrote", out.name, "sha256", sha(data), "|", len(data), "bytes (" + fmt + ") |", n, "cells | diff +%d -%d | sched %s"
          % (facts["diff_lines"]["added"], facts["diff_lines"]["removed"], json.dumps(facts["sched"])))
    return out


if __name__ == "__main__":
    build()
