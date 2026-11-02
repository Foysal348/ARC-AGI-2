# ==================================================================== CELL 0 — mode + fallback submission (stdlib only)
import os, sys, json, time, random, hashlib, re, shutil, subprocess, threading, traceback, zipfile, math
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures as cf

NOTEBOOK_T0 = time.perf_counter()
# S1 (CLOCK): Kaggle's 12 h counts from kernel start; R5's cell 0 began 7 s later. Anchor the clock at the kernel
# process start (capped at 10 min in case the value is odd).
try:
    import psutil as _ps
    _since_kernel_start = max(0.0, min(600.0, time.time() - _ps.Process(os.getpid()).create_time()))
except Exception:
    _since_kernel_start = 0.0
NOTEBOOK_T0 -= _since_kernel_start
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
ARC_ROOT = Path(os.environ.get("Q27X_ARC_ROOT", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"))
if not ARC_ROOT.exists():
    for _alt in ("/kaggle/input/arc-prize-2026-arc-agi-2",):
        if Path(_alt).exists():
            ARC_ROOT = Path(_alt)
WORKING = Path(os.environ.get("Q27X_WORKING", "/kaggle/working")); WORKING.mkdir(parents=True, exist_ok=True)
OUT = WORKING / "q27x"; OUT.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = WORKING / "submission.json"

def _load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

test_challenges = _load_json(ARC_ROOT / "arc-agi_test_challenges.json")

def write_json_atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def identity_submission(challenges):
    return {tid: [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in task["test"]]
            for tid, task in challenges.items()}

write_json_atomic(SUBMISSION_PATH, identity_submission(test_challenges))
print("MODE:", "COMPETITION RERUN (hidden test)" if RERUN else "COMMIT / INTERACTIVE (evaluation)",
      "| fallback submission.json:", len(test_challenges), "tasks | RUN_ID", RUN_ID,
      "| clock anchored", round(_since_kernel_start, 1), "s before cell 0")


# ==================================================================== CELL 1 — NB5 configuration (E25 FN-Duck)
# E15B / E21 profile mechanics; Flash-Next values. Every NB1 / NB2 output that changes the engine is a parameter here.
NB5_DEFAULT_PROFILE = "NB6COMMIT"
PROFILE = "RERUN_PROD" if RERUN else os.environ.get("Q27X_PROFILE", NB5_DEFAULT_PROFILE).strip().upper()   # NB6 E1: the hidden rerun = RERUN_PROD
E11_HARNESS = os.environ.get("E11_HARNESS") == "1"            # local CPU harness only (fake server, no GPU)
_NB5_T = {"time_budget_sec": 710 * 60, "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 2 * 60,
          "verify_margin_sec": 1 * 60, "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60}   # 11:50 hard stop
# SMOKE = real evaluation tasks, the Duck's real multi-turn continuation, exact-match scored (addendum 4): two fast ones
# (S1COMMIT4) + the known-answer 0934a4d8 (RC2's FP8 116k answer was wrong, 15/27 cells; 27B xhigh solves it). Never
# tuned on: it is a check, recorded per KV dtype.
NB5_SMOKE_TASKS = ["2ba387bc", "1818057f", "0934a4d8"]
# paired quality A/B (addendum 5, user-approved; OFF by default; runs only after NB0's FP8 gate = PASS; GPU needs the
# user's go): the 10 tasks the 27B solved on every one of >= 3 of our attempts (Puzzle Capability Ledger,
# ledger_all120.csv) + the known-answer 0934a4d8; the same seeds, budgets, prompts and harness in both configs
# revision 4: SMOKE = a 37-min solve window inside a hard 40-min gate (the paired replay must fit before it); the
# gate decides EVAL / switch to variant A / abort; it never moves the 11:45 solve end or the 11:50 watchdog
NB5_SMOKE_GATE_SEC = int(os.environ.get("NB5_SMOKE_GATE_SEC", 40 * 60)) if E11_HARNESS else 40 * 60
NB5_QUALITY_TASKS = ["1ae2feb7", "38007db0", "3e6067c3", "65b59efc", "d8e07eb2", "1818057f", "409aa875", "5961cc34",
                     "7491f3cf", "b99e7126", "0934a4d8"]
NB5_QUALITY_WINDOW = int(os.environ.get("NB5_QUALITY_WINDOW_SEC", 150 * 60))   # a bound per config (expected 1.3 / 1.8 h)
NB5_NB0_FP8 = os.environ.get("NB5_NB0_FP8_GATE", "").strip().upper()             # NB0's FP8 verdict: PASS | FAIL | ""
# ---- the paired FP8-vs-BF16 KV ARC accuracy test NB5QUALITY_R2 (lead 2026-10-31, user-approved; spec R2): the 15 tasks
# the lead chose from resources/Puzzle Capability Ledger/tokens_to_solve.csv (our 27B Duck + Poetiq runs on the public
# eval set), in this order, with their group labels; ONE notebook, phase Q_FP8_K3 then Q_BF16_K3 (separate boots), the
# same seeds / budgets / prompts / Duck config / task order / capacity levers, equal solve windows inside 3 h
NB5_QAB_GROUPS = {"solid": ["1818057f", "45a5af55", "bf45cf4b", "2ba387bc", "8f215267", "38007db0", "58f5dbd5"],
                  "sometimes": ["247ef758", "f931b4a8", "b5ca7ac4", "65b59efc", "221dfab4"],
                  "never": ["2b83f449", "16b78196", "64efde09"]}
NB5_QAB_TASKS = [t for g in ("solid", "sometimes", "never") for t in NB5_QAB_GROUPS[g]]
NB5_QAB_GROUP_OF = {t: g for g, ts in NB5_QAB_GROUPS.items() for t in ts}
# A-9: R2 never requires NB5_NB0_FP8_GATE == PASS; NB0's verdict is baked and recorded, the FP8 run is labelled
NB5_QAB_NB0 = {"verdict": "NO_GO_FP8_TF_DIVERGENCE", "full_prompt_rank1": {"2k": 0.90, "8k": 0.93, "40k": 0.749, "90k": 0.661},
               "source": "E25 Experiment Results/E25_GPU_wave1_NB0_NB1_2026-10-31.md (NB0 esfersami50 sha 37a427a2)"}
NB5_QAB_FP8_LABEL = "UNVALIDATED_FP8_UNDER_TEST"
NB5_CAPS16 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64]                                        # NB1 CAPTURE_HI (FULL decode covers 16 x (1 + K3) = 64)
# phase = [name, tasks ("smoke" | "all"), solve window sec (None = until the hard stop), MTP ("on" | "off"), commit
# variant, KV dtype (None = NB5_KV_DTYPE), options {"early_exit": SMOKE exits as soon as its gates pass, "budget":
# both attempts' generated-token budget for this phase}]
# ---- NB6 E2 (production SUBMISSION, lead 2026-11-01): the levers of NB5QUALITY_R2 (Kaggle-run 2026-11-01, sha 9c71ba89,
# FP8 phase 13 / 15, pool 214, 9 workers): FP8 KV, MTP K3, commit variant A, KV pin ladder (RC2 pin + 1.0 GiB per card =
# 3,553,580,524 B first, then 0.75 / 0.5 / 0), #60490, measured F, slots admission, launch seqs table, CAPS16 captures,
# MERLIN_SHARE_DRAFT_WORKSPACE=1, gate 1.0, 104k per attempt. Attempt policy = E15B SUBMIT (LB 22.92): its
# solve_all_r6 call passes no skip / extra / b1-mode argument -> skip_b1_if_accepted False, extra_attempts False,
# b1_accepted_mode "" (two attempts per task, plain priority). No fixed LOCAL_ANALYZER_SEED (E15B SUBMIT has none).
# Engine attempts = E18 R2's ladder (e18_ladder: <= 3, any failure before solving -> the next pin step, 30 / 15 / 15 min).
NB6_COMMIT_TASKS = ["1818057f", "2ba387bc", "45a5af55"]      # NB5 QUALITY's solid group, solved in both phases
NB6_LEVERS = dict(nb5_gate=1.0, nb5_attempt={"b0": 104000, "b1": 104000},
                  nb5_env={"E12_B_SKIP_B1_IF_ACCEPTED": "0", "E12_B_B1_ACC_MODE": "0", "E12_B_EXTRA_ATTEMPTS": "0"},
                  admit_mode="slots", f_rule="measured", pin_ladder=True, patch_60490=True, seqs_rule="table", caps16=True,
                  lever_env={"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, no_pure_decode=True, continue_after_failed_boot=True,
                  e18_ladder=True,          # NB6 P3: E18 R2 cell 6 L468-507
                  nb5_phases=[["EVAL", "all", None, "on", "A", "fp8"]])
PROFILE_TABLE = {
    "NB5EVAL120": dict(_NB5_T, source="eval", n_tasks=120,
                       nb5_phases=[["SMOKE", "smoke", 37 * 60, "on", "split", None, {"early_exit": True}], ["EVAL", "all", None, "on", "C"]]),
    "NB5EVAL30": dict(_NB5_T, source="eval", n_tasks=30,
                      nb5_phases=[["SMOKE", "smoke", 37 * 60, "on", "split", None, {"early_exit": True}], ["EVAL", "all", None, "on", "C"]]),
    # paired accuracy run (PLAN section 0): the same 30 tasks, MTP on then off, equal windows
    "NB5EVAL30MTP": dict(_NB5_T, source="eval", n_tasks=30,
                         nb5_phases=[["SMOKE", "smoke", 37 * 60, "on", "split", None, {"early_exit": True}], ["EVAL", "all", 280 * 60, "on", "C"],
                                     ["EVALOFF", "all", 280 * 60, "off", "C"]]),
    # the paired quality A/B (addendum 5): (FP8 KV, MTP K3) vs (BF16 KV, MTP K3); streams = the no-preemption ceiling at
    # a 120k worst-case context = pool // (ceil(120000 / B) + F) (RC2 pools 149 / 78, F 6 -> 4 / 2); attempts of
    # 104k (8k prompt + 104k + 8k tools = 120k), gate 1.0 (the ceiling itself), two attempts per task, no stage 3, one
    # fixed per-request seed. MTP off is NOT in this A/B (NB0 measures it by teacher forcing)
    "NB5QUALITY": dict(_NB5_T, source="eval", task_list=NB5_QUALITY_TASKS, nb5_gate=1.0,
                       nb5_attempt={"b0": 104000, "b1": 104000},
                       nb5_env={"E12_B_EXTRA_ATTEMPTS": "0", "LOCAL_ANALYZER_SEED": str(20261005)},
                       nb5_phases=[["Q_FP8_K3", "all", NB5_QUALITY_WINDOW, "on", "C", "fp8"],
                                   ["Q_BF16_K3", "all", NB5_QUALITY_WINDOW, "on", "C", "bf16"]]),
    # the long multi-task SMOKE (PLAN v3.6): diagnostic only, never in a production / EVAL profile; no early exit, no EVAL
    "NB5SMOKEDIAG": dict(_NB5_T, source="eval", n_tasks=120,
                         nb5_phases=[["SMOKE", "smoke", 37 * 60, "on", "split", None, {"early_exit": False}]]),
    # budget sweep (PLAN v3.6, off by default): EVAL30, the same 30 tasks per budget, equal 160-min windows
    "NB5BUDGET": dict(_NB5_T, source="eval", n_tasks=30,
                      nb5_phases=[[f"B{k // 1000:03d}", "all", 160 * 60, "on", "C", None, {"budget": k}]
                                  for k in (60000, 80000, 100000, 112000)]),
    "RERUN_ALL": dict(_NB5_T, source="test", n_tasks=None, nb5_phases=[["EVAL", "all", None, "on", "C"]]),
    # spec R2 (lead 2026-10-31): 3 h hard runtime incl. install and both boots; equal solve windows (fixed when the first
    # phase starts); commit variant A (C is not validated, A-11); no SMOKE (the boot's fail-fast probes stay); 104k per
    # attempt, 2 attempts, no stage 3, one fixed seed; levers a (KV pin ladder + #60490) and b (non-numeric smaller F,
    # measured F), the same in both phases; concurrency (simplified, user decision): workers = max_num_seqs =
    # max(1, floor(pool / (ceil(65536 / B) + F))) of the boot, launch seqs from SEQS_TABLE per pin step, CAPS16 captures
    "NB5QUALITY_R2": dict(_NB5_T, time_budget_sec=180 * 60, source="eval", task_list=NB5_QAB_TASKS, task_order=NB5_QAB_TASKS,
                          nb5_gate=1.0, nb5_attempt={"b0": 104000, "b1": 104000},
                          nb5_env={"E12_B_EXTRA_ATTEMPTS": "0", "LOCAL_ANALYZER_SEED": str(20261005)},
                          quality_ab=True, equal_windows=True, no_pure_decode=True, admit_mode="slots", f_rule="measured",
                          pin_ladder=True, patch_60490=True, seqs_rule="table", caps16=True,
                          lever_env={"MERLIN_SHARE_DRAFT_WORKSPACE": "1"},
                          continue_after_failed_boot=True,
                          nb5_phases=[["Q_FP8_K3", "all", None, "on", "A", "fp8"], ["Q_BF16_K3", "all", None, "on", "A", "bf16"]]),
    # NB6 E2: the hidden rerun (every hidden test task, ORDER2, one EVAL phase); E18 R2 RERUN_ALL budget 12 h - 3 min
    "RERUN_PROD": dict(_NB5_T, time_budget_sec=12.0 * 3600 - 3 * 60, source="test", n_tasks=None, **NB6_LEVERS),
    # NB6 E2: the commit check before Submit: 3 fast eval tasks, the same levers, scored; 50-min budget (hard stop
    # 48 min, solve end 45 min = ~26 min of solving after a ~17-min FP8 boot as in NB5 QUALITY; F8: a boot ready by
    # ~33 min still passes the phase budget check)
    "NB6COMMIT": dict(_NB5_T, time_budget_sec=50 * 60, source="eval", task_list=NB6_COMMIT_TASKS, **NB6_LEVERS),
}
if E11_HARNESS and os.environ.get("NB5H_TASKS"):                 # local harness only: a few tasks, short windows
    PROFILE_TABLE["NB5HARNESS"] = dict(_NB5_T, source="eval", task_list=os.environ["NB5H_TASKS"].split(","),
                                       nb5_phases=json.loads(os.environ["NB5H_PHASES"]))
if PROFILE not in PROFILE_TABLE:
    raise ValueError(f"unknown profile {PROFILE}; choose {list(PROFILE_TABLE)}")
P = PROFILE_TABLE[PROFILE]
if PROFILE == "NB5QUALITY" and NB5_NB0_FP8 != "PASS":
    # addendum 5: only after NB0's FP8 gate; a clear FP8 FAIL skips the A/B (production then uses BF16)
    raise RuntimeError(f"NB5QUALITY needs NB5_NB0_FP8_GATE=PASS (got {NB5_NB0_FP8!r}): A/B skipped")
assert not any(_p.get("parity") for _p in PROFILE_TABLE.values()), "NB5-2: the E12 DFlash parity path is not part of NB5"
NB5_QAB = ({"nb0": NB5_QAB_NB0, "nb0_gate_env": NB5_NB0_FP8 or None, "nb0_gate_required": False, "fp8_label": NB5_QAB_FP8_LABEL,
            "groups": NB5_QAB_GROUPS, "tasks": NB5_QAB_TASKS} if P.get("quality_ab") else None)   # A-9: recorded, never a gate
for _k, _v in (P.get("nb5_env") or {}).items():
    os.environ.setdefault(_k, _v)                             # read by the E15B block below / by Duck at import
SEED = 20261005
TASK_SAMPLE_SEED = 20261005
POETIQ_NUM_EXPERTS = POETIQ_MAX_ITERATIONS = 0                # (printed by the E10 task-order cell; no Poetiq here)

# ---- time (E15B / E21 CELL 1 block, verbatim)
TOTAL_BUDGET_SEC = float(os.environ.get("Q27X_TIME_BUDGET", P["time_budget_sec"]))
# v1.3: deadlines derived in a guaranteed order (v1.2 had DRAIN after HARD_STOP for EVAL12, so the drain never ran)
VERIFY_DEADLINE = NOTEBOOK_T0 + TOTAL_BUDGET_SEC - P.get("verify_margin_sec", 8 * 60)   # reporting only since v1.4
HARD_STOP = NOTEBOOK_T0 + TOTAL_BUDGET_SEC - P.get("hard_margin_sec", 25 * 60)        # abort everything, write outputs
DRAIN_DEADLINE = min(NOTEBOOK_T0 + TOTAL_BUDGET_SEC * P["drain_frac"], HARD_STOP - 8 * 60)    # force-close thinking
ADMIT_DEADLINE = min(NOTEBOOK_T0 + TOTAL_BUDGET_SEC * P["admit_frac"], DRAIN_DEADLINE - 7 * 60)  # no new task after this
if "drain_margin_sec" in P:   # v1.4.4 (RERUN): fixed offsets from the hard stop, so the slots stay full until the drain
    DRAIN_DEADLINE = HARD_STOP - P["drain_margin_sec"]
    ADMIT_DEADLINE = DRAIN_DEADLINE - P["admit_margin_sec"]
assert NOTEBOOK_T0 < ADMIT_DEADLINE < DRAIN_DEADLINE < HARD_STOP < VERIFY_DEADLINE < NOTEBOOK_T0 + TOTAL_BUDGET_SEC

# ---- NB5 engine + workload parameters. Values marked PLACEHOLDER must be replaced by NB1 / NB2 measurements.
INP = Path(os.environ.get("E23_INPUT", "/kaggle/input"))
TMP = Path(os.environ.get("E23_TMP", "/tmp"))
NB5_KV_DTYPE = os.environ.get("NB5_KV_DTYPE", "fp8").strip().lower()   # PROVISIONAL: RC2 TF FP8-vs-BF16 under study in NB1
if NB5_NB0_FP8 == "FAIL":
    NB5_KV_DTYPE = "bf16"                                     # addendum 5: NB0 failed FP8 clearly -> BF16
assert NB5_KV_DTYPE in ("fp8", "bf16"), NB5_KV_DTYPE
NB5_KV_BYTES = {"fp8": 2479838700, "bf16": 2464415744}         # RC2 pins (Merlin L4 rule); NB1 may change them
# NB1 capacity table (addendum 2), JSON: {"fp8": {"pool_blocks": n, "f_blocks": n, "kv_bytes": n}, "bf16": {...}};
# the live server values (cache_config_info, the max-concurrency line) override it; then the PLACEHOLDERS below
NB5_NB1_CAPACITY = json.loads(os.environ.get("NB5_NB1_CAPACITY_JSON", "{}") or "{}")
NB5_POOL_PLACEHOLDER = {"fp8": 149, "bf16": 78}               # PLACEHOLDER: RC2 (513,940 / 269,042 tok, block 4096)
NB5_F_PLACEHOLDER = 6                                         # PLACEHOLDER = RC2 measured (149 / 3.92 - 32 = 6.0); NB1 refines
NB5_ADMIT_MODE = os.environ.get("NB5_ADMIT_MODE", P.get("admit_mode", "projected"))  # projected (default) | live | future | slots
# R2 'slots' (lead 2026-10-31, simplified): the engine schedules; the reservation is only a safety cap of `workers`
# attempts in flight (1 slot each); C-7 still denies an attempt whose projected blocks exceed the pool
NB5_F_RULE = P.get("f_rule", "max")                          # 'measured' (R2, lever b): F from the server's own line
NB5_SEQS_RULE = P.get("seqs_rule")                           # R2: launch --max-num-seqs from nb5_lib.SEQS_TABLE[kv][pin]
NB5_LEVER_ENV = dict(P.get("lever_env") or {})               # lever b (R2): non-numeric Merlin env; dropped if it fails
# 'future' = a PLACEHOLDER hook for a later NB1-memory-telemetry mode (nb5_lib.future_admission_blocks returns None):
# it behaves exactly as projected now and is recorded as such
assert NB5_ADMIT_MODE in ("projected", "live", "future", "slots"), NB5_ADMIT_MODE
NB5_THINK_FLOOR = int(os.environ.get("NB5_THINK_FLOOR", 60000)) if E11_HARNESS else 60000   # user floor, every request
NB5_CTX_MARGIN = 2048                                         # PLAN v3.6: safe output = min(cap, 131072 - prompt - 2048)
NB5_GATE = float(P.get("nb5_gate", 0.95))                    # one gate: projected blocks <= 0.95 x pool
_att = P.get("nb5_attempt") or {"b0": 112000, "b1": 100000}
NB5_ATTEMPT = {"b0": int(os.environ.get("NB5_ATTEMPT_B0", _att["b0"])), "b1": int(os.environ.get("NB5_ATTEMPT_B1", _att["b1"]))}
NB5_RESERVE = int(os.environ.get("NB5_RESERVE", 4096))        # the commit turn's reserve (E15B #48)
NB5_CAP = NB5_ATTEMPT["b0"] - NB5_RESERVE                     # the longest single request (uninterrupted thinking)
NB5_WINDOW = 131072 - 512                                     # Duck context window = max-model-len - Duck's safety margin
NB5_PROMPT_ALLOW = NB5_TOOL_ALLOW = 8192                      # projected peak = prompt + budget + tool results
NB5_COMMIT_MODE = os.environ.get("NB5_COMMIT_MODE", "")      # "" = per phase (SMOKE split, EVAL C); else C | A | split
NB5_EXTRA_ARGS = json.loads(os.environ.get("NB5_EXTRA_ARGS_JSON", "[]") or "[]")   # NB1 / NB2 levers (trailing vllm args)
if P.get("caps16"):                                           # trailing args win (NB2 CAPS16 rule: FULL covers seqs x (1 + K))
    NB5_EXTRA_ARGS = NB5_EXTRA_ARGS + ["--compilation-config", json.dumps(
        {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": NB5_CAPS16}, separators=(",", ":"))]
NB5_BASE_ENV = json.loads(os.environ.get("NB5_BASE_ENV_JSON", "{}") or "{}")       # Merlin knobs, e.g. {"DRAFT_HEAD": "int8"}
NB5_BOOT_LEVERS = {}                                          # the lever env of the current boot try (set by nb5_boot)
NB5_MTP_OFF_ARGS = ["--speculative-config", "None"]           # vLLM optional_type: "None" -> no speculation (last wins)
NB5_BOOT_EXTRA = []                                           # set per boot by the runner
NB5_READY_TIMEOUT = 30 * 60                                   # spec section 2: readiness gate <= 30 min per boot
NB5_RELAUNCH_MIN_SEC = 30 * 60                                # a Merlin relaunch needs ~17-20 min before it serves
NB5_RAM_NEED_MIB = (69 + 4) * 1024                            # Merlin's floor 69 GiB + 4 GiB margin
NB5_PORT = int(os.environ.get("NB5_PORT", 8001))                # RC2 FP8 port; the local harness picks a free one
if not E11_HARNESS:                                           # user floor: thinking >= 60k (never below without approval)
    assert min(NB5_ATTEMPT.values()) >= 60000, NB5_ATTEMPT
assert NB5_CAP <= 131072 - 512 and max(NB5_ATTEMPT.values()) <= 120000, NB5_ATTEMPT
SGL_MODEL_PATH = INP / "models/foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1"   # tokenizer
SGL_DRAFT_PATH = Path("/nonexistent/nb5-no-separate-drafter")  # (read by the rev 6 preflight; MTP is built in)
SGL_MAX_RUNNING = 8                                           # Duck env placeholder; the phase sets the real streams
SGL_RELAUNCH_TIMEOUT_SEC = NB5_READY_TIMEOUT
SGL_URL = None
CONFIG = {"PROFILE": PROFILE, "RERUN": RERUN, "profile_row": P}
os.environ["E12_RAW_REASONING_HISTORY"] = "1"                 # E14-P (E15B CELL 1b): Duck keeps the raw reasoning
# ---- E12: Duck REPL agent (Experiment 12, user option A 2026-10-19). Duck's own settings are in arc2_duck.DUCK_ENV
# (= dfranzen milestone-2 cell 4): 12,288 output per request, yield at 2,048 generated tokens, preserved thinking,
# 128 Ki context with drain to 58 Ki. Approved exception to the 60k-per-request floor (thinking accumulates in context).
E12_BRANCHES = int(os.environ.get("E12_BRANCHES", 2))                 # independent branches per task -> attempt_1/2
_E12_REV5 = True                                # rev 5 build: every profile (incl. the hidden rerun) runs the fixes
_E12_REV6 = True                                # rev 6 build (user 2026-10-21): code-first view + coverage scheduler
_E12_REV4 = PROFILE == "EVAL6" or _E12_REV5                            # rev 4 defaults (pre-run capacity check, 2026-10-20)
E12_STREAMS = int(os.environ.get("E12_STREAMS", 16 if _E12_REV4 else 0))   # Duck ARC3_MAX_ACTIVE_STREAMS; 0 = engine concurrency
E12_MAX_GEN_PER_BRANCH = int(os.environ.get("E12_MAX_GEN", NB5_ATTEMPT["b0"])) or None   # NB5: Flash-Next attempt 1   # E15B rev 3f: attempt 1 = 50,000 (attempt 2: E15B block)   # Duck max_generated_tokens_per_game
# rev 4: at most N branches alive (resident contexts <= KV 867k / 128k context = 6; EVAL30: 30 live -> 8.4 M re-prefill),
# branches of a task start together; and up to 60,000 output per request (EVAL30: 29/30 first requests cut at 12,288)
E12_LIVE_BRANCHES = int(os.environ.get("E12_LIVE", 12 if _E12_REV4 else 0)) or None   # 6 x 131k = 786k <= KV 867k; S12 (user): 12 x rev 5 context p50 58k = 697k
# rev 5 B5 (evidence, 2026-10-21): no productive EVAL6 reply exceeded 44,779 tokens; the 7 replies that hit 60,000 made no
# tool call; at 49,152 the input budget is 81,408 (4 of the 6 real-overflow trims would have fitted). 40k would have cut one.
E12_DUCK_MAX_OUTPUT = int(os.environ.get("E12_DUCK_MAX_OUTPUT", (NB5_ATTEMPT["b0"] - NB5_RESERVE)
                                         if _E12_REV4 else 0)) or None   # rev 6: the first-request cap; later caps are per request
# rev 5 B9 (user 2026-10-21): Qwen3.8 model-card thinking temperature 1.0 (Duck milestone 0.7)
E12_DUCK_TEMPERATURE = float(os.environ.get("E12_DUCK_TEMPERATURE", 1.0 if _E12_REV5 else 0)) or None
E12_REASONING_EFFORT = os.environ.get("E12_REASONING_EFFORT", "xhigh").strip().lower()   # NB5: forced xhigh   # "" = model default (xhigh)
if E12_REASONING_EFFORT:   # user 2026-10-21: low-effort variant of rev 5 (Qwen3.8 template: xhigh / medium / low)
    assert E12_REASONING_EFFORT in ("xhigh", "medium", "low"), E12_REASONING_EFFORT
    os.environ["E12_REASONING_EFFORT"] = E12_REASONING_EFFORT
if _E12_REV5:   # B1-B4, B6-B8 in arc2_duck (E12_ARC2_REV5); B3 counts reasoning with the served model's tokenizer
    os.environ["E12_ARC2_REV5"] = "1"
    os.environ["E12_TOKENIZER_JSON"] = str(SGL_MODEL_PATH / "tokenizer.json")
# rev 6 (user 2026-10-21; fix list "Rev 6 gap check", both plan reviews): facts view + trusted helpers + auto-check +
# state-aware resumes (arc2_duck E12_ARC2_REV6); request caps 12,288 -> 24,576 after the first successful Python call,
# 32,768 after two no-call cuts (thinking kept in history, branch allowance >= 60k, soft cap 100k); first exposure: branch 0
# of every task in ORDER2 order, paused at about 28k generated tokens, then stage 2 by evidence (resume / branch 1 with the
# grid view = the representation hedge); 10 workers, context window sized to the KV pool in CELL 6; per-request deadline.
E12_R6 = {"E12_R6_CAP_FIRST": NB5_CAP, "E12_R6_CAP_PROGRESS": NB5_CAP, "E12_R6_CAP_RESCUE": NB5_CAP,   # NB5: no 12k/24k/32k
          "E12_R6_FIRST_EXPOSURE": 0, "E12_R6_B1_VIEW": "facts", "E12_R6_MIN_ADMIT_SEC": 300,   # E15B #3, #1, #49
          "E12_CHECK_VERIFY_CAP": 24, "E12_AUTOCHECK_CAP": 24, "E12_AUTOCHECK_PER_CALL": 3, "E12_DUCK_DRAIN_TOKENS": 16384,
          # rev 6c (user 2026-10-21: 12 streams): the per-stream window = 1.0 x KV / 12 (= 73.4k at 881k). 0.9 would give
          # 66.1k (history 41.5k at the 24,576 cap); D66 ran 12 slots at context 155,648 with KV peak 90 % and 0 preemptions,
          # rev 6e G4-1 (user 2026-10-21): workers stay 12; no new task or branch while KV usage >= 0.95
          "E12_R6_FACTS_MAX_CHARS": 3600, "E12_R6_SHOW_MAX_CELLS": 900, "E12_R6_KV_FRACTION": 1.0,
          "E12_R6_P90_CUTOFF": 0,
          # rev 6b (gap check vs the problem->fix map): facts <= 800 Qwen tokens (Q-5; 120 eval tasks: full p50 1,237 ->
          # bounded p50 736 / p90 837 / max 1,396), fair-share stage 2 (R6-S3) with a 24,576-token minimum chunk
          "E12_R6_FACTS_MAX_TOKENS": 800, "E12_R6_FAIR_SHARE": 0,   # E15B #45: no chunks / pauses
          # rev 6e (user 2026-10-21: "take all", fix list G4-1..G4-18; G4B evidence): KV gate 0.95 polled every 5 s (G4-1);
          # branch-1 first cap 24,576 (G4-2; G4B b1 first calls at 12.4-18.0k); show() up to 900 cells per grid, 1,000 per
          # call (G4-4; 30x30 = 2,033 Qwen tokens of the 3,072 tool cap); provisional nudge (G4-6: 27,000 in rev 6e; 24,000 since rev 6f, H6-1; G4B
          # solved branches defined transform at 18,383 / 26,228); branch-1 indexed view <= 12,000 tokens (G4-11);
          # minimum stage-2 chunk 12,288 (G4-17: every G4B share 3.8-7.5k was below 24,576); cache diagnostic (G4-8)
          "E12_R6_KV_ADMIT": NB5_GATE, "E12_R6_KV_POLL": 5.0, "E12_R6_CAP_FIRST_B1": NB5_CAP, "E12_R6_SHOW_CALL_CELLS": 1000,
          "E12_R6_NUDGE_TOKENS": 24000, "E12_R6_B1_GRID_MAX_TOKENS": 12000, "E12_R6_MIN_CHUNK": NB5_CAP,
          "E12_R6_CACHE_DIAG": 0}   # NB5: the SGLang cache diagnostic is not run on vLLM
# rev 6f (user 2026-10-22, the H6 set of the three GPT rev 6e audits): the provisional nudge at 24,000 tokens, sent in the
# next tool result (H6-1: in G4B no branch 0 started a new turn at >= 27k before the 28k pause); the cache diagnostic
# runs AFTER solving (H6-3); every commit window starts after the preflight (H6-6); one isolation retry (H6-17)
if _E12_REV6:
    os.environ["E12_ARC2_REV6"] = "1"
    for _k, _v in E12_R6.items():
        E12_R6[_k] = type(_v)(os.environ.get(_k, _v))
        os.environ[_k] = str(E12_R6[_k])
# audits (verified): non-streaming read timeout 900 s < a 60k reply at ~61 tok/s per stream (978 s) -> 1,800 s.
# The 100k per-branch cap is soft (checked after each request). max_tokens 60,000 is a ceiling, not a minimum thinking.
# The milestone context (131,072, drain 58 Ki) is kept: input budget 70,560, trim target 35,280. Only the last message is protected:
# a tool-less reply before the resume opener could be dropped (EVAL6: 5 resets; rev 5 B3 counts reasoning correctly).
E12_DUCK_REQUEST_TIMEOUT = int(os.environ.get("E12_DUCK_REQUEST_TIMEOUT", 3600 if _E12_REV4 else 0)) or None   # E15B #47
for _k, _v in (("E12_DUCK_MAX_OUTPUT", E12_DUCK_MAX_OUTPUT), ("E12_DUCK_REQUEST_TIMEOUT", E12_DUCK_REQUEST_TIMEOUT),
               ("E12_DUCK_TEMPERATURE", E12_DUCK_TEMPERATURE)):
    if _v:
        os.environ[_k] = str(_v)                                         # read by arc2_duck (CELL 3 / CELL 6)
# E15B (user 2026-10-24, Exp 15 Notes/02; rev 3f 2026-10-25): config B = 2 independent attempts per task, attempt 1 =
# 50,000 and attempt 2 = 45,000 generated tokens (attempt 1 on every
# task first, then attempt 2 with its own agent and the facts view); no pause / chunk / resume; request cap = remaining
# budget - 4,096 (a final commit turn); fixed 100,000-token Duck window (no trim inside an attempt); 16 real streams
E15B = {"E12_B_MODE": 1, "E12_B_ATTEMPT_TOKENS": E12_MAX_GEN_PER_BRANCH or NB5_ATTEMPT["b0"], "E12_B_RESERVE": NB5_RESERVE,
        "E12_B_WINDOW": NB5_WINDOW, "E12_B_ATTEMPT_TOKENS_B1": NB5_ATTEMPT["b1"],
        "E12_B_SKIP_B1_IF_ACCEPTED": 0, "E12_B_EXTRA_ATTEMPTS": 1, "E12_B_MAX_ATTEMPTS": 8, "E12_B_WEAK_COMMIT": 1, "E12_B_B1_ACC_MODE": 2}
for _k, _v in E15B.items():
    E15B[_k] = int(os.environ.get(_k, _v))
    os.environ[_k] = str(E15B[_k])
assert E15B["E12_B_ATTEMPT_TOKENS"] == E12_MAX_GEN_PER_BRANCH, "the attempt budget and the session cap must agree"
assert 0 < E15B["E12_B_ATTEMPT_TOKENS_B1"] <= E15B["E12_B_ATTEMPT_TOKENS"], "attempt 2's budget (rev 3f)"
E12_SOLVE_WINDOW_SEC = P.get("solve_window_sec")                      # SMOKE1 only
E12_STOP_BEFORE_HARD_SEC = 180   # branches stop; outputs are written after it. rev 3 (GPT re-audit): a submit at the
                                 # deadline can still verify for <= 60 s, then CELL 6/7 write; 60 s left no margin
E12_MAMBA_TRACK_64 = os.environ.get("E12_MAMBA_TRACK_64", "0") == "1"  # Duck's --mamba-track-interval/--page-size 64 (measure)
# rev 2 (GPT audits, validated 2026-10-19): Duck request snapshots (the exact messages sent; large) for the smoke only,
# and a per-sandbox-process address-space limit (Duck sets CPU / file size / open files only)
E12_REQUEST_LOGS = os.environ.get("E12_REQUEST_LOGS", "1" if PROFILE == "SMOKE1" else "0") == "1"
E12_SANDBOX_MEM_MB = int(os.environ.get("E12_SANDBOX_MEM_MB", 4096))
CONFIG["e15b"] = dict(E15B)
CONFIG["e12"] = {k: globals()[k] for k in ("E12_BRANCHES", "E12_STREAMS", "E12_MAX_GEN_PER_BRANCH", "E12_SOLVE_WINDOW_SEC",
                                           "E12_LIVE_BRANCHES", "E12_DUCK_MAX_OUTPUT", "E12_DUCK_REQUEST_TIMEOUT",
                                           "E12_DUCK_TEMPERATURE", "_E12_REV5", "E12_REASONING_EFFORT",
                                           "E12_STOP_BEFORE_HARD_SEC", "E12_MAMBA_TRACK_64", "E12_REQUEST_LOGS",
                                           "E12_SANDBOX_MEM_MB", "_E12_REV6")}
if _E12_REV6:
    CONFIG["e12"]["rev6"] = dict(E12_R6)
CONFIG["e12"]["duck_commit"] = "10882e332d883af3e61616df8c8a17d03a328a75"
print("E12:", json.dumps(CONFIG["e12"]))
for _k in ("E12_B_ATTEMPT_TOKENS", "E12_B_ATTEMPT_TOKENS_B1"):
    assert E11_HARNESS or E15B[_k] >= 60000, (_k, E15B[_k])   # the 60k floor holds after any env override too
E21_VARIANT = "NB5"
E21_SERVED = {"fp8": "q27", "bf16": "q27"}                     # RC2's served model name
E21_VLLM_MAX_MODEL_LEN = 131072
E21_HALF = None
E21_HALF_END_PC = HARD_STOP - E12_STOP_BEFORE_HARD_SEC
E21_STATE = {"halves": {}}
E15B_SOLVE_END_PC = float("inf")


Q27X_PAYLOAD_SLUG = "foysalemonshanto/q27-e15b-payload"      # NB5: the foysalemonshanto copy (esfersami50 second)
Q27X_PAYLOAD_DIRS = ["datasets/foysalemonshanto/q27-e15b-payload", "datasets/esfersami50/q27-e15b-payload"]
Q27X_PAYLOAD_USED = {}                                       # NB5: payload name -> the file that was used


def q27x_payload(name, sha):
    """One file of the payload dataset (Notes/21: Kaggle commit runs of a 745 kB notebook never started, so the big
    payloads live in a dataset), found by its exact name under Q27X_PAYLOAD_DIR or /kaggle/input (at most 3 directory
    levels deep) and verified by sha256 before use."""
    roots = [Path(x) for x in os.environ.get("Q27X_PAYLOAD_DIR", "").split(os.pathsep) if x] + [Path("/kaggle/input")]
    seen = []
    for r in roots:
        for pat in [d_ + "/" + name for d_ in Q27X_PAYLOAD_DIRS] + [name, "*/" + name, "*/*/" + name, "*/*/*/" + name]:   # NB5
            for f in (sorted(r.glob(pat)) if r.is_dir() else []):
                b = f.read_bytes()
                if hashlib.sha256(b).hexdigest() == sha:
                    Q27X_PAYLOAD_USED[name] = {"path": str(f), "sha256": sha, "rejected_before": list(seen)}   # NB5
                    print("payload", name, "<-", f, flush=True)
                    return b
                seen.append(str(f))
    raise FileNotFoundError(f"payload {name} (sha256 {sha[:12]}) not found or wrong under {[str(r) for r in roots]}; "
                            f"seen {seen}; attach the dataset {Q27X_PAYLOAD_SLUG}")


# ---- NB5 library (new code, E25 Build/nb5_lib.py; verbatim)
NB5_LIB_SRC = (lambda _t: _t if __import__("hashlib").sha256(_t.encode()).hexdigest() == "18efcbc1dde9c590a14ac1937b45dec563ff76ed4d92b16632d273dec2c88500" else (_ for _ in ()).throw(RuntimeError("nb5_lib: embedded source sha256 mismatch")))(__import__("zlib").decompress(__import__("base64").b64decode("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")).decode("utf-8"))
(TMP / "nb5_lib.py").write_text(NB5_LIB_SRC, encoding="utf-8")
sys.path.insert(0, str(TMP))
import nb5_lib as NB5L
CONFIG["nb5"] = {"kv_dtype": NB5_KV_DTYPE, "kv_bytes": NB5_KV_BYTES, "nb1_capacity": NB5_NB1_CAPACITY,
                 "pool_placeholder": NB5_POOL_PLACEHOLDER, "f_placeholder": NB5_F_PLACEHOLDER, "gate": NB5_GATE,
                 "attempt": NB5_ATTEMPT, "reserve": NB5_RESERVE, "admit_mode": NB5_ADMIT_MODE, "think_floor": NB5_THINK_FLOOR,
                 "smoke_gate_sec": NB5_SMOKE_GATE_SEC, "cap": NB5_CAP, "window": NB5_WINDOW,
                 "prompt_allow": NB5_PROMPT_ALLOW, "tool_allow": NB5_TOOL_ALLOW, "commit_mode": NB5_COMMIT_MODE,
                 "extra_args": NB5_EXTRA_ARGS, "base_env": NB5_BASE_ENV, "phases": P["nb5_phases"],
                 "smoke_tasks": NB5_SMOKE_TASKS, "lib_sha256": hashlib.sha256(NB5_LIB_SRC.encode()).hexdigest(),
                 "payload_used": Q27X_PAYLOAD_USED,   # filled when CELL 3 loads the Duck package
                 "f_rule": NB5_F_RULE, "seqs_rule": NB5_SEQS_RULE, "lever_env": NB5_LEVER_ENV,
                 "pin_ladder": bool(P.get("pin_ladder")), "patch_60490": bool(P.get("patch_60490")),
                 "equal_windows": bool(P.get("equal_windows")), "quality_ab": NB5_QAB}
print("NB5:", json.dumps(CONFIG["nb5"], default=str), flush=True)


# ==================================================================== CELL 3 — E12: Duck package (verbatim) + ARC-2 adapter
# Tufa Labs' Duck harness as forked by Daniel Franzen (da-fr/arc-agi-3-solution @ 10882e332d, Apache-2.0,
# LICENSE + NOTICE inside). E15B: inference/agent/tool_agent.py (yield after the batch, ARC3_YIELD_AFTER_BATCH) and
# python_tool_sandbox.py (sys shim / time / import messages, ARC2_SANDBOX_SYS_SHIM) carry two switched-off-by-default
# patches (Exp 15 src/duck_b); the other 18 files are byte-identical to that commit (sha256 manifest below);
# arc2_duck.py is the only new code (ARC-2 prompt, prelude with check/submit, host re-verification, driver, selection).
import base64 as _b64, io as _io, logging as _logging
E12_DIR = Path(os.environ.get("E12_DIR", "/tmp/e12_duck"))
_E12_ZIP_BYTES = q27x_payload("q27x_e15b_duck_9e4b89285136.bin", "9e4b89285136616d523028ac042dcca6afe82175081a1481d584c998ab3ee566")   # Notes/21: payload dataset
# E15B #17: the sandbox self-test task (identity on every pair): fixed, so an unusual first hidden task cannot fail it
E15B_SELFTEST_TASK = {"train": [{"input": [[1, 0], [0, 2]], "output": [[1, 0], [0, 2]]},
                                {"input": [[3, 3, 0]], "output": [[3, 3, 0]]}], "test": [{"input": [[4]]}]}
_E12_MANIFEST = {
"arc2_duck.py": "b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128",
"vendor/LICENSE": "3ddf9be5c28fe27dad143a5dc76eea25222ad1dd68934a047064e56ed2fa40c5",
"vendor/NOTICE.txt": "1e2b756570ed5a6d85d06e61e458021c1221a5f72ad33165ea29bb85b9053cf1",
"vendor/inference/__init__.py": "d92fed573b404870f7c6a1cef8f3e5d696195e2886e9e9183485c304fe5090f8",
"vendor/inference/agent/__init__.py": "8c561093771390b6f64fc0e87186982cd04045e8b6a2cf8d7f17e494762b4385",
"vendor/inference/agent/action_names.py": "8dd40e957046977e811eb2c2eb13fce905dbecf77f8dc7f846540f4d48af963d",
"vendor/inference/agent/noop_repeat_guard.py": "e9172d58687e9043c3445f87e11bdf74c4104d8b0bed5fb6b54745a6aa58655a",
"vendor/inference/agent/priority_pace_reference.json": "f29c16e336aef6b0dbf73ec056f554bd5192ac02f8deeda70ad152ef1214d0ab",
"vendor/inference/agent/priority_scheduler.py": "3a13659001458ea1a45ee3a812babe1b0012409eeabb23d681395796941b8e94",
"vendor/inference/agent/prompts.py": "14965a99bc55a61dad59fb79038f56c0ba1d1368350c8487b5a7b9edef0488ac",
"vendor/inference/agent/python_tool_sandbox.py": "d29d807eef107a2d9aa60354dc370782739e2b9604a8247f89279ee10ae0bf3c",
"vendor/inference/agent/runtime_state.py": "b5c55d561a63e2dcf0422ee7fd339923a9fb909c779302bbf02b8ea49ccc607f",
"vendor/inference/agent/tool_agent.py": "f7454201484f2416b9c9a7d3a365bc048901e4a384a00ce1e1a2e6ec458c7b4c",
"vendor/inference/agent/vision_context.py": "51d54a7aa9f1cd8e41b3fcc00ffd1a9cd1ada6fcde261d09faa49561fd843466",
"vendor/inference/utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
"vendor/inference/utils/animation.py": "6c14b613b3a7c66ce3373162d6faa1469a1ddb3ca4ab882299998c5cf14add96",
"vendor/inference/utils/frame_diff.py": "6ac3112773ffa6b1562b1e8f35a8018bd0dec9de7a78b12e7b13c529d4f46af8",
"vendor/inference/utils/grid_utils.py": "1e43809c9a4a463c65f897c48abb8e6cf4b28a32ac4ec32a07e9adcf06ab0c03",
"vendor/inference/utils/openai_compat.py": "fea0a75081372769fd854997b99e7826b334dcc5966876f31f587d19ae961165",
"vendor/inference/utils/retained_functions.py": "0d8b267cb387565c04d4977116020cc04d84ad9d9175d2bf4bc4613703cc5a01",
"vendor/inference/utils/run_artifacts.py": "789d541b395043379c23f5b0b24044e50fadab02e7f86f64cdd54714f4cab339",
"vendor/inference/utils/segmentation.py": "0d19669e5a36351e41a6b94b7c68f5b258829de78441a03a8749687c262b452c",
"vendor/inference/utils/viewer_artifacts.py": "cb5043dab8f5294608cda3675408ced904b849ae4985829b4ffd2d84b8e5d807"
}
if E12_DIR.exists():
    shutil.rmtree(E12_DIR)
with zipfile.ZipFile(_io.BytesIO(_E12_ZIP_BYTES)) as _z:
    _z.extractall(E12_DIR)
_bad = [p for p, h in _E12_MANIFEST.items() if hashlib.sha256((E12_DIR / p).read_bytes()).hexdigest() != h]
assert not _bad, ("E12 package hash mismatch", _bad)
sys.path[:0] = [str(E12_DIR), str(E12_DIR / "vendor")]
import arc2_duck as E12A
# Duck reads its settings at import: set them first. The base URL is the engine cell's (same port); the stream count is
# set again from the engine's concurrency before the gate is first used (read at call time).
os.environ.update(E12A.duck_env(f"http://127.0.0.1:{int(os.environ.get('E11_PORT', 30000))}/v1", "pending", SGL_MAX_RUNNING))
# E21: vLLM 0.19 / 0.32 read only message["reasoning"] of an assistant history message (vllm/entrypoints/chat_utils.py
# 1512 in v0.19.0; vllm/renderers/chat_utils.py 2083 in v0.32.0); "reasoning_content" (DUCK_ENV, read by SGLang) would
# be dropped from the prompt. E12-V run 2 sent "reasoning" and its prompts grew by the generated tokens (6,882 + 12,288
# -> 19,215). Duck reads the key at call time (tool_agent._reasoning_history_keys).
os.environ["ARC3_REASONING_HISTORY_KEY"] = "reasoning"
_E12_TA, _E12_DP, _e12_run_sandboxed, _ = E12A._import_duck()
# E21 rev 2 (VAL_api_D1 F1, verified in the vLLM source): vLLM 0.19 / 0.32 report finish_reason "tool_calls" whenever
# the parser found a call, also for a reply cut at max_tokens (chat_completion/serving.py 0.19 L1550-1560, 0.32
# L1099-1116); SGLang 0.5.21 kept "length", which arc2_duck's cut-call handling (rev 5 B6 / R5-2, the cut-code harvest,
# the R6 "cut" opener) keys on. Restore "length" from the token count (Duck's package files stay byte-identical).
_E21_CC_ORIG = _E12_TA.ToolAgent._chat_completion
E21_LENGTH_RESTORED = {"n": 0}
E21_CALL_WITH_CONTENT = {"n": 0}     # rev 2b (VAL_api_D1 F4): a tool call + visible text


def _e21_chat_completion(self, messages, **kw):
    res = _E21_CC_ORIG(self, messages, **kw)
    try:
        u = res.usage if isinstance(res.usage, dict) else {}
        cap = kw.get("max_tokens")
        cap = self._max_output_tokens if cap is None else cap
        ct, pt = u.get("completion_tokens"), u.get("prompt_tokens")
        _m = res.message if isinstance(res.message, dict) else {}
        if _m.get("tool_calls") and str(_m.get("content") or "").strip():
            E21_CALL_WITH_CONTENT["n"] += 1
        if str(res.finish_reason or "") == "tool_calls" and cap and isinstance(ct, int):
            lim = int(cap) if not isinstance(pt, int) else min(int(cap), E21_VLLM_MAX_MODEL_LEN - int(pt))   # 0.32 clamps
            if ct >= lim:
                res.finish_reason = "length"
                E21_LENGTH_RESTORED["n"] += 1
    except Exception:
        pass
    return res


_E12_TA.ToolAgent._chat_completion = _e21_chat_completion
E12_SANDBOX_LIMIT = E12A.install_sandbox_memory_limit(E12_SANDBOX_MEM_MB)   # rev 2: every sandbox child, before its code


class _E12LevelCountFilter(_logging.Filter):   # Duck's per-game level-count warning has no ARC-2 meaning (priority uses 10)
    def filter(self, record):
        return "total level count unavailable" not in record.getMessage()


_logging.getLogger(_E12_TA.__name__).addFilter(_E12LevelCountFilter())

# sandbox self-test with our own scripted code (the engine cell runs only if it passes)
_t = time.perf_counter()
try:
    _st_task = E15B_SELFTEST_TASK      # E15B #17 (code audit M4): a fixed task, never the first hidden task
    _st = _e12_run_sandboxed(
        code=E12A.task_prelude(_st_task) + "def _e12_ident(g):\n    return g\nres = check(_e12_ident, verbose=False)\nprint(len(res), sum(1 for r in res if 'ok' in r))",
        timeout_seconds=30, initial_state={"current_frame": None, "history": [], "valid_actions": [], "last_action_call_result": {}},
        action_handler=lambda a, stale_after=None: {"action_result": {}, "state": {}}, kept_functions=[], retain_imports=True)
    _n = len(_st_task["train"])
    E10_SANDBOX_SELFTEST = {"ok": (_st.get("stdout") or "").strip() == f"{_n} {_n}" and not _st.get("error"),
                            "stdout": (_st.get("stdout") or "")[:200], "error": (_st.get("error") or "")[:500],
                            "sec": round(time.perf_counter() - _t, 2), "python": sys.version.split()[0]}
except Exception as _e:
    E10_SANDBOX_SELFTEST = {"ok": False, "error": f"{type(_e).__name__}: {_e}"[:500]}
# rev 2: memory limit evidence (reported; the run continues without it rather than not at all). The 5 GiB request is
# made only after the read-back confirmed the limit, so it can never really allocate.
_mem = {"limit": dict(E12_SANDBOX_LIMIT)}
try:
    _st_kw = dict(timeout_seconds=30, initial_state={"current_frame": None, "history": [], "valid_actions": [],
                                                     "last_action_call_result": {}},
                  action_handler=lambda a, stale_after=None: {"action_result": {}, "state": {}}, kept_functions=[],
                  retain_imports=True)
    if E12_SANDBOX_LIMIT.get("readback_ok"):
        _big = _e12_run_sandboxed(code="x = bytearray(5 * 1024 ** 3)\nprint(len(x))", **_st_kw)
        _small = _e12_run_sandboxed(code="x = bytearray(128 * 1024 ** 2)\nprint(len(x))", **_st_kw)
        _mem.update(over_limit_rejected="MemoryError" in (_big.get("error") or ""),
                    small_ok=(_small.get("stdout") or "").strip() == str(128 * 1024 ** 2))
        _mem["ok"] = bool(_mem["over_limit_rejected"] and _mem["small_ok"])
    else:
        _mem["ok"] = False
except Exception as _e:
    _mem.update(ok=False, error=f"{type(_e).__name__}: {_e}"[:300])
_mem["limit"] = dict(E12_SANDBOX_LIMIT)
E10_SANDBOX_SELFTEST["memory_limit"] = _mem
if E12_SANDBOX_MEM_MB > 0 and not _mem.get("ok"):
    print("E12 WARNING: sandbox memory limit NOT verified:", json.dumps(_mem), flush=True)
    if os.name == "posix":   # rev 3 (GPT re-audit): on Kaggle (Linux) never run model code without the limit
        E10_SANDBOX_SELFTEST["ok"] = False
        E10_SANDBOX_SELFTEST["error"] = "memory limit not verified (fail closed): " + json.dumps(_mem)[:300]
        print("E12: solver disabled (fail closed); CELL 0 fallback submission stays", flush=True)
CONFIG["e12"]["sandbox_selftest"] = E10_SANDBOX_SELFTEST
if _E12_REV5:   # rev 5 R5-4: is B3 counting reasoning with the tokenizer (else a fallback ratio)? recorded, never fatal
    CONFIG["e12"]["rev5_tokenizer"] = E12A.tokenizer_status()
    print("E12 rev 5 reasoning tokenizer:", json.dumps(CONFIG["e12"]["rev5_tokenizer"]), flush=True)
print("E12 package OK:", len(_E12_MANIFEST), "files | sandbox self-test:", json.dumps(E10_SANDBOX_SELFTEST), flush=True)


# ==================================================================== NB5 (E25): commit-turn variant C, request trace, admission
# _E21_CC_ORIG is the name E21's wrapper (installed above as ToolAgent._chat_completion) calls: NB5 rebinds it to a
# dispatcher, so E21's finish_reason restore stays around BOTH paths. A commit request (E15B #48: after the commit text)
# goes as variant C (CONFIG nb5 commit mode; 'split' = a stable (task, branch) hash, the paired SMOKE comparison).
_NB5_CC_BASE = _E21_CC_ORIG                      # Duck's own ToolAgent._chat_completion
NB5_GUARD = {"requests": 0, "tokenized": 0, "estimated": 0, "clamps": 0, "clamped_tokens": 0, "floor_events": 0,
             "events": [], "tokenize_retries": 0, "local_tokenizer": 0, "no_request": {}, "estimates": []}
NB5_LOCAL_TOK = {"verified": False, "why": "not loaded yet"}     # A-14: set in CELL 8 from RC2's pinned-hash check
NB5_LOCAL_MARGIN = 256                           # A-14: + tokens on a local count (the server render is the reference)


class NB5NoRequest(Exception):
    """A request that must not be sent (A-13 / A-14 / C-7); carries the attempt verdict."""
    def __init__(self, verdict, info=None):
        super().__init__(verdict)
        self.verdict, self.info = verdict, info or {}


def _nb5_peak_len(self):
    """C-9: the attempt's own projected peak tokens min(131072, prompt_allow + budget + tool_allow) (CELL 5 capacity)."""
    pk = NB5_BLOCKS.get("peak_b0" if int(getattr(self, "arc2_branch", 0) or 0) == 0 else "peak_b1")
    return int(min(MAX_LEN, pk)) if pk else MAX_LEN


def _nb5_guard(self, messages, kw):
    """Revision 3 + A-13 / A-14 / C-9, on EVERY request (first turn and every tool turn): the server's own render length
    (/tokenize, 2 retries; + 2 for variant C's longer tail); if /tokenize still fails, the LOCAL Flash-Next tokenizer +
    template only when their pinned hashes were verified at load (+ NB5_LOCAL_MARGIN), else NO request
    (INCONCLUSIVE_CONTEXT_LENGTH); every estimate is recorded. max_tokens <= the attempt's projected peak (C-9: so the
    reservation is a true upper bound) - prompt - margin; no room -> NO request (CONTEXT_EXHAUSTED, A-13). The 60k
    thinking floor (nb5_lib.context_guard): an attempt that can no longer reach it ends its thinking path through Duck's
    own budget-exhaustion route."""
    ids, how, tries = None, None, []
    payload = NB5L.chat_payload(_E12_TA, self, messages, kw)
    for _k in range(3):                              # A-14: /tokenize + 2 retries
        try:
            ids = NB5L.tokenize_chat(SGL_URL + "/v1", payload, timeout=120)
            how = "tokenize"
            break
        except Exception as _e:
            tries.append(f"{type(_e).__name__}: {_e}"[:160])
            if _k < 2:
                NB5_GUARD["tokenize_retries"] += 1
                time.sleep(0.25 * (_k + 1))
    NB5_GUARD["requests"] += 1
    est = None
    if ids is not None:
        plen = len(ids) + 2
        NB5_GUARD["tokenized"] += 1
    else:
        try:
            est = int(self._estimate_request_input_tokens(messages, tools=kw.get("tools")))
        except Exception:
            est = None
        local = None
        if NB5_LOCAL_TOK.get("verified"):
            try:
                local = len(NB5L.local_chat_ids(payload, NB5_LOCAL_TOK["hf"], NB5_LOCAL_TOK["template"], NB5_LOCAL_TOK["encode"],
                                                NB5_LOCAL_TOK.get("default_kwargs")))
            except Exception as _e:
                tries.append(f"local render: {type(_e).__name__}: {_e}"[:160])
        NB5_GUARD["estimated"] += 1
        if len(NB5_GUARD["estimates"]) < 500:
            NB5_GUARD["estimates"].append({"task": self.arc2_task_id, "branch": self.arc2_branch, "duck_estimate": est,
                                           "local_tokens": local, "verified": bool(NB5_LOCAL_TOK.get("verified")),
                                           "tries": tries, "t": round(time.time(), 1)})
        if local is None:
            raise NB5NoRequest("INCONCLUSIVE_CONTEXT_LENGTH", {"tries": tries, "duck_estimate": est,
                                                               "local_why": NB5_LOCAL_TOK.get("why")})
        plen, how = local + 2 + NB5_LOCAL_MARGIN, "local_tokenizer"
        NB5_GUARD["local_tokenizer"] += 1
    req = int(kw.get("max_tokens") or self._max_output_tokens)
    inc = bool(getattr(self, "_arc2_b", False) and self._arc2_b_in_commit()) or NB5L.is_commit(self)
    g = NB5L.context_guard(plen, req, int(self.generated_tokens), int(getattr(self, "_arc2_b_budget", 10 ** 9)),
                           int(getattr(self, "_arc2_b_reserve", NB5_RESERVE)), inc, _nb5_peak_len(self), NB5_THINK_FLOOR,
                           NB5_CTX_MARGIN)
    g["how"] = how
    if g["no_room"]:
        raise NB5NoRequest("CONTEXT_EXHAUSTED", dict(g, task=self.arc2_task_id, branch=self.arc2_branch))
    if g["clamped"]:
        kw["max_tokens"] = g["cap"]
        NB5_GUARD["clamps"] += 1
        NB5_GUARD["clamped_tokens"] += g["requested"] - g["cap"]
    if g["floor_event"]:
        NB5_GUARD["floor_events"] += 1
        self._arc2_b_budget = g["new_budget"]          # Duck's own route: last thinking request, then the commit turn
        self._nb5_budget_limited = True                # the attempt's verdict (record + report): BUDGET_LIMITED
    if (g["clamped"] or g["floor_event"]) and len(NB5_GUARD["events"]) < 500:
        NB5_GUARD["events"].append(dict(g, task=self.arc2_task_id, branch=self.arc2_branch, generated=int(self.generated_tokens),
                                        t=round(time.time(), 1)))
    return (ids if how == "tokenize" else None), g


def _nb5_no_request(self, verdict, info):
    """The request is NOT sent. The attempt ends through Duck's own route without a request: its budget is set to what it
    generated and the commit flag is raised, so Duck's done_now() is true at the next should_stop() check (Duck checks it
    before every request); the session cap follows. A synthetic empty reply (0 tokens) goes back to Duck. The verdict is
    the attempt's record verdict; the tracer counts it apart from the requests."""
    NB5_GUARD["no_request"][verdict] = NB5_GUARD["no_request"].get(verdict, 0) + 1
    self._nb5_verdict = getattr(self, "_nb5_verdict", None) or verdict
    self._nb5_budget_limited = True
    n = getattr(self, "_nb5_no_request_n", 0) + 1
    self._nb5_no_request_n = n
    try:
        self._arc2_b_budget = int(self.generated_tokens)
        self._arc2_b_commit_sent = True
        s_ = NB5_ACTIVE.get((self.arc2_task_id, self.arc2_branch))
        if s_ is not None:
            s_.max_generated_tokens = max(1, int(self.generated_tokens))
    except Exception:
        pass
    try:
        NB5_TRACE.note_no_request(self, verdict, info)
    except Exception:
        pass
    if n > 3:                                        # never a silent loop: the attempt ends through Duck's failure path
        raise RuntimeError(f"nb5: {verdict}: no request sent ({n} times)")
    return _E12_TA._ChatCompletionResult(message={"role": "assistant", "content": "", "reasoning": None}, finish_reason="stop",
                                         usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                                         served_by="nb5_no_request")


NB5_TRACE = NB5L.Tracer()
NB5_C = {"enabled": True, "mode": "C", "n": 0, "probe": None}
NB5_TAILS = None                                 # (TAIL_A, TAIL_C) from the Flash-Next tokenizer (CELL 5)
NB5_RES = None                                   # the phase's Reservation (CELL 5)
NB5_BLOCKS = {"b0": None, "b1": None, "start": None, "f": None, "block": 4096, "peak_b0": None, "peak_b1": None}
NB5_ACTIVE = {}                                  # (task, branch) -> the running BranchSession (SMOKE early exit)
NB5_EARLY = {"armed": False, "fired": None}
NB5_POOL_CAP = {"blocks": None}                  # 'slots' (R2): the boot's gate x pool, for the C-7 check


def _nb5_cc(self, messages, **kw):
    t0, res, err, info, kind = time.time(), None, None, None, "chat"
    try:
        _ids, _g = _nb5_guard(self, messages, kw)
    except NB5NoRequest as _nr:
        return _nb5_no_request(self, _nr.verdict, _nr.info)
    NB5_TRACE.enter()
    try:
        info = {"guard": _g}
        if NB5L.is_commit(self):
            v = NB5L.commit_variant(NB5_C["mode"], self)
            kind = "commit_A"
            try:
                NB5_TRACE.keep_commit_sample(v, NB5L.chat_payload(_E12_TA, self, messages, kw), self)
            except Exception:
                pass
            if v == "C" and NB5_C["enabled"] and NB5_TAILS:
                NB5_C["n"] += 1
                res, info = NB5L.commit_c_call(_E12_TA, self, messages, kw, SGL_URL + "/v1", NB5_TAILS[0], NB5_TAILS[1],
                                               kw.get("request_timeout_seconds") or self._timeout, NB5_C["n"], ids=_ids)
                info["guard"] = _g
                kind = "commit_C" if res is not None else "commit_A_fallback"
        if res is None:
            res = _NB5_CC_BASE(self, messages, **kw)
        return res
    except BaseException as _e:
        err = _e
        raise
    finally:
        NB5_TRACE.exit()
        try:
            NB5_TRACE.record(self, kind, kw, res, t0, time.time(), error=err, info=info, messages=messages)
            if NB5_ADMIT_MODE == "live" and NB5_RES is not None and res is not None:
                _u = res.usage if isinstance(res.usage, dict) else {}
                _ctx = int(_u.get("prompt_tokens") or 0) + int(_u.get("completion_tokens") or 0)
                NB5_RES.update((self.arc2_task_id, self.arc2_branch),
                               NB5L.cdiv(max(1, _ctx), NB5_BLOCKS["block"]) + NB5_BLOCKS["f"])
        except Exception:
            pass


_E21_CC_ORIG = _nb5_cc
_NB5_RUN0 = E12A.BranchSession.run


def _nb5_session_run(self, pause_at_tokens=None, tag=None):
    """C2: an attempt starts only when its projected peak blocks fit the reservation (deadline = the solve end minus
    the min-admit time); running attempts are never touched; the blocks are released when the attempt's run ends.
    C-7: a projected need above the whole capacity never starts (KV_NEED_ABOVE_POOL, no request). 'slots' (R2): the
    engine schedules; the reservation only caps the attempts in flight at `workers` (1 slot each), C-7 checks the
    projected blocks against NB5_POOL_CAP (the boot's gate x pool)."""
    res = NB5_RES
    if res is None or self.status == "done":
        return _NB5_RUN0(self, pause_at_tokens=pause_at_tokens, tag=tag)
    key = (self.task_id, self.branch)
    if NB5_EARLY.get("fired"):                     # PLAN v3.6: SMOKE passed its gates -> no new attempt
        self.status = "done"
        self.record.update(note="nb5: SMOKE early exit (gates passed)", nb5_reservation="early_exit")
        self.write()
        return self.status
    slots = NB5_ADMIT_MODE == "slots"
    need = (int(NB5_BLOCKS["b0" if self.branch == 0 else "b1"] or 1) if NB5_ADMIT_MODE != "live"
            else int(NB5_BLOCKS["start"] or 1))
    if (need > int(NB5_POOL_CAP["blocks"] or 0)) if slots else res.above_pool(need):
        self.status = "done"
        self.record.update(note="nb5: KV_NEED_ABOVE_POOL (projected need above the whole pool)", nb5_reservation="denied",
                           nb5_verdict="KV_NEED_ABOVE_POOL")
        self.write()
        return self.status
    if not res.acquire(key, 1 if slots else need, self.deadline - float(E12_R6["E12_R6_MIN_ADMIT_SEC"])):
        self.status = "done"
        self.record.update(note="nb5: no KV reservation before the admit deadline", nb5_reservation="denied")
        self.write()
        return self.status
    self.record["nb5_reservation"] = {"blocks": need, "slots": slots, "used_after_grant": res.used}
    NB5_ACTIVE[key] = self
    try:
        return _NB5_RUN0(self, pause_at_tokens=pause_at_tokens, tag=tag)
    finally:
        NB5_ACTIVE.pop(key, None)
        res.release(key)
        _v = getattr(self.agent, "_nb5_verdict", None)
        if _v:
            self.record["nb5_verdict"] = _v                    # CONTEXT_EXHAUSTED / INCONCLUSIVE_CONTEXT_LENGTH / KV_...
        elif getattr(self.agent, "_nb5_budget_limited", False):
            self.record["nb5_verdict"] = "BUDGET_LIMITED"     # published by the scheduler right after this returns


E12A.BranchSession.run = _nb5_session_run
print("NB5: commit dispatcher + request trace + reservation hook installed", flush=True)


# E10: original task-order and evaluation profiles, but one Poetiq program per task/expert/iteration.
if P['source']=='eval':
    _ch=_load_json(ARC_ROOT/'arc-agi_evaluation_challenges.json')
    _sol=_load_json(ARC_ROOT/'arc-agi_evaluation_solutions.json')
    if P.get('task_list'):
        assert all(t in _ch for t in P['task_list']), 'Evaluation task ID missing'
        TASK_IDS=sorted(P['task_list'])
    else:
        TASK_IDS=sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch),P['n_tasks']))
    CHALLENGES={t:_ch[t] for t in TASK_IDS}
    SOLUTIONS={t:_sol[t] for t in TASK_IDS}
else:
    CHALLENGES=dict(test_challenges)
    TASK_IDS=sorted(CHALLENGES) # no Codex ordering
    SOLUTIONS=None
N_HIDDEN_UNITS=sum(len(t['test']) for t in test_challenges.values())
print('E10 Poetiq:',len(TASK_IDS),'tasks,',sum(len(CHALLENGES[t]['test']) for t in TASK_IDS),
      'queries | experts',POETIQ_NUM_EXPERTS,'iterations per expert',POETIQ_MAX_ITERATIONS)


# ==================================================================== CELL 5 — NB5 engine: Merlin Flash-Next (RC2 FP8 boot path)
# Verbatim donors (build_e25_nb5.py cuts and asserts them): RC2 builder (e23_lab, CELL 1 LAB load + inputs, CELL 2 to the
# stock fallback, CELL 3 prompts, E23 step runner, kv_census / sampled_check / prefill_probe); E21 (metrics, stats
# lines, counters, content / contract / cache probes). New: the NB5 glue at the end of this cell.
import urllib.request as _urlreq, signal as _signal, statistics as _statistics
import tarfile                                   # NB5-FATAL-1: RC2's merlin_install opens the Merlin payload tar
MODE = "FULL"                                   # E23 step runner (NTOK 128); RC2's MODE
FAKE = os.environ.get("E23_FAKE_SERVER") if E11_HARNESS else None
T_START = time.time() - (time.perf_counter() - NOTEBOOK_T0)      # RC2's epoch clock on the Duck notebook's origin
HARD_STOP_EPOCH = T_START + TOTAL_BUDGET_SEC                      # 11:50: the e23 watchdog kills every live server
BUDGET = int(TOTAL_BUDGET_SEC)                                   # RC2's name (its input log line)
MAX_LEN = 131072
ENG_OUT = OUT / "engine"
ENG_OUT.mkdir(parents=True, exist_ok=True)
LOGS = ENG_OUT / "logs"
LOGS.mkdir(exist_ok=True)
TRACES = ENG_OUT / "traces"
TRACES.mkdir(exist_ok=True)
R = {"what": "E25 NB5 FN-Duck engine", "max_len": MAX_LEN, "setup": {}, "sections": {}, "errors": [], "order": []}
_SAVE_LOCK = threading.Lock()


def save():
    with _SAVE_LOCK:
        tmp = ENG_OUT / "nb5_engine.json.tmp"
        tmp.write_text(json.dumps(R, indent=1, default=str))
        tmp.replace(ENG_OUT / "nb5_engine.json")


def left():
    return HARD_STOP_EPOCH - time.time()


def log(*a):
    print(f"[{(time.time() - T_START) / 60:6.1f} min]", *a, flush=True)


COMP = INP / "competitions" / "arc-prize-2026-arc-agi-2"
ENGINE_READY = False; llm = None; engine = None; CONCURRENCY = 0; E11_SERVER = None; E12_VLLM_CONCURRENCY = 0
ENGINE_INFO = {"backend": None}
HELP = {}
# ---- RC2 CELL 1: the lab module + every request traced (verbatim)
LAB_SRC = (lambda _t: _t if __import__("hashlib").sha256(_t.encode()).hexdigest() == "ef393fbb2ff7669ee4c2c99ee841b60da66514a3b86b7c0ce81fd5f64689ab5e" else (_ for _ in ()).throw(RuntimeError("e23_lab: embedded source sha256 mismatch")))(__import__("zlib").decompress(__import__("base64").b64decode("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")).decode("utf-8"))
(TMP / "e23_lab.py").write_text(LAB_SRC)
sys.path.insert(0, str(TMP))
import e23_lab as L
REQ_LOG = TRACES / "requests.jsonl"                           # memory rule: the full text of EVERY request
_stream_run = L.Stream.run

def _traced_run(self):
    try:
        _stream_run(self)
    finally:
        if not self.err and any(m is not None and not (isinstance(m, (int, float)) and math.isfinite(m))
                                for m in self.margins):
            self.err = "NONFINITE_LOGPROB"           # final audit H2: NaN / Inf logprobs are never data
        try:
            r = self.row()
            pr = self.body.get("prompt") or []
            r.update(port=self.port, sampling={k: v for k, v in self.body.items() if k != "prompt"}, prompt_len=len(pr),
                     prompt_sha=hashlib.sha256(json.dumps(pr).encode()).hexdigest()[:16],
                     ttft=round(self.t_first - self.t_start, 2) if self.t_first and self.t_start else None,
                     sec=round(self.t_end - self.t_start, 2) if self.t_end and self.t_start else None, t=time.time())
            with _SAVE_LOCK, open(REQ_LOG, "a") as fh:
                fh.write(json.dumps(r) + "\n")
        except Exception:
            pass
L.Stream.run = _traced_run

# ---- RC2 CELL 2 (verbatim to the stock fallback)
def run(name, cmd, timeout=3600, env=None):
    t = time.time()
    try:
        with open(LOGS / f"{name}.log", "w") as f:
            r = subprocess.run(["bash", "-c", "set -o pipefail; " + cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                               env={**os.environ, **(env or {})})
        rc = r.returncode
    except subprocess.TimeoutExpired:
        rc = -9
    txt = (LOGS / f"{name}.log").read_text(errors="replace")
    R["setup"][name] = {"rc": rc, "sec": round(time.time() - t, 1), "tail": txt[-1500:]}
    log(name, "rc", rc, f"{time.time() - t:.0f}s")
    save()
    return rc, txt


SHIM = TMP / "e23_libcuda"          # FlashInfer JIT links -lcuda; Kaggle has only libcuda.so.1
SHIM.mkdir(exist_ok=True)
for d in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/compat", "/usr/lib64"):
    c = sorted(Path(d).glob("libcuda.so.1")) if Path(d).is_dir() else []
    if c and not (SHIM / "libcuda.so").exists():
        (SHIM / "libcuda.so").symlink_to(c[0])


def host_facts():
    """A0: what the Kaggle box gives us (glibc, driver, GPUs, PCIe, peer access, cgroup memory, /dev/shm, disk)."""
    f = {}
    for k, cmd in (("glibc", "ldd --version | head -1"), ("uname", "uname -a"),
                   ("nvidia_smi", "nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader"),
                   ("cuda_version", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"), ("shm", "df -h /dev/shm | tail -1"),
                   ("tmp_disk", f"df -h {TMP} | tail -1"), ("nproc", "nproc"),
                   ("cgroup_max", "cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes"),
                   ("cgroup_cur", "cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes"),
                   ("meminfo", "grep -E 'MemTotal|MemAvailable' /proc/meminfo"), ("ulimit_l", "ulimit -l"),
                   ("omp", "echo OMP_NUM_THREADS=$OMP_NUM_THREADS"), ("topo", "nvidia-smi topo -m; nvidia-smi topo -p2p r"),
                   ("pcie_now", "nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current,power.limit --format=csv,noheader")):
        try:
            f[k] = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60).stdout.strip()[:600]
        except Exception as e:
            f[k] = repr(e)[:200]
    try:
        import torch
        n = torch.cuda.device_count()
        f["peer_access"] = [[torch.cuda.can_device_access_peer(i, j) if i != j else None for j in range(n)] for i in range(n)]
    except Exception as e:
        f["peer_access"] = repr(e)[:200]
    return f


def host_mem():
    """MiB. avail_mib = max - (current - inactive_file), Merlin's own rule (RC1's guard counted page cache as used and
    skipped arm 3 with MemAvailable 55 GB); /dev/shm counts against the cgroup."""
    out = {}
    try:
        cur = int(Path("/sys/fs/cgroup/memory.current").read_text())
        mx = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        st = dict(l.split() for l in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines() if len(l.split()) == 2)
        out.update(cgroup_cur_mib=cur >> 20, cgroup_max_mib=None if mx == "max" else int(mx) >> 20,
                   **{k + "_mib": int(st[k]) >> 20 for k in ("anon", "file", "shmem", "inactive_file") if k in st})
        if mx != "max":
            out["avail_mib"] = (int(mx) - (cur - int(st.get("inactive_file", 0)))) >> 20
    except Exception:
        pass
    try:
        for l in open("/proc/meminfo"):
            if l.startswith(("MemAvailable", "MemTotal")):
                out[l.split(":")[0] + "_mib"] = int(l.split()[1]) >> 10
    except Exception:
        pass
    try:
        out["dev_shm_used_mib"] = shutil.disk_usage("/dev/shm").used >> 20
    except Exception:
        pass
    return out


TELEMETRY = ENG_OUT / "telemetry.jsonl"   # NB5: engine folder


def _telemetry():
    """Host RAM + GPU memory every 10 s for the whole run (Merlin's floor ~69 GiB inside a ~173 GiB cgroup)."""
    while True:
        try:
            with open(TELEMETRY, "a") as fh:
                fh.write(json.dumps({"t": round(time.time() - T_START, 1), "host": host_mem(), "gpu": L.gpu_mem()}) + "\n")
        except Exception:
            pass
        time.sleep(10)


threading.Thread(target=_telemetry, daemon=True).start()


SHM = Path(os.environ.get("RC2_SHM", "/dev/shm"))


def shm_names():
    try:
        return {p.name for p in SHM.iterdir()}
    except Exception:
        return set()


def shm_cleanup(before):
    """After a boot's server is dead: only /dev/shm entries that appeared during THIS boot (absent from `before`) and
    match the engine's prefixes are unlinked; everything else is recorded, never touched."""
    out = {"host_before": host_mem(), "removed": [], "kept": []}
    try:
        for p in sorted(SHM.iterdir()):
            try:
                sz = p.stat().st_size >> 20
            except Exception:
                sz = None
            if p.name not in before and re.match(r"(torch_|psm_|vllm|ple|nccl|sem\.mp-|cuda\.shm)", p.name, re.I):
                try:
                    p.unlink() if not p.is_dir() else shutil.rmtree(p)
                    out["removed"].append([p.name, sz])
                except Exception as e:
                    out["kept"].append([p.name, sz, repr(e)[:80]])
            else:
                out["kept"].append([p.name, sz])
    except Exception as e:
        out["error"] = repr(e)[:200]
    out["host_after"] = host_mem()
    out["removed"], out["kept"] = out["removed"][:60], out["kept"][:60]
    return out


E18_SILENCE_SEC = float(os.environ.get("E18_SILENCE_SEC", 600))   # NB6 P1: E18 R2 cell 6 (B-M3)


class MServer(L.Server):
    """Merlin passes --shutdown-timeout 60 and the PLE transport unlinks its /dev/shm storages only on a graceful
    teardown: SIGTERM, wait 75 s, then E23's kill (SIGTERM + 40 s + SIGKILL of the whole session)."""
    def wait_ready(self, timeout, silence=E18_SILENCE_SEC):
        """NB6 P1 = E18 R2 E18Server.wait_ready: ready = /health answers AND this server's own "GPU KV cache size"
        line is in its log (B-L8); a launch whose log stops growing for `silence` seconds is a stall (B-M3)."""
        last, last_t = -1, time.perf_counter()
        while time.perf_counter() - self.t0 < timeout:
            if not self.alive():
                return False, f"exited rc={self.proc.returncode}"
            try:
                _urlreq.urlopen(self.url + "/health", timeout=10).read()
                if "GPU KV cache size" in self.log():
                    self.ready_sec = round(time.perf_counter() - self.t0, 1)
                    return True, f"ready in {time.perf_counter() - self.t0:.0f}s"
            except Exception:
                pass
            sz = self.log_path.stat().st_size if self.log_path.exists() else 0
            if sz != last:
                last, last_t = sz, time.perf_counter()
            elif time.perf_counter() - last_t > silence:
                return False, f"stall ({silence:.0f} s of log silence)"
            time.sleep(1 if FAKE else 5)
        return False, "timeout"

    def kill(self, fast=False):
        if fast and self.proc is not None:     # NB6 P1 = E18 R2 E18Server.kill(fast=True): at once after the rerun's solve
            try:
                os.killpg(self.proc.pid, 15) if hasattr(os, "killpg") else self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=0.1)
            except Exception:
                pass
            try:
                os.killpg(self.proc.pid, 9) if hasattr(os, "killpg") else self.proc.kill()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=30)
            except Exception:
                pass
            return
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 15) if hasattr(os, "killpg") else self.proc.terminate()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=75)
            except Exception:
                pass
        super().kill()


def boot_gate():
    """Merlin's pin leaves ~0.9 GiB free per card: every boot needs near-empty GPUs (< 150 MiB, not E23's 1024)."""
    g = L.wait_gpu_release(300, fake=bool(FAKE), gpus="0,1,2,3", free_mib=150)
    return {"ok": g, "gpus": L.gpu_mem(), "host": host_mem()}       # callers launch only on ok is True


CKPT = TMP / "fn_merlin"
SHA = {"thread": None, "ok": None, "bad": [], "sec": None}


def merge_ckpt():
    """Symlink the 5 parts + 10 PLE shards into one directory; names and sizes now, sha256 in a background thread."""
    SHA.update(thread=None, ok=None, bad=[], sec=None)        # a stale result never gates a new merge
    shutil.rmtree(CKPT, ignore_errors=True)
    CKPT.mkdir(parents=True)
    expected = {}
    for k, (d, man) in PARTS.items():
        for n, (size, sha) in man["files"].items():
            expected[n] = (d / man.get("model_subdir", ".") / n, size, sha)
    for n, (size, sha) in PLE_EXPECTED.items():
        expected[n] = (PLE_DIR / n, size, sha)
    bad = []
    for n, (src, size, sha) in sorted(expected.items()):
        if not src.is_file():
            bad.append(f"missing {n}")
            continue
        if not FAKE and src.stat().st_size != size:
            bad.append(f"{n}: {src.stat().st_size} bytes, expected {size}")
        try:
            (CKPT / n).symlink_to(src)
        except OSError:
            if not FAKE:
                raise
            shutil.copyfile(src, CKPT / n)               # local Windows harness only (no symlink privilege)
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text())
    need = set(idx["weight_map"].values())
    bad += [f"index needs {n}, not merged" for n in sorted(need - set(expected))]
    cfg = json.loads((CKPT / "config.json").read_text())
    if (cfg.get("text_config") or {}).get("ple_embedding_dtype") != "float8_e4m3fn":
        bad.append("config.json: text_config.ple_embedding_dtype is not float8_e4m3fn")
    rec = {"files": len(expected), "index_files": len(need), "bad": bad[:20], "ok": not bad}

    def hash_all():
        t = time.time()
        import concurrent.futures as CF

        def one(item):
            n, (src, size, sha) = item
            if FAKE:
                return n, True
            with open(src, "rb") as fh:
                return n, hashlib.file_digest(fh, "sha256").hexdigest() == sha
        with CF.ThreadPoolExecutor(6) as ex:
            res = list(ex.map(one, sorted(expected.items())))
        SHA.update(ok=all(ok for _, ok in res), bad=[n for n, ok in res if not ok], sec=round(time.time() - t, 1))
        R["setup"]["sha256"] = {k: SHA[k] for k in ("ok", "bad", "sec")}
        save()
    if rec["ok"]:
        SHA["thread"] = threading.Thread(target=hash_all, daemon=True)
        SHA["thread"].start()
    return rec


def sha_gate(timeout):
    """The first launch waits for the full sha256 check (Merlin itself never hashes shards)."""
    th = SHA["thread"]
    if th is not None:
        th.join(max(1, timeout))
    if th is None or th.is_alive():
        return False, "sha256 check not finished"
    return bool(SHA["ok"]), ("ok" if SHA["ok"] else f"sha256 mismatch: {SHA['bad'][:5]}")


def prefetch_bg(paths, tag):
    """rev 3: warm the page cache of a cold 17-30 GB /kaggle/input checkpoint in the background (many readers)."""
    def go():
        R["setup"]["prefetch_" + tag] = L.prefetch([p for p in paths if p]) if not FAKE else {"fake": True}
        save()
    th = threading.Thread(target=go, daemon=True)
    th.start()
    return th


MERLIN_ENG = None


def merlin_install():
    """P2 payload (.bin = a tar Kaggle does not unpack) -> install_merlin_offline.sh (UV_OFFLINE); payload deleted after;
    a failure is cached so later Merlin steps fail fast (rev 3)."""
    global MERLIN_ENG
    if MERLIN_ENG is not None:
        if MERLIN_ENG.get("failed"):
            raise RuntimeError("Merlin install failed earlier: " + MERLIN_ENG["failed"])
        return MERLIN_ENG
    if not MERLIN_OK or MERLIN_PAY is None:
        MERLIN_ENG = {"failed": "Merlin checkpoint / env not attached or incomplete"}
        raise RuntimeError(MERLIN_ENG["failed"])
    if FAKE:
        MERLIN_ENG = {"fake": True}
        return MERLIN_ENG
    try:
        pay_root = TMP / "merlin_pay"
        shutil.rmtree(pay_root, ignore_errors=True)
        pay_root.mkdir()
        t = time.time()
        with tarfile.open(MERLIN_PAY) as tf:
            tf.extractall(pay_root, filter="data")
        R["setup"]["merlin_extract_sec"] = round(time.time() - t, 1)
        pay = pay_root / "merlin_offline"
        shutil.rmtree(TMP / "merlin_off", ignore_errors=True)
        rc, txt = run("install_merlin", f"bash {pay}/install_merlin_offline.sh {pay} {TMP}/merlin_off", timeout=5400)
        kv = dict(re.findall(r"^(MERLIN_[A-Z]+)=(.*)$", txt, re.M))
        shutil.rmtree(pay_root, ignore_errors=True)
        if rc or "INSTALL OK" not in txt:
            raise RuntimeError("Merlin offline install failed (logs/install_merlin.log)")
        sys.path.insert(0, kv["MERLIN_TREE"] + "/tools/merlin")
        import launch as ML
        site = subprocess.run([kv["MERLIN_VENV"] + "/bin/python", "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                              capture_output=True, text=True).stdout.strip()
        MERLIN_ENG = {"kv": kv, "ML": ML, "cuda_home": site + "/nvidia/cu13"}
        return MERLIN_ENG
    except Exception as e:
        MERLIN_ENG = {"failed": f"{type(e).__name__}: {e}"[:300]}
        raise


MERLIN_CC = {"mode": "gcc13"}       # rev 3c: Merlin README:229 / install-env.sh: FlashInfer JIT at first start needs GCC >= 13



KV_FP8, KV_BF16 = 2479838700, 2464415744      # Merlin's own L4 pins (23,034 MiB cards; preflight.kv_bytes_for_vram, 150 / 79 blocks)
MERLIN_PROFILE = "fn-quad-24g-sm89"   # NB5: PROFILE is the Duck profile name
XHIGH = json.dumps({"enable_thinking": True, "reasoning_effort": "xhigh"})
HOST_PHASE = {}


def merlin_fn_server(name, kv, port, kv_bytes):
    """Merlin's launch.build() for Flash-Next on the L4 profile: KV pin lowered for 22.49 GiB cards, the checkpoint's own
    chat template at xhigh (Merlin passes medium otherwise), Merlin's host_phase checks recorded, E23's env overrides."""
    eng = merlin_install()
    extra = ["--served-model-name", "q27", "--max-model-len", str(MAX_LEN), "--port", str(port),
             "--default-chat-template-kwargs", XHIGH] + list(NB5_BOOT_EXTRA)   # NB5: NB1/NB2 levers, MTP off
    if FAKE:
        return MServer(name, [sys.executable, FAKE, str(CKPT) + "-w4a16", "--tensor-parallel-size", "2", "--max-num-seqs",
                              "8" if kv == "fp8" else "3", "--port", str(port), "--max-model-len", str(MAX_LEN),
                              "--kv-cache-dtype", NB5L.KV_DTYPE_NAME[kv], "--kv-cache-memory-bytes", str(kv_bytes)] + list(NB5_BOOT_EXTRA),   # NB5: the fake reports dtype / pin (C-11 read-back, the ladder)
                       dict(os.environ, **NB5_BASE_ENV, **NB5_BOOT_LEVERS), port, LOGS)
    kvv = eng["kv"]
    gcc_bin = str(Path(kvv["MERLIN_CC"]).parent) if kvv.get("MERLIN_CC") else ""
    base = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH", "CUDA_HOME", "LD_PRELOAD", "CC", "CXX")}
    base.update(GPUS="0,1,2,3", PORT=str(port), KV_BYTES=str(kv_bytes), P2P="off",
                VENV=kvv["MERLIN_VENV"], TREE=kvv["MERLIN_TREE"], CHAT_TEMPLATE="", WARM="0", CACHE_ROOT=f"{TMP}/merlin-cache",
                PATH=f"{kvv['MERLIN_VENV']}/bin:{gcc_bin}:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}",
                CUDA_HOME=eng["cuda_home"])
    base.update(NB5_BASE_ENV)                          # NB5: Merlin knobs from NB1 / NB2 (build() reads DRAFT_HEAD etc.)
    base.update(NB5_BOOT_LEVERS)                       # NB5 R2: this boot try's lever env (lever b)
    if kv == "fp8":
        base["SCALES"] = str(CKPT / "qsa_kv_scales_262k.json")
    plan = eng["ML"].build("flash-next", MERLIN_PROFILE, str(CKPT), base, extra=extra, kv=kv, checkpoint="flash-next", status="untested")
    try:                                               # audits: tree products, venv, checkpoint marker, PLE dtype, toolchain,
        eng["ML"].host_phase(plan)                     # scales file: a failure stops this boot before any server starts
        HOST_PHASE[name] = "ok"
    except BaseException as e:
        HOST_PHASE[name] = f"{type(e).__name__}: {e}"[:600]
        R["setup"].setdefault("host_phase_error", {})[name] = HOST_PHASE[name]
        save()
        raise RuntimeError("MERLIN_HOST_PHASE_FAILED: " + HOST_PHASE[name])
    env = dict(plan.env)
    env.update(LD_PRELOAD=kvv["MERLIN_SHIM"], CUDA_HOME=eng["cuda_home"], VLLM_USE_RUST_FRONTEND="0", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", LIBRARY_PATH=f"{SHIM}:{os.environ.get('LIBRARY_PATH', '')}",
               PATH=base["PATH"], VLLM_CACHE_ROOT=f"{TMP}/merlin-cache", TRITON_CACHE_DIR=f"{TMP}/merlin-triton",
               FLASHINFER_WORKSPACE_BASE=f"{TMP}/merlin-fi", CUDA_DEVICE_ORDER="PCI_BUS_ID", VLLM_PLE_OFFLOAD_READY_TIMEOUT="3600")
    for k in ("CC", "CXX", "CUDAHOSTCXX"):
        env.pop(k, None)
    if MERLIN_CC["mode"] == "gcc13" and kvv.get("MERLIN_CC") and kvv.get("MERLIN_CXX"):
        env.update(CC=kvv["MERLIN_CC"], CXX=kvv["MERLIN_CXX"], CUDAHOSTCXX=kvv["MERLIN_CXX"])
        lib13 = Path(kvv["MERLIN_CC"]).parents[1] / "lib"
        if (lib13 / "libstdc++.so.6").exists():
            env["LD_LIBRARY_PATH"] = f"{lib13}:{env.get('LD_LIBRARY_PATH', '')}"
    try:                                               # Merlin's own scheme: PORT 8000..8355 -> 29500+d / 30000+100d
        mp, vp = eng["ML"].distributed_ports(str(port))
        env.update(MASTER_PORT=str(mp), VLLM_PORT=str(vp))
    except Exception:
        env.update(MASTER_PORT=str(29500 + port - 8000), VLLM_PORT=str(30000 + 100 * (port - 8000)))
    env.update(NB5_BASE_ENV)                           # NB5Q-ENV-1: after build(), which strips every MERLIN_* variable (= NB1)
    env.update(NB5_BOOT_LEVERS)                        # NB5Q-ENV-1: lever b reaches the server env
    argv = plan.argv()
    HELP[name] = L.help_check("merlin", argv[:argv.index("serve") + 1], env, argv)
    R["setup"].setdefault("merlin_plans", {})[name] = {"argv_tail": argv[-60:], "status": plan.status, "host_phase": HOST_PHASE[name],
                                                       "env_keys": {k: env[k] for k in sorted(env) if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES"))},
                                                       "lever_env": {k: env.get(k) for k in sorted({**NB5_BASE_ENV, **NB5_BOOT_LEVERS})}}   # NB5Q-ENV-1
    save()
    return MServer(name, argv, env, port, LOGS)



# ---- RC2 CELL 1 input block, wrapped
def nb5_rc2_inputs():
    # RC2 CELL 1 input discovery (exact Model paths), GPU preflight, watchdog
    global PLE_EXPECTED, MD, PART_DIRS, PARTS, k, d, mf, PLE_DIR, MERLIN_PAY, INTEL, WHEELHOUSE, ARC, MERLIN_OK, ABORT, NVSMI_OK
    PLE_EXPECTED = json.loads(r"""{"model-plefp8-00000.safetensors": [5200027199, "dc2e845b7edd35bda92834fba3626bf7d199e28d6aceac99fee654aade390cfd"], "model-plefp8-00001.safetensors": [5200027202, "899eaa0716e28594468a1389ee58cb065c23907f1270de3831f5ecb0a4f82d56"], "model-plefp8-00002.safetensors": [5200027198, "06fd8a11abf0419a669f89397b8d70dd6ff42d401e6b2a037c65e49704faaf71"], "model-plefp8-00003.safetensors": [5200027189, "c6aaa1fc08e84eced3c8151ac8679ed943888eba0fcef2556963693430f95bd9"], "model-plefp8-00004.safetensors": [5200027190, "d94e97c96d3ea09208614da016960f3f4b429f47a044c898a51b493f42f74ba2"], "model-plefp8-00005.safetensors": [5200027190, "03d5d4792e14a4ab55bae50bb459624b01786f818bbcfcaed1a5d0235af484c6"], "model-plefp8-00006.safetensors": [5200027190, "586cccbc12383021bc9bc02f206d9b19fbd1672373ed9a5d91e3b0ce34c2418f"], "model-plefp8-00007.safetensors": [5200027190, "c4a23bcc10f3cde6b633e82c282a9a518e3a64d433628283e1bc592c94cf3d6c"], "model-plefp8-00008.safetensors": [5200027190, "2dc8098c0d020bff277c9cf499a6b908e17836b70a5f949dfa24793371c9a87e"], "model-plefp8-00009.safetensors": [4400023163, "61de98b89bb79f386795787d7a76827a26f1e292c26edbb0a1b613da455f5a9c"]}""")                     # 10 FP8 PLE shards: [size, sha256]
    # exact Kaggle mount paths (foysalemonshanto Models, checked through the API 2026-10-31; no recursive search)
    MD = INP / "models" / "foysalemonshanto"
    PART_DIRS = {1: MD / "qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1",
                 2: MD / "qwen3-8-flash-next-w4a16-merlin-p2/transformers/default/1",
                 3: MD / "qwen3-8-flash-next-w4a16-merlin-p3/transformers/default/1",
                 4: MD / "qwen3-8-flash-next-w4a16-merlin-p4/transformers/default/1",
                 5: MD / "qwen3-8-flash-next-w4a16-merlin-p5/transformers/default/1"}
    PARTS = {}
    for k, d in PART_DIRS.items():                                  # the five checkpoint parts (Models from the kernel outputs)
        mf = d / f"q27_fn_merlin_part{k}_manifest.json"
        PARTS[k] = (d, json.loads(mf.read_text())) if mf.is_file() else None
    PLE_DIR = MD / "qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1"      # our FP8 PLE overlay = albucino's shards
    PLE_DIR = PLE_DIR if (PLE_DIR / "model-plefp8-00000.safetensors").is_file() else None
    MERLIN_PAY = MD / "e23-merlin-env/other/v1/1/merlin_offline_tar.bin"
    MERLIN_PAY = MERLIN_PAY if MERLIN_PAY.is_file() else None
    INTEL = INP / "models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
    INTEL = INTEL if (INTEL / "config.json").is_file() else None
    WHEELHOUSE = INP / "datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse/install_wheelhouse.py"
    WHEELHOUSE = WHEELHOUSE if WHEELHOUSE.is_file() else None
    ARC = COMP / "arc-agi_evaluation_challenges.json"
    for _nb6_arc in [ARC_ROOT / "arc-agi_evaluation_challenges.json"] + ([ARC_ROOT / "arc-agi_test_challenges.json"] if RERUN else []):
        if not ARC.is_file() and _nb6_arc.is_file():   # NB6 E7: CELL 0's ARC_ROOT; rerun: the test file (train pairs) for the probe prompts
            ARC = _nb6_arc
    if not ARC.is_file():
        raise FileNotFoundError(f"{ARC}: attach the competition data")
    MERLIN_OK = all(PARTS.values()) and PLE_DIR is not None and MERLIN_PAY is not None
    R["inputs"] = {"parts": {k: (str(v[0]) if v else None) for k, v in PARTS.items()}, "ple_dir": str(PLE_DIR),
                   "merlin_pay": str(MERLIN_PAY), "intel": str(INTEL), "wheelhouse": str(WHEELHOUSE), "arc": str(ARC),
                   "merlin_inputs_ok": MERLIN_OK}
    ABORT = {"why": None}
    NVSMI_OK = bool(L.gpu_mem()) and not FAKE
    R["setup"]["gpus_at_start"] = L.gpu_mem()
    if not FAKE and not os.environ.get("RC2_CPU_KIT") and len(R["setup"]["gpus_at_start"] or []) != 4:   # CPU kit: no GPUs
        R["errors"].append(f"expected 4 GPUs, nvidia-smi shows {R['setup']['gpus_at_start']}")
        save()
        raise RuntimeError("GPU_PREFLIGHT_FAILED: " + repr(R["setup"]["gpus_at_start"]))
    L.start_watchdog(HARD_STOP_EPOCH, lambda: ABORT.update(why="HARD_STOP reached: watchdog killed live servers"))
    log("RC2 | budget", BUDGET // 60, "min |", json.dumps(R["inputs"])[:900])
    save()


# ---- RC2 CELL 3, wrapped
def nb5_rc2_prompts():
    # RC2 CELL 3: A0 (host facts, merge, sha256 thread), tokenizer, template, prompts
    global _t, TOKDIR, MERLIN_PROMPTS, _src, _n, Tokenizer, AutoTokenizer, TOK, _HF, _tpl, _tasks, _ids, _grid, _task_text, _INTRO, _PH, _render, _RX, _RM, _head_txt, _tail_txt, _HEAD, _TAIL, TPL_MSG, TPL_IDS, _TASK_IDS, prompt, TOK_SHA, TPL_SHA, BANDS, NSTREAM, PROMPTS, MAX_NEW, ORACLE_P, LONG_P, TF_P
    # ==================================================================== CELL 3 — prompts (Flash-Next tokenizer + its own template at xhigh)
    R["sections"]["A0"] = {}
    _t = time.time()
    R["sections"]["A0"]["host"] = host_facts()
    R["sections"]["A0"]["merge"] = merge_ckpt() if MERLIN_OK else {"ok": False, "why": "Merlin inputs missing", "inputs": R["inputs"]}
    R["sections"]["A0"]["host_mem"] = host_mem()
    log("A0 host", json.dumps(R["sections"]["A0"]["host"])[:1200])
    log("A0 merge", json.dumps(R["sections"]["A0"]["merge"])[:800])
    save()
    TOKDIR = TMP / "fn_tok"
    shutil.rmtree(TOKDIR, ignore_errors=True)
    TOKDIR.mkdir()
    # the Merlin checkpoint's tokenizer when the merge is OK; otherwise the Intel (stock fallback) checkpoint's own
    MERLIN_PROMPTS = bool(R["sections"]["A0"]["merge"].get("ok")) and (CKPT / "tokenizer.json").exists()
    _src = CKPT if MERLIN_PROMPTS else INTEL
    if _src is None or not (_src / "tokenizer.json").exists():
        raise FileNotFoundError("no tokenizer: the Merlin merge failed and the Intel fallback checkpoint is not attached")
    for _n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (_src / _n).exists():
            shutil.copyfile(_src / _n, TOKDIR / _n)
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    TOK = Tokenizer.from_file(str(TOKDIR / "tokenizer.json"))
    _HF = AutoTokenizer.from_pretrained(str(TOKDIR))
    _tpl = ((TOKDIR / "chat_template.jinja").read_text() if (TOKDIR / "chat_template.jinja").exists()
            else json.loads((TOKDIR / "tokenizer_config.json").read_text())["chat_template"])
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


    _INTRO = "Find the transformation rule of each ARC task below and solve the test inputs.\n\n"
    _PH = "E13_BODY"


    def _render(effort):
        return _HF.apply_chat_template([{"role": "user", "content": _INTRO + _PH}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True, reasoning_effort=effort, chat_template=_tpl)


    _RX, _RM = _render("xhigh"), _render("medium")
    assert _RX.count(_PH) == 1
    _head_txt, _tail_txt = _RX.split(_PH)
    _HEAD = TOK.encode(_head_txt, add_special_tokens=False).ids
    _TAIL = TOK.encode(_tail_txt, add_special_tokens=False).ids
    TPL_MSG = [{"role": "user", "content": _INTRO + _task_text(_ids[0])}]
    TPL_IDS = TOK.encode(_HF.apply_chat_template(TPL_MSG, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                                 reasoning_effort="xhigh", chat_template=_tpl), add_special_tokens=False).ids
    R["prompt_render"] = {"head_text": _head_txt[-600:], "tail_text": _tail_txt, "head_tokens": len(_HEAD), "tail_tokens": len(_TAIL),
                          "xhigh_differs_from_medium": _RX != _RM, "xhigh_sentence": "xhigh" in _head_txt,
                          "template_sha256": hashlib.sha256(_tpl.encode()).hexdigest(),
                          "tokenizer_json_sha256": hashlib.sha256((TOKDIR / "tokenizer.json").read_bytes()).hexdigest()}
    _TASK_IDS = {k: TOK.encode(_task_text(k) + "\n\n", add_special_tokens=False).ids for k in _ids}


    def prompt(band, variant):
        body, i = [], (variant * 7) % len(_ids)
        need = band - len(_HEAD) - len(_TAIL)
        while len(body) < need:
            body += _TASK_IDS[_ids[i % len(_ids)]]
            i += 1
        return _HEAD + body[:need] + _TAIL


    TOK_SHA = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"   # = the 27B Merlin tokenizer.json
    TPL_SHA = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"   # = the 27B FP8 chat_template.jinja
    R["prompt_render"]["source"] = "merlin" if MERLIN_PROMPTS else "intel_fallback"
    R["prompt_render"]["hashes_ok"] = (R["prompt_render"]["tokenizer_json_sha256"] == TOK_SHA and
                                       R["prompt_render"]["template_sha256"] == TPL_SHA)
    if MERLIN_PROMPTS:                    # the pinned hashes belong to the Merlin checkpoint only (Intel's are recorded)
        assert R["prompt_render"]["hashes_ok"], "Flash-Next tokenizer / template differ from the pinned Merlin checkpoint"
    assert R["prompt_render"]["xhigh_sentence"] and R["prompt_render"]["xhigh_differs_from_medium"], "xhigh render missing"
    BANDS = {"2k": 2048, "8k": 8192, "40k": 40000}      # NB5: sampled check, warm-up, pure decode window
    NSTREAM = 8
    PROMPTS = {b: [prompt(n, v) for v in range(NSTREAM)] for b, n in BANDS.items()}
    MAX_NEW = {b: MAX_LEN - n - 512 for b, n in BANDS.items()}
    ORACLE_P = [PROMPTS["2k"][0], PROMPTS["2k"][1], PROMPTS["8k"][2]]
    LONG_P = [PROMPTS["40k"][3]]
    TF_P = {}
    R["prompts"] = {b: {"n": len(p), "len": len(p[0])} for b, p in PROMPTS.items()}
    log("prompts", json.dumps(R["prompts"]), "| render", json.dumps({k: v for k, v in R["prompt_render"].items() if "text" not in k}))
    save()


# ---- RC2 CELL 4 (= the E23 step runner, verbatim)
WIN = float(os.environ.get("E23_WIN_SEC") or 90.0)   # 120 s FULL windows
EST_SCALE = float(os.environ.get("E23_EST_SCALE") or 1.0) if FAKE else 1.0   # local harness only: walk the whole order
SM_TF = MODE == "SMOKE"                                            # SMOKE teacher-forced: the 40k row only
ROWS = {}            # oracle greedy rows by key (same weights)
TF = {}              # teacher-forced rows by key
NTOK = 128 if MODE == "FULL" else 64
LIVE = {"rec": None}                 # rev 3c: windows land in the step record before every save (a hard kill keeps them)
COMPILE_ERR = re.compile(r"ninja: build stopped|Error building extension|error: command '.*(nvcc|g\+\+|gcc|c\+\+)' failed|"
                         r"GLIBCXX_[0-9.]+' not found|unsupported GNU version|fatal error: ")
OOM_ERR = re.compile(r"OutOfMemoryError|CUDA out of memory|out of memory")
XID = re.compile(r"NVRM: Xid|\bXid \d+|CUDA error: an illegal memory access|illegal memory access was encountered|"
                 r"unspecified launch failure|CUDA error: misaligned address")       # rev 3e: device faults, not any CUDA error


def step(sec, name, est_min, fn):
    """One measured step: skipped when the budget cannot hold it, recorded on any exception, never PASS by default."""
    rec = R["sections"].setdefault(sec, {}).setdefault(name, {"status": "NOT_RUN"})
    R["order"].append(f"{sec}/{name}")
    if ABORT["why"]:
        rec.update(status="SKIPPED_ABORT", why=ABORT["why"])
        log("---", sec, name, "SKIPPED_ABORT")
        save()
        return rec
    if left() < est_min * EST_SCALE * 60 + 300:
        rec.update(status="SKIPPED_BUDGET", left_min=round(left() / 60, 1), est_min=est_min)
        log("---", sec, name, "SKIPPED_BUDGET", f"(est {est_min} min, left {left() / 60:.0f} min)")
        save()
        return rec
    LIVE["rec"] = rec
    t = time.time()
    log(">>>", sec, name, f"(est {est_min} min, left {left() / 60:.0f} min)")
    try:
        out = fn() or {}
        rec.update(out)
        if rec.get("status") in (None, "NOT_RUN"):
            rec["status"] = "DONE"
    except Exception as e:
        rec.update(status="ERROR", error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2500:])
    rec["sec"] = round(time.time() - t, 1)
    LIVE["rec"] = None
    if ABORT["why"] and rec["status"] in ("DONE", "SERVER_DIED", "LAUNCH_FAILED", "BODY_ERROR"):
        rec["status"] = "PARTIAL_HARD_STOP"
    log("<<<", sec, name, rec.get("status"), f"{rec['sec']:.0f}s")
    save()
    return rec


def with_servers(servers, timeout, body):
    """Start, wait ready, record facts + memory, run body(servers), always kill; then the GPU-release gate on these
    servers' GPUs only; abort the lab if they stay held."""
    out = {"servers": {}}
    timeout = max(60, min(timeout, left() - 120))
    try:
        for s in servers:
            s.start()
        for s in servers:
            ok, why = s.wait_ready(timeout)
            out["servers"][s.name] = {"ready": ok, "why": why, **s.facts(), "argv_tail": s.argv[-44:], "gpus": s.gpus,
                                      "help_check": HELP.get(s.name)}
            if not ok:
                out["status"] = "LAUNCH_FAILED"
                out["servers"][s.name]["log_tail"] = s.log()[-4000:]
                _xl = [l for l in (s.log() or "").splitlines() if XID.search(l)][:8]   # rev 3e/3f: start-up / capture faults
                out["servers"][s.name].update(xid=bool(_xl), xid_lines=_xl)
                return out
        mem0 = L.gpu_mem()
        for s in servers:
            out["servers"][s.name].update(gpu_mem_after_ready=mem0, host_rss_mb=s.rss_mb())
        try:                             # rev 3: an exception in the body keeps everything measured so far
            out.update(body(servers) or {})
        except Exception as e:
            out.update(status="BODY_ERROR", body_error=f"{type(e).__name__}: {e}"[:800], body_tb=traceback.format_exc()[-2000:])
        for s in servers:
            out["servers"][s.name].update(s.facts(), gpu_mem_end=L.gpu_mem(), host_rss_mb_end=s.rss_mb(),
                                          alive_after_body=s.alive(), rc=None if s.alive() else s.proc.returncode,
                                          xid_lines=[l for l in (s.log() or "").splitlines() if XID.search(l)][:8])
            out["servers"][s.name]["xid"] = bool(out["servers"][s.name]["xid_lines"])
            if not s.alive():
                out["status"] = "SERVER_DIED"
                out["servers"][s.name]["log_tail"] = s.log()[-4000:]
        return out
    finally:
        _nb6_final = RERUN and bool((globals().get("NB5_REPORT") or {}).get("phases"))   # NB6 P2: the rerun's solve ran
        for s in servers:
            s.kill(fast=True) if _nb6_final else s.kill()       # NB6 P2: E18 R2 kill(fast=RERUN)
        gpus = ",".join(sorted({g for s in servers for g in s.gpus.split(",")}))
        rel = True if _nb6_final else L.wait_gpu_release(180, fake=bool(FAKE), gpus=gpus)   # NB6 P2: E18 has no wait after it
        if rel is False:                 # rev 3c: one more kill + 300 s before the lab aborts
            for s in servers:
                s.kill()
            rel = L.wait_gpu_release(300, fake=bool(FAKE), gpus=gpus)
        if rel is None and NVSMI_OK:     # nvidia-smi worked at start: a failure now is not "released"
            rel = False
        out["gpu_released"] = rel
        if rel is False:
            ABORT["why"] = f"GPUs {gpus} still held after {[s.name for s in servers]}"
            out["status"] = "FAIL_GPU_RELEASE"


def oracle(port, key=None, refs=()):
    """Greedy rows (2k, 2k, 8k) + one 40k row; saved under `key` (a target-only oracle) or compared with ROWS[ref]."""
    rows = L.greedy(port, ORACLE_P + LONG_P, n_tok=NTOK, timeout=max(120, min(900, left() / 6)))
    clean = all(not r.get("error") and r.get("tokens") and not L.degenerate(r.get("text")) for r in rows)
    if key and clean:            # rev 3d: an errored / empty / garbage reference is never stored (consumers: NO_ORACLE)
        ROWS[key] = rows
    elif key:
        R["errors"].append(f"oracle {key}: reference not stored (error / empty / garbage row)")
    res = {"garbage": [L.degenerate(r["text"]) for r in rows], "errors": [r["error"] for r in rows],
           "text_head": [r["text"][:120] for r in rows]}
    if not FAKE:                 # rev 3h: chat-path render vs the production xhigh render (recorded per arm)
        res["template_probe"] = L.template_probe(port, TPL_MSG, TPL_IDS)
    for ref in refs:
        res["vs_" + ref] = L.compare(ROWS.get(ref), rows) if ROWS.get(ref) else {"status": "INCONCLUSIVE_NO_ORACLE"}
    return res


def tforced(port, key=None, ref=None):
    """Teacher-forced 40k / 90k prompt logprobs (last 256 positions); saved under `key` or compared with TF[ref].
    rev 3c: on stock vLLM only on its own short launch (prompt logprobs = chunk x 248k-vocab logits: OOM at 0.90)."""
    rows = {b: L.teacher_forced(port, ids) for b, ids in TF_P.items() if not (SM_TF and b != "40k")}
    if key and L.tf_complete(rows):                                 # rev 3e: complete references only
        TF[key] = rows
    elif key:
        R["errors"].append(f"teacher-forced {key}: reference not stored (incomplete band)")
    if ref:
        return {b: L.compare_tf((TF.get(ref) or {}).get(b), rows[b]) for b in rows}
    return {b: {"n": (rows[b] or {}).get("n"), "error": (rows[b] or {}).get("error")} for b in rows}


def levels(port, band, lv, server=None, **kw):
    """rev 3c: the guard includes admission (prefill) time; a band stops once KV-bound (the fit is recorded); a dead
    server stops the band; every window is written into the live step record before save()."""
    out, fit = {}, None
    for n in lv:
        k = f"{band}@{n}"
        need = WIN + 90 + min(3600, n * BANDS[band] / 1500)
        if ABORT["why"] or left() < need + 120:
            out[k] = {"status": "SKIPPED_BUDGET", "need_sec": round(need)}
        elif fit is not None:
            out[k] = {"status": "SKIPPED_KV_BOUND", "streams_fit": fit}
        elif server is not None and not server.alive():
            out[k] = {"status": "SKIPPED_SERVER_DEAD"}
        else:
            try:                         # rev 3: one failing level never loses the others
                out[k] = L.steady(port, PROMPTS[band], n, secs=WIN, max_new=MAX_NEW[band], **kw)
            except Exception as e:
                out[k] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:300]}
            rm = out[k].get("running_max")
            if out[k].get("status") == "CAPACITY_LIMITED" or (rm is not None and 0 < rm < n and
                                                              str(out[k].get("status")).startswith("INCONCLUSIVE")):
                fit = rm
        if LIVE["rec"] is not None:
            LIVE["rec"].setdefault("live_windows", {})[k] = out[k]
        save()
    return out


# ---- RC2 CELL 5 checks (verbatim)
def kv_census(s):
    log_t = s.log() or ""
    lines = log_t.splitlines()
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log_t)
    conc = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", log_t)
    pick = lambda rx, n=24: [l[:240] for l in lines if re.search(rx, l, re.I)][:n]
    return {"kv_tokens": [int(x.replace(",", "")) for x in kv][:2], "max_concurrency": conc[:2],
            "scale_lines": pick(r"K/V scale|kv scale|scaling factor|DEFAULT 1\.0|static K/V"),
            "engine_lines": pick(r"WNA16 MoE backend|MarlinExperts|PLE offload|engram|cudagraph_mode|CUDAGraphMode|PIECEWISE"),
            "startup_lines": pick(r"Loading weights took|Model loading took|torch\.compile takes|torch\.compile took|"
                                  r"Graph capturing finished|init engine|PLE.*ready|Available KV cache memory"),
            "oom_lines": [l[:240] for l in lines if OOM_ERR.search(l)][:6], "host_mem": host_mem()}


def sampled_check(s):
    """RC1 died at its first sampled request (FlashInfer sampling JIT: ld cannot find -lcuda): one short sampled request
    right after ready, then the engine must still be alive."""
    g = L.Stream(s.port, PROMPTS["2k"][1], 64, temperature=TEMP, seed=5)
    g.start()
    g.join(900)
    if g.is_alive():
        g.stop()
        g.join(10)
    r = g.row()
    return {"tokens": len(r["tokens"]), "error": r["error"], "finish": r["finish_reason"], "alive_after": s.alive(),
            "garbage": L.degenerate(r["text"]), "ok": len(r["tokens"]) == 64 and not r["error"] and s.alive()}


def prefill_probe(port):
    """Time to first token of one cold 40k prompt (1 new token): the measured prefill rate the ladder admission uses."""
    ids = PROMPTS["40k"][7]
    g = L.Stream(port, ids, 1, temperature=0.0)
    g.start()
    g.join(1800)
    if g.is_alive():
        g.stop()
        g.join(10)
    sec = (g.t_first or g.t_end or time.perf_counter()) - (g.t_start or time.perf_counter())
    ok = not g.err and g.t_first is not None
    if ok:
        PREFILL.update(tok_s=round(len(ids) / max(1e-3, sec), 1), measured=True)
    return {"prompt": len(ids), "ttft_sec": round(sec, 1), "tok_s": round(len(ids) / max(1e-3, sec), 1), "error": g.err, "ok": ok}




# ---- E21 engine functions
# ---- E21 donor, verbatim: _e12v_request_json
def _e12v_request_json(url, payload=None, timeout=30):   # = Duck's request_json
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = _urlreq.Request(url, data=data, headers={"Content-Type": "application/json"})
    with _urlreq.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ---- E21 donor, verbatim: _E12V_PROM_RE
_E12V_PROM_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


# ---- E21 donor, verbatim: _E12V_ALIAS
_E12V_ALIAS = {"vllm:num_requests_running": "num_running_reqs", "vllm:num_requests_waiting": "num_queue_reqs",
               "vllm:kv_cache_usage_perc": "token_usage"}


# ---- E21 donor, verbatim: metrics_snapshot
def metrics_snapshot():
    if not ENGINE_READY:
        return []
    try:
        txt = _urlreq.urlopen(SGL_URL + "/metrics", timeout=10).read().decode("utf-8", "replace")
        rows = []
        for line in txt.splitlines():
            m = _E12V_PROM_RE.match(line) if line and not line.startswith("#") else None
            if not m or m.group(1).endswith(("_bucket", "_created")):
                continue
            name = m.group(1)
            if name in _E12V_ALIAS or any(k in name for k in ("prefix_cache", "spec_decode", "generation_tokens_total",
                                                               "prompt_tokens_total", "num_preemptions")):
                try:
                    rows.append({"name": _E12V_ALIAS.get(name, name), "labels": m.group(2), "value": float(m.group(3))})
                except ValueError:
                    pass
        return rows
    except Exception as e:
        return [{"error": repr(e)[:300]}]


# ---- E21 donor, verbatim: _E12V_STAT_RE
_E12V_STAT_RE = re.compile(r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
                           r"Running: (\d+) reqs, Waiting: (\d+) reqs(?:, Deferred: \d+ reqs)?(?:, KV fetch: [^%]*?completed waiting)?"
                           r"(?:, Preemptions: (\d+))?, GPU KV cache usage: ([\d.]+)%(?:, Prefix cache hit rate: ([\d.]+)%)?")


# ---- E21 donor, verbatim: e21_stat_rows
def e21_stat_rows(text):
    """vLLM's 10-s stats lines as dicts (both versions, every optional field)."""
    out = []
    for m in _E12V_STAT_RE.finditer(text):
        g = m.groups()
        out.append({"prompt_tps": float(g[0]), "gen_tps": float(g[1]), "running": int(g[2]), "waiting": int(g[3]),
                    "preempt": int(g[4]) if g[4] is not None else 0, "kv_pct": float(g[5]),
                    "prefix_pct": float(g[6]) if g[6] is not None else None})
    return out


# ---- E21 donor, verbatim: e11_decode_summary
def e11_decode_summary():
    """E12-V's stats-line summary on the rev 2 regex (preemption / deferred lines included) + the counter snapshot."""
    if E11_SERVER is None:
        return None
    txt = E11_SERVER.log_text()
    rows = e21_stat_rows(txt)
    med = lambda v: round(_statistics.median(v), 1) if v else None
    acc = re.findall(r"Mean acceptance length: ([\d.]+)", txt)
    pre = [r["prefix_pct"] for r in rows if r["prefix_pct"] is not None]
    out = {"n_lines": len(rows),
           "running_hist": {str(k): sum(1 for r in rows if r["running"] == k) for k in sorted({r["running"] for r in rows})},
           "gen_tok_s_median_at_concurrency": med([r["gen_tps"] for r in rows if r["running"] >= E12_VLLM_CONCURRENCY]),
           "gen_tok_s_median_all_busy": med([r["gen_tps"] for r in rows if r["running"] > 0]),
           "waiting_max": max((r["waiting"] for r in rows), default=None),
           "kv_usage_pct_max": max((r["kv_pct"] for r in rows), default=None),
           "prefix_cache_hit_pct_last_rolling": pre[-1] if pre else None,
           "preemptions_in_stats_lines": sum(r["preempt"] for r in rows),
           "spec_mean_acceptance_median": med([float(a) for a in acc]),
           "accept_len_median": med([float(a) for a in acc]),      # the key CELL 7 reads
           "counters": ENGINE_INFO.get("final_counters") or e21_counters()}
    try:
        write_json_atomic(OUT / "sglang_decode_summary.json", out)   # bundled under this name by CELL 7
    except Exception:
        pass
    return out


# ---- E21 donor, verbatim: _E21_COUNTERS
_E21_COUNTERS = ("vllm:num_preemptions_total", "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
                 "vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:request_success_total",
                 "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_drafts_total")


# ---- E21 donor, verbatim: e21_counters
def e21_counters():
    """VAL_engine_A2 M3 / VAL_api_D2 F6-F7: the server's own counters (preemptions, whole-run prefix hits, tokens), summed
    over label rows. Taken before every kill (a dead server loses them)."""
    try:
        rows = e21_prom()
    except Exception as e:
        return {"error": repr(e)[:200]}
    out = {}
    for r in rows:
        if r.name in _E21_COUNTERS:
            out[r.name] = out.get(r.name, 0.0) + float(r.value)
    q, h = out.get("vllm:prefix_cache_queries_total"), out.get("vllm:prefix_cache_hits_total")
    out["prefix_hit_pct_whole_run"] = round(100 * h / q, 2) if q else None
    return out


# ---- E21 donor, verbatim: e21_snapshot_counters
def e21_snapshot_counters():
    if E11_SERVER is not None and E11_SERVER.alive():
        c = e21_counters()
        if "error" not in c:
            ENGINE_INFO["final_counters"] = c


# ---- E21 donor, verbatim: E21_CONTENT_PROBES
E21_CONTENT_PROBES = [("Reply with exactly one number: 7 * 6 =", "42", "number"),          # audit 1
                      ("Repeat exactly, nothing else: ARC-GRID-17", "ARC-GRID-17", "copy"),   # audit 2
                      ("Complete the sequence with the next number only: 2, 4, 6, 8,", "10", "number")]


# ---- E21 donor, verbatim: E21_NUMBER_WORDS
E21_NUMBER_WORDS = {"4": "four", "10": "ten", "42": "forty-two"}


# ---- E21 donor, verbatim: e21_garbage
def e21_garbage(gen, limit=80):
    """rev 2i: empty, too long for the question, or < 90 % ASCII (the 0.19 INT4 multilingual token soup)."""
    g = (gen or "").strip()
    return not g or len(g) > limit or sum(ch.isascii() for ch in g) < 0.9 * len(g)


# ---- E21 donor, verbatim: e21_content_ok
def e21_content_ok(gen, want, kind, limit=80):
    """rev 2i (GPT INT4 audits 1+2): a clean reply that holds the answer (number: the last number in it, or its word)."""
    g = (gen or "").strip()
    if e21_garbage(g, limit):
        return False
    if kind == "number":
        return re.findall(r"-?\d+", g)[-1:] == [want] or E21_NUMBER_WORDS.get(want, "\0") in g.lower()
    return want in g


# ---- E21 donor, verbatim: e21_content_probes
def e21_content_probes(served):
    """rev 2i: three more greedy, thinking-off probes after the 2 + 2 smoke. A garbage reply (or one cut at the limit) is
    fatal in every mode; a clean but wrong answer is fatal in commit / smoke and a recorded warning in the hidden rerun,
    so a formatting surprise can never leave the rerun without an engine."""
    res = []
    for prompt, want, kind in E21_CONTENT_PROBES:
        r = _e12v_request_json(SGL_URL + "/v1/chat/completions", payload={
            "model": served, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
        c = r["choices"][0]
        gen = (c["message"].get("content") or "").strip()
        res.append({"want": want, "generated": gen[:120], "finish": c.get("finish_reason"),
                    "clean": not e21_garbage(gen) and c.get("finish_reason") != "length",
                    "ok": e21_content_ok(gen, want, kind) and c.get("finish_reason") != "length"})
    if not all(x["clean"] for x in res):
        raise RuntimeError(f"content probes: garbage reply {json.dumps(res, ensure_ascii=True)[:600]}")
    if not all(x["ok"] for x in res):
        if not RERUN:
            raise RuntimeError(f"content probes: wrong answer {json.dumps(res, ensure_ascii=True)[:600]}")
        print("E21 WARNING (rerun, not fatal): content probes wrong answer", json.dumps(res, ensure_ascii=True)[:300], flush=True)
    return res


# ---- E21 donor, verbatim: e21_prom
def e21_prom():
    """The server's /metrics as rows with .name / .labels / .value (the e11_prom() shape CELL 6 reads)."""
    from types import SimpleNamespace
    txt = _urlreq.urlopen(SGL_URL + "/metrics", timeout=10).read().decode("utf-8", "replace")
    rows = []
    for line in txt.splitlines():
        m = _E12V_PROM_RE.match(line) if line and not line.startswith("#") else None
        if not m:
            continue
        try:
            rows.append(SimpleNamespace(name=m.group(1), labels=m.group(2) or "", value=float(m.group(3))))
        except ValueError:
            pass
    return rows


# ---- E21 donor, verbatim: E21_PROBE_TOOLS
E21_PROBE_TOOLS = [{"type": "function", "function": {"name": "python", "description": E12A.ARC2_TOOL_DESCRIPTION,
                    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": (
                        "Python code to run. The snippet is ephemeral and is not saved across tool calls.")}},
                        "required": ["code"]}}}]


# ---- E21 donor, verbatim: e21_contract_probe
def e21_contract_probe(served):
    """rev 2 (VAL_api_D1 F2, VAL_engine_A1 M1): the Duck contract on this server, before a 5.5-h half. Turn 1: tools +
    thinking (reasoning_effort low) asking for one python call. Hard failures: an HTTP error, tool-call arguments that are
    not a JSON object with a string 'code', think / tool-call markup leaking into content, a degenerate reply. If a call
    came back, turn 2 sends it as history (reasoning under 'reasoning', tool result) and the prompt must grow by about
    turn 1's prompt + completion (a dropped reasoning shows up as a shortfall, the E12-V check). No call within the
    budget (thinking ran long) is inconclusive, not a failure."""
    out = {"ok": False}
    kw = {"enable_thinking": True, "preserve_thinking": True, "reasoning_effort": "low"}
    msgs = [{"role": "user", "content": "Use the python tool exactly once to run print(6 * 7). Keep your thinking very short."}]
    r1 = _e12v_request_json(SGL_URL + "/v1/chat/completions", payload={
        "model": served, "messages": msgs, "tools": E21_PROBE_TOOLS, "tool_choice": "auto", "temperature": 0.0,
        "max_tokens": 3072, "chat_template_kwargs": kw}, timeout=600)
    m, u = r1["choices"][0]["message"], r1.get("usage") or {}
    reas, cont, calls = m.get("reasoning") or m.get("reasoning_content") or "", m.get("content") or "", m.get("tool_calls") or []
    out.update(finish=r1["choices"][0].get("finish_reason"), reasoning_chars=len(reas), content=cont[:200],
               tool_calls=len(calls), prompt=u.get("prompt_tokens"), completion=u.get("completion_tokens"))
    bad = [k for k in ("<think>", "</think>", "<tool_call>", "<function=") if k in cont]
    if bad:
        raise RuntimeError(f"contract: markup in content {bad}: {cont[:300]!r}")
    txt = (reas + cont).strip()
    if len(txt) >= 40 and len(set(txt.replace(" ", "").replace("\n", ""))) <= 2:
        raise RuntimeError(f"contract: degenerate reply {txt[:80]!r}")
    for c in calls:
        a = json.loads(c["function"]["arguments"])
        if not (isinstance(a, dict) and isinstance(a.get("code"), str) and a["code"].strip()):
            raise RuntimeError(f"contract: tool-call arguments {c['function']['arguments'][:200]!r}")
    if not calls:
        out.update(turn2="inconclusive: no tool call in turn 1")
        e21_soft(out, "cache", lambda: e21_cache_probe(served))
        out["ok"] = True
        return out
    hist = msgs + [{"role": "assistant", "reasoning": reas, "tool_calls": calls},
                   {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "content": "42"}]
    r2 = _e12v_request_json(SGL_URL + "/v1/chat/completions", payload={
        "model": served, "messages": hist, "tools": E21_PROBE_TOOLS, "tool_choice": "auto", "temperature": 0.0,
        "max_tokens": 512, "chat_template_kwargs": kw}, timeout=600)
    p2 = (r2.get("usage") or {}).get("prompt_tokens")
    short = (int(out["prompt"]) + int(out["completion"])) - int(p2)
    out.update(turn2_prompt=p2, turn2_shortfall=short)
    if reas and short > max(128, len(reas) // 6):
        e21_soft(out, "history", lambda: (_ for _ in ()).throw(
            RuntimeError(f"contract: history prompt {p2} short by {short} tokens (reasoning dropped?)")))
    e21_soft(out, "cache", lambda: e21_cache_probe(served))
    out["ok"] = True
    return out


# ---- E21 donor, verbatim: e21_soft
def e21_soft(out, key, fn):
    """rev 2f: a check that is fatal in commit / smoke (falls back) but only a recorded warning in the hidden rerun."""
    try:
        out[key] = fn()
    except Exception as e:
        if not RERUN:
            raise
        out.setdefault("warnings", []).append(f"{key}: {e}"[:600])
        print("E21 WARNING (rerun, not fatal):", key, str(e)[:300], flush=True)


# ---- E21 donor, verbatim: e21_cache_probe
def e21_cache_probe(served):
    """rev 2b (VAL_api_D2 F5): the align-mode prefix-cache path that serves 67-94 % of every later Duck prompt. A shared
    ~5,000-token prefix with a marker per request: the cold request, then the same request again (it must hit the cache,
    >= 1 block) at temperature 0, thinking off; both replies must contain the request's own marker and be equal."""
    import random as _rnd
    _r7 = _rnd.Random(7)                       # rev 2f (VAL2_config_smoke F6): one RNG (the old code repeated one digit)
    rows = " ".join(str(_r7.randrange(10)) for _ in range(6000))
    res = []
    for i in range(2):
        mark = f"QZ{_rnd.randrange(10 ** 6, 10 ** 7)}K{i}"
        msgs = [{"role": "user", "content": "Grid digits for context: " + rows + f"\n\nReply with exactly this code: {mark}"}]
        body = {"model": served, "messages": msgs, "temperature": 0.0, "max_tokens": 96,
                "chat_template_kwargs": {"enable_thinking": False}}
        r1 = _e12v_request_json(SGL_URL + "/v1/chat/completions", payload=body, timeout=300)
        r2 = _e12v_request_json(SGL_URL + "/v1/chat/completions", payload=body, timeout=300)
        g1 = r1["choices"][0]["message"].get("content") or ""
        g2 = r2["choices"][0]["message"].get("content") or ""
        u2 = r2.get("usage") or {}
        c2 = ((u2.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        res.append({"prompt": u2.get("prompt_tokens"), "cached_repeat": c2, "own_marker": mark in g1 and mark in g2,
                    "equal": g1 == g2})
    if not all(x["own_marker"] and x["equal"] and x["cached_repeat"] >= 1024 for x in res):
        raise RuntimeError(f"cache probe failed: {json.dumps(res)}")
    return res

# ==================================================================== NB5 glue (new)
TEMP = 0.6                                       # RC2's sampled-check temperature (the Duck itself runs at T 1.0)
PREFILL = {"tok_s": 700.0, "measured": False}
NB5_SERVERS = []                                 # with_servers' list; a relaunch replaces the element (same name)
NB5_BOOTS = []
NB5_RC2_FP8_ARGV_TAIL = json.loads(r"""["127.0.0.1", "--port", "8001", "--tensor-parallel-size", "2", "--pipeline-parallel-size", "2", "--enable-expert-parallel", "--disable-custom-all-reduce", "--speculative-config", "{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"probabilistic\"}", "--max-model-len", "262144", "--max-num-seqs", "8", "--max-num-batched-tokens", "1024", "--gpu-memory-utilization", "0.96", "--kv-cache-memory-bytes", "2479838700", "--kv-cache-dtype", "fp8_e4m3", "--limit-mm-per-prompt", "{\"image\":42,\"video\":0}", "--mm-processor-cache-gb", "0", "--mm-processor-kwargs", "{\"size\":{\"longest_edge\":8388608,\"shortest_edge\":65536}}", "--compilation-config", "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[1,2,3,4,6,8,12,16,24,32]}", "--enable-prefix-caching", "--mamba-cache-mode", "align", "--enable-prompt-tokens-details", "--no-enable-flashinfer-autotune", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3", "--override-generation-config", "{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0,\"repetition_penalty\":1.0}", "--default-chat-template-kwargs", "{\"enable_thinking\": true, \"reasoning_effort\": \"medium\"}", "--shutdown-timeout", "60", "--block-size", "4096", "--use-replayssm", "--prefix-match-unit", "64", "--served-model-name", "q27", "--max-model-len", "131072", "--port", "8001", "--default-chat-template-kwargs", "{\"enable_thinking\": true, \"reasoning_effort\": \"xhigh\"}"]""")
NB5_CAPACITY = {}
NB5_FINAL_KILL = False
NB5_BOOT_KV = NB5_KV_DTYPE
NB5_BOOT_DECODE = {"tok_s": None}               # this boot's pure decode rate (the token-weighted duty factor)
NB5_F_MARGIN = 0                                 # PLAN v3.6: + blocks per admission after a real preemption (carried)


class NB5Server(MServer):
    """The E21 cells call log_text() and kill(fast=...); RC2's MServer has log() and kill()."""
    def log_text(self):
        return self.log()

    def kill(self, fast=False, wait=None):
        super().kill()


MServer = NB5Server


def E11Server(*_a, **_k):
    """NB5-2: E21 CELL 6/7's E12 DFlash parity relaunch (P['parity']) names E11Server; no NB5 profile sets 'parity'
    (asserted in CELL 1), so that path is unreachable. Defined explicitly so it can never become a NameError."""
    raise RuntimeError("NB5-2: the E12 DFlash parity relaunch (E11Server) is not part of NB5")


# ---- vLLM #60490 backport (R2 lever a; NB1's strict applier + the sha-pinned patch, verified before and after)
P60490 = "# vLLM PR #60490 (issue #60489) rebased onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5,\n# tree 07c16a79d077f1e06c19d0eea84efd64f0b63540 = the tree the Kaggle payload installs, install_merlin.log).\n# Same code as upstream; only the import context differs (Merlin's import block). One marker comment added.\n# pre-patch sha256  a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653\n# post-patch sha256 88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b\n--- a/vllm/models/qwen4_exp/nvidia/qsa.py\n+++ b/vllm/models/qwen4_exp/nvidia/qsa.py\n@@ -18,8 +18,9 @@\n from torch import nn\n from transformers import Qwen4ExpTextConfig\n \n+import vllm.envs as envs\n from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n-from vllm.config import VllmConfig\n+from vllm.config import VllmConfig, get_current_vllm_config_or_none\n from vllm.config.cache import CacheConfig, CacheDType\n from vllm.distributed import get_tensor_model_parallel_world_size\n from vllm.forward_context import get_forward_context\n@@ -45,6 +46,7 @@\n )\n from vllm.platforms import current_platform\n from vllm.platforms.interface import DeviceCapability\n+from vllm.utils.math_utils import cdiv\n from vllm.utils.torch_utils import (\n     LayerNameType,\n     _encode_layer_name,\n@@ -75,6 +77,7 @@\n from . import model\n from .indexer_qsa import QSAIndexer\n from .ops.qsa import qsa_e4m3_uses_native_cast\n+from .ops.qsa_indexer import _TOPK_WORKSPACE_BYTES\n \n logger = init_logger(__name__)\n \n@@ -782,6 +785,31 @@\n         if isinstance(metadata, list):\n             metadata = metadata[0]\n         if not isinstance(metadata, dict):\n+            # E25 NB1: backport of vLLM PR #60490 (issue #60489).\n+            # Profiling/dummy run: the indexer allocates its selection scratch\n+            # on demand, so touch its peak here for the memory profiler (as\n+            # sparse_attn_indexer does): the prefill logits budget or the\n+            # worst-case decode logits [decode tokens, compressed columns],\n+            # plus the top-k workspace.\n+            logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024\n+            cfg = get_current_vllm_config_or_none()\n+            if cfg is not None:\n+                spec = cfg.speculative_config\n+                num_spec = spec.num_speculative_tokens if spec is not None else 0\n+                decode_tokens = min(\n+                    cfg.scheduler_config.max_num_seqs * (num_spec + 1),\n+                    cfg.scheduler_config.max_num_batched_tokens,\n+                )\n+                block_size = cfg.cache_config.block_size\n+                columns = cdiv(cfg.model_config.max_model_len, block_size) * cdiv(\n+                    block_size, self.indexer.compress_ratio\n+                )\n+                logits_bytes = max(logits_bytes, decode_tokens * columns * 4)\n+            _ = torch.empty(\n+                logits_bytes + _TOPK_WORKSPACE_BYTES,\n+                dtype=torch.uint8,\n+                device=output.device,\n+            )\n             output.zero_()\n             return\n         main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])\n"
P60490_REL = "vllm/models/qwen4_exp/nvidia/qsa.py"
P60490_PRE, P60490_POST = "a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653", "88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b"
P60490_MARKER = "# E25 NB1: backport of vLLM PR #60490 (issue #60489)."
def apply_unified(text, patch):
    """Strict unified-diff applier (no fuzz, no offset): every context / removed line must match at its stated line,
    else ValueError. Lines before the first hunk (comments, ---/+++ headers) are ignored."""
    src = text.splitlines(True)
    hunks, cur = [], None
    for ln in patch.splitlines(True):
        if ln.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
            if not m:
                raise ValueError("bad hunk header: " + ln[:80])
            cur = [int(m.group(1)), int(m.group(2) or 1), []]
            hunks.append(cur)
        elif cur is not None and ln[:1] in (" ", "-", "+"):
            cur[2].append(ln)
        elif cur is not None and ln.startswith("\\"):
            raise ValueError("'No newline at end of file' hunks are not supported")
    if not hunks:
        raise ValueError("patch has no hunks")
    out, pos = [], 0
    for start, n_old, body in hunks:
        i = start - 1
        if i < pos:
            raise ValueError(f"overlapping hunk at line {start}")
        out += src[pos:i]
        old_seen = 0
        for ln in body:
            tag, s = ln[0], ln[1:]
            if tag in " -":
                if i >= len(src) or src[i] != s:
                    raise ValueError(f"context mismatch at line {i + 1}: {s[:60]!r}")
                i += 1
                old_seen += 1
            if tag in " +":
                out.append(s)
        if old_seen != n_old:
            raise ValueError(f"hunk at line {start}: {old_seen} old lines, header says {n_old}")
        pos = i
    return "".join(out + src[pos:])


def nb5_apply_60490():
    """After merlin_install, before the first boot (NB1 apply_60490 without the worker extension): the installed tree's
    qsa.py must have Merlin ad42e37's pre-sha; the strict applier's result must have the post-sha; atomic replace; marker,
    py_compile with the Merlin venv and the vllm origin in the tree are checked. Any miss raises (the file is untouched
    unless the post-sha matched): the caller then keeps the RC2 pin (no ladder above it)."""
    if FAKE:
        return {"ok": None, "fake": True, "why": "local harness: no Merlin tree"}
    eng = merlin_install()
    tree, py = Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"
    f = tree / P60490_REL
    raw = f.read_bytes()
    h0 = hashlib.sha256(raw).hexdigest()
    rec = {"file": str(f), "sha_before": h0, "pre_expected": P60490_PRE, "post_expected": P60490_POST}
    if h0 == P60490_POST:
        rec["applied"] = "already_applied"
    elif h0 != P60490_PRE:
        raise RuntimeError(f"PATCH_60490_PRE_SHA_MISMATCH: {h0} (expected Merlin ad42e37 {P60490_PRE})")
    else:
        new = apply_unified(raw.decode("utf-8"), P60490).encode("utf-8")
        if hashlib.sha256(new).hexdigest() != P60490_POST:
            raise RuntimeError("PATCH_60490_POST_SHA_MISMATCH: " + hashlib.sha256(new).hexdigest())
        tmp = f.with_name(f.name + ".nb5tmp")
        tmp.write_bytes(new)
        os.replace(tmp, f)
        rec["applied"] = "applied"
    for pyc in (f.parent / "__pycache__").glob("qsa.*.pyc"):
        pyc.unlink()
    data = f.read_bytes()
    rec.update(sha_after=hashlib.sha256(data).hexdigest(), marker=P60490_MARKER.encode() in data)
    cp = subprocess.run([py, "-c", "import py_compile, sys\nfor f in sys.argv[1:]: py_compile.compile(f, doraise=True)", str(f)],
                        capture_output=True, text=True, timeout=300)
    rec["py_compile"] = {"rc": cp.returncode, "err": cp.stderr[-400:]}
    envp = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    envp["PYTHONPATH"] = str(tree)
    cp2 = subprocess.run([py, "-c", "import importlib.util as u; print(u.find_spec('vllm').origin)"], capture_output=True,
                         text=True, timeout=300, env=envp)
    rec["vllm_origin"] = cp2.stdout.strip()[-400:]
    origin_ok = rec["vllm_origin"].startswith(str(tree))
    if rec["sha_after"] != P60490_POST or not rec["marker"] or cp.returncode or not origin_ok:
        raise RuntimeError("PATCH_60490_VERIFY_FAILED: " + json.dumps({k: rec.get(k) for k in ("sha_after", "marker", "py_compile",
                                                                                              "vllm_origin")})[:600])
    rec["ok"] = True
    return rec


def nb5_host_floor():
    """KIT-6: the host RAM floor. Real runs: always Merlin's 69 GiB + 4 GiB. FAKE mode only (the CPU-kit fake server;
    the Kaggle CPU kit has a 30 GiB cgroup): NB5_FAKE_HOST_FLOOR_MIB overrides it. Ignored without FAKE."""
    v = os.environ.get("NB5_FAKE_HOST_FLOOR_MIB")
    if FAKE and v:
        try:
            return int(v), "NB5_FAKE_HOST_FLOOR_MIB (fake server only)"
        except ValueError:
            pass
    return NB5_RAM_NEED_MIB, "Merlin 69 GiB + 4 GiB margin"


def e11_kill_server(wait=True):
    """NB5: CELL 7 calls this after every phase; the server must live on for the next phase (the boot's with_servers
    kills it). Counters are snapshot first (E21)."""
    e21_snapshot_counters()
    if NB5_FINAL_KILL and E11_SERVER is not None:
        E11_SERVER.kill(fast=not wait)


def nb5_gate_kv(kv_server):
    """Duck's KV gate value (E15B G4-1): the reservation's projected fraction. Merlin's raw usage gauge also counts the
    retained prefixes of finished turns (RC2: 0.30 with one 47k request; 0.97 at 8k x 8 with 0 preemptions), so the
    raw gauge would hold admissions that fit; it is recorded (kv_usage_server) and drives the live hold instead."""
    return NB5_RES.frac() if NB5_RES is not None else kv_server


def nb5_metrics_text():
    try:
        return _urlreq.urlopen(SGL_URL + "/metrics", timeout=15).read().decode("utf-8", "replace")
    except Exception:
        return ""


def nb5_host_gate(tag):
    """Spec section 2: drop the page cache of the checkpoint files (merged symlinks -> Model mounts), then Merlin's
    floor (max - (current - inactive_file) >= 69 + 4 GiB). Fail closed on a measured shortfall; recorded either way."""
    roots = [CKPT] + [d for d, _ in [v for v in (PARTS or {}).values() if v]] + ([PLE_DIR] if PLE_DIR else [])
    _need, _src = nb5_host_floor()
    rec = NB5L.ram_gate(host_mem, _need, drop_fn=lambda: NB5L.drop_page_cache(roots),
                        wait_sec=0 if FAKE else 120)
    rec.update(tag=tag, floor_source=_src)
    R["setup"].setdefault("host_gates", []).append(rec)
    save()
    return rec


def nb5_capacity(srv, kv):
    """Capacity from measured inputs (addendum 2): the server's caps (max_num_seqs, block, max-concurrency line), the
    usable pool from cache_config_info (live) -> NB1's table -> the gauge quantization -> the PLACEHOLDER; F = max(NB1's F
    or the PLACEHOLDER, F_log). Every source is recorded."""
    caps = NB5L.server_caps(srv.log())
    mt = nb5_metrics_text()
    live = NB5L.pool_from_metrics(mt)
    nb1 = (NB5_NB1_CAPACITY or {}).get(kv) or {}
    if live:
        pool, src = live["usable"], "cache_config_info"
    elif nb1.get("pool_blocks"):
        pool, src = int(nb1["pool_blocks"]), "nb1_table"
    else:
        pool, src = NB5_POOL_PLACEHOLDER[kv], "PLACEHOLDER_rc2"
    f_par = int(nb1.get("f_blocks") or NB5_F_PLACEHOLDER)
    f_src = "nb1_table" if nb1.get("f_blocks") else "PLACEHOLDER_rc2_measured_6"
    block = caps.get("block_size") or 4096
    plan = NB5L.capacity_plan(pool, block, NB5_ATTEMPT, NB5_PROMPT_ALLOW, NB5_TOOL_ALLOW, MAX_LEN, f_par, NB5_GATE,
                              seqs=caps.get("max_num_seqs"), conc_tokens=caps.get("conc_tokens"), conc_x=caps.get("conc_x"),
                              f_rule=NB5_F_RULE)
    plan.update(pool_source=src, f_source=("f_log (server max-concurrency line)" if NB5_F_RULE == "measured" and plan.get("f_log")
                                           is not None else f_src), caps=caps, kv_dtype=kv, live_pool=live, nb1=nb1)
    if NB5_ADMIT_MODE == "slots":                  # R2 (user decision): workers = max_num_seqs = the 64k-context rule
        _wr = NB5L.workers_rule(pool, block, plan["f_eff"])
        _seq = caps.get("max_num_seqs")
        _w = min(_wr, int(_seq)) if _seq else _wr
        plan["r2_workers"] = {"rule": _wr, "rule_text": "max(1, floor(pool / (ceil(65536 / B) + F)))", "pool": pool,
                              "f_eff": plan["f_eff"], "launch_max_num_seqs": _seq, "workers": _w,
                              "rule_vs_launch": None if not _seq else ("EQUAL" if _wr == int(_seq) else
                                                                       ("RULE_BELOW_LAUNCH" if _wr < int(_seq) else "RULE_ABOVE_LAUNCH")),
                              "projected_120k_workers": plan["workers"],
                              "captures_cover_workers": NB5L.captures_cover(NB5_CAPS16 if P.get("caps16") else [32], _w)}
        if plan["workers"] >= 1:                   # C-8 stays: no attempt kind fits the pool -> NO_GO (workers 0)
            plan["workers"] = _w
    if caps.get("max_num_seqs") is None:
        plan["warning"] = "max_num_seqs not found in the server log: workers = pool streams"
    plan["other_dtype_plan"] = {k: NB5L.capacity_plan(NB5_POOL_PLACEHOLDER[k] if not (NB5_NB1_CAPACITY.get(k) or {}).get("pool_blocks")
                                                      else int(NB5_NB1_CAPACITY[k]["pool_blocks"]), block, NB5_ATTEMPT,
                                                      NB5_PROMPT_ALLOW, NB5_TOOL_ALLOW, MAX_LEN, f_par, NB5_GATE,
                                                      seqs=8 if k == "fp8" else 3)["per_attempt"]
                                for k in ("fp8", "bf16")}
    return plan


def nb5_chat(body, timeout=600):
    return _e12v_request_json(SGL_URL + "/v1/chat/completions", payload=body, timeout=timeout)


def nb5_complete(ids, max_tokens, timeout=600, **extra):
    body = dict({"model": E21_SERVED[NB5_KV_DTYPE], "prompt": list(ids), "max_tokens": int(max_tokens), "temperature": 0.0,
                 "stream": False}, **extra)
    return _e12v_request_json(SGL_URL + "/v1/completions", payload=body, timeout=timeout)


def nb5_commit_probe(served):
    """W3 GPU check before the Duck (addendum 1 + 2): a real tool turn, then the commit request rendered by the server
    (/tokenize) -> tail check -> render parity with the chat path (prompt_tokens) -> variant C via /v1/completions (cache
    hit, tool call, think markup, finish stop with a closed think block) -> the SAME ids through the chat path with
    continue_final_message (the server's qwen3 reasoning + qwen3_coder tool parsers) -> parser agreement -> the same C
    ids with a cache_salt (cold prefix) vs warm: recurrent-state correctness under prefix reuse (greedy)."""
    out = {"c_usable": False}
    kw = {"enable_thinking": True, "reasoning_effort": "low"}
    sys_ = {"role": "system", "content": "You are a careful assistant. When asked, use the python tool."}
    u1 = {"role": "user", "content": "NB5 commit probe: use the python tool exactly once to run print(6 * 7). Keep your thinking very short."}
    r1 = nb5_chat({"model": served, "messages": [sys_, u1], "tools": E21_PROBE_TOOLS, "tool_choice": "auto",
                   "temperature": 0.0, "max_tokens": 3072, "chat_template_kwargs": kw})
    m1 = r1["choices"][0]["message"]
    calls = m1.get("tool_calls") or []
    out["turn1"] = {"tool_calls": len(calls), "finish": r1["choices"][0].get("finish_reason"),
                    "usage": r1.get("usage")}
    if not calls:          # thinking ran long: our own scripted turn (the render / cache / parser checks still apply)
        calls = [{"id": "chatcmpl-tool-nb5probe", "type": "function",
                  "function": {"name": "python", "arguments": json.dumps({"code": "print(6 * 7)"})}}]
        m1 = {"content": "", "reasoning": "I will run print(6 * 7) with the python tool."}
        out["turn1"]["scripted"] = True
    hist = [sys_, u1, {"role": "assistant", "content": m1.get("content") or "", "reasoning": m1.get("reasoning") or "",
                       "tool_calls": calls},
            {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "content": "42"},
            {"role": "user", "content": "Your budget is almost spent. In this reply, call python once with print(7 * 6) and nothing else."}]
    payload = {"model": served, "messages": hist, "tools": E21_PROBE_TOOLS, "chat_template_kwargs": kw}
    ids = NB5L.tokenize_chat(SGL_URL + "/v1", payload)
    ids_c = NB5L.variant_c_ids(ids, NB5_TAILS[0], NB5_TAILS[1])
    out.update(render_tokens=len(ids), tail=ids[-4:], tail_ok=ids_c is not None)
    r2 = nb5_chat({"model": served, "messages": hist, "tools": E21_PROBE_TOOLS, "temperature": 0.0, "max_tokens": 1,
                   "chat_template_kwargs": kw})
    out["render_parity"] = (r2.get("usage") or {}).get("prompt_tokens") == len(ids)
    out["chat_prompt_tokens"] = (r2.get("usage") or {}).get("prompt_tokens")
    if ids_c is None:
        return out
    r3 = nb5_complete(ids_c, 512)
    c3, u3 = r3["choices"][0], r3.get("usage") or {}
    msg3, n3, leak3 = NB5L.parse_c_reply(_E12_TA, c3.get("text") or "")
    out["variant_c"] = {"tool_calls": n3, "finish": c3.get("finish_reason"), "think_markup_in_reply": leak3,
                        "cached": ((u3.get("prompt_tokens_details") or {}).get("cached_tokens")),
                        "prompt": u3.get("prompt_tokens"), "completion": u3.get("completion_tokens"),
                        "args": [(c["function"] or {}).get("arguments") for c in (msg3.get("tool_calls") or [])],
                        "natural_finish": c3.get("finish_reason") == "stop" and not leak3, "text_head": (c3.get("text") or "")[:300]}
    try:          # the server's own parsers on the hand-closed think: continue_final_message with a fixed prefill renders
        # C + the prefill (HF / vLLM trim only the end of a NON-empty final message); the SAME ids go to /v1/completions
        # (Duck's markup parser) and to chat (the server's qwen3 reasoning + qwen3_coder tool parsers), greedy
        pre = {"role": "assistant", "content": "I will call the tool now.", "reasoning": ""}
        tb = {"model": served, "messages": hist + [pre], "tools": E21_PROBE_TOOLS, "chat_template_kwargs": kw,
              "add_generation_prompt": False, "continue_final_message": True}
        ids_cfm = list(_e12v_request_json(SGL_URL + "/tokenize", payload=tb, timeout=120).get("tokens") or [])
        out["cfm_starts_with_c"] = ids_cfm[:len(ids_c)] == ids_c
        if out["cfm_starts_with_c"]:
            r4 = nb5_chat(dict(tb, temperature=0.0, max_tokens=512))
            m4 = r4["choices"][0]["message"]
            a4 = [(c.get("function") or {}).get("arguments") for c in (m4.get("tool_calls") or [])]
            r4c = nb5_complete(ids_cfm, 512)
            m4c, n4c, _ = NB5L.parse_c_reply(_E12_TA, r4c["choices"][0].get("text") or "")
            a4c = [(c.get("function") or {}).get("arguments") for c in (m4c.get("tool_calls") or [])]
            norm_ = lambda xs: [json.loads(x) if isinstance(x, str) and x.strip().startswith("{") else x for x in xs]
            out["server_parsers"] = {"tool_calls": len(a4), "client_tool_calls": n4c, "reasoning_chars": len(m4.get("reasoning") or ""),
                                     "content_head": (m4.get("content") or "")[:200], "finish": r4["choices"][0].get("finish_reason"),
                                     "args_equal_client_parse": norm_(a4) == norm_(a4c)}
    except Exception as _e:
        out["server_parsers"] = {"error": repr(_e)[:300]}
    try:          # recurrent-state correctness under prefix reuse: a cold prefix (cache_salt) vs the warm one, greedy
        r5 = nb5_complete(ids_c, 128, cache_salt=f"nb5-cold-{time.time_ns()}")
        r6 = nb5_complete(ids_c, 128)
        t5, t6 = r5["choices"][0].get("text") or "", r6["choices"][0].get("text") or ""
        cw = ((r6.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        cc = ((r5.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        k = next((i for i, (a, b) in enumerate(zip(t5, t6)) if a != b), None)
        out["cold_vs_warm"] = {"cold_cached": cc, "warm_cached": cw, "equal": t5 == t6, "first_diff_char": k,
                               "verdict": ("INCONCLUSIVE_NO_COLD" if cc else ("EQUAL" if t5 == t6 else "DIFFER"))}
    except Exception as _e:
        out["cold_vs_warm"] = {"error": repr(_e)[:300], "verdict": "INCONCLUSIVE_ERROR"}
    out["c_usable"] = bool(out["tail_ok"] and out["render_parity"])
    return out


def nb5_paired_commit_replay(tag, deadline_pc=None):
    """Addendum 2: the first REAL Duck commit request of the phase (kept by the tracer), replayed greedily as variant A
    (chat, thinking on) and variant C (ids): cache hit, generated tokens, tool call and code validity of each; with an
    MTP-off boot the same C replay is compared later (target-only parity)."""
    samp = NB5_TRACE.commit_samples.get("C") or NB5_TRACE.commit_samples.get("A")
    if not samp:
        return {"skipped": "no commit request in this phase"}
    pl = json.loads(json.dumps(samp["payload"]))
    _left = lambda: (deadline_pc - time.perf_counter()) if deadline_pc else 1200.0
    if _left() < 60:
        return {"skipped": "less than 60 s before the SMOKE gate", "task": samp["task"]}
    out = {"task": samp["task"], "branch": samp["branch"]}
    import ast as _ast

    def _code_ok(args):
        try:
            a = json.loads(args) if isinstance(args, str) else (args or {})
            _ast.parse(a.get("code") or "")
            return bool(a.get("code"))
        except Exception:
            return False
    try:
        ids = NB5L.tokenize_chat(SGL_URL + "/v1", pl)
        ids_c = NB5L.variant_c_ids(ids, NB5_TAILS[0], NB5_TAILS[1])
        if ids_c is not None:
            r = nb5_complete(ids_c, min(int(pl.get("max_tokens") or 2048), 2048), timeout=max(10, _left() / 2))
            c = r["choices"][0]
            m, n, lk = NB5L.parse_c_reply(_E12_TA, c.get("text") or "")
            u = r.get("usage") or {}
            out["C"] = {"cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "prompt": u.get("prompt_tokens"),
                        "completion": u.get("completion_tokens"), "tool_calls": n, "finish": c.get("finish_reason"),
                        "code_valid": any(_code_ok(x["function"].get("arguments")) for x in (m.get("tool_calls") or [])),
                        "text": (c.get("text") or "")[:4000]}
        ra = nb5_chat(dict(pl, temperature=0.0, max_tokens=min(int(pl.get("max_tokens") or 4096), 4096)), timeout=max(10, _left()))
        ca, ua = ra["choices"][0], ra.get("usage") or {}
        ma = ca.get("message") or {}
        out["A"] = {"cached": (ua.get("prompt_tokens_details") or {}).get("cached_tokens"), "prompt": ua.get("prompt_tokens"),
                    "completion": ua.get("completion_tokens"), "tool_calls": len(ma.get("tool_calls") or []),
                    "finish": ca.get("finish_reason"),
                    "code_valid": any(_code_ok((x.get("function") or {}).get("arguments")) for x in (ma.get("tool_calls") or [])),
                    "reasoning_chars": len(ma.get("reasoning") or "")}
        if "C" in out:
            out["same_tool_call_presence"] = bool(out["A"]["tool_calls"]) == bool(out["C"]["tool_calls"])
    except Exception as _e:
        out["error"] = repr(_e)[:400]
    return out


def nb5_warmup(port, W):
    """Addendum 1.3: the JIT-during-inference kernels (RC2: _fill_num_accepted, causal_conv1d_update, gdn_recoverssm
    verify, ple_conv, ple_conv_writeback) compile at the first sampled decode / first concurrent verify. Every metric
    window starts after this warm-up; JIT lines inside a later window are reported per phase."""
    j0 = len(NB5L.jit_kernels(E11_SERVER.log()))
    t0 = time.time()
    rows = []
    for level, band, n_new in ((1, "2k", 256), (W, "8k", 384), (min(W, 2), "40k", 64), (W, "2k", 256)):
        ss = [L.Stream(port, PROMPTS[band][i % len(PROMPTS[band])], n_new, temperature=1.0, seed=900 + i) for i in range(level)]
        for g in ss:
            g.start()
        for g in ss:
            g.join(max(30, min(900, left() - 600)))
        for g in ss:
            if g.is_alive():
                g.stop()
                g.join(10)
        rows.append({"level": level, "band": band, "tokens": [len(g.tokens) for g in ss], "errors": [g.err for g in ss if g.err]})
    ks = NB5L.jit_kernels(E11_SERVER.log())
    return {"sec": round(time.time() - t0, 1), "rows": rows, "jit_before": j0, "jit_after": len(ks), "jit_kernels": ks,
            "log_offset_after": len(E11_SERVER.log())}


def nb5_pure_decode(port, W, plan):
    """Pure decode: L.steady (the window starts only after prefill settled; STEADY needs running == level the whole
    window) at the Duck's own sampling (T 1.0) and level = workers, 40k prompts when they fit, else 8k."""
    band = "40k" if W * (NB5L.cdiv(40000 + 8192, plan["block_size"]) + plan["f_eff"]) <= plan["capacity_blocks"] else "8k"
    r = L.steady(port, PROMPTS[band], W, secs=WIN, max_new=MAX_NEW[band], temperature=1.0, prefill_tok_s=PREFILL["tok_s"])
    r["band"] = band
    r["engine_rates"] = NB5L.engine_rates(r, pp=2)
    r["note"] = ("decode-rate window on RC2's ARC-text prompts (concatenated real tasks): a rate, never an acceptance / K "
                 "decision input (PLAN never-do: synthetic prompts for acceptance)")
    return r


def nb5_probes(srv):
    """The short correctness gates before any Duck work (RC2 + E21): census, thinking-off arithmetic, a sampled request,
    content probes, the tool contract (history 'reasoning' key, turn-2 growth, cache probe), the commit probe.
    Returns (ok, record); garbage / dead is fatal everywhere, a tool-contract issue only outside the rerun."""
    o = {"census": kv_census(srv), "probes": L.chat_probes(srv.port), "sampled": sampled_check(srv)}
    o["fatal"] = None
    if (not FAKE and o["probes"].get("ok") is not True) or o["sampled"].get("ok") is not True or o["sampled"].get("garbage"):
        o["fatal"] = "SHORT_CORRECTNESS_GATE_FAILED"
        return False, o
    served = E21_SERVED[NB5_KV_DTYPE]
    try:
        o["content_probes"] = e21_content_probes(served)
    except Exception as _e:
        o["content_probes"] = {"error": repr(_e)[:600]}
        o["fatal"] = "CONTENT_PROBES_FAILED"
        return False, o
    try:
        o["contract"] = e21_contract_probe(served)
    except Exception as _e:
        o["contract"] = {"error": repr(_e)[:600]}
        if not RERUN:
            o["fatal"] = "CONTRACT_PROBE_FAILED"
            return False, o
    try:
        o["commit_probe"] = nb5_commit_probe(served)
    except Exception as _e:
        o["commit_probe"] = {"error": repr(_e)[:600], "c_usable": False}
    NB5_C["enabled"] = bool((o["commit_probe"] or {}).get("c_usable"))
    NB5_C["probe"] = o["commit_probe"]
    if not PREFILL["measured"]:
        o["prefill"] = prefill_probe(srv.port)
    return True, o


def nb5_reset_duck_state():
    """E21 gotchas: Duck's priority gate is a process-wide singleton built with ARC3_MAX_ACTIVE_STREAMS (fresh per
    phase); the scheduler / server-down globals of arc2_duck are cleared."""
    with _E12_TA._PRIORITY_GATE_LOCK:
        _E12_TA._PRIORITY_GATE = None
    _E12_TA._GATE_STATS.update(enqueued=0, blocked=0, blocked_ms=0, handover_ms=0)
    E12A.SERVER_DOWN.clear(); E12A.SERVER_WAITS.update(waits=0, sec=0.0); E12A.SERVER_LAST.clear(); E12A.SCHED_LAST.clear()
    os.environ["ARC3_REASONING_HISTORY_KEY"] = "reasoning"
    os.environ["LOCAL_ANALYZER_BASE_URL"] = os.environ["OPENAI_BASE_URL"] = SGL_URL + "/v1"
    os.environ["ARC3_MAX_ACTIVE_STREAMS"] = str(int(CONCURRENCY))


def nb5_sampler(stop, rows, path, workers):
    """Every 15 s during a phase: server running / waiting / raw KV gauge / generated / preemptions, the client's
    in-flight requests and the reservation. Live observation (addendum 2): server waiting > 0 while the client has no
    more requests in flight than run -> the reservation holds new grants (two samples in a row); a real preemption ->
    F + 1 for later admissions. Never touches running work."""
    last_pre, bad = None, 0
    while not stop.wait(float(os.environ.get("NB5_SAMPLER_SEC", "15"))):
        try:
            rr = e21_prom()
            g = lambda n: sum(float(x.value) for x in rr if x.name == n)
            s = {"t": round(time.time(), 1), "running": int(g("vllm:num_requests_running")), "waiting": int(g("vllm:num_requests_waiting")),
                 "kv_server": round(max([float(x.value) for x in rr if x.name == "vllm:kv_cache_usage_perc"] or [0.0]), 4),
                 "gen": g("vllm:generation_tokens_total"), "preempt": g("vllm:num_preemptions_total")}
        except Exception as _e:
            s = {"t": round(time.time(), 1), "error": repr(_e)[:120]}
        s["inflight"] = NB5_TRACE.inflight
        s["workers"] = workers
        if NB5_RES is not None:                   # PLAN v3.6: does Merlin's defer and Duck's projected gate restrict each other?
            _need = min(int(NB5_BLOCKS["b0"] or 1), int(NB5_BLOCKS["b1"] or 1))
            _done = sum(1 for v in (globals().get("E12_RECORDS") or {}).values() for r in v if r.get("status") == "done")
            s.update(duck_waiting=NB5_RES.waiting, duck_held=len(NB5_RES.held), free_blocks=NB5_RES.cap - NB5_RES.used,
                     work_left=(_done + len(NB5_RES.held)) < 2 * len(TASK_IDS))
            s["free_not_admitted"] = bool(s["free_blocks"] >= _need and s["duck_held"] < workers and s["work_left"]
                                          and not NB5_EARLY.get("fired") and not NB5_RES.waiting)
        if NB5_EARLY.get("armed") and not NB5_EARLY.get("fired"):
            try:
                if nb5_smoke_ready(s):
                    nb5_smoke_early_exit()
            except Exception as _e:
                NB5_EARLY["error"] = repr(_e)[:200]
        if NB5_RES is not None:
            s["reserved"] = NB5_RES.used
            if "waiting" in s and NB5_ADMIT_MODE != "slots":     # R2 'slots': the engine schedules (no client hold)
                bad = bad + 1 if (s["waiting"] > 0 and s["inflight"] <= s["running"] + s["waiting"] and s["running"] < s["inflight"]) else 0
                NB5_RES.set_live_hold(f"server waiting {s['waiting']} running {s['running']} in-flight {s['inflight']}" if bad >= 2 else None)
            if "preempt" in s:
                if last_pre is not None and s["preempt"] > last_pre and NB5_ADMIT_MODE == "slots":
                    NB5_EARLY["preempted"] = True          # R2: preemptions allowed (counted, speed only); no F correction
                elif last_pre is not None and s["preempt"] > last_pre:
                    NB5_RES.correct(1, f"server preemptions {last_pre} -> {s['preempt']}")
                    NB5_EARLY["preempted"] = True
                last_pre = s["preempt"]
        rows.append(s)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(s) + "\n")
        except Exception:
            pass


def nb5_smoke_ready(sample):
    """PLAN v3.6 SMOKE early exit: the functional checks already passed before the phase (probes, render parity, one
    short tool turn); the task-level gates: requests served without errors, >= 1 attempt finished, >= 1 variant-C
    commit turn with a tool call (unless C is disabled), no preemption, the server alive."""
    tr = NB5_TRACE.summary()
    done = [r for v in (globals().get("E12_RECORDS") or {}).values() for r in v if r.get("status") == "done"]
    cv = (tr.get("commit_by_variant") or {}).get("C") or {}
    return bool(tr.get("requests") and not tr.get("errors") and done and (cv.get("with_tool_call") or not NB5_C["enabled"])
                and not NB5_EARLY.get("preempted") and E11_SERVER is not None and E11_SERVER.alive())


def nb5_smoke_early_exit():
    """No new attempt (the reservation hook refuses), and every running attempt ends at its next request boundary
    (Duck's own done_now: generated >= the session cap; later requests clamped to now). In-flight requests finish."""
    NB5_EARLY["fired"] = round(time.perf_counter() - NOTEBOOK_T0, 1)
    for s_ in list(NB5_ACTIVE.values()):
        try:
            s_.max_generated_tokens = max(1, int(getattr(s_.agent, "generated_tokens", 0) or 0))
            if s_.agent is not None:
                s_.agent._arc2_deadline = time.monotonic()
        except Exception:
            pass
    log("NB5 SMOKE early exit: gates passed at", NB5_EARLY["fired"], "s")


def nb5_interaction(rows, interval):
    """Per phase (PLAN v3.6): the server waiting while Duck holds attempts, free projected capacity with work left but no
    admission, Duck holding with the server idle-able; in samples and seconds."""
    rr = [r for r in rows if "duck_waiting" in r and "waiting" in r]
    f = lambda c: sum(1 for r in rr if c(r))
    out = {"samples": len(rr), "interval_sec": interval,
           "server_waiting": f(lambda r: r["waiting"] > 0),
           "duck_holding": f(lambda r: r["duck_waiting"] > 0),
           "server_waiting_while_duck_holds": f(lambda r: r["waiting"] > 0 and r["duck_waiting"] > 0),
           "free_projected_capacity_no_admission": f(lambda r: r.get("free_not_admitted"))}
    out.update({k + "_sec": round(v * interval, 1) for k, v in list(out.items()) if k not in ("samples", "interval_sec")})
    return out


def nb5_qab_phase_record(name, rep, rows, preemptions):
    """R2 per phase: the actual solve window, the capacity levers of this boot (pool blocks, pin, F, workers, max_num_seqs,
    the lever env), speed (max streams reached, mean running, preemptions -> speed PREEMPTED, accuracy untouched,
    tok/s), the per-task accuracy table (nb5_lib.qab_tasks) and its per-group summary, the FP8 label."""
    run = dict(E12_RUN or {})
    ws = (run.get("window_shift") or {}).get("window_start_sec") or run.get("started_sec")
    we_pc = E15B_SOLVE_END_PC if E15B_SOLVE_END_PC != float("inf") else None
    to_epoch = lambda pc: time.time() - (time.perf_counter() - pc)
    ws_ep = to_epoch(NOTEBOOK_T0 + float(ws)) if ws is not None else None
    we_ep = to_epoch(we_pc) if we_pc else None
    win = {"start_sec": ws, "end_sec": round(we_pc - NOTEBOOK_T0) if we_pc else None,
           "actual_sec": round(we_pc - NOTEBOOK_T0 - float(ws), 1) if (we_pc and ws is not None) else None,
           "planned_sec": (NB5_REPORT["phases"].get(name, {}).get("window_plan") or {}).get("planned_sec"),
           "solver_returned_sec": run.get("finished_sec")}
    b = NB5_BOOTS[-1] if NB5_BOOTS else {}
    tr = (b.get("tries") or [{}])[-1]
    cap = NB5_CAPACITY or {}
    busy = [r_ for r_ in rows if isinstance(r_.get("running"), int) and (r_.get("inflight") or 0) > 0]
    tasks = NB5L.qab_tasks(E12_RECORDS, SOLUTIONS, rep.get("eval_unit_records"), NB5_QAB_GROUP_OF, P["task_list"],
                           {"b0": E15B["E12_B_ATTEMPT_TOKENS"], "b1": E15B["E12_B_ATTEMPT_TOKENS_B1"]}, ws_ep, we_ep,
                           (NB5_TRACE.summary() or {}).get("by_attempt"))
    return {"window": win, "order": rep.get("order"),
            "fp8_status": NB5_QAB_FP8_LABEL if NB5_BOOT_KV == "fp8" else "BF16_REFERENCE",
            "capacity_levers": {"pool_blocks": cap.get("pool_blocks"), "pool_source": cap.get("pool_source"),
                                "capacity_blocks": cap.get("capacity_blocks"), "f_eff": cap.get("f_eff"), "f_log": cap.get("f_log"),
                                "f_source": cap.get("f_source"), "pin_gib": tr.get("pin_gib"), "kv_bytes": tr.get("kv_bytes"),
                                "lever_env": tr.get("levers"), "max_num_seqs_launch": tr.get("max_num_seqs_launch"),
                                "workers": int(CONCURRENCY), "r2_workers": cap.get("r2_workers"),
                                "captures": NB5_CAPS16 if P.get("caps16") else None,
                                "ladder_tries": [{"name": t_.get("name"), "status": t_.get("status"),
                                                  "failed": (t_.get("ladder") or {}).get("failed")} for t_ in b.get("tries") or []],
                                "levers_dropped": b.get("levers_dropped"), "config_readback": (tr.get("config_readback") or {}).get("ok")},
            "speed": {"max_running": max([r_["running"] for r_ in rows if isinstance(r_.get("running"), int)] or [0]),
                      "max_inflight": max([int(r_.get("inflight") or 0) for r_ in rows] or [0]),
                      "mean_running_busy": round(sum(r_["running"] for r_ in busy) / len(busy), 2) if busy else None,
                      "preemptions": int(preemptions or 0), "speed_status": "PREEMPTED" if preemptions else "CLEAN",
                      "accuracy_affected_by_preemption": False,
                      "solver_tok_s": (NB5_REPORT["phases"].get(name, {}).get("solver_window") or {}).get("tok_s"),
                      "phase_tok_s": (NB5_REPORT["phases"].get(name, {}).get("phase_window") or {}).get("tok_s")},
            "qab": {"tasks": tasks, "summary": NB5L.qab_summary(tasks)}}


def nb5_phase(spec, mtp_tag):
    """One Duck run on the live server: fresh Duck state (workload reset), counters / log offset at the start
    (timestamp-delta metric reset), E15B CELL 6 + 7 (E21's wrapped functions), the NB5 phase record."""
    global TASK_IDS, CHALLENGES, SOLUTIONS, OUT, E21_HALF, E21_HALF_END_PC, E12_SOLVE_WINDOW_SEC, E15B_SOLVE_END_PC, NB5_RES
    global NB5_F_MARGIN, E12_MAX_GEN_PER_BRANCH, CONCURRENCY, E12_STREAMS, E12_VLLM_CONCURRENCY, E12_RECORDS
    name, sel, window, _mtp, cmode = spec[:5]
    opts = (spec[6] if len(spec) > 6 else None) or {}
    rec = NB5_REPORT["phases"].setdefault(name, {"spec": spec, "boot": mtp_tag})
    ids = [t for t in NB5_SMOKE_TASKS if t in NB5_EVAL_CH] if sel == "smoke" else list(NB5_ALL_IDS)
    if not ids:
        rec["status"] = "SKIPPED_NO_TASKS"
        return rec
    if P.get("equal_windows") and sel != "smoke":
        # R2: equal solve windows, fixed when the FIRST paired phase starts (after its boot: a slow first boot shortens
        # every window equally); a later phase gets the same window unless the hard stop leaves less (recorded UNEQUAL)
        if NB5_EQ["window_sec"] is None:
            _left_ph = [s_[0] for s_ in P["nb5_phases"] if s_[0] == name or s_[0] not in NB5_REPORT["phases"]]
            NB5_EQ["plan"] = NB5L.equal_windows(time.perf_counter(), HARD_STOP - E12_STOP_BEFORE_HARD_SEC,
                                                time.perf_counter() - float(NB5_EQ["try_t0_pc"] or time.perf_counter()),
                                                float(os.environ.get("NB5_PHASE_TAIL_SEC", 300)) if E11_HARNESS else 300.0,
                                                max(1, len(_left_ph)))
            NB5_EQ["window_sec"] = NB5_EQ["plan"]["window_sec"]
        window = float(NB5_EQ["window_sec"])
        _room = (HARD_STOP - E12_STOP_BEFORE_HARD_SEC) - time.perf_counter() - 60.0
        rec["window_plan"] = {"planned_sec": window, "room_sec": round(_room, 1), "plan": NB5_EQ["plan"], "unequal": False}
        if _room < window:
            rec["window_plan"].update(unequal=True, short_by_sec=round(window - _room, 1))
            window = max(0.0, _room)
        if window < (60 if E11_HARNESS else 5 * 60):
            rec.update(status="SKIPPED_BUDGET", left_min=round(left() / 60, 1), window_sec=round(window, 1))
            return rec
    else:
        need = (window or 0) + 10 * 60
        if left() < need + 5 * 60:
            rec.update(status="SKIPPED_BUDGET", left_min=round(left() / 60, 1), need_min=round(need / 60, 1))
            return rec
    src_ch, src_sol = (NB5_EVAL_CH, NB5_EVAL_SOL) if sel == "smoke" else (NB5_ALL_CH, NB5_ALL_SOL)
    TASK_IDS = sorted(ids)
    CHALLENGES = {t: src_ch[t] for t in TASK_IDS}
    SOLUTIONS = {t: src_sol[t] for t in TASK_IDS} if src_sol is not None else None
    OUT = BASE_OUT / f"phase_{name}"
    OUT.mkdir(parents=True, exist_ok=True)
    E21_HALF, E21_HALF_END_PC, E12_SOLVE_WINDOW_SEC = name, HARD_STOP - E12_STOP_BEFORE_HARD_SEC, window
    E15B_SOLVE_END_PC = float("inf")
    nb5_reset_duck_state()
    NB5_C["mode"] = NB5_COMMIT_MODE or cmode
    _base = (dict(E15B), E12_MAX_GEN_PER_BRANCH, dict(NB5_BLOCKS), CONCURRENCY)
    if opts.get("budget"):                         # PLAN v3.6 NB5BUDGET: this phase's attempt budget (both attempts)
        _bud = int(opts["budget"])
        assert E11_HARNESS or _bud >= 60000, _bud
        E15B["E12_B_ATTEMPT_TOKENS"] = E15B["E12_B_ATTEMPT_TOKENS_B1"] = E12_MAX_GEN_PER_BRANCH = _bud
        os.environ["E12_B_ATTEMPT_TOKENS"] = os.environ["E12_B_ATTEMPT_TOKENS_B1"] = str(_bud)
        _pl = NB5L.capacity_plan(NB5_CAPACITY["pool_blocks"], NB5_CAPACITY["block_size"], {"b0": _bud, "b1": _bud},
                                 NB5_PROMPT_ALLOW, NB5_TOOL_ALLOW, MAX_LEN, NB5_CAPACITY["f_eff"], NB5_GATE,
                                 seqs=(NB5_CAPACITY.get("caps") or {}).get("max_num_seqs"))
        NB5_BLOCKS.update(b0=_pl["per_attempt"]["b0"]["blocks"], b1=_pl["per_attempt"]["b1"]["blocks"])
        CONCURRENCY = E12_STREAMS = E12_VLLM_CONCURRENCY = int(_pl["workers"])
        rec["budget_plan"] = {"budget": _bud, "workers": _pl["workers"], "blocks": _pl["per_attempt"]["b0"]["blocks"]}
        if int(_pl["workers"]) < 1:                # C-8: this budget fits no stream -> no solver for this phase
            rec.update(status="NO_GO_INSUFFICIENT_KV", budget_plan=dict(rec["budget_plan"], verdict=_pl.get("verdict")))
            E15B.update(_base[0])
            E12_MAX_GEN_PER_BRANCH = _base[1]
            NB5_BLOCKS.update(_base[2])
            CONCURRENCY = E12_STREAMS = E12_VLLM_CONCURRENCY = _base[3]
            os.environ["E12_B_ATTEMPT_TOKENS"] = str(_base[0]["E12_B_ATTEMPT_TOKENS"])
            os.environ["E12_B_ATTEMPT_TOKENS_B1"] = str(_base[0]["E12_B_ATTEMPT_TOKENS_B1"])
            return rec
        nb5_reset_duck_state()
    # 'slots' (R2): the reservation is only a safety cap of `workers` attempts in flight (the engine schedules)
    NB5_RES = NB5L.Reservation(int(CONCURRENCY) if NB5_ADMIT_MODE == "slots" else NB5_CAPACITY["capacity_blocks"])
    NB5_RES.f_extra = NB5_F_MARGIN                 # PLAN v3.6: a preemption margin carries into later phases
    NB5_EARLY.update(armed=bool(opts.get("early_exit")), fired=None, preempted=False)
    NB5_TRACE.new_phase(name, OUT / "duck")
    c0, off0, t0 = e21_counters(), len(E11_SERVER.log()), time.time()
    h0 = NB5L.hist_sums(e21_prom())
    g0 = dict(NB5_GUARD, events=[])
    _pc0 = time.perf_counter()
    # revision 4: SMOKE's solve window ends >= 3 min before its 40-min gate (the replay fits in between); never past the
    # notebook's own solve end
    _gate_pc = min(_pc0 + NB5_SMOKE_GATE_SEC, HARD_STOP - E12_STOP_BEFORE_HARD_SEC) if name == "SMOKE" else None
    if name == "SMOKE":
        window = max(60.0, min(float(window or NB5_SMOKE_GATE_SEC), _gate_pc - _pc0 - 180.0))
        E12_SOLVE_WINDOW_SEC = window
        rec["smoke_gate_deadline_min"] = round((_gate_pc - _pc0) / 60, 1)
    E12_RECORDS = {}                               # the sampler never reads the previous phase's records
    rows, stop = [], threading.Event()
    th = threading.Thread(target=nb5_sampler, args=(stop, rows, OUT / "nb5_sampler.jsonl", int(CONCURRENCY)), daemon=True,
                          name="nb5_sampler")
    th.start()
    rec.update(status="RUNNING", tasks=len(TASK_IDS), start_sec=round(time.perf_counter() - NOTEBOOK_T0), commit_mode=NB5_C["mode"],
               commit_c_enabled=NB5_C["enabled"], workers=int(CONCURRENCY))
    try:
        e21_cell6()
        e21_cell7()
        rec["status"] = "DONE"
    except BaseException:
        rec.update(status="ERROR", error=traceback.format_exc()[-3000:])
    finally:
        stop.set()
        th.join(timeout=30)
        c1, t1 = e21_counters(), time.time()
        txt = E11_SERVER.log()
        sl = txt[off0:]
        run = dict(E12_RUN or {})
        solve_sec = (run.get("finished_sec") or 0) - (run.get("started_sec") or 0)
        bs = run.get("b_summary") or {}
        rep = {}
        try:
            rep = json.loads((OUT / "e12_run_report.json").read_text(encoding="utf-8"))
        except Exception as _e:
            rep = {"error": repr(_e)[:200]}
        rec.update(end_sec=round(time.perf_counter() - NOTEBOOK_T0), counters=NB5L.counter_delta(c0, c1),
                   tracer=NB5_TRACE.summary(), reservation=NB5_RES.snapshot(),
                   running_check=NB5L.running_check(rows, int(CONCURRENCY)),
                   steady_mixed=NB5L.steady_mixed(e21_stat_rows(sl), int(CONCURRENCY)),
                   jit_kernels_in_phase=NB5L.jit_kernels(sl), deferred=NB5L.deferred_lines(sl),
                   latency_means=NB5L.hist_delta(h0, NB5L.hist_sums(e21_prom())),
                   solver_window={"solve_sec": solve_sec, "server_gen_tokens": bs.get("server_gen_tokens"),
                                  "tok_s": NB5L.rate(bs.get("server_gen_tokens"), solve_sec)},
                   phase_window={"sec": round(t1 - t0, 1), "gen_tokens": NB5L.counter_delta(c0, c1).get("vllm:generation_tokens_total"),
                                 "tok_s": NB5L.rate(NB5L.counter_delta(c0, c1).get("vllm:generation_tokens_total"), t1 - t0)},
                   bands=NB5L.solve_bands(E12_RECORDS, SOLUTIONS),
                   task_solve=NB5L.task_solve_times(E12_RECORDS, SOLUTIONS), kv_dtype=NB5_BOOT_KV,
                   quality=(NB5L.quality_table(E12_RECORDS, SOLUTIONS, NB5_TRACE.summary().get("turns"),
                                               ((NB5L.hist_delta(h0, NB5L.hist_sums(e21_prom())).get("time_to_first_token_seconds") or {})
                                                .get("mean_sec"))) if name.startswith("Q_") else None),
                   score={k: rep.get(k) for k in ("eval_query_hits", "eval_query_total", "eval_pass1_hits", "eval_pass2_hits",
                                                  "eval_task_norm_pct")},
                   b_summary={k: bs.get(k) for k in ("attempts", "attempt2_started", "accepted", "at_budget", "commit_turns",
                                                     "commit_with_tool_call", "weak_accepts", "server_running_mean",
                                                     "server_gen_tokens", "counted_gen_tokens", "extra_started")},
                   rev6_workers=run.get("rev6_workers"), streams=run.get("streams"), solver_error=run.get("error"),
                   samples=len(rows),
                   guard={k: NB5_GUARD[k] - g0[k] for k in ("requests", "tokenized", "estimated", "clamps", "clamped_tokens",
                                                            "floor_events")} | {"events": NB5_GUARD["events"][-50:]},
                   duty=NB5L.duty_factor(sl, t1 - t0, NB5L.counter_delta(c0, c1).get("vllm:generation_tokens_total"),
                                         NB5_BOOT_DECODE.get("tok_s")),
                   interaction=nb5_interaction(rows, float(os.environ.get("NB5_SAMPLER_SEC", "15"))),
                   budget_limited_attempts=sum(1 for v in (E12_RECORDS or {}).values() for r in v
                                               if r.get("nb5_verdict") == "BUDGET_LIMITED"),
                   early_exit={k: NB5_EARLY.get(k) for k in ("armed", "fired", "preempted", "error")},
                   paired_split=(NB5L.variant_outcomes(E12_RECORDS, SOLUTIONS, NB5_C["mode"]) if NB5_C["mode"] == "split" else None))
        _pre = int(rec["counters"].get("vllm:num_preemptions_total") or 0)
        if _pre and name == "SMOKE":
            NB5_F_MARGIN += 1                          # PLAN v3.6 gate: any SMOKE preemption -> +1 F for every later admission
        NB5_F_MARGIN = max(NB5_F_MARGIN, NB5_RES.f_extra)
        rec["preemption_gate"] = {"preemptions": _pre, "verdict": "PREEMPTED" if _pre else "PASS",
                                  "f_margin_after": NB5_F_MARGIN}
        if P.get("quality_ab") and name.startswith("Q_"):
            try:
                rec.update(nb5_qab_phase_record(name, rep, rows, _pre))
            except Exception as _e:
                rec["qab_error"] = traceback.format_exc()[-1500:]
        if opts.get("budget"):                     # restore the base budgets / capacity for any later phase
            E15B.update(_base[0])
            E12_MAX_GEN_PER_BRANCH = _base[1]
            NB5_BLOCKS.update(_base[2])
            CONCURRENCY = E12_STREAMS = E12_VLLM_CONCURRENCY = _base[3]
            os.environ["E12_B_ATTEMPT_TOKENS"] = str(_base[0]["E12_B_ATTEMPT_TOKENS"])
            os.environ["E12_B_ATTEMPT_TOKENS_B1"] = str(_base[0]["E12_B_ATTEMPT_TOKENS_B1"])
        NB5_EARLY.update(armed=False)
        try:
            rec["paired_commit_replay"] = (nb5_paired_commit_replay(name, _gate_pc) if name == "SMOKE" or mtp_tag.endswith("nomtp")
                                           else None)
        except Exception as _e:
            rec["paired_commit_replay"] = {"error": repr(_e)[:300]}
        NB5_RES = None
        if name == "SMOKE":
            rec["smoke_gate_at_min"] = round((time.perf_counter() - _pc0) / 60, 1)
            rec["smoke_gate_within_bound"] = time.perf_counter() <= _gate_pc + 30
        nb5_write_report()
    return rec


def nb5_smoke_gate(rec):
    """SMOKE -> EVAL (fail closed): the phase ran, requests were served, the server lived, no preemption, the request
    error share < 50 %. A broken commit variant C (>= 4 C commits, none with a tool call) switches EVAL to variant A."""
    tr = rec.get("tracer") or {}
    why = []
    if rec.get("status") != "DONE":
        why.append(f"status {rec.get('status')}")
    if not tr.get("requests"):
        why.append("no request served")
    elif tr.get("errors", 0) / max(1, tr["requests"]) >= 0.5:
        why.append(f"errors {tr['errors']} of {tr['requests']}")
    if (rec.get("preemption_gate") or {}).get("verdict") == "PREEMPTED":
        rec["preemption_action"] = f"+1 F margin for every later admission (now {NB5_F_MARGIN})"   # reported, not an abort
    if E11_SERVER is None or not E11_SERVER.alive():
        why.append("server not alive")
    cv = (tr.get("commit_by_variant") or {}).get("C") or {}
    av = (tr.get("commit_by_variant") or {}).get("A") or {}
    sp = rec.get("paired_split") or {}
    # PLAN v3.6: the C -> A decision = the >= 4-commits-without-a-call safety trigger, OR corruption only under C
    # (unparsable code / think markup in a closed-think reply / non-grid candidates), OR >= 2 fewer exact-match attempts
    # under C with >= 3 commit-turn attempts in each variant (small samples: recorded either way)
    reasons = []
    if cv.get("requests", 0) >= 4 and not cv.get("with_tool_call"):
        reasons.append("no tool call in any of >= 4 variant-C commit requests")
    c_bad = cv.get("corrupt", 0) + (sp.get("C") or {}).get("invalid_grid_candidates", 0)
    a_bad = av.get("corrupt", 0) + (sp.get("A") or {}).get("invalid_grid_candidates", 0)
    if c_bad and not a_bad:
        reasons.append(f"corruption under C only ({c_bad})")
    if ((sp.get("C") or {}).get("attempts", 0) >= 3 and (sp.get("A") or {}).get("attempts", 0) >= 3
            and sp["C"]["exact"] <= sp["A"]["exact"] - 2):       # a 1-attempt difference on 3 + 3 is noise
        reasons.append(f"exact-match C {sp['C']['exact']}/{sp['C']['attempts']} vs A {sp['A']['exact']}/{sp['A']['attempts']}")
    rec["variant_decision"] = {"C": {"commit": cv, "split": sp.get("C"), "corrupt": c_bad},
                               "A": {"commit": av, "split": sp.get("A"), "corrupt": a_bad}, "switch_reasons": reasons}
    if reasons:
        NB5_C["enabled"] = False
        rec["commit_c_disabled_for_eval"] = "; ".join(reasons)
    if rec.get("smoke_gate_within_bound") is False:
        why.append("SMOKE gate passed its 40-min bound")
    rec["gate_decision"] = "ABORT" if why else ("EVAL_SWITCH_TO_A" if rec.get("commit_c_disabled_for_eval") else "EVAL")
    return why


def e21_relaunch():
    """E15B's watchdog relaunch on Merlin (NB5): the dead / hung server is killed, the GPUs released, the host gate
    passed, the same plan booted under the same name (its old log is copied first), at most once per boot, only with
    >= NB5_RELAUNCH_MIN_SEC before the solve end. The new server replaces the element of with_servers' list."""
    global E11_SERVER, SGL_URL
    t = globals().get("E15B_SOLVE_END_PC", float("inf")) - time.perf_counter() - 5 * 60
    if t < NB5_RELAUNCH_MIN_SEC:
        return False, f"only {round(t)} s left for a relaunch"
    e21_snapshot_counters()
    old = E11_SERVER
    n = len(NB5_REPORT.setdefault("relaunches", []))
    try:
        cp = old.log_path.with_name(old.log_path.stem + f"_dead{n}.log")
        shutil.copyfile(old.log_path, cp)
        wd = (E12_RUN or {}).get("watchdog") or {}
        if wd.get("old_logs"):
            wd["old_logs"][-1] = str(cp)
        old.kill()
        rel = L.wait_gpu_release(300, fake=bool(FAKE), gpus="0,1,2,3", free_mib=150)
        gate = nb5_host_gate(f"relaunch{n}")
        if rel is False or gate.get("ok") is False:
            NB5_REPORT["relaunches"].append({"ok": False, "gpu_release": rel, "host_gate": gate.get("ok")})
            return False, "GPU release / host RAM gate failed"
        srv = merlin_fn_server(old.name, NB5_BOOT_KV, old.port, NB5_KV_BYTES[NB5_BOOT_KV])
        srv.start()
        NB5_SERVERS[NB5_SERVERS.index(old)] = srv
        ok, why = srv.wait_ready(max(60.0, min(20 * 60, t)))   # NB6 P5: E18 R2 relaunch wait_ready(20 * 60)
        sc = sampled_check(srv) if ok else None
        ok = bool(ok and sc and sc.get("ok"))
        NB5_REPORT["relaunches"].append({"ok": ok, "why": why, "sampled": sc, "old_log": str(cp)})
        if ok:
            E11_SERVER, SGL_URL = srv, srv.url
        return ok, why
    except Exception as _e:
        NB5_REPORT["relaunches"].append({"ok": False, "error": repr(_e)[:400]})
        return False, repr(_e)[:300]


NB5_PIN = {"start_gib": None, "levers": None, "patch_60490": None}   # R2: the next boot starts at this boot's working step
NB5_EQ = {"window_sec": None, "plan": None, "try_t0_pc": None}        # R2: the equal solve window (set at the first phase)


def nb5_boot_try(trec, name, kv_bytes, phases, mtp_on):
    """One launch of one boot (fail closed): GPU release < 150 MiB, page-cache drop + Merlin RAM floor, ready <= 30 min,
    the C-11 config read-back, the short gates, capacity (C-8: 0 workers = NO_GO), warm-up, the pure decode window
    (commit only, not R2), then the phases of this boot (A-12: a phase that is not DONE or has a solver error ends the
    boot as PHASE_FAILED, the partial submission / report stay). Everything is recorded in trec."""
    global E11_SERVER, SGL_URL, ENGINE_READY, CONCURRENCY, E12_STREAMS, E12_VLLM_CONCURRENCY, ENGINE_INFO, NB5_CAPACITY
    trec["t0_sec"] = round(time.perf_counter() - NOTEBOOK_T0)
    NB5_EQ["try_t0_pc"] = time.perf_counter()
    trec["boot_gate"] = boot_gate()
    if trec["boot_gate"]["ok"] is not True:
        trec["status"] = "GPU_NOT_EMPTY"
        return trec
    trec["host_gate"] = nb5_host_gate(name)
    if trec["host_gate"].get("ok") is False:
        trec["status"] = "HOST_RAM_GATE_FAILED"
        return trec
    shm0 = shm_names()
    srv = merlin_fn_server(name, NB5_BOOT_KV, NB5_PORT, kv_bytes)
    trec["argv_tail"] = srv.argv[-60:]
    trec["server_env_levers"] = {k: srv.env.get(k) for k in sorted({**NB5_BASE_ENV, **NB5_BOOT_LEVERS})}   # NB5Q-ENV-1: final server env
    trec["diff_vs_rc2_fp8"] = NB5L.argv_diff(NB5_RC2_FP8_ARGV_TAIL, srv.argv[-60:])
    NB5_SERVERS[:] = [srv]

    def body(ss):
        global E11_SERVER, SGL_URL, ENGINE_READY, CONCURRENCY, E12_STREAMS, E12_VLLM_CONCURRENCY, ENGINE_INFO, NB5_CAPACITY
        s = ss[0]
        E11_SERVER, SGL_URL = s, s.url
        trec["ready_sec"] = round(time.perf_counter() - NOTEBOOK_T0)
        trec["config_readback"] = NB5L.config_readback(s.log(), s.argv, NB5_BOOT_KV, mtp_on)    # C-11
        if not trec["config_readback"]["ok"] and not RERUN:
            trec["status"] = "INCONCLUSIVE_CONFIG_MISMATCH"
            for sp in phases:
                NB5_REPORT["phases"].setdefault(sp[0], {"spec": sp, "boot": name}).update(
                    status="INCONCLUSIVE_CONFIG_MISMATCH", config_readback=trec["config_readback"]["checks"])
            return {}
        ok, pr = nb5_probes(s)
        trec["probes"] = pr
        if not ok:
            trec["status"] = "NO_GO_" + str(pr.get("fatal"))
            return {}
        NB5_CAPACITY = nb5_capacity(s, NB5_BOOT_KV)
        trec["capacity"] = NB5_CAPACITY
        W = int(NB5_CAPACITY["workers"])
        if W < 1 or NB5_CAPACITY.get("verdict") == "NO_GO_INSUFFICIENT_KV":                     # C-8
            trec["status"] = "NO_GO_INSUFFICIENT_KV"
            return {}
        NB5_BLOCKS["b0"] = NB5_CAPACITY["per_attempt"]["b0"]["blocks"]
        NB5_BLOCKS["b1"] = NB5_CAPACITY["per_attempt"]["b1"]["blocks"]
        NB5_BLOCKS.update(f=NB5_CAPACITY["f_eff"], block=NB5_CAPACITY["block_size"],
                          start=NB5L.cdiv(NB5_PROMPT_ALLOW, NB5_CAPACITY["block_size"]) + NB5_CAPACITY["f_eff"],
                          peak_b0=NB5_CAPACITY["per_attempt"]["b0"]["projected_peak_tokens"],          # C-9
                          peak_b1=NB5_CAPACITY["per_attempt"]["b1"]["projected_peak_tokens"])
        NB5_POOL_CAP["blocks"] = NB5_CAPACITY["capacity_blocks"]
        CONCURRENCY = E12_STREAMS = E12_VLLM_CONCURRENCY = W
        ENGINE_INFO = {"backend": f"merlin-{NB5_BOOT_KV}-{'mtp3' if mtp_on else 'nomtp'}", "kv_tokens": (NB5_CAPACITY.get("caps") or {}).get("kv_tokens"),
                       "concurrency": W, "capacity": NB5_CAPACITY, "max_model_len": MAX_LEN, "cfg": {"weights": NB5_BOOT_KV}}
        ENGINE_READY = True
        log("NB5 boot", name, "| workers", W, "| capacity", json.dumps({k: NB5_CAPACITY.get(k) for k in
                                                                         ("pool_blocks", "pool_source", "f_eff", "f_source", "capacity_blocks",
                                                                          "r2_workers")}))
        trec["admission"] = {"mode": NB5_ADMIT_MODE, "future_hook": NB5L.future_admission_blocks(None, None),
                             "note": ("future = placeholder, behaves as projected" if NB5_ADMIT_MODE == "future" else
                                      "slots = a safety cap of workers attempts in flight; the engine schedules (preemptions "
                                      "allowed, counted)" if NB5_ADMIT_MODE == "slots" else None)}
        trec["warmup"] = nb5_warmup(s.port, W)
        if not RERUN and not P.get("no_pure_decode"):
            try:
                trec["pure_decode"] = nb5_pure_decode(s.port, W, NB5_CAPACITY)
                if trec["pure_decode"].get("status") == "STEADY":
                    NB5_BOOT_DECODE["tok_s"] = trec["pure_decode"].get("tok_s")
            except Exception as _e:
                trec["pure_decode"] = {"error": repr(_e)[:300]}
        trec["counters_after_warmup"] = e21_counters()
        trec["phases_started"] = True
        for spec in phases:
            r = nb5_phase(spec, name)
            if spec[0] == "SMOKE":
                why = nb5_smoke_gate(r)
                r["gate"] = why or "PASS"
                if why and not RERUN:
                    log("NB5 SMOKE gate failed:", why)
                    trec["status"] = "SMOKE_GATE_FAILED"
                    break
            elif r.get("status") != "DONE" or r.get("solver_error"):                                # A-12
                trec.update(status="PHASE_FAILED", failed_phase=spec[0],
                            phase_error=str(r.get("error") or r.get("solver_error") or r.get("status"))[-2000:])
                log("NB5 PHASE_FAILED:", spec[0], trec["phase_error"][-300:])
                break
            if not s.alive() and E11_SERVER is s:
                break
        trec["counters_end"] = e21_counters()
        return {}
    o = with_servers(NB5_SERVERS, globals().get("NB6_READY_SEC", NB5_READY_TIMEOUT), body)   # NB6 P3
    ENGINE_READY = False
    trec.update(servers=o.get("servers"), gpu_released=o.get("gpu_released"), shm_cleanup=shm_cleanup(shm0))
    trec["log_text_for_ladder"] = srv.log() or ""
    trec["status"] = trec.get("status") or o.get("status") or "DONE"
    return trec


def nb5_boot(tag, phases, mtp_on, kv=None):
    """One Merlin boot. R2 lever a (KV pin ladder, NB1's descending rule): kv_cache_memory_bytes = the RC2 pin + 1.0 /
    0.75 / 0.5 / 0 GiB per card, highest first (the second boot starts at the first boot's working step: the same
    settings in both phases); a step down only on a launch failure or an OOM line before any phase. Lever b: the
    non-numeric lever env (MERLIN_SHARE_DRAFT_WORKSPACE=1) is dropped when an error line names it, or once all steps
    failed with it (then the RC2 pin without it). The launch --max-num-seqs follows SEQS_TABLE[kv][pin]. Every try is
    recorded (pin, pool, F, failure); the working try's record is merged into the boot record."""
    global NB5_BOOT_EXTRA, NB5_BOOT_KV, NB5_BOOT_LEVERS
    NB5_BOOT_KV = kv or NB5_KV_DTYPE
    brec = {"tag": tag, "mtp": mtp_on, "kv_dtype": NB5_BOOT_KV, "kv_bytes": NB5_KV_BYTES[NB5_BOOT_KV], "tries": []}
    NB5_BOOTS.append(brec)
    base_extra = list(NB5_EXTRA_ARGS) + ([] if mtp_on else list(NB5_MTP_OFF_ARGS))
    if P.get("pin_ladder"):
        start = NB5_PIN["start_gib"]
        if start is None and P.get("patch_60490") and not (NB5_PIN.get("patch_60490") or {}).get("ok") and not FAKE:
            start = 0.0                            # #60490 not verified: no step above the RC2 pin (fail safe)
        tries = NB5L.pin_tries(NB5_KV_BYTES[NB5_BOOT_KV], start)
    else:
        tries = [{"gib": 0.0, "kv_bytes": NB5_KV_BYTES[NB5_BOOT_KV]}]
    levers = dict(NB5_PIN["levers"] if NB5_PIN["levers"] is not None else NB5_LEVER_ENV)
    nolever_left = bool(levers)
    i, trec = 0, {}
    while i < len(tries):
        tr = tries[i]
        if brec["tries"] and left() < 25 * 60:
            brec["ladder_stop"] = f"only {round(left() / 60, 1)} min left: no further boot try"
            break
        NB5_BOOT_LEVERS = dict(levers)
        seqs = NB5L.seqs_for(NB5_BOOT_KV, tr["gib"]) if NB5_SEQS_RULE == "table" else None
        NB5_BOOT_EXTRA = base_extra + (["--max-num-seqs", str(seqs)] if seqs else [])
        name = (tag if not P.get("pin_ladder") else f"{tag}_pin{int(round(tr['gib'] * 100))}" + ("" if levers or not NB5_LEVER_ENV else "_nolever"))
        trec = {"name": name, "pin_gib": tr["gib"], "kv_bytes": tr["kv_bytes"], "max_num_seqs_launch": seqs, "levers": dict(levers)}
        brec["tries"].append(trec)
        globals()["NB6_READY_SEC"] = ((30 if len(brec["tries"]) == 1 else 15) * 60 if P.get("e18_ladder")
                                      else NB5_READY_TIMEOUT)          # NB6 P3: E18 R2 cell 6 L485
        nb5_boot_try(trec, name, tr["kv_bytes"], phases, mtp_on)
        fail = NB5L.boot_failure(trec.get("status"), trec.pop("log_text_for_ladder", ""), levers, trec.get("phases_started", False))
        if P.get("e18_ladder"):   # NB6 P3: E18 R2 retries on ANY failed attempt (not ready, stall, smoke) before solving
            fail["failed"] = trec.get("status") != "DONE" and not trec.get("phases_started", False)
        trec["ladder"] = fail
        log("NB5 boot try", name, trec.get("status"), "| ladder", json.dumps({k: fail[k] for k in ("failed", "lever_suspect")}))
        nb5_write_report()
        if not fail["failed"] or not P.get("pin_ladder") or ABORT.get("why") or (P.get("e18_ladder") and len(brec["tries"]) >= 3):
            break                                  # NB6 P3: E18 R2 ladder = at most 3 engine attempts
        if fail["lever_suspect"]:
            for k_ in fail["lever_suspect"]:
                levers.pop(k_, None)
            brec.setdefault("levers_dropped", []).append({"keys": fail["lever_suspect"], "why": "named in an error line", "try": name})
            continue                               # the same pin step without that lever
        if i == len(tries) - 1 and nolever_left and levers:
            brec.setdefault("levers_dropped", []).append({"keys": sorted(levers), "why": "every pin step failed with the lever env",
                                                          "try": name})
            levers, nolever_left = {}, False
            continue                               # the RC2 pin without the levers
        i += 1
    for k_ in ("boot_gate", "host_gate", "argv_tail", "server_env_levers", "diff_vs_rc2_fp8", "config_readback", "probes", "capacity", "admission",
               "warmup", "pure_decode", "counters_after_warmup", "counters_end", "servers", "gpu_released", "shm_cleanup",
               "failed_phase", "phase_error", "phases_started", "ready_sec", "t0_sec"):
        if k_ in trec:
            brec[k_] = trec[k_]
    brec["pin_used"] = {"gib": trec.get("pin_gib"), "kv_bytes": trec.get("kv_bytes"), "levers": trec.get("levers"),
                        "server_env_levers": trec.get("server_env_levers"),
                        "max_num_seqs_launch": trec.get("max_num_seqs_launch"),
                        "pool_blocks": (trec.get("capacity") or {}).get("pool_blocks"), "f_eff": (trec.get("capacity") or {}).get("f_eff"),
                        "failures": [{"name": t_["name"], "status": t_.get("status"), "oom_lines": (t_.get("ladder") or {}).get("oom_lines")}
                                     for t_ in brec["tries"] if (t_.get("ladder") or {}).get("failed")]}
    if trec and not (trec.get("ladder") or {}).get("failed") and trec.get("status") not in ("GPU_NOT_EMPTY", "HOST_RAM_GATE_FAILED"):
        NB5_PIN.update(start_gib=trec.get("pin_gib"), levers=dict(trec.get("levers") or {}))
    brec["status"] = trec.get("status") or brec.get("ladder_stop") and "SKIPPED_BUDGET" or "NO_TRY"
    return brec


def nb5_write_report():
    try:
        NB5_REPORT["minutes"] = round((time.perf_counter() - NOTEBOOK_T0) / 60, 1)
        write_json_atomic(BASE_OUT / "nb5_report.json", NB5_REPORT)
    except Exception as _e:
        print("NB5 report write failed:", repr(_e)[:200], flush=True)


# ==================================================================== CELL 6/7 — E21: the Duck run and its report as functions
# Both are E15B R3K2's cells (build_e21.py: the changed lines are listed in the Experiment 21 README),
# wrapped so each half runs them once (every module-level name is declared global: same behaviour).
def e21_cell6():
    # E15B CELL 6 (R3K2) with the vLLM probes / watchdog (build_e21.py), run once per half
    global _E12_INDEX, _e12_order2_score, _e12_total_cells, E12_ORDER, E12_RECORDS, E12_RUN, _E12_LOCK, E12_OUT, _tempfile, _E12_WRITE_LOCK, _e12_write_json_atomic, e12_write_submission, _e12_branch_done_r6, _e12_g4b_window, _e12_shifted_end, _e21_progress, _e15b_gen_tokens_total, _e15b_all_server_logs, _e15b_server_log_summary, _e12_capacity_probe, _e12_cache_diag, _e12_no_spec, _e12_parity, _e12_r6_preflight, _e12_branch_done, _E12_STOP, _e12_heartbeat, _e, _end, _ra, _hid_end, _start, _per, _, _kv, _first, _workers, _shift, _e15b_solve_end, E15B_SOLVE_END_PC, _e15b_tok0, _WD_POLL, _WD_STALL, _WD_FAIL, _WD_MIN_LEFT, _e15b_watchdog, _att, _ev, _bud, _g, _run, _q, _S, _full, _left
    # ==================================================================== CELL 6 — E12: Duck REPL agent, every task, E12_BRANCHES branches
    # Every branch is admitted at once; Duck's priority gate keeps ARC3_MAX_ACTIVE_STREAMS conversations active (first
    # branches in ORDER2 order before any second branch). A branch ends when the host verifies a submitted program on every
    # train pair, at the deadline, or at the optional token cap. Full thinking: Duck's transcript.txt per branch.
    _E12_INDEX = {t: i for i, t in enumerate(TASK_IDS)}


    def _e12_order2_score(tid):   # = S49 ORDER2 ([>=3 train] + [1 test]) with the cells tie-break, sorted-id order last
        _t = CHALLENGES[tid]
        return int(len(_t["train"]) >= 3) + int(len(_t["test"]) == 1)


    def _e12_total_cells(tid):
        _t = CHALLENGES[tid]
        _c = lambda g: len(g) * len(g[0]) if g and g[0] else 0
        return sum(_c(p["input"]) + _c(p["output"]) for p in _t["train"]) + sum(_c(p["input"]) for p in _t["test"])


    E12_ORDER = sorted(TASK_IDS, key=lambda t: (-_e12_order2_score(t), _e12_total_cells(t), _E12_INDEX[t]))
    if P.get("task_order"):      # NB5 R2: the lead's fixed task order (the same in both paired phases)
        E12_ORDER = [t for t in P["task_order"] if t in TASK_IDS] + [t for t in E12_ORDER if t not in P["task_order"]]
    E12_RECORDS = {t: [] for t in TASK_IDS}
    E12_RUN = {"started_sec": None, "model_id": None, "streams": None, "deadline_in_sec": None, "error": None,
               "branches_done": 0, "branches_accepted": 0, "finished_sec": None}
    _E12_LOCK = threading.Lock()
    E12_OUT = OUT / "duck"
    E12_OUT.mkdir(parents=True, exist_ok=True)


    import tempfile as _tempfile
    _E12_WRITE_LOCK = threading.Lock()   # rev 2 (GPT audits, reproduced): branches finish concurrently


    def _e12_write_json_atomic(path, obj):
        """A unique temp file per write + os.replace (CELL 0's helper reuses one '<name>.tmp' for every writer)."""
        fd, tmp = _tempfile.mkstemp(prefix=Path(path).name + ".", suffix=".tmp", dir=str(Path(path).parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(obj, f)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


    def e12_write_submission():
        with _E12_WRITE_LOCK:                          # one writer at a time; the snapshot is taken inside, so it is fresh
            sub = identity_submission(test_challenges)
            if RERUN:
                with _E12_LOCK:
                    recs = {t: list(v) for t, v in E12_RECORDS.items() if v}
                for t, rows in E12A.build_submission({t: CHALLENGES[t] for t in recs}, recs).items():
                    if t in sub and len(sub[t]) == len(rows):
                        sub[t] = rows
            _e12_write_json_atomic(SUBMISSION_PATH, sub)   # evaluation: keeps the competition-format test fallback


    def _e12_branch_done_r6(rec, _recs=E12_RECORDS, _run=E12_RUN, _lock=_E12_LOCK, _half=E21_HALF):
        # rev 6: a branch reports every pause and its finish; keep the latest per branch.
        # E21 rev 2c (GPT audits 1-3, CRITICAL, reproduced on this source): the callback is bound to ITS half's records, run
        # state and lock (default arguments, evaluated when this half's CELL 6 defines it). A half-A worker that is still
        # alive (a straggler past join grace) can never write into half B's records, and never writes half B's submission.
        with _lock:
            rows = _recs.setdefault(rec["task_id"], [])
            # rev 6g H7-1 (GPT rev 6f audits): callbacks can finish out of order; never let an older record replace a newer one
            _old = next((r for r in rows if r.get("branch") == rec.get("branch")), None)
            if _old is not None and int(rec.get("publish_seq") or 0) <= int(_old.get("publish_seq") or 0):
                _run["stale_callbacks_dropped"] = _run.get("stale_callbacks_dropped", 0) + 1
                return
            new = rec.get("branch") not in {r.get("branch") for r in rows}
            rows[:] = [r for r in rows if r.get("branch") != rec.get("branch")] + [rec]
            _run["branches_started"] = _run.get("branches_started", 0) + int(new)
            _run["branches_done"] = sum(1 for v in _recs.values() for r in v if r.get("status") == "done")
            _run["branches_accepted"] = sum(1 for v in _recs.values() for r in v if r.get("accepted"))
        if _half != E21_HALF:
            _run["late_cross_half_callbacks"] = _run.get("late_cross_half_callbacks", 0) + 1
            return
        if time.perf_counter() >= globals().get("E15B_SOLVE_END_PC", float("inf")):
            _run["late_writes_skipped"] = _run.get("late_writes_skipped", 0) + 1     # E15B B1
            return
        try:
            e12_write_submission()
        except Exception as _e:
            print("E12 submission write failed:", repr(_e)[:200], flush=True)


    def _e12_g4b_window(hidden_solve_end_sec, solve_start_sec, n_tasks, hidden_tasks=240):
        # gate 4b (user 2026-10-21): this run gets the hidden run's compute per task. H6-16: a function so the real
        # values (42,720 / 1,463 / 12 -> 2,063 s) are unit-tested, not only under the harness's 300 s cap
        per = max(0.0, float(hidden_solve_end_sec) - float(solve_start_sec)) / float(hidden_tasks)
        return per, per * int(n_tasks)


    def _e12_shifted_end(end, setup_sec, hard_end):
        # SA-1 / H6-6: a commit window starts after the preflight (and nothing else runs before solving since H6-3); the end
        # moves by the setup time, never past the hard stop
        return min(float(hard_end), float(end) + max(0.0, float(setup_sec)))


    def _e21_progress():
        # E21 rev 2h (GPT v3 HIGH-1, VAL2_dryrun F1): (requests running or waiting, prompt + generated tokens) from /metrics; None when
        # /metrics fails or lacks a field (then the stall rule never acts)
        try:
            _r = e21_prom()
        except Exception:
            return None
        _run = [x.value for x in _r if x.name in ("vllm:num_requests_running", "vllm:num_requests_waiting")]
        _tok = [x.value for x in _r if x.name in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total")]
        return (sum(_run), sum(_tok)) if (_run and _tok) else None      # running + waiting on the server


    def _e15b_gen_tokens_total():
        # E15B #32 on vLLM (E21): the server's own generated-token counter (vllm:generation_tokens_total, summed over rows)
        try:
            _v = [float(r.value) for r in e21_prom() if getattr(r, "name", "") == "vllm:generation_tokens_total"
                  and getattr(r, "value", None) is not None]
            return sum(_v) if _v else None
        except Exception:
            return None


    def _e15b_all_server_logs():
        # B6 (auditor 2 F6, auditor 3 L85 / L125): every server log of the solve (the crashed ones first), not only the last
        _t = []
        for _p in ((E12_RUN.get("watchdog") or {}).get("old_logs") or []):
            try:
                _t.append(Path(_p).read_text(encoding="utf-8", errors="replace"))
            except Exception:
                pass
        try:
            _t.append(E11_SERVER.log_text())
        except Exception:
            pass
        return "\n".join(_t)


    def _e15b_server_log_summary(text):
        # E15B #42 / #53 on vLLM (E21): vLLM's 10-s engine stats lines (rev 2: e21_stat_rows, every optional field incl.
        # Preemptions / Deferred) instead of SGLang decode lines: running-level histogram (lines with work), queue > 0 share,
        # generation throughput, prefix-cache hit rate (rolling), KV usage, MTP acceptance, preemptions, the startup KV line
        import re as _re
        rows = e21_stat_rows(text)
        work = [r for r in rows if r["running"] > 0 or r["waiting"] > 0]
        _hist, _hq, _tpl = {}, {}, {}
        for r in work:
            k = r["running"]
            _hist[k] = _hist.get(k, 0) + 1
            if r["waiting"] > 0:
                _hq[k] = _hq.get(k, 0) + 1
            _tpl.setdefault(k, []).append(r["gen_tps"])
        _tp = sorted(r["gen_tps"] for r in work)
        _acc = sorted(float(a) for a in _re.findall(r"Mean acceptance length: ([\d.]+)", text))
        _kv = _re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
        _pre = [r["prefix_pct"] for r in rows if r["prefix_pct"] is not None]
        _n = len(work)
        return {"decode_lines": _n, "running_hist": dict(sorted(_hist.items())),
                "running_hist_queue_gt0": dict(sorted(_hq.items())),
                "startup": {"kv_tokens_logged": int(_kv[0].replace(",", "")) if _kv else None},
                "accept_len_p50": _acc[len(_acc) // 2] if _acc else None,
                "gen_tok_s_p50_by_running": {k: sorted(v)[len(v) // 2] for k, v in sorted(_tpl.items())},
                "running_mean_lines": round(sum(k * v for k, v in _hist.items()) / _n, 2) if _n else None,
                "queue_gt0_line_share": round(sum(_hq.values()) / _n, 3) if _n else None,
                "gen_tok_s_p50": _tp[len(_tp) // 2] if _tp else None,
                "gen_tok_s_mean": round(sum(_tp) / len(_tp), 1) if _tp else None,
                "prefix_cache_hit_pct_last_rolling": _pre[-1] if _pre else None,
                "kv_usage_pct_max": max((r["kv_pct"] for r in rows), default=None),
                "preemptions_in_stats_lines": sum(r["preempt"] for r in rows),
                "preempt_lines": sum(1 for r in rows if r["preempt"]),
                "oom_lines": sum(1 for _l in text.splitlines() if "out of memory" in _l.lower()),
                "traceback_lines": sum(1 for _l in text.splitlines() if "Traceback" in _l)}


    def _e12_capacity_probe():   # E21: vLLM 0.19 / 0.32 gauges (the max over label rows); vLLM has no separate Mamba gauge
        try:
            rows = [r for r in e21_prom() if getattr(r, "value", None) is not None]
        except Exception:
            rows = None
        if not rows:
            # E21 rev 2c (GPT audit 1 HIGH-4): Duck's KV gate admits on a missing sample (fail-open). On vLLM a failed
            # /metrics read while requests run means "unknown", so report a full pool: admissions wait; with nothing
            # running Duck's idle override still admits (no deadlock); the watchdog handles a dead server.
            E21_STATE["probe_failures"] = E21_STATE.get("probe_failures", 0) + 1
            _t0 = E21_STATE.setdefault("probe_fail_since", time.monotonic())
            if time.monotonic() - _t0 > 120:     # rev 2d: a long outage -> Duck's own fail-open (counted), not 1 stream forever
                E21_STATE["probe_fail_open"] = E21_STATE.get("probe_fail_open", 0) + 1
                return {"kv_usage": None, "running": None, "queue": None, "mamba_usage": None, "probe_failed": True}
            return {"kv_usage": 1.0, "running": None, "queue": None, "mamba_usage": None, "probe_failed": True}
        E21_STATE.pop("probe_fail_since", None)
        pick = lambda name: [float(r.value) for r in rows if r.name == name]
        out = {}
        for key, name in (("kv_usage", "vllm:kv_cache_usage_perc"), ("running", "vllm:num_requests_running"),
                          ("queue", "vllm:num_requests_waiting")):
            v = pick(name)
            out[key] = max(v) if v else None
        out["mamba_usage"] = None
        out["kv_usage_server"] = out.get("kv_usage")      # NB5: recorded
        out["kv_usage"] = nb5_gate_kv(out.get("kv_usage"))  # NB5: the live-block reservation fraction
        return out


    def _e12_cache_diag(budget_sec=240.0):
        # rev 6e G4-8 (GPT G4B audits): a bounded exact-prompt replay for the prefix-cache misses (G4B: 20/180 requests with
        # 0 cached tokens while KV <= 0.44; suspects SGLang #22935 split 0-hit and #36935 mamba checkpoint eviction).
        # Prompt A (about 9k tokens) warm -> exact replay -> two divergent suffixes -> 12 distinct long prompts at once
        # (churn) -> replay and a divergent suffix again. max_tokens 1, thinking off. Commit mode only.
        import random as _rnd, concurrent.futures as _cf
        t_end = time.monotonic() + float(budget_sec)

        def _blob(seed, rows=150):
            r = _rnd.Random(seed)
            return "\n".join(" ".join(str(r.randrange(10)) for _ in range(30)) for _ in range(rows))

        def _ask(msgs):
            body = {"model": E12_RUN["model_id"], "max_tokens": 1, "temperature": 0.0, "messages": msgs,
                    "chat_template_kwargs": {"enable_thinking": False}}
            t0 = time.monotonic()
            req = _urlreq.Request(SGL_URL + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                  headers={"Content-Type": "application/json"})
            r = json.loads(_urlreq.urlopen(req, timeout=max(5.0, t_end - time.monotonic())).read())
            u = r.get("usage") or {}
            det = u.get("prompt_tokens_details") if isinstance(u.get("prompt_tokens_details"), dict) else {}
            return {"prompt": u.get("prompt_tokens"), "cached": int(det.get("cached_tokens") or 0),
                    "cached_reported": "cached_tokens" in det, "sec": round(time.monotonic() - t0, 2)}

        A = {"role": "system", "content": "E12-CACHE-DIAG reference A\n" + _blob(1)}
        q = lambda text: {"role": "user", "content": text}
        steps = []

        def step(name, msgs):
            if time.monotonic() > t_end - 5:
                steps.append({"step": name, "skipped": "budget"})
                return
            try:
                res = _ask(msgs)
            except Exception as _e:
                res = {"error": repr(_e)[:200]}
            res["step"] = name
            res.update({"g_" + k: v for k, v in _e12_capacity_probe().items()})
            steps.append(res)
        step("warm", [A, q("Reply with one word.")])
        step("replay_exact", [A, q("Reply with one word.")])
        step("divergent_1", [A, q("Divergent continuation one. Reply with one word.")])
        step("divergent_2", [A, q("Divergent continuation two. Reply with one word.")])
        if time.monotonic() < t_end - 30:
            churn = [[{"role": "system", "content": f"E12-CACHE-DIAG churn {i}\n" + _blob(100 + i)}, q("Reply with one word.")]
                     for i in range(max(1, int(E12_RUN.get("streams") or 12)))]

            def _safe(m):
                try:
                    return _ask(m)
                except Exception as _e:
                    return {"error": repr(_e)[:120]}
            with _cf.ThreadPoolExecutor(len(churn)) as _ex:
                cr = list(_ex.map(_safe, churn))
            steps.append({"step": "churn", "n": len(cr), "errors": sum(1 for x in cr if "error" in x),
                          "prompt_sum": sum(int(x.get("prompt") or 0) for x in cr),
                          "sec_max": max((x.get("sec") or 0) for x in cr),
                          **{"g_" + k: v for k, v in _e12_capacity_probe().items()}})
        step("replay_after_churn", [A, q("Reply with one word.")])
        step("divergent_after_churn", [A, q("Divergent continuation three. Reply with one word.")])
        by = {x["step"]: x for x in steps if "prompt" in x}
        pa = (by.get("warm") or {}).get("prompt")
        out = {"budget_sec": budget_sec, "prompt_tokens_A": pa, "steps": steps,
               "summary": {k: (by[k]["cached"], by[k]["prompt"]) for k in ("warm", "replay_exact", "divergent_1", "divergent_2",
                                                                             "replay_after_churn", "divergent_after_churn")
                           if k in by}}
        print("E12 cache diag (cached, prompt):", json.dumps(out["summary"]), flush=True)
        return out


    def _e12_no_spec(cmd):   # the same server command without --speculative-* (each takes one value)
        out, skip = [], False
        for x in cmd:
            if skip:
                skip = False
                continue
            if str(x).startswith("--speculative-"):
                skip = True
                continue
            out.append(x)
        return out


    def _e12_parity():
        # R6-X1 target-only parity (user 2026-10-21): the rev 6 system prompt + first opener of each task, with tools, T 0,
        # one request at a time, max_tokens 2,048, thinking on and off. DFlash2 twice (control), then a relaunch without
        # speculative decoding. Judged on equality and on what Duck acts on (tool-call presence and arguments, finish).
        global E11_SERVER
        TA, _, _, wrs = E12A._import_duck()
        Agent = E12A.make_agent_class()
        probes = []
        for tid in TASK_IDS:
            ag = Agent(task=CHALLENGES[tid], task_id=tid, branch=0, view="facts", model=E12_RUN["model_id"],
                       base_url=SGL_URL + "/v1", provider="vllm")
            d = E12_OUT / f"parity_{tid}"
            d.mkdir(parents=True, exist_ok=True)
            sp = d / TA.RUNTIME_STATE_FILENAME
            wrs(sp, current_frame=None, history=[])
            for think in (True, False):
                probes.append((tid, think, {"model": E12_RUN["model_id"], "max_tokens": 2048, "temperature": 0.0, "top_p": 1.0,
                                            "messages": [{"role": "system", "content": ag._system_prompt},
                                                         {"role": "user", "content": E12A.R6_TEXT["first"] + E12A.R6_COMPLETE}],
                                            "tools": ag._tools(sp), "tool_choice": "auto",
                                            "chat_template_kwargs": {"preserve_thinking": True, "enable_thinking": think}}))

        def _ask(body):
            req = _urlreq.Request(SGL_URL + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                  headers={"Content-Type": "application/json"})
            r = json.loads(_urlreq.urlopen(req, timeout=900).read())
            m = r["choices"][0]["message"]
            return {"reasoning": str(m.get("reasoning_content") or ""), "content": str(m.get("content") or ""),
                    "calls": [str((tc.get("function") or {}).get("arguments", "")) for tc in (m.get("tool_calls") or [])],
                    "finish": r["choices"][0].get("finish_reason"), "tokens": (r.get("usage") or {}).get("completion_tokens")}

        def _round(tag):
            rows = []
            for tid, think, body in probes:
                try:
                    rows.append(_ask(body))
                except Exception as _e:
                    rows.append({"error": repr(_e)[:300]})
            (E12_OUT / f"parity_{tag}.json").write_text(json.dumps(rows), encoding="utf-8")
            return rows
        out = {"probes": [{"task": t, "thinking": k} for t, k, _ in probes]}
        a, b = _round("dflash_a"), _round("dflash_b")
        try:
            cmd, env = list(E11_SERVER.cmd), E11_SERVER.env
            e11_kill_server(wait=True)
            srv = E11Server(_e12_no_spec(cmd), env, "target_only")
            srv.start()
            ok, why = srv.wait_ready(min(SGL_RELAUNCH_TIMEOUT_SEC, max(60, HARD_STOP - time.perf_counter() - 600)))
            E11_SERVER = srv
            out["target_only_launch"] = {"ok": ok, "why": why, "spec_flags_removed": [x for x in cmd if str(x).startswith("--speculative-")]}
            t = _round("target_only") if ok else [{"error": "target-only launch failed"} for _ in probes]
        except Exception as _e:
            out["target_only_launch"] = {"ok": False, "why": repr(_e)[:300]}
            t = [{"error": "target-only launch failed"} for _ in probes]

        def _pre(x, y):
            n = 0
            for c1, c2 in zip(x, y):
                if c1 != c2:
                    break
                n += 1
            return n
        cmp_ = []
        for (tid, think, _), ra, rb, rt in zip(probes, a, b, t):
            row = {"task": tid, "thinking": think}
            if "error" not in ra and "error" not in rb:
                row["control_identical"] = ra == rb
            if "error" not in ra and "error" not in rt:
                ta, tt = ra["reasoning"] + ra["content"], rt["reasoning"] + rt["content"]
                row.update(target_identical=ra == rt, common_prefix_chars=_pre(ta, tt), dflash_chars=len(ta), target_chars=len(tt),
                           same_tool_call_presence=bool(ra["calls"]) == bool(rt["calls"]), same_tool_args=ra["calls"] == rt["calls"],
                           same_finish=ra["finish"] == rt["finish"], dflash_tokens=ra["tokens"], target_tokens=rt["tokens"])
            cmp_.append(row)
        out["compare"] = cmp_
        ok_rows = [r for r in cmp_ if "target_identical" in r]
        out["summary"] = {"probes": len(cmp_), "control_identical": sum(1 for r in cmp_ if r.get("control_identical")),
                          "target_identical": sum(1 for r in ok_rows if r["target_identical"]),
                          "same_tool_call_presence": sum(1 for r in ok_rows if r["same_tool_call_presence"]),
                          "same_finish": sum(1 for r in ok_rows if r["same_finish"]), "compared": len(ok_rows)}
        print("E12 parity:", json.dumps(out["summary"]), flush=True)
        return out


    def _e12_r6_preflight(n):
        # R6-X1 (bounded, inside this session): the drafter's tap-layer config (SGLang #37475) and an isolation smoke for
        # #36548 (n concurrent requests with distinct markers; a marker in another request's reply = contamination).
        out = {}
        try:
            _cfg = json.loads((SGL_DRAFT_PATH / "config.json").read_text(encoding="utf-8"))
            _flat = dict(_cfg)
            for _v in list(_cfg.values()):
                if isinstance(_v, dict):
                    _flat.update(_v)
            out["drafter_target_layer_config"] = {k: v for k, v in _flat.items() if "layer" in k.lower()
                                                  and ("target" in k.lower() or "tap" in k.lower())}
        except Exception as _e:
            out["drafter_config_error"] = repr(_e)[:200]
        try:
            import concurrent.futures as _cf, random as _rnd
            marks = ["ZQ" + str(_rnd.randrange(10 ** 6, 10 ** 7)) + "K" + str(i) for i in range(n)]

            def _one(m):
                body = {"model": E12_RUN["model_id"], "max_tokens": 24, "temperature": 0.0,
                        "messages": [{"role": "user", "content": "Reply with exactly this code and nothing else: " + m}],
                        "chat_template_kwargs": {"enable_thinking": False}}
                req = _urlreq.Request(SGL_URL + "/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                      headers={"Content-Type": "application/json"})
                r = json.loads(_urlreq.urlopen(req, timeout=60).read())     # E21 rev 2b (C1 M3): <= ~2 min per preflight
                return str(r["choices"][0]["message"].get("content") or "")
            with _cf.ThreadPoolExecutor(n) as _ex:
                replies = list(_ex.map(_one, marks))
            out.update(E12A.isolation_verdict(marks, replies))     # C6-2: PASS / FAIL / INCONCLUSIVE
        except Exception as _e:
            out.update(isolation_error=repr(_e)[:300], isolation_state="INCONCLUSIVE", isolation_ok=None)
        try:   # R6-X1 parity CONTROL: the same request twice at T 0, one at a time; equal replies = a deterministic baseline.
            # A target-only (no DFlash) comparison needs a separate server launch and is never run beside production.
            pc = []
            for _m in ("PARITY-A-" + str(n), "PARITY-B-" + str(n)):
                _a, _b = _one(_m), _one(_m)
                pc.append({"same": _a == _b, "own_marker": _m in _a})
            out["parity_control"] = {"pairs": len(pc), "identical": sum(x["same"] for x in pc),
                                     "own_marker": sum(x["own_marker"] for x in pc),
                                     "note": "same config, batch size 1, T 0; target-only parity = separate launch (not run)"}
        except Exception as _e:
            out["parity_control"] = {"error": repr(_e)[:300]}
        print("E12 rev 6 preflight:", json.dumps(out)[:800], flush=True)
        return out


    def _e12_branch_done(rec):
        with _E12_LOCK:
            E12_RECORDS.setdefault(rec["task_id"], []).append(rec)
            E12_RUN["branches_done"] += 1
            E12_RUN["branches_accepted"] += int(bool(rec.get("accepted")))
        try:
            e12_write_submission()
        except Exception as _e:
            print("E12 submission write failed:", repr(_e)[:200], flush=True)


    _E12_STOP = threading.Event()


    def _e12_heartbeat(every=300, _stop=_E12_STOP):     # E21 rev 2c: bound to this half's stop event (GPT audit 1)
        while not _stop.wait(every):
            snap = {r.get("name"): r.get("value") for r in metrics_snapshot() if isinstance(r, dict) and "name" in r}
            keep = {k: v for k, v in snap.items() if any(s in k for s in ("num_running_reqs", "num_queue_reqs", "gen_throughput",
                                                                            "token_usage", "cache_hit_rate"))}
            print(f"E12 [{(time.perf_counter() - NOTEBOOK_T0) / 60:.0f} min] branches done {E12_RUN['branches_done']}"
                  f"/{len(TASK_IDS) * E12_BRANCHES}, accepted {E12_RUN['branches_accepted']} | {json.dumps(keep)[:400]}", flush=True)


    if ENGINE_READY and E10_SANDBOX_SELFTEST.get("ok"):
        try:
            try:
                E12_RUN["model_id"] = json.loads(_urlreq.urlopen(SGL_URL + "/v1/models", timeout=30).read())["data"][0]["id"]
            except Exception as _e:
                E12_RUN["model_id"] = E21_SERVED[(ENGINE_INFO.get("cfg") or {}).get("weights", "fp8")]   # E21: the served name
                print("E12: /v1/models failed, using the model path:", repr(_e)[:200], flush=True)
            E12_RUN["streams"] = min(int(CONCURRENCY), E12_STREAMS or int(CONCURRENCY))
            os.environ["ARC3_MAX_ACTIVE_STREAMS"] = str(E12_RUN["streams"])
            _end = min(HARD_STOP - E12_STOP_BEFORE_HARD_SEC, E21_HALF_END_PC)     # E21: this half's latest end
            if E12_SOLVE_WINDOW_SEC:
                _end = min(_end, time.perf_counter() + E12_SOLVE_WINDOW_SEC)
            if P.get("hidden_equiv"):   # gate 4b (user 2026-10-21): this run gets the hidden run's compute per task
                _ra = PROFILE_TABLE["RERUN_ALL"]
                _hid_end = _ra["time_budget_sec"] - _ra["hard_margin_sec"] - E12_STOP_BEFORE_HARD_SEC
                _start = time.perf_counter() - NOTEBOOK_T0
                _per, _ = _e12_g4b_window(_hid_end, _start, len(TASK_IDS))          # hidden test = 240 tasks
                E12_RUN["g4b"] = {"hidden_solve_end_sec": _hid_end, "solve_start_sec": round(_start), "hidden_tasks": 240,
                                  "sec_per_task": round(_per, 1), "tasks": len(TASK_IDS), "window_sec": round(_per * len(TASK_IDS))}
                _end = min(_end, time.perf_counter() + _per * len(TASK_IDS))
                print("E12 gate 4b:", json.dumps(E12_RUN["g4b"]), flush=True)
            E12_RUN["deadline_in_sec"] = round(_end - time.perf_counter())
            E12_RUN["started_sec"] = round(time.perf_counter() - NOTEBOOK_T0)
            print("E12 solving:", len(E12_ORDER), "tasks x", E12_BRANCHES, "branches | streams", E12_RUN["streams"],
                  "| model", E12_RUN["model_id"], "| stops in", E12_RUN["deadline_in_sec"], "s | order head", E12_ORDER[:5], flush=True)
            threading.Thread(target=_e12_heartbeat, daemon=True, name="e12_heartbeat").start()
            # rev 6 R6-S5 (GPT plan review 2): per-stream allowance = 90 % of the KV pool / streams, so 10 resident histories
            # plus their reply reserve fit; the reserve itself follows each request's cap (R6-D1)
            _kv = int(ENGINE_INFO.get("kv_tokens") or 866744)
            E12_RUN["rev6_kv_tokens"] = _kv
            # E15B #46 (Exp 15 Notes/02): a FIXED window, not KV / streams (16 streams -> ~54k would trim earlier than rev 6g);
            # an attempt's history stays <= ~50k, the KV admission gate (0.95) covers memory
            E12_RUN["rev6_window"] = min(131072, int(E15B["E12_B_WINDOW"]))
            os.environ["E12_DUCK_CONTEXT_WINDOW"] = str(E12_RUN["rev6_window"])
            E12_RUN["rev6_preflight"] = _e12_r6_preflight(E12_RUN["streams"])
            if E12_RUN["rev6_preflight"].get("isolation_state") == "INCONCLUSIVE":
                # H6-17 (GPT rev 6e audit 3; user 2026-10-22): a missing marker without a leak is retried once before the
                # 6-stream fail-safe; a FAIL never is
                _first = E12_RUN["rev6_preflight"]
                E12_RUN["rev6_preflight"] = _e12_r6_preflight(E12_RUN["streams"])
                E12_RUN["rev6_preflight"]["retry_of"] = {k: _first.get(k) for k in ("isolation_state", "isolation_own_marker",
                                                                                      "isolation_leaks")}
            # C6-2 (rev 6c audits): full concurrency only on an isolation PASS; FAIL / INCONCLUSIVE / error -> 6 streams
            _workers = E12A.workers_for_isolation(E12_RUN["rev6_preflight"].get("isolation_ok"), E12_RUN["streams"])
            if E12_RUN["rev6_preflight"].get("isolation_state") != "PASS":
                print("E12 rev 6 WARNING: isolation preflight", E12_RUN["rev6_preflight"].get("isolation_state"),
                      "-> workers", _workers, flush=True)
            E12_RUN["rev6_workers"] = _workers
            print("E12 rev 6: window", E12_RUN["rev6_window"], "| workers", _workers, "| first exposure",
                  E12_R6["E12_R6_FIRST_EXPOSURE"], "| branch 1 view", E12_R6["E12_R6_B1_VIEW"], flush=True)
            if not RERUN and (E12_SOLVE_WINDOW_SEC or (P.get("hidden_equiv") and E12_RUN.get("g4b"))):
                # SA-1 + H6-6 (GPT rev 6e audit 2): every commit window (gate 4b and the plain solve window) starts after the
                # preflight, bounded by the hard stop; the per-task compute of gate 4b stays the hidden run's. The hidden
                # rerun keeps its absolute end (no window). H6-3: nothing else runs before solving
                _shift = (time.perf_counter() - NOTEBOOK_T0) - E12_RUN["started_sec"]     # started_sec: set when _end was fixed
                _end = _e12_shifted_end(_end, _shift, min(HARD_STOP - E12_STOP_BEFORE_HARD_SEC, E21_HALF_END_PC))
                E12_RUN["window_shift"] = {"shift_sec": round(_shift, 1),
                                           "window_start_sec": round(time.perf_counter() - NOTEBOOK_T0)}
                if E12_RUN.get("g4b"):
                    E12_RUN["g4b"].update(window_start_sec=E12_RUN["window_shift"]["window_start_sec"], shift_sec=round(_shift, 1))
                E12_RUN["deadline_in_sec"] = round(_end - time.perf_counter())
                print("E12: the solve window starts after the preflight (shift", round(_shift, 1), "s), stops in",
                      E12_RUN["deadline_in_sec"], "s", flush=True)
            # E15B #15 (code audit H1): a server watchdog for the solve. A dead SGLang process (or /health failing for
            # 180 s) is relaunched with the same command (at most 2 times, only with >= 20 min left); while it is down the
            # branches wait (arc2_duck.SERVER_DOWN) instead of burning their consecutive-failure budget.
            E12_RUN["watchdog"] = {"relaunches": [], "health_fail_sec_max": 0, "old_logs": [], "tok_acc": 0.0, "tok_last": None,
                                   "health_probes": 0}
            _e15b_solve_end = time.perf_counter() + max(0.0, _end - time.perf_counter())
            E15B_SOLVE_END_PC = _e15b_solve_end       # B1: per-branch submission writes stop here (the final write stays)
            _e15b_tok0 = _e15b_gen_tokens_total()
            E12_RUN["watchdog"]["tok_base"] = _e15b_tok0

            # A7 (Notes/07): the limits come from env only for the CPU harness's fault test; Kaggle uses the defaults
            _WD_POLL = float(os.environ.get("E15B_WD_POLL_SEC", "10"))
            _WD_STALL = float(os.environ.get("E15B_WD_STALL_SEC", "600"))     # E21 rev 2h: no-progress limit
            _WD_FAIL = float(os.environ.get("E15B_WD_FAIL_SEC", "180"))
            _WD_MIN_LEFT = float(os.environ.get("E15B_WD_MIN_LEFT_SEC", str(20 * 60)))
            E12_RUN["watchdog"]["limits"] = {"poll_sec": _WD_POLL, "fail_sec": _WD_FAIL, "min_left_sec": _WD_MIN_LEFT}

            def _e15b_watchdog(_wd_stop=_E12_STOP, _wd_run=E12_RUN, _wd_half=E21_HALF, _e15b_solve_end=_e15b_solve_end):
                # E21 rev 2f (VAL2_cells F1): this half's objects bound at def time (a module-level binding is global
                # after wrap() and half B would rebind it)
                # E21: the E15B watchdog on vLLM: a dead server (or /health failing for _WD_FAIL s while its log is silent)
                # is relaunched once with the same half's ready config (e21_relaunch, CELL 5); while it is down the
                # branches wait (arc2_duck.SERVER_DOWN) instead of burning their consecutive-failure budget
                _bad, _sz_last = None, {}
                _wd = _wd_run["watchdog"]
                _prog = {"v": None, "t": time.monotonic()}      # rev 2h: the last progress value and when it changed
                _wd.update(stall_sec=_WD_STALL, stalls=0, no_progress_sec_max=0)
                while not _wd_stop.wait(_WD_POLL):
                    if E21_HALF != _wd_half:          # rev 2c: a half-A watchdog never acts in half B
                        return
                    _srv = E11_SERVER
                    _dead = _srv is None or not _srv.alive()
                    _stall = False
                    if not _dead:
                        _tk = _e15b_gen_tokens_total()      # B6: the last counter value of THIS server (it resets on relaunch)
                        if _tk is not None:
                            _wd["tok_last"] = _tk
                        # rev 2h (GPT v3 HIGH-1): /health 200 + a silent log while running requests make no progress = hung
                        _pr = _e21_progress()
                        if _pr is not None:
                            if not _pr[0] or _pr[1] != _prog["v"]:
                                _prog.update(v=_pr[1], t=time.monotonic())
                            else:
                                _np = time.monotonic() - _prog["t"]
                                _wd["no_progress_sec_max"] = max(_wd["no_progress_sec_max"], round(_np))
                                _stall = _np >= _WD_STALL
                    if _stall:
                        _wd["stalls"] += 1
                        print("E15B watchdog: no progress for", round(time.monotonic() - _prog["t"]), "s with",
                              _pr[0], "requests running -> treated as dead", flush=True)
                    elif not _dead:
                        try:     # B2: a growing server log = alive; /health only when it is silent
                            _sz = os.path.getsize(_srv.log_path)
                        except Exception:
                            _sz = None
                        if _sz is not None and _sz != _sz_last.get(id(_srv)):
                            _sz_last[id(_srv)] = _sz
                            _bad = None
                            continue
                        try:
                            _wd["health_probes"] += 1
                            _urlreq.urlopen(SGL_URL + "/health", timeout=10).read()
                            _bad = None
                            continue
                        except Exception:
                            _bad = _bad or time.monotonic()
                            _wd["health_fail_sec_max"] = max(_wd["health_fail_sec_max"],
                                                                             round(time.monotonic() - _bad))
                            if time.monotonic() - _bad < _WD_FAIL:
                                continue
                    _left = _e15b_solve_end - time.perf_counter()
                    _rec = {"t": round(time.perf_counter() - NOTEBOOK_T0), "dead": _dead, "left_sec": round(_left),
                            "stall": _stall}
                    if len(_wd["relaunches"]) >= 2 or _left < _WD_MIN_LEFT:
                        _rec["action"] = "none (limit or < 20 min left)"
                        _wd["relaunches"].append(_rec)
                        print("E15B watchdog: server down, no relaunch:", json.dumps(_rec), flush=True)
                        return
                    if E21_HALF != _wd_half:          # rev 2f: re-check right before acting
                        return
                    E12A.SERVER_DOWN.set()
                    print("E15B watchdog: server down -> relaunch", json.dumps(_rec), flush=True)
                    try:
                        try:
                            _wd["old_logs"].append(str(_srv.log_path))
                        except Exception:
                            pass
                        if _wd.get("tok_last") is not None and _wd.get("tok_base") is not None:   # B6
                            _wd["tok_acc"] += max(0.0, _wd["tok_last"] - _wd["tok_base"])
                        _wd["tok_base"], _wd["tok_last"] = 0.0, None
                        _ok, _why = e21_relaunch()
                        _rec.update(ok=_ok, why=_why)
                    except Exception as _e:
                        _rec.update(ok=False, why=repr(_e)[:300])
                    finally:
                        E12A.SERVER_DOWN.clear()
                        _wd["relaunches"].append(_rec)
                        _bad = None
                        _prog.update(v=None, t=time.monotonic())
                    print("E15B watchdog:", json.dumps(_rec), flush=True)
                    if not _rec.get("ok") and len(_wd["relaunches"]) >= 2:    # B3: a failed relaunch retries once
                        return
            threading.Thread(target=_e15b_watchdog, daemon=True, name="e15b_watchdog").start()
            if P.get("parity"):          # R6-X1 diagnostic profile: no solving (the fallback submission stays)
                E12_RUN["parity"] = _e12_parity()
            else:
              E12A.solve_all_r6({t: CHALLENGES[t] for t in E12_ORDER}, E12_ORDER, out_dir=E12_OUT,
                              deadline=time.monotonic() + max(0.0, _end - time.perf_counter()), base_url=SGL_URL + "/v1",
                              model_id=E12_RUN["model_id"], branches=E12_BRANCHES, max_generated_tokens=E12_MAX_GEN_PER_BRANCH,
                              save_request_logs=E12_REQUEST_LOGS, on_branch_done=_e12_branch_done_r6, workers=_workers,
                              first_exposure_tokens=E12_R6["E12_R6_FIRST_EXPOSURE"] or None, b1_view=E12_R6["E12_R6_B1_VIEW"],
                              min_admit_sec=E12_R6["E12_R6_MIN_ADMIT_SEC"], capacity_probe=_e12_capacity_probe,
                              kv_admit_limit=E12_R6["E12_R6_KV_ADMIT"], kv_poll_sec=E12_R6["E12_R6_KV_POLL"],
                              min_chunk_tokens=E12_R6["E12_R6_MIN_CHUNK"],
                              fair_share=bool(E12_R6["E12_R6_FAIR_SHARE"]),
                              stage2_p90_cutoff=bool(E12_R6["E12_R6_P90_CUTOFF"]),
                              join_grace_sec=90.0,       # E15B #54: room for the final harvest / partial test (30 s grace)
                              max_generated_tokens_b1=E15B["E12_B_ATTEMPT_TOKENS_B1"],
                              skip_b1_if_accepted=bool(E15B["E12_B_SKIP_B1_IF_ACCEPTED"]),     # rev 3i: R
                              extra_attempts=bool(E15B["E12_B_EXTRA_ATTEMPTS"]),               # rev 3i: stage 3
                              max_attempts=int(E15B["E12_B_MAX_ATTEMPTS"]),
                              # rev 3j: 1 = after the attempt 3s (notebook 1), 2 = before the attempt 3s (notebook 2)
                              b1_accepted_mode={0: "", 1: "after3", 2: "before3"}[int(E15B["E12_B_B1_ACC_MODE"])],   # NB6 E4: 0 -> "" (E15B SUBMIT)
                              weak_commit=bool(E15B["E12_B_WEAK_COMMIT"]))    # rev 3f: attempt 2's session cap
            E12_RUN["rev6_scheduler"] = dict(E12A.SCHED_LAST)
            # E21 rev 2c (GPT audits 1-2): the deadline contract as evidence (planned solve end vs when the solver returned)
            E12_RUN["solve_overrun_sec"] = round(max(0.0, time.perf_counter() - _e15b_solve_end), 1)
            E12_RUN["planned_solve_end_sec"] = round(_e15b_solve_end - NOTEBOOK_T0, 1)
            E12_RUN["solve_end_epoch"] = time.time() + (_e15b_solve_end - time.perf_counter())   # rev 2f: request t / t0 are epoch
            E12_RUN["stragglers"] = (E12_RUN["rev6_scheduler"] or {}).get("stragglers")
            if E12_RUN["solve_overrun_sec"] > 120:
                E12_RUN["deadline_violation"] = True
                print("E21 WARNING: the solver returned", E12_RUN["solve_overrun_sec"], "s after its deadline", flush=True)
            try:      # E15B #29, #30, #39: per-attempt summary + real concurrency from the 30-s server gauges
                _att = [r for v in E12_RECORDS.values() for r in v]
                _ev = [r.get("b_events") or {} for r in _att]
                # rev 3f: each attempt's own budget (attempt 1 = E12_B_ATTEMPT_TOKENS, attempt 2 = E12_B_ATTEMPT_TOKENS_B1)
                _bud = lambda r: int(E15B["E12_B_ATTEMPT_TOKENS_B1"] if r.get("branch") >= 1 else E15B["E12_B_ATTEMPT_TOKENS"])
                _g = [x for x in (E12_RUN["rev6_scheduler"].get("gauges") or []) if isinstance(x, list) and len(x) >= 5
                      and isinstance(x[3], (int, float))]
                _run = [x[3] for x in _g]
                _q = [x[4] for x in _g if isinstance(x[4], (int, float))]
                # A1 (Notes/07): the 30-s samples in which every client stream was busy (workers_running >= streams)
                _S = E12_RUN.get("streams") or 0
                _full = [x for x in _g if len(x) >= 6 and isinstance(x[5], (int, float)) and _S and x[5] >= _S]
                E12_RUN["b_summary"] = {
                    "attempts": len(_att), "attempt2_started": sum(1 for r in _att if r.get("branch") == 1),
                    "accepted": sum(1 for r in _att if r.get("accepted")),
                    "at_budget": sum(1 for r in _att if int(r.get("generated_tokens") or 0) >= _bud(r) - 64),
                    # no host-accepted candidate at all (an accepted exact check counts too), but a transform was written
                    "at_budget_with_progress": sum(1 for r in _att if int(r.get("generated_tokens") or 0)
                                                   >= _bud(r) - 64
                                                   and not any(x.get("accepted") for x in (r.get("submissions") or [])
                                                               + (r.get("checked") or []))
                                                   and ((r.get("rev6_events") or {}).get("first_transform_tokens") is not None)),
                    "commit_turns": sum(1 for e in _ev if e.get("commit_at_tokens") is not None),
                    "commit_with_tool_call": sum(1 for e in _ev if e.get("commit_tool_call")),
                    "submit_tool_routed": sum(int(e.get("submit_tool_routed") or 0) for e in _ev),
                    "reasoning_harvest_ran": sum(1 for e in _ev if (e.get("reasoning_harvest") or {}).get("ran")),
                    "reasoning_harvest_new_accepted": sum(int((e.get("reasoning_harvest") or {}).get("new_accepted") or 0) for e in _ev),
                    "trimmed_requests": sum(1 for r in _att for q in (r.get("requests") or []) if q.get("trimmed")),
                    "worker_errors": E12_RUN["rev6_scheduler"].get("worker_errors"),
                    "setup_failed": E12_RUN["rev6_scheduler"].get("setup_failed"),     # rev 3h: Auditor 4 #4
                    "cut_code_saved": sum(int(e.get("cut_code_saved") or 0) for e in _ev),     # rev 3h
                    "cut_code_harvested": sum(1 for e in _ev if (e.get("reasoning_harvest") or {}).get("cut_code_blocks")),
                    # rev 3i: R + stage 3
                    "b1_skipped": E12_RUN["rev6_scheduler"].get("b1_skipped"),
                    "extra_started": E12_RUN["rev6_scheduler"].get("extra_started"),
                    "extra_by_attempt": E12_RUN["rev6_scheduler"].get("extra_by_attempt"),
                    "extra_accepted": sum(1 for r in _att if int(r.get("branch") or 0) >= 2 and r.get("accepted")),
                    "b1_deferred": E12_RUN["rev6_scheduler"].get("b1_deferred"),                 # rev 3j
                    "b1_deferred_started": E12_RUN["rev6_scheduler"].get("b1_deferred_started"),
                    "weak_accepts": E12_RUN["rev6_scheduler"].get("weak_accepts"),               # rev 3k W1
                    "weak_only_tasks_continued": E12_RUN["rev6_scheduler"].get("weak_only_tasks_continued"),
                    "stage2_errors": E12_RUN["rev6_scheduler"].get("stage2_errors"),
                    "server_running_mean": round(sum(_run) / len(_run), 2) if _run else None,
                    "server_queue_gt0_share": round(sum(1 for v in _q if v > 0) / len(_q), 3) if _q else None,
                    "gauge_samples": len(_g), "streams": E12_RUN.get("streams"),
                    "client_full_samples": len(_full),
                    "server_running_mean_client_full": round(sum(x[3] for x in _full) / len(_full), 2) if _full else None,
                    "server_at_streams_share_client_full": (round(sum(1 for x in _full if x[3] >= _S) / len(_full), 3)
                                                            if _full else None),
                    "kv_stale_checks": E12_RUN["rev6_scheduler"].get("kv_stale_checks"),      # A4
                    # #34: accepted submissions vs harvested exact checks; #31 roll-up; #35 the effort actually sent
                    "accepted_submit": sum(int((r.get("b_candidate") or {}).get("submitted_accepted") or 0) > 0 for r in _att),
                    "harvested_only": sum(1 for r in _att if int((r.get("b_candidate") or {}).get("harvested") or 0) > 0
                                          and not int((r.get("b_candidate") or {}).get("submitted_accepted") or 0)),
                    "never_wrote_candidate": sum(1 for r in _att if not (r.get("b_candidate") or {}).get("candidate_source_written")),
                    "wrote_never_checked": sum(1 for r in _att if (r.get("b_candidate") or {}).get("candidate_source_written")
                                               and not (r.get("b_candidate") or {}).get("check_executed")),
                    "partial_test_runs": sum(1 for e in _ev if (e.get("partial_test") or {}).get("name")),
                    "effective_effort": E12_REASONING_EFFORT or "xhigh (template default)",
                    # #32: server-side generated tokens in the solve vs the attempts' count
                    # B6: per server (the counter resets on a relaunch): the finished servers' deltas + the current one
                    "server_gen_tokens": (lambda a, b: round((E12_RUN["watchdog"].get("tok_acc") or 0.0) + a - b)
                                          if a is not None and b is not None else None)(
                        _e15b_gen_tokens_total(), E12_RUN["watchdog"].get("tok_base")),
                    "counted_gen_tokens": sum(int(r.get("generated_tokens") or 0) for r in _att),
                    "server_down_waits": dict(E12A.SERVER_WAITS), "watchdog": E12_RUN.get("watchdog")}
                try:
                    E12_RUN["b_summary"]["server_log"] = _e15b_server_log_summary(_e15b_all_server_logs())   # #42 / #53; B6
                except Exception as _e:
                    E12_RUN["b_summary"]["server_log"] = {"error": repr(_e)[:200]}
                print("E15B summary:", json.dumps(E12_RUN["b_summary"]), flush=True)
                if _run and E12_RUN["b_summary"]["server_running_mean"] < (E12_RUN.get("streams") or 0) - 0.5:
                    print("E15B WARNING: mean server running", E12_RUN["b_summary"]["server_running_mean"], "< streams - 0.5",
                          flush=True)
            except Exception as _e:
                E12_RUN["b_summary"] = {"error": repr(_e)[:300]}
            # H6-3 (GPT rev 6e audits): the cache diagnostic runs AFTER solving, so the solver starts from the same server
            # state as the hidden rerun (which never runs it). Its time comes from what is left before the output margin;
            # the solve deadline never moves for it. Skipped when under 90 s are left
            if not RERUN and not P.get("parity") and E12_R6.get("E12_R6_CACHE_DIAG"):
                _left = HARD_STOP - E12_STOP_BEFORE_HARD_SEC - time.perf_counter() - 30
                if (E12_RUN.get("rev6_scheduler") or {}).get("stragglers"):      # rev 6g H7-4
                    E12_RUN["cache_diag"] = {"skipped": "a solver straggler may still be sending requests"}
                elif _left >= 90:
                    try:
                        E12_RUN["cache_diag"] = _e12_cache_diag(budget_sec=min(240.0, _left))
                        E12_RUN["cache_diag"]["ran"] = "after solving"
                    except Exception as _e:
                        E12_RUN["cache_diag"] = {"error": repr(_e)[:300]}
                else:
                    E12_RUN["cache_diag"] = {"skipped": "less than 90 s before the output margin", "left_sec": round(_left)}
        except BaseException:
            E12_RUN["error"] = traceback.format_exc()[-3000:]
            print("E12 solver error (submission.json keeps what was written):", E12_RUN["error"], flush=True)
    else:
        E12_RUN["error"] = f"not run: ENGINE_READY={ENGINE_READY}, sandbox self-test={E10_SANDBOX_SELFTEST.get('ok')}"
        print("E12:", E12_RUN["error"], "-> CELL 0 fallback submission", flush=True)
    _E12_STOP.set()
    E12_RUN["finished_sec"] = round(time.perf_counter() - NOTEBOOK_T0)
    try:
        e12_write_submission()
    except Exception:
        print("E12 final submission write failed:", traceback.format_exc()[-1000:], flush=True)


def e21_cell7():
    # E15B CELL 7 (R3K2) with a per-half bundle name, run once per half
    global _recs, _gen, _usage_summary, branches, t, r, subs, _all_u, report, _r6_metrics, _r6_metrics_more, _r6e_metrics, _zero_classes, _r6e_concurrency, hits, pass1, pass2, total, task_norm, recs_eval, th, qi, att, gold, a, b, acc, harv, _e15b_has, _B_TOK, _pa, _rows, _rs, _golds, _f, a0, a1, n0, n1, hit, r0, bp, k, v, _x, _ok, _hi, _e, _e11e, _gh, _solved, _yt, _sc, _pack, z, fp
    # ==================================================================== CELL 7 — E12: report, score (evaluation) and bundle
    try:
        with _E12_LOCK:
            _recs = {t: list(v) for t, v in E12_RECORDS.items()}
        _gen = E12A.build_submission({t: CHALLENGES[t] for t in TASK_IDS}, _recs)
        write_json_atomic(OUT / "e12_generated_attempts.json", _gen)

        def _usage_summary(u):
            p = sum(int(x.get("prompt") or 0) for x in u); c = sum(int(x.get("cached") or 0) for x in u)
            return {"requests": len(u), "prompt_tokens": p, "cached_tokens": c, "cached_pct": round(100 * c / p, 1) if p else None,
                    "completion_tokens": sum(int(x.get("completion") or 0) for x in u),
                    "cached_pct_by_request": [round(100 * int(x.get("cached") or 0) / int(x["prompt"]), 1) if x.get("prompt") else None
                                              for x in u][:400]}
        branches = []
        for t in TASK_IDS:
            for r in _recs.get(t, []):
                subs = r.get("submissions") or []
                branches.append({"task": t, "branch": r.get("branch"), "accepted": bool(r.get("accepted")), "turns": r.get("turns"),
                                 "failures": r.get("failures"), "note": r.get("note"), "generated_tokens": r.get("generated_tokens"),
                                 "total_tokens": r.get("total_tokens"), "sec": round((r.get("t1") or 0) - (r.get("t0") or 0), 1),
                                 "submissions": len(subs), "train_exact_per_submission": [sum(map(bool, s.get("train_exact") or []))
                                                                                           for s in subs],
                                 "usage": _usage_summary(r.get("usage") or []),
                                 # rev 5 metrics (absent before rev 5): harvested check() candidates and event counters
                                 "checked_verified": len(r.get("checked") or []),
                                 "checked_kept": sum(1 for s in (r.get("checked") or []) if s.get("accepted")),
                                 "rev5_events": r.get("rev5_events"),
                                 # rev 6 (absent before rev 6)
                                 "view": r.get("view"), "status": r.get("status"), "runs": r.get("runs"),
                                 "progress_best": (r.get("progress") or {}).get("best"), "rev6_events": r.get("rev6_events"),
                                 "stage2_class": r.get("stage2_class"),
                                 "overfit_flags": sorted({f for v in (r.get("checked") or []) + (r.get("submissions") or [])
                                                          if v.get("accepted") for f in (v.get("overfit_flags") or [])})})
        _all_u = [x for t in TASK_IDS for r in _recs.get(t, []) for x in (r.get("usage") or [])]
        report = {"run_id": RUN_ID, "profile": PROFILE, "rerun": RERUN, "configuration": CONFIG, "engine": ENGINE_INFO,
                  "e12_run": E12_RUN, "order": E12_ORDER, "branches": branches, "usage_all": {k: v for k, v in _usage_summary(_all_u).items()
                                                                                            if k != "cached_pct_by_request"},
                  "tasks_planned": len(TASK_IDS), "queries_planned": sum(len(CHALLENGES[t]["test"]) for t in TASK_IDS)}
        def _r6_metrics(rows):   # R6-M1: one report contract (censored first-check = the branch total)
            import statistics as _st
            med = lambda v: _st.median(v) if v else None
            ev = [r.get("rev6_events") or {} for r in rows]
            tot = lambda k: sum(int(e.get(k) or 0) for e in ev)
            ft = [e.get("first_tool_tokens") for e in ev if e.get("first_tool_tokens") is not None]
            fc = [(e.get("first_check_tokens") if e.get("first_check_tokens") is not None else r.get("generated_tokens") or 0)
                  for e, r in zip(ev, rows)]
            hist = lambda k: {kk: sum(int((e.get(k) or {}).get(kk, 0)) for e in ev)
                              for kk in sorted({x for e in ev for x in (e.get(k) or {})})}
            vs = sorted(x for e in ev for x in (e.get("verify_sec") or []))
            b0 = [r for r in rows if r.get("branch") == 0]
            return {"branches": len(rows),
                    "tasks_exposed": len({r.get("task") for r in b0 if (r.get("generated_tokens") or 0) > 0}),
                    "tasks_meaningful": len({r.get("task") for r in rows
                                             if (r.get("rev6_events") or {}).get("first_tool_tokens") is not None}),
                    "first_tool_tokens_median": med(ft), "first_check_tokens_median_censored": med(fc),
                    # G4-13 (GPT G4B audit 1 M1-B): the observed median is over checked branches only (selection bias)
                    "first_check_tokens_median_observed": med([e["first_check_tokens"] for e in ev
                                                               if e.get("first_check_tokens") is not None]),
                    "first_check_tokens_median_observed_b0": med([e["first_check_tokens"] for e, r in zip(ev, rows)
                                                                  if e.get("first_check_tokens") is not None and r.get("branch") == 0]),
                    "checked_branches": sum(1 for e in ev if e.get("first_check_tokens") is not None),
                    "never_checked": sum(1 for e in ev if e.get("first_check_tokens") is None),
                    "autocheck_calls": tot("autocheck_calls"), "autocheck_functions": tot("autocheck_functions"),
                    "grid_rows_suppressed": tot("grid_rows_suppressed"), "length_cut_no_call": tot("length_cut_no_call"),
                    "nudges_replaced": tot("nudges_replaced"), "timeouts_clamped": tot("timeouts_clamped"),
                    "verify_skipped_deadline": tot("verify_skipped_deadline"),
                    "accepted_duplicate_outputs": tot("accepted_duplicate_outputs"),
                    "resumes": hist("resumes"), "caps": hist("caps"),
                    "verify_sec_p50": vs[len(vs) // 2] if vs else None, "verify_sec_p90": vs[int(len(vs) * 0.9)] if vs else None,
                    "status": {s: sum(1 for r in rows if r.get("status") == s) for s in ("paused", "done", "new")},
                    **_r6_metrics_more([dict(r, task=t) for t in TASK_IDS for r in _recs.get(t, [])])}

        def _r6_metrics_more(rows):   # rev 6b additions (fix-list R6-M1; raw branch records; label-free, no truth)
            ev = [r.get("rev6_events") or {} for r in rows]
            q = lambda v, f: sorted(v)[min(len(v) - 1, int(f * len(v)))] if v else None
            tot = lambda k: sum(int(e.get(k) or 0) for e in ev)
            rc = tot("reasoning_chars")
            ft = [e.get("facts_tokens") for e in ev if e.get("facts_tokens") is not None]
            acc = [(r, v) for r in rows for v in (r.get("checked") or []) + (r.get("submissions") or []) if v.get("accepted")]
            # G4-7 / G4-13 (GPT G4B audits): a check() and a submit() of one program are 2 events, 1 program
            uniq, preds, progs = {}, {}, {}
            for r, v in acc:
                uniq.setdefault((r.get("task"), r.get("branch"), v.get("name"), json.dumps(v.get("test_outputs"))), v)
                preds.setdefault(r.get("task"), set()).add(json.dumps(v.get("test_outputs")))
                # H6-9 (GPT rev 6e audits 2/3): a program = its source + retained dependencies (src_key), across branches
                progs.setdefault(r.get("task"), set()).add(v.get("src_key") or (r.get("branch"), v.get("name"),
                                                                                 json.dumps(v.get("test_outputs"))))
            flags = {}
            for v in uniq.values():
                for f in v.get("overfit_flags") or []:
                    flags[f] = flags.get(f, 0) + 1
            by_task = {}
            for r, v in acc:
                by_task.setdefault(r.get("task"), {}).setdefault(r.get("branch"), set()).add(json.dumps(v.get("test_outputs")))
            both = [t for t, d in by_task.items() if len(d) >= 2]
            agree = sum(1 for t in both if set.intersection(*by_task[t].values()))
            b1 = {}
            for r in rows:
                if r.get("branch") == 1 and r.get("stage2_class") is not None:
                    own = by_task.get(r.get("task"), {})
                    new = len(own.get(1, set()) - own.get(0, set()))
                    c = b1.setdefault(str(r["stage2_class"]), {"branches": 0, "new_distinct_verified": 0, "tokens": 0})
                    c["branches"] += 1
                    c["new_distinct_verified"] += new
                    c["tokens"] += int(r.get("generated_tokens") or 0)
            runs = [x for r in rows for x in (r.get("runs") or [])]
            osh = [x["overshoot"] for x in runs if "overshoot" in x]
            rp = [x["first_uncached"] for x in runs if x.get("stage") == 2 and "first_uncached" in x]
            pe = [a / b for r in rows for a, b in ((r.get("rev5_events") or {}).get("prompt_estimates") or []) if b]
            us = [u for r in rows for u in (r.get("usage") or [])]
            try:
                dec = e11_decode_summary() or {}
            except Exception:
                dec = {}
            rh = dec.get("running_hist") or {}
            return {"facts_tokens_p50": q(ft, .5), "facts_tokens_p90": q(ft, .9), "facts_tokens_max": max(ft) if ft else None,
                    "hand_reading_grid_row_share": round(tot("reasoning_grid_row_chars") / rc, 4) if rc else None,
                    "reasoning_digit_share": round(tot("reasoning_digit_chars") / rc, 4) if rc else None,
                    "reasoning_repeat_share": round(tot("reasoning_repeat_chars") / rc, 4) if rc else None,
                    "tool_calls": tot("tool_calls"), "redefinition_share": round(tot("redefinitions") / max(1, tot("tool_calls")), 4),
                    "infra_errors": tot("infra_errors"), "trim_reasoning_only_blocks": tot("trim_reasoning_only_blocks"),
                    "facts_over_target": sum(1 for e in ev if e.get("facts_over_target")),
                    "accepted_events": len(acc), "accepted_candidates": len(uniq), "unique_verified_programs": len(uniq),
                    # H7-5: the field above is keyed (task, branch, name, outputs), i.e. name / output candidates; the program
                    # count by source + dependencies is verified_programs_by_source
                    "unique_name_output_candidates": len(uniq),
                    "distinct_test_predictions": sum(len(v) for v in preds.values()),
                    # H6-9: the old field counted distinct predictions; both are reported now
                    "verified_programs_by_source": sum(len(v) for v in progs.values()),
                    "verified_programs_per_task": {t: len(v) for t, v in sorted(progs.items())},
                    "distinct_predictions_per_task": {t: len(v) for t, v in sorted(preds.items())},
                    "overfit_flags": flags, "overfit_flags_counted_on": "unique programs",
                    "tasks_with_two_branch_candidates": len(both), "tasks_where_branches_agree": agree,
                    "branch1_yield_by_stage2_class": b1,
                    "overshoot_p50": q(osh, .5), "overshoot_max": max(osh) if osh else None,
                    "stage2_resume_uncached_p50": q(rp, .5), "stage2_resume_uncached_p90": q(rp, .9),
                    "prompt_estimate_over_actual_p50": round(q(pe, .5), 3) if pe else None,
                    "uncached_prompt_tokens": sum(int(u.get("prompt") or 0) - int(u.get("cached") or 0) for u in us),
                    "generated_tokens": sum(int(r.get("generated_tokens") or 0) for r in rows),
                    "real_running_max": max((int(k) for k, v in rh.items() if v), default=None) if rh else None,
                    "accept_len_median": dec.get("accept_len_median"),
                    "fair_share_last": ((E12_RUN.get("rev6_scheduler") or {}).get("shares") or [None])[-1],
                    **_r6e_metrics(rows, ev, b1, us)}

        def _r6e_metrics(rows, ev, b1, us):   # rev 6e: G4-14 / G4-15 / G4-18 (label-free, no truth)
            import statistics as _st
            med = lambda v: round(_st.median(v), 1) if v else None
            n = len(rows)
            ftt = [e.get("first_transform_tokens") for e in ev]
            obs = [x for x in ftt if x is not None]
            cens = [x if x is not None else int(r.get("generated_tokens") or 0) for x, r in zip(ftt, rows)]
            best = {}
            for r in rows:
                b = ((r.get("progress") or {}).get("best") or {})
                k = f"{b.get('k')}/{b.get('n')}" if b else "none"
                best[k] = best.get(k, 0) + 1
            b1_tok = sum(c["tokens"] for c in b1.values())
            b1_new = sum(c["new_distinct_verified"] for c in b1.values())
            reqs = [q for r in rows for q in (r.get("requests") or [])]
            later = [q for q in reqs if q.get("i", 0) > 0 and q.get("prompt")]
            zero = [q for q in later if not q.get("cached")]
            runs2 = [x for r in rows for x in (r.get("runs") or []) if x.get("stage") == 2]

            def _yield(sel):   # H6-14 / H7-3: new verified predictions per 100k generated tokens (all / b0 / b1, by stage)
                rr = [(x, r) for r in rows for x in (r.get("runs") or []) if sel(x, r)]
                tok = sum(int(x.get("tokens") or 0) for x, _ in rr)
                nb = sum(int(x.get("new_in_branch") or 0) for x, _ in rr)
                nt = sum(int(x.get("new_for_task") or 0) for x, _ in rr)
                return {"runs": len(rr), "tokens": tok, "new_in_branch": nb, "new_for_task": nt,
                        "per_100k_for_task": round(1e5 * nt / tok, 3) if tok else None,
                        "per_100k_in_branch": round(1e5 * nb / tok, 3) if tok else None}
            nud = [(e, r) for e, r in zip(ev, rows) if e.get("provisional_nudge_tokens") is not None]
            gap = lambda k: [e[k] - e["provisional_nudge_tokens"] for e, _ in nud if e.get(k) is not None]
            return {
                # H6-14 (GPT rev 6e audits): the allocation question is decided on these
                "verified_yield": {"all": _yield(lambda x, r: True), "b0": _yield(lambda x, r: r.get("branch") == 0),
                                   "b1": _yield(lambda x, r: r.get("branch") == 1),
                                   "stage1": _yield(lambda x, r: x.get("stage") == 1),
                                   "stage2": _yield(lambda x, r: x.get("stage") == 2)},
                # H6-18: nudge efficacy (H6-1: sent in a tool result at 24k, else in an opener)
                "nudge": {"branches": len(nud), "via": {k: sum(1 for e, _ in nud if e.get("provisional_nudge_via") == k)
                                                         for k in ("tool", "opener")},
                          # H7-10: emitted (above) vs seen by the model in a later request
                          "seen": sum(1 for e, _ in nud if e.get("provisional_nudge_seen_request") is not None),
                          "emitted_not_seen": sum(1 for e, _ in nud if e.get("provisional_nudge_seen_request") is None),
                          "tokens_p50": med([e["provisional_nudge_tokens"] for e, _ in nud]),
                          "transform_after": len(gap("first_transform_after_nudge")),
                          "tokens_to_transform_p50": med(gap("first_transform_after_nudge")),
                          "check_after": len(gap("first_check_after_nudge")),
                          "tokens_to_check_p50": med(gap("first_check_after_nudge")),
                          "accepted": sum(1 for _, r in nud if r.get("accepted")
                                          or any(v.get("accepted") for v in (r.get("checked") or [])))},
                "progress_same_name_new_source": sum(int(e.get("progress_same_name_new_source") or 0) for e in ev),
                "indexed_view_over_target": sum(1 for e in ev if e.get("view_over_target")),             # H7-8
                "tasks_with_response": len({r.get("task") for r in rows if r.get("branch") == 0
                                            and (r.get("generated_tokens") or 0) > 0}),                  # H7-7
                # G4-14
                "first_transform_tokens_median_observed": med(obs), "first_transform_tokens_median_censored": med(cens),
                "branches_with_transform": len(obs), "branches_with_transform_within_20k": sum(1 for x in obs if x <= 20000),
                "share_with_transform_within_20k": round(sum(1 for x in obs if x <= 20000) / n, 3) if n else None,
                "best_single_function_pairs": dict(sorted(best.items())),
                "b1_new_distinct_verified_per_100k_tokens": round(1e5 * b1_new / b1_tok, 3) if b1_tok else None,
                "provisional_nudges": sum(1 for e in ev if e.get("provisional_nudge_tokens") is not None),
                # G4-18
                "progress_reports": sum(int(e.get("progress_reports") or 0) for e in ev),
                "progress_same_improved": sum(int(e.get("progress_same_improved") or 0) for e in ev),
                "progress_new_hypothesis": sum(int(e.get("progress_new_hypothesis") or 0) for e in ev),
                "stage2_runs": len(runs2), "stage2_runs_improved": sum(1 for x in runs2 if x.get("improved")),
                # G4-15 (null prompt_tokens_details = 0 cached: SGLang 0.5.21 omits it when nothing was cached)
                "requests_instrumented": len(reqs), "later_requests": len(later), "later_zero_hit": len(zero),
                "later_zero_hit_prefix_intact": sum(1 for q in zero if q.get("prefix_ok")),
                "later_zero_hit_trimmed": sum(1 for q in zero if q.get("trimmed")),
                "later_zero_hit_idle_sec_p50": med([q["idle_sec"] for q in zero if q.get("idle_sec") is not None]),
                "later_idle_sec_p50": med([q["idle_sec"] for q in later if q.get("idle_sec") is not None]),
                "later_zero_hit_mamba_p50": med([q["mamba"] for q in zero if q.get("mamba") is not None]),
                "later_prefix_intact_share": round(sum(1 for q in later if q.get("prefix_ok")) / len(later), 3) if later else None,
                # H6-10: character-prefix share and tools / template changes for the zero-hit later requests (still a proxy)
                "later_zero_hit_char_prefix_full": sum(1 for q in zero if q.get("serialized_char_prefix") is not None
                                                       and q.get("prev_chars") and q["serialized_char_prefix"] >= q["prev_chars"] - 2),
                # H7-13 (GPT rev 6f audit 3): unexplained = the serialized prompt kept its prefix, tools / template unchanged,
                # not trimmed, and still 0 cached; otherwise the prompt changed (reason). A proxy, never proof of eviction
                "later_zero_hit_class": _zero_classes(zero),
                "later_zero_hit_tools_or_template_changed": sum(1 for q in zero if q.get("tools_same") is False
                                                                or q.get("template_same") is False),
                "cached_not_reported_requests": sum(1 for q in reqs if q.get("cached_reported") is False),
                "cache_null_detail_requests": sum(1 for u in us if u.get("cached") is None),
                "cache_note": "aggregate over all requests; a missing prompt_tokens_details means 0 cached (vLLM 0.19 / 0.32, as SGLang 0.5.21)",
                "concurrency": _r6e_concurrency()}

        def _zero_classes(zero):   # H7-13
            out = {}
            for q in zero:
                full = (q.get("serialized_char_prefix") is not None and q.get("prev_chars")
                        and q["serialized_char_prefix"] >= q["prev_chars"] - 2)
                if q.get("trimmed"):
                    k = "prompt_changed:trimmed"
                elif q.get("tools_same") is False:
                    k = "prompt_changed:tools"
                elif q.get("template_same") is False:
                    k = "prompt_changed:template"
                elif not full:
                    k = "prompt_changed:prefix"
                else:
                    k = "unexplained_cache_miss"
                out[k] = out.get(k, 0) + 1
            return out

        def _r6e_concurrency():   # G4-15 / Q5: decode rate by running count, split by "a prefill batch ran just before"
            import re as _re, statistics as _st
            try:
                text = E11_SERVER.log_text() if E11_SERVER is not None else ""
            except Exception:
                return None
            g = lambda pat, ln: (lambda m: float(m.group(1)) if m else None)(_re.search(pat, ln))
            by, after = {}, False
            for ln in text.splitlines():
                if "TP0]" not in ln:
                    continue
                if "Prefill batch" in ln:
                    after = True
                    continue
                if "Decode batch" not in ln:
                    continue
                n = g(r"#running-req: (\d+)", ln)
                row = by.setdefault(str(int(n)) if n is not None else "?", {"lines": 0, "tok": [], "tok_after_prefill": [],
                                                                             "accept": [], "mamba": []})
                row["lines"] += 1
                t = g(r"gen throughput \(token/s\): ([\d.]+)", ln)
                if t is not None:
                    (row["tok_after_prefill"] if after else row["tok"]).append(t)
                for k, pat in (("accept", r"accept len: ([\d.]+)"), ("mamba", r"mamba usage: ([\d.]+)")):
                    v = g(pat, ln)
                    if v is not None:
                        row[k].append(v)
                after = False
            m = lambda v: round(_st.median(v), 2) if v else None
            return {k: {"lines": v["lines"], "tok_s_p50_clean": m(v["tok"]), "tok_s_p50_after_prefill": m(v["tok_after_prefill"]),
                        "share_after_prefill": round(len(v["tok_after_prefill"]) / max(1, len(v["tok"]) + len(v["tok_after_prefill"])), 3),
                        "accept_len_p50": m(v["accept"]), "mamba_usage_p50": m(v["mamba"])}
                    for k, v in sorted(by.items(), key=lambda kv: (len(kv[0]), kv[0]))}
        if _E12_REV6:
            report["rev6"] = {"metrics": _r6_metrics(branches), "scheduler": E12_RUN.get("rev6_scheduler"),
                              "window": E12_RUN.get("rev6_window"), "workers": E12_RUN.get("rev6_workers"),
                              "preflight": E12_RUN.get("rev6_preflight"), "g4b": E12_RUN.get("g4b"),
                              "parity": E12_RUN.get("parity"), "cache_diag": E12_RUN.get("cache_diag")}
        if SOLUTIONS is not None:
            hits = pass1 = pass2 = total = 0; task_norm = 0.0; recs_eval = []
            for t in TASK_IDS:
                th = 0
                for qi, att in enumerate(_gen[t]):
                    gold = SOLUTIONS[t][qi]; a = att["attempt_1"] == gold; b = att["attempt_2"] == gold
                    hits += a or b; pass1 += a; pass2 += b; total += 1; th += a or b
                    acc = [s for r in _recs.get(t, []) for s in (r.get("submissions") or []) if s.get("accepted")]
                    harv = [s for r in _recs.get(t, []) for s in (r.get("checked") or []) if s.get("accepted")]
                    recs_eval.append({"task": t, "query": qi, "hit": bool(a or b), "pass1": bool(a), "pass2": bool(b),
                                      "accepted_programs": len(acc),
                                      "accepted_programs_correct": sum(1 for s in acc if qi < len(s.get("test_outputs") or [])
                                                                       and s["test_outputs"][qi] == gold),
                                      "harvested_programs": len(harv),
                                      "harvested_programs_correct": sum(1 for s in harv if qi < len(s.get("test_outputs") or [])
                                                                        and s["test_outputs"][qi] == gold)})
                task_norm += th / max(1, len(_gen[t]))
            report.update(eval_query_hits=hits, eval_query_total=total, eval_pass1_hits=pass1, eval_pass2_hits=pass2,
                          eval_task_norm_pct=round(100 * task_norm / max(1, len(TASK_IDS)), 2), eval_unit_records=recs_eval)
            try:    # E15B A2 (Exp 15 Notes/07): what each attempt found. A task counts for an attempt when that attempt has a
                # right output for EVERY test input: "acc" = among its host-accepted candidates, "any" = also partial ones
                def _e15b_has(r, qi, gold, acc_only):
                    return any(qi < len(s.get("test_outputs") or []) and s["test_outputs"][qi] == gold
                               for s in (r.get("submissions") or []) + (r.get("checked") or []) if s.get("accepted") or not acc_only)
                _B_TOK = int(E15B["E12_B_ATTEMPT_TOKENS"]) if "E15B" in globals() else 35000
                _pa = dict.fromkeys(("tasks", "b1_started", "hit", "b0_acc", "b1_acc", "both_acc", "b0_only_acc", "b1_only_acc",
                                     "b0_any", "b1_any", "b1_only_any", "hit_without_b0_any", "b0_budget_partial",
                                     "b0_budget_partial_b1_acc"), 0)
                _rows = []
                for t in TASK_IDS:
                    _rs = {r.get("branch"): r for r in _recs.get(t, [])}
                    _golds = SOLUTIONS[t]
                    _f = lambda b, acc: b in _rs and all(_e15b_has(_rs[b], qi, g, acc) for qi, g in enumerate(_golds))
                    a0, a1, n0, n1 = _f(0, True), _f(1, True), _f(0, False), _f(1, False)
                    hit = all(att["attempt_1"] == g or att["attempt_2"] == g for att, g in zip(_gen[t], _golds))
                    r0 = _rs.get(0) or {}
                    bp = (int(r0.get("generated_tokens") or 0) >= _B_TOK - 64
                          and not any(x.get("accepted") for x in (r0.get("submissions") or []) + (r0.get("checked") or []))
                          and (r0.get("rev6_events") or {}).get("first_transform_tokens") is not None)
                    for k, v in (("tasks", True), ("b1_started", 1 in _rs), ("hit", hit), ("b0_acc", a0), ("b1_acc", a1),
                                 ("both_acc", a0 and a1), ("b0_only_acc", a0 and not a1), ("b1_only_acc", a1 and not a0),
                                 ("b0_any", n0), ("b1_any", n1), ("b1_only_any", n1 and not n0),
                                 ("hit_without_b0_any", hit and not n0), ("b0_budget_partial", bp),
                                 ("b0_budget_partial_b1_acc", bp and a1)):
                        _pa[k] += int(bool(v))
                    _rows.append({"task": t, "hit": hit, "b0_acc": a0, "b1_acc": a1, "b0_any": n0, "b1_any": n1,
                                  "b0_budget_partial": bp, "b0_tokens": r0.get("generated_tokens"),
                                  "b1_tokens": (_rs.get(1) or {}).get("generated_tokens")})
                report["eval_b_attempts"] = dict(_pa, rows=_rows)
                # rev 3i: what stage 3 added (a task hit whose only right accepted answer came from attempt 3+)
                _x = {"tasks_with_attempt3plus": 0, "attempt3plus_acc_tasks": 0, "hit_only_from_attempt3plus": 0}
                for t in TASK_IDS:
                    _rs = {r.get("branch"): r for r in _recs.get(t, [])}
                    _golds = SOLUTIONS[t]
                    _ok = lambda b: b in _rs and all(_e15b_has(_rs[b], qi, g, True) for qi, g in enumerate(_golds))
                    _hi = [b for b in _rs if (b or 0) >= 2]
                    if _hi:
                        _x["tasks_with_attempt3plus"] += 1
                        if any(_ok(b) for b in _hi):
                            _x["attempt3plus_acc_tasks"] += 1
                            if not _ok(0) and not _ok(1):
                                _x["hit_only_from_attempt3plus"] += 1
                report["eval_b_attempts"].update(_x)
                print("E15B stage 3 (eval):", json.dumps(_x), flush=True)
                print("E15B attempts (eval):", json.dumps(_pa), flush=True)
            except Exception as _e:
                report["eval_b_attempts"] = {"error": repr(_e)[:300]}
        try:
            report["e11"] = {"decode_summary": e11_decode_summary(),
                             "shim": engine.stats if isinstance(getattr(engine, "stats", None), dict) else None,
                             "server_log": str(E11_SERVER.log_path) if E11_SERVER is not None else None}
        except Exception as _e11e:
            report["e11"] = {"error": repr(_e11e)[:300]}
        if _E12_REV6 and isinstance(report.get("rev6"), dict):      # rev 6g H7-12 (GPT rev 6f audit 2): per GPU-hour
            try:
                _gh = 4 * max(1, (E12_RUN.get("finished_sec") or 0) - (E12_RUN.get("started_sec") or 0)) / 3600.0
                _solved = len({u["task"] for u in (report.get("eval_unit_records") or [])}
                              - {u["task"] for u in (report.get("eval_unit_records") or []) if not u.get("hit")})
                _yt = (((report["rev6"].get("metrics") or {}).get("verified_yield") or {}).get("all") or {}).get("new_for_task")
                report["rev6"]["per_gpu_hour"] = {
                    "gpu_hours": round(_gh, 3),
                    "correct_inputs": round(report["eval_query_hits"] / _gh, 3) if report.get("eval_query_hits") is not None else None,
                    "tasks_solved": round(_solved / _gh, 3) if report.get("eval_unit_records") else None,
                    "new_for_task_predictions": round(_yt / _gh, 3) if _yt is not None else None}
            except Exception as _e:
                report["rev6"]["per_gpu_hour"] = {"error": repr(_e)[:200]}
        write_json_atomic(OUT / "e12_run_report.json", report)
        if _E12_REV6:
            print("E12 rev 6 metrics:", json.dumps(report["rev6"]["metrics"])[:1500], flush=True)
            _sc = report["rev6"].get("scheduler") or {}
            # A3-F4 (rev 6c audits): full coverage is claimed only when stage1_left == 0
            print("E12 rev 6 coverage: tasks initialized", _sc.get("tasks_initialized"), "| request sent",
                  _sc.get("tasks_with_request_sent"), "| response", (report["rev6"].get("metrics") or {}).get("tasks_with_response"),
                  "/", len(TASK_IDS), "| stage-1 starts left",
                  _sc.get("stage1_left"), "| workers", report["rev6"].get("workers"), "(fixed)", "| KV holds", _sc.get("kv_holds"),
                  "for", _sc.get("kv_hold_sec"), "s | idle overrides", _sc.get("kv_idle_overrides"),
                  "| KV samples", _sc.get("kv_samples"), "missing", _sc.get("kv_missing"),
                  "| stragglers", _sc.get("stragglers"), flush=True)
        print("E12 RUN REPORT:", OUT / "e12_run_report.json", "| eval", report.get("eval_query_hits"), "/", report.get("eval_query_total"),
              "| task-norm", report.get("eval_task_norm_pct"), "| accepted branches", E12_RUN["branches_accepted"],
              "| cached", report["usage_all"].get("cached_pct"), "%", flush=True)
        e12_write_submission()
        _pack = WORKING / f"Q27X_E21_{E21_VARIANT}_HALF{E21_HALF}_DUCK_BUNDLE.zip"
        with zipfile.ZipFile(_pack, "w", zipfile.ZIP_DEFLATED) as z:
            for fp in (SUBMISSION_PATH, OUT / "e12_run_report.json", OUT / "e12_generated_attempts.json",
                       OUT / "sglang_decode_summary.json", OUT / "sglang_requests.jsonl"):
                if fp.exists():
                    z.write(fp, arcname=fp.name)
            for fp in sorted(E12_OUT.rglob("*")):     # transcripts (full thinking) + branch results
                if fp.is_file() and (fp.name in ("transcript.txt", "branch_result.json") or fp.name.endswith("requests.jsonl")):
                    z.write(fp, arcname="duck/" + fp.relative_to(E12_OUT).as_posix())
            if E11_SERVER is not None and E11_SERVER.log_path.exists():
                z.write(E11_SERVER.log_path, arcname=E11_SERVER.log_path.name)
        print("E12 bundle:", _pack, round(_pack.stat().st_size / 1e6, 1), "MB", flush=True)
    except BaseException:
        print("E12 reporting error (submission.json still exists):", traceback.format_exc()[-3000:], flush=True)
    try:
        e11_kill_server(wait=False)
    except Exception:
        pass


# ==================================================================== CELL 8 — NB5: boots and phases
# Commit: warm-up -> pure decode window -> SMOKE (2 tasks, A/C split) -> SMOKE gate -> EVAL (eval subset, scored)
# [-> MTP-off boot -> EVALOFF]. Hidden rerun: warm-up -> EVAL on every hidden task (CELL 0's fallback stays otherwise).
BASE_OUT = OUT
NB5_ALL_IDS, NB5_ALL_CH, NB5_ALL_SOL = list(TASK_IDS), dict(CHALLENGES), (dict(SOLUTIONS) if SOLUTIONS is not None else None)
NB5_EVAL_CH = _load_json(ARC_ROOT / "arc-agi_evaluation_challenges.json") if not RERUN else {}
NB5_EVAL_SOL = (_load_json(ARC_ROOT / "arc-agi_evaluation_solutions.json")
                if not RERUN and (ARC_ROOT / "arc-agi_evaluation_solutions.json").exists() else None)
NB5_REPORT = {"run_id": RUN_ID, "profile": PROFILE, "rerun": RERUN, "config": CONFIG.get("nb5"), "boots": NB5_BOOTS,
              "phases": {}, "errors": []}
try:
    if not E10_SANDBOX_SELFTEST.get("ok"):
        raise RuntimeError("sandbox self-test failed: " + json.dumps(E10_SANDBOX_SELFTEST)[:300])
    nb5_rc2_inputs()
    if not MERLIN_OK:
        raise RuntimeError("Merlin inputs missing: " + json.dumps(R.get("inputs"), default=str)[:600])
    nb5_rc2_prompts()            # RC2 A0 (host facts, merge + background sha256) + the Flash-Next tokenizer / xhigh template
    NB5_TAILS = NB5L.tails(lambda s_: TOK.encode(s_, add_special_tokens=False).ids)
    NB5_REPORT["tails"] = NB5_TAILS
    NB5_REPORT["prompt_render"] = {k: v for k, v in R.get("prompt_render", {}).items() if "text" not in k}
    # A-14: the local tokenizer + template are a /tokenize fallback only when RC2's pinned hashes were verified at load
    NB5_LOCAL_TOK.update(verified=bool(MERLIN_PROMPTS and R.get("prompt_render", {}).get("hashes_ok")), hf=_HF, template=_tpl,
                         encode=lambda s_: TOK.encode(s_, add_special_tokens=False).ids, default_kwargs=json.loads(XHIGH),
                         why=None if (MERLIN_PROMPTS and R.get("prompt_render", {}).get("hashes_ok")) else
                         "tokenizer / template hashes not verified (or not the Merlin checkpoint)")
    NB5_REPORT["local_tokenizer"] = {"verified": NB5_LOCAL_TOK["verified"], "why": NB5_LOCAL_TOK["why"]}
    step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")}
                                                           if not FAKE else {"fake": True})(merlin_install())})
    _ok, _why = (sha_gate(min(1500, left() - 900)) if not RERUN
                 else (True, "rerun: no sha gate (E18 R2 / E15B SUBMIT have none)"))   # NB6 P6
    R["setup"]["sha_gate"] = {"ok": _ok, "why": _why}
    if not _ok:
        raise RuntimeError("sha256 gate: " + _why)
    if P.get("patch_60490"):                       # R2 lever a: #60490 before the first boot (fail safe: RC2 pin only)
        try:
            NB5_PIN["patch_60490"] = nb5_apply_60490()
        except Exception as _e:
            NB5_PIN["patch_60490"] = {"ok": False, "error": f"{type(_e).__name__}: {_e}"[:800]}
        R["setup"]["patch_60490"] = NB5_REPORT["patch_60490"] = NB5_PIN["patch_60490"]
        log("NB5 #60490:", json.dumps(NB5_PIN["patch_60490"], default=str)[:600])
    _groups = []
    for _sp in P["nb5_phases"]:
        _sp = list(_sp) + [None] * (7 - len(_sp))
        _key = (_sp[3], _sp[5] or NB5_KV_DTYPE)
        if not _groups or _groups[-1][0] != _key:
            _groups.append([_key, []])
        _groups[-1][1].append(_sp)
    for (_mtp, _kv), _phs in _groups:
        _b = nb5_boot(f"merlin_{_kv}_{'mtp' if _mtp == 'on' else 'nomtp'}", _phs, _mtp == "on", _kv)
        log("NB5 boot", _b["tag"], _b.get("status"))
        nb5_write_report()
        if _b.get("status") not in ("DONE",):
            # R2: the second paired phase still runs after a failed first boot (recorded); a GPU-release abort stops all
            if not P.get("continue_after_failed_boot") or ABORT.get("why"):
                break
except BaseException:
    NB5_REPORT["errors"].append(traceback.format_exc()[-3000:])
    print("NB5 error (submission.json keeps what was written):", traceback.format_exc()[-2000:], flush=True)
NB5_FINAL_KILL = True
for _s in list(NB5_SERVERS):
    try:
        _s.kill()
    except Exception:
        pass
nb5_write_report()


# ==================================================================== CELL 9 — NB5 report: the four averages, counters, bundle
try:
    _ph = NB5_REPORT.get("phases") or {}
    _main = _ph.get("EVAL") or next(iter(_ph.values()), {})
    _gen_total = sum(float(((b.get("counters_end") or {}).get("vllm:generation_tokens_total")) or 0) for b in NB5_BOOTS)
    _nb_sec = time.perf_counter() - NOTEBOOK_T0
    NB5_REPORT["throughput"] = {
        "pure_decode": [{k: (b.get("pure_decode") or {}).get(k) for k in ("band", "level", "status", "tok_s", "running_mean",
                                                                         "mean_acceptance_length", "step_ms", "engine_rates")} | {"boot": b["tag"]}
                        for b in NB5_BOOTS if b.get("pure_decode")],
        "steady_mixed": {n: p.get("steady_mixed") for n, p in _ph.items()},
        "solver_window": {n: p.get("solver_window") for n, p in _ph.items()},
        "whole_notebook": {"gen_tokens_all_boots": _gen_total, "sec": round(_nb_sec, 1), "tok_s": NB5L.rate(_gen_total, _nb_sec)},
        # revision 1: productive decode time / wall time and the token-weighted version, next to the rates
        "duty_factor": {**{n: p.get("duty") for n, p in _ph.items()},
                        "whole_notebook": NB5L.duty_factor("\n".join(f.read_text(encoding="utf-8", errors="replace")
                                                                      for f in sorted(LOGS.glob("server_*.log"))),
                                                           _nb_sec, _gen_total,
                                                           next((b["pure_decode"].get("tok_s") for b in NB5_BOOTS
                                                                 if (b.get("pure_decode") or {}).get("status") == "STEADY"), None))},
        "note": "pure decode = a settled decode-only window at running == workers; steady mixed = stats lines with every slot "
                "running (turn prefill included); solver window = server tokens / solve seconds; whole notebook = every boot's "
                "tokens / the notebook's seconds (boots, probes, warm-ups included). All after the JIT warm-up."}
    if "EVAL" in _ph and "EVALOFF" in _ph:     # the paired MTP on / off accuracy comparison (same tasks, equal windows)
        _a, _b = _ph["EVAL"], _ph["EVALOFF"]
        NB5_REPORT["mtp_pair"] = {"score_on": _a.get("score"), "score_off": _b.get("score"),
                                  "accepted_on": (_a.get("bands") or {}).get("accepted"), "accepted_off": (_b.get("bands") or {}).get("accepted"),
                                  "correct_on": (_a.get("bands") or {}).get("correct"), "correct_off": (_b.get("bands") or {}).get("correct"),
                                  "solver_tok_s_on": (_a.get("solver_window") or {}).get("tok_s"),
                                  "solver_tok_s_off": (_b.get("solver_window") or {}).get("tok_s"),
                                  "target_only_commit_replay": _b.get("paired_commit_replay")}
    # revision 5: the gates only a GPU run can pass, with what THIS run measured; a fake-server pass is not evidence
    _cp = ((NB5_BOOTS[0].get("probes") or {}).get("commit_probe") or {}) if NB5_BOOTS else {}
    _vc, _sp, _cw = _cp.get("variant_c") or {}, _cp.get("server_parsers") or {}, _cp.get("cold_vs_warm") or {}
    _v = lambda ok: None if ok is None else ("PASS" if ok else "FAIL")
    NB5_REPORT["gpu_only_gates"] = {
        "evidence": "FAKE_SERVER (local harness): NOT production evidence" if FAKE else "GPU run",
        "token_id_completion_semantics": {"rule": "server render ends '<think>\\n', /tokenize = chat prompt_tokens, the C reply ends "
                                                  "with finish stop, a closed think, a parsed call",
                                          "tail_ok": _cp.get("tail_ok"), "render_parity": _cp.get("render_parity"),
                                          "natural_finish": _vc.get("natural_finish"), "calls": _vc.get("tool_calls"),
                                          "verdict": _v(bool(_cp.get("tail_ok") and _cp.get("render_parity") and _vc.get("natural_finish")
                                                             and _vc.get("tool_calls"))) if _cp else None},
        "parser_correctness": {"rule": "the same continue_final_message ids: server qwen3 / qwen3_coder parse == Duck's markup parse",
                               "cfm_starts_with_c": _cp.get("cfm_starts_with_c"), "args_equal": _sp.get("args_equal_client_parse"),
                               "verdict": _v(_sp.get("args_equal_client_parse")) if "args_equal_client_parse" in _sp else "INCONCLUSIVE"},
        "mtp_on_state_consistency": {"rule": "greedy C reply with a cold prefix (cache_salt) == with the warm prefix; MTP-off replay "
                                             "(NB5EVAL30MTP) for target-only parity", "cold_vs_warm": _cw.get("verdict"),
                                     "target_only_replay": (_ph.get("EVALOFF") or {}).get("paired_commit_replay") or "not run",
                                     "verdict": {"EQUAL": "PASS", "DIFFER": "FAIL"}.get(_cw.get("verdict"), "INCONCLUSIVE")},
        "real_prefix_reuse": {"rule": "whole-run prefix hit >= 88 % (PLAN W2), commit turns cached", "phases": {
            n: {"client_prefix_hit_pct": (p.get("tracer") or {}).get("prefix_hit_pct"),
                "server_prefix_hit_pct": (p.get("counters") or {}).get("prefix_hit_pct"),
                "block_floor_pct": (p.get("tracer") or {}).get("prefix_hit_pct_at_block_floor"),
                "retained_session_hit_share": (p.get("tracer") or {}).get("retained_session_hit_share"),
                "commit_C_cached_pct": (((p.get("tracer") or {}).get("commit_by_variant") or {}).get("C") or {}).get("cached_pct")}
            for n, p in _ph.items()},
            "verdict": _v(((_main.get("counters") or {}).get("prefix_hit_pct") or 0) >= 88.0) if _main.get("counters") else None}}
    if FAKE:
        for _k in ("token_id_completion_semantics", "parser_correctness", "mtp_on_state_consistency", "real_prefix_reuse"):
            NB5_REPORT["gpu_only_gates"][_k]["verdict"] = "LOCAL_ONLY (" + str(NB5_REPORT["gpu_only_gates"][_k]["verdict"]) + ")"
    _bs = {n: p for n, p in _ph.items() if (p.get("budget_plan") or {}).get("budget")}
    if _bs:                                    # PLAN v3.6 NB5BUDGET: solve rate, concurrency, tok/s per budget
        NB5_REPORT["budget_sweep"] = {n: {"budget": p["budget_plan"]["budget"], "workers": p["budget_plan"]["workers"],
                                          "task_norm_pct": (p.get("score") or {}).get("eval_task_norm_pct"),
                                          "query_hits": (p.get("score") or {}).get("eval_query_hits"),
                                          "server_running_mean": (p.get("b_summary") or {}).get("server_running_mean"),
                                          "solver_tok_s": (p.get("solver_window") or {}).get("tok_s"),
                                          "budget_limited_attempts": p.get("budget_limited_attempts")} for n, p in _bs.items()}
    _q = {n: p for n, p in _ph.items() if n.startswith("Q_")}
    if _q:                                     # addendum 5: per config and the paired task x config table
        NB5_REPORT["quality_ab"] = {
            "configs": {n: {"kv": p.get("kv_dtype"), "workers": p.get("workers"), "score": p.get("score"),
                            "exact_tasks": sum(1 for v in (p.get("quality") or {}).values() if v.get("exact_all_tests")),
                            "complete_tasks": sum(1 for v in (p.get("quality") or {}).values() if v.get("complete")),
                            "output_tokens": sum(v.get("output_tokens") or 0 for v in (p.get("quality") or {}).values()),
                            "tool_call_rate": (p.get("tracer") or {}).get("tool_call_rate"),
                            "ttft_mean_sec": ((p.get("latency_means") or {}).get("time_to_first_token_seconds") or {}).get("mean_sec"),
                            "phase_sec": (p.get("phase_window") or {}).get("sec")} for n, p in _q.items()},
            "paired": {t: {n: {k: (p.get("quality") or {}).get(t, {}).get(k) for k in ("exact_all_tests", "complete", "output_tokens",
                                                                                         "wall_sec", "tool_call_rate", "first_request_latency_sec")}
                           for n, p in _q.items()} for t in (P.get("task_list") or NB5_QUALITY_TASKS)}}
    if P.get("quality_ab"):                    # R2: per phase + the paired FP8 / BF16 summary (all / finished in both)
        _pc = {n: {"kv": p.get("kv_dtype"), "status": p.get("status"), "fp8_status": p.get("fp8_status"),
                   "window": p.get("window"), "window_plan": p.get("window_plan"), "capacity_levers": p.get("capacity_levers"),
                   "speed": p.get("speed"), "order": p.get("order"), "score": p.get("score"),
                   "tasks": (p.get("qab") or {}).get("tasks") or {}, "summary": (p.get("qab") or {}).get("summary")}
               for n, p in _q.items()}
        _qa = NB5_REPORT.setdefault("quality_ab", {})
        _qa.update(nb0=NB5_QAB_NB0, fp8_label=NB5_QAB_FP8_LABEL, groups=NB5_QAB_GROUPS, per_config=_pc)
        # Q-1 / Q-2 / Q-3 / Q-5: windows check (both DONE, numeric, |delta| <= 2 s, none unequal), paired, deployment_comparison
        # (all tasks / finished in both / missing-incomplete-failed matrix), precision_comparison UNVERIFIED, phase_diff
        _qa.update(NB5L.qab_report(_pc))
        for n, v in _pc.items():
            print(NB5L.qab_table_text(v["tasks"], n), flush=True)
            print(n, "summary:", json.dumps(v.get("summary"), default=str), "| levers:", json.dumps(v.get("capacity_levers"), default=str)[:1500],
                  "| speed:", json.dumps(v.get("speed"), default=str), "| window:", json.dumps(v.get("window"), default=str), flush=True)
        _pa = _qa["paired"]
        for _blk in ("all", "finished_in_both"):
            for _g, _s in ((_pa.get(_blk) or {}).items() if isinstance(_pa.get(_blk), dict) else []):
                print(f"PAIRED [{_blk}] {_g}: tasks {_s['tasks']} | solved FP8 {_s['solved_FP8']} BF16 {_s['solved_BF16']} | both "
                      f"{_s['both']} FP8-only {_s['FP8_only']} BF16-only {_s['BF16_only']} neither {_s['neither']} | discordant "
                      f"{_s['discordant']} (McNemar p {_s['mcnemar_p']}; paired n {_s['paired_n']}, min attainable p "
                      f"{_s['min_attainable_p_paired_n']} / {_s['min_attainable_p_at_discordant']} at this discordant count; "
                      f"{_s['reading']}) {_s['discordant_tasks']}", flush=True)
        if "skipped" in _pa:
            print("PAIRED:", _pa["skipped"], flush=True)
        print("deployment_comparison:", _qa["deployment_comparison"]["status"], json.dumps(_qa["windows_check"]), "| matrix:",
              json.dumps({n: (p["outcomes"], p["tasks_missing"]) for n, p in _qa["deployment_comparison"]
                          ["missing_incomplete_failed"]["phases"].items()}), "| precision_comparison: UNVERIFIED | phase_diff:",
              json.dumps(_qa["phase_diff"]["differences"], default=str)[:2500], flush=True)
    nb5_write_report()
    print("NB5 REPORT:", json.dumps({"throughput": NB5_REPORT["throughput"], "phases": {n: {k: p.get(k) for k in (
        "status", "score", "workers", "running_check", "commit_mode")} for n, p in _ph.items()}}, default=str)[:6000], flush=True)
except BaseException:
    print("NB5 report error:", traceback.format_exc()[-2000:], flush=True)
try:
    _pack = WORKING / "Q27X_NB5_BUNDLE.zip"
    with zipfile.ZipFile(_pack, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in [BASE_OUT / "nb5_report.json", SUBMISSION_PATH, ENG_OUT / "nb5_engine.json", ENG_OUT / "telemetry.jsonl"] \
                + sorted(LOGS.glob("*.log")) + sorted(TRACES.glob("*")) + sorted(BASE_OUT.glob("phase_*/e12_run_report.json")) \
                + sorted(BASE_OUT.glob("phase_*/nb5_sampler.jsonl")):
            if fp.exists() and fp.is_file():
                z.write(fp, arcname=fp.relative_to(WORKING).as_posix() if WORKING in fp.parents else fp.name)
    print("NB5 bundle:", _pack, round(_pack.stat().st_size / 1e6, 1), "MB", flush=True)
except BaseException:
    print("NB5 bundle error:", traceback.format_exc()[-1500:], flush=True)
