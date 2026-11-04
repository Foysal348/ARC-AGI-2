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
E31_CFG = {'think': 104000, 'answer': 8192, 'window_min': 170, 'workers': 24, 'temperature': 0.6, 'drain_min': 23}
# ---- E31 (user 2026-11-01): the line above is the ONE place to change this run (no rebuild): think = the phase-1 cap
# (>= 60,000, the user floor), answer = phase-2 tokens per test input (x the task's test inputs), window_min = the solve-window
# CAP (the direct solver returns as soon as every task is done; never a wait), workers = client streams (<= the 24 server
# seqs), temperature (ORDER2: 0.6), drain_min = the forced-close reserve at the window end (covers 3 test inputs x answer at
# ~18 tok/s per stream; inside the window: no new task from T - 12 min, every request aborted at T - 1 min). The budget
# follows it: boot ~15 min + window_min + a 20-min reserve (nb5_phase's solve end HARD_STOP - 3 min and its 1-min
# room, hard margin 5, a late boot up to ~11 min); the deadman = the budget (HARD_STOP + 5 min).
if os.environ.get("E11_HARNESS") == "1" and os.environ.get("E31_CFG_JSON"):   # local / CPU-kit harness only (short fake runs)
    E31_CFG = dict(E31_CFG, **json.loads(os.environ["E31_CFG_JSON"]))
E31_BUDGET_MIN = 15 + int(E31_CFG["window_min"]) + 20
# ---- E26 K100 (user 2026-11-01): the commit's ABSOLUTE DEADMAN (E31: HARD_STOP + 5 min = E31_BUDGET_MIN), armed first (CELL 0 has written the fallback
# submission). A daemon thread sleeps until NOTEBOOK_T0 + K100_DEADMAN_SEC, then: one marker line (stdout fd + stderr fd +
# /kaggle/working/k100q_deadman.txt), SIGKILL every child (psutil tree; else os.killpg on the servers' groups + pkill -9 -f
# vllm), os._exit(0). It uses no lock, no main thread, no cell. The hidden rerun (RERUN_PROD) has no deadman.
K100_DEADMAN_SEC = 12300   # E31: the built default (E31_CFG window_min 170 -> budget 205 min = HARD_STOP 200 + 5)
K100_DEADMAN_SEC = int(E31_BUDGET_MIN * 60)   # E31: follows E31_CFG['window_min'] (HARD_STOP + 5 min, the K100Q rule)
K100_DEADMAN = {"armed": False, "fire_pc": None, "pgids": []}


def k100_deadman(t_fire_pc, marker, poll=5.0, use_psutil=True):
    """Sleep until perf_counter >= t_fire_pc, then the marker, SIGKILL every child process (group), os._exit(0)."""
    while True:
        d = t_fire_pc - time.perf_counter()
        if d <= 0:
            break
        time.sleep(min(poll, d))
    line = f"@@K100Q_DEADMAN_FIRED@@ t={round(time.perf_counter() - NOTEBOOK_T0, 1)}s pid={os.getpid()}\n"
    for fd in (1, 2):
        try:
            os.write(fd, line.encode())
        except Exception:
            pass
    try:
        with open(marker, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass
    killed = False
    if use_psutil:
        try:
            import psutil
            for c in psutil.Process(os.getpid()).children(recursive=True):
                try:
                    c.kill()
                except Exception:
                    pass
            killed = True
        except Exception:
            killed = False
    g = globals()
    pg = list(K100_DEADMAN.get("pgids") or [])
    for s in list(g.get("NB5_SERVERS") or []) + [g.get("E11_SERVER")]:
        pid = getattr(getattr(s, "proc", None), "pid", None)
        if pid:
            pg.append(pid)                            # MServer starts each server in its own session (pgid = pid)
    if hasattr(os, "killpg"):
        for x in pg:
            try:
                os.killpg(int(x), 9)
            except Exception:
                pass
    if not killed:
        try:
            subprocess.run(["pkill", "-9", "-f", "vllm"], timeout=10)
        except Exception:
            pass
    os._exit(0)


if not RERUN:
    K100_DEADMAN.update(fire_pc=NOTEBOOK_T0 + K100_DEADMAN_SEC, marker=str(WORKING / "k100q_deadman.txt"))
    K100_DEADMAN["thread"] = threading.Thread(target=k100_deadman, args=(K100_DEADMAN["fire_pc"], K100_DEADMAN["marker"]),
                                              daemon=True, name="k100q_deadman")
    K100_DEADMAN["thread"].start()
    K100_DEADMAN["armed"] = True
    print(f"K100 DEADMAN armed: fires at +{K100_DEADMAN_SEC / 60:.0f} min after the notebook start (in "
          f"{round((K100_DEADMAN['fire_pc'] - time.perf_counter()) / 60, 1)} min)", flush=True)
# E15B / E21 profile mechanics; Flash-Next values. Every NB1 / NB2 output that changes the engine is a parameter here.
NB5_DEFAULT_PROFILE = "E31DIRECT30"   # E31
PROFILE = "RERUN_PROD" if RERUN else os.environ.get("Q27X_PROFILE", NB5_DEFAULT_PROFILE).strip().upper()   # NB6 E1: the hidden rerun = RERUN_PROD
E11_HARNESS = os.environ.get("E11_HARNESS") == "1"            # local CPU harness only (fake server, no GPU)
_NB5_T = {"time_budget_sec": 710 * 60, "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 2 * 60,
          "verify_margin_sec": 1 * 60, "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60}   # 11:50 hard stop
NB5_SMOKE_TASKS = ["2ba387bc", "1818057f", "0934a4d8"]
NB5_SMOKE_GATE_SEC = int(os.environ.get("NB5_SMOKE_GATE_SEC", 40 * 60)) if E11_HARNESS else 40 * 60
NB5_QUALITY_TASKS = ["1ae2feb7", "38007db0", "3e6067c3", "65b59efc", "d8e07eb2", "1818057f", "409aa875", "5961cc34",
                     "7491f3cf", "b99e7126", "0934a4d8"]
NB5_QUALITY_WINDOW = int(os.environ.get("NB5_QUALITY_WINDOW_SEC", 150 * 60))   # a bound per config (expected 1.3 / 1.8 h)
NB5_NB0_FP8 = os.environ.get("NB5_NB0_FP8_GATE", "").strip().upper()             # NB0's FP8 verdict: PASS | FAIL | ""
NB5_QAB_GROUPS = {"solid": ["1818057f", "45a5af55", "bf45cf4b", "2ba387bc", "8f215267", "38007db0", "58f5dbd5"],
                  "sometimes": ["247ef758", "f931b4a8", "b5ca7ac4", "65b59efc", "221dfab4"],
                  "never": ["2b83f449", "16b78196", "64efde09"]}
NB5_QAB_TASKS = [t for g in ("solid", "sometimes", "never") for t in NB5_QAB_GROUPS[g]]
NB5_QAB_GROUP_OF = {t: g for g, ts in NB5_QAB_GROUPS.items() for t in ts}
NB5_QAB_NB0 = {"verdict": "NO_GO_FP8_TF_DIVERGENCE", "full_prompt_rank1": {"2k": 0.90, "8k": 0.93, "40k": 0.749, "90k": 0.661},
               "source": "E25 Experiment Results/E25_GPU_wave1_NB0_NB1_2026-10-31.md (NB0 esfersami50 sha 37a427a2)"}
NB5_QAB_FP8_LABEL = "UNVALIDATED_FP8_UNDER_TEST"
NB5_CAPS16 = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64]                                        # NB1 CAPTURE_HI (FULL decode covers 16 x (1 + K3) = 64)
NB6_COMMIT_TASKS = ["1818057f", "2ba387bc", "45a5af55"]      # NB5 QUALITY's solid group, solved in both phases
NB6_LEVERS = dict(nb5_gate=1.0, nb5_attempt={"b0": 104000, "b1": 104000},
                  nb5_env={"E12_B_SKIP_B1_IF_ACCEPTED": "0", "E12_B_B1_ACC_MODE": "0", "E12_B_EXTRA_ATTEMPTS": "0"},
                  admit_mode="slots", f_rule="measured", pin_ladder=True, patch_60490=True, seqs_rule="table", caps16=True,
                  lever_env={"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, no_pure_decode=True, continue_after_failed_boot=True,
                  e18_ladder=True,          # NB6 P3: E18 R2 cell 6 L468-507
                  nb5_phases=[["EVAL", "all", None, "on", "A", "fp8"]])
# ---- E26 K100 (lead 2026-11-01): klee100 3bpw-MTP on NB5 QUALITY's Duck + Merlin path. Boot = klee100 LOAD rev 3's K3 (sha
# 7773e63f): FP8 E4M3 QSA KV + the static scales sidecar, MTP K3, --language-model-only, #60490, the K3 pin ladder (launch OOM ->
# next lower rung), the read-only worker extension (method-map gate). Duck = NB5QUALITY_R2's FP8 phase, unchanged.
K100_PINS = [7933526016, 6859784192, 5270142976, 3553580524, 2479838700]                     # klee100 LOAD rev 3 K3_PINS steps (KL.k3_ladder; build facts k3_pins)
K100_PIN_STEPS_GIB = tuple((b - 2479838700) / 2 ** 30 for b in K100_PINS)   # nb5_lib.pin_tries: GiB per card over Merlin's FP8 pin
K100_SEQS_TABLE = {5.08: 21, 4.08: 18, 2.6: 14, 1.0: 9, 0.0: 6}   # launch --max-num-seqs per K3 rung (GiB step): NB5 R2's rule on the expected pool (F 6)
K100_CAPS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 84]   # Merlin capture_sizes_for(21, K=3): FULL covers 21 x 4
K100_EXTRA_FLAGS = ["--language-model-only", "--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt"]   # klee100 LOAD k1_boot
K100_ENV = {"VLLM_SERVER_DEV_MODE": "1"}     # klee100 LOAD DIAG_ENV: /collective_rpc for the read-only worker extension
K100_WINDOW_SEC = 6600                 # user 2026-11-01 (30 tasks; NB5's Q_FP8_K3 window was 4,049.8 s for its 15)
K100_BOOT_LEVERS = dict(seqs_rule="k100", caps16=False, k100_caps=True, k100=True)
K100_PROFILES = ("E31DIRECT30", "RERUN_PROD")   # E31
K100_EXPECTED = [{'kv_bytes': 7933526016, 'gib_step': 5.08, 'gib_per_card': 7.389, 'pool_usable_est': 479, 'f_est': 6, 'workers_est': 21, 'captures_cover': True}, {'kv_bytes': 6859784192, 'gib_step': 4.08, 'gib_per_card': 6.389, 'pool_usable_est': 414, 'f_est': 6, 'workers_est': 18, 'captures_cover': True}, {'kv_bytes': 5270142976, 'gib_step': 2.6, 'gib_per_card': 4.908, 'pool_usable_est': 317, 'f_est': 6, 'workers_est': 14, 'captures_cover': True}, {'kv_bytes': 3553580524, 'gib_step': 1.0, 'gib_per_card': 3.31, 'pool_usable_est': 214, 'f_est': 6, 'workers_est': 9, 'captures_cover': True}, {'kv_bytes': 2479838700, 'gib_step': 0.0, 'gib_per_card': 2.31, 'pool_usable_est': 149, 'f_est': 6, 'workers_est': 6, 'captures_cover': True}]   # per rung: pool / workers estimated from the v1 boot (319 blocks at 5,270,142,976 B, F 6)
# user 2026-11-01: 30 tasks = NB5's 15 (its order; the paired W4 comparison) + 10 hard_long + 5 sometimes_new
K100_NEW_GROUPS = {'hard_long': ['3dc255db', '7491f3cf', 'd59b0160', '35ab12c3', 'e8686506', '7666fa5d', '898e7135', 'edb79dae', '2d0172a1', '8e5c0c38'], 'sometimes_new': ['136b0064', 'db0c5428', '800d221b', 'a395ee82', '20270e3b']}
K100_TASKS = NB5_QAB_TASKS + [t for g in ("hard_long", "sometimes_new") for t in K100_NEW_GROUPS[g]]
NB5_QAB_GROUPS = dict(NB5_QAB_GROUPS, **K100_NEW_GROUPS)      # NB5's three groups + the two new ones (records / reports)
NB5_QAB_GROUP_OF = {t: g for g, ts in NB5_QAB_GROUPS.items() for t in ts}
# ---- E28 K100 (user 2026-11-01, Exp 28 README): EVAL120 6 h with every tested speed lever on klee100. T1 = 104,000 per
# attempt (= K100Q, reserve 4,096) for A / B1 / C; B2 (cap-cut re-runs) = 120,000 inside K100Q's 131,072 max-model-len and
# the normal Duck window 130,560 (the 120k rule below holds for B2 too); I8 = DRAFT_HEAD=int8 (NB4T's
# env, an input of Merlin's build()); the A / B / C scheduler (e28_sched_patch.py, E28_SCHED=1); no ORDER2; workers 21 (the
# R2 rule) with K100Q's 'slots' admission (the engine schedules KV; preemptions = recompute, counted per attempt)
E28_T1 = 104000                    # A / B1 / C attempts
E28_B2 = 120000                    # B2 (cap-cut) re-runs only
E28_B2_WINDOW = 131072 - 512             # = NB5_WINDOW (K100Q's max-model-len 131,072 is unchanged)
E28_B2_TIMEOUT = 9000             # E28_B2_REQUEST_TIMEOUT (user 2026-11-01)
E28_WINDOW_SEC = 21600               # the fixed 6-h solve window
E28_I8_ENV = {"DRAFT_HEAD": "int8"}     # NB4T arm I8 (Merlin launch.py:319-322 -> "draft_lm_head_quant":"int8")
E28_I8 = {"want": False, "on": False, "static": None, "dropped": None, "readback": None}   # set in CELL 8 / per boot try
assert E28_B2 <= 120000 and E28_B2 - 4096 <= E28_B2_WINDOW and E28_T1 == 104000   # the 120k rule, B2 included
E28_LEVERS = dict(K100_BOOT_LEVERS, nb5_gate=1.0, nb5_attempt={"b0": E28_T1, "b1": E28_T1},
                  nb5_env={"E12_B_SKIP_B1_IF_ACCEPTED": "0", "E12_B_B1_ACC_MODE": "0", "E12_B_EXTRA_ATTEMPTS": "0"},   # NB6 / E15B SUBMIT
                  admit_mode="slots", e28_workers_rule=True, f_rule="measured", pin_ladder=True, patch_60490=True,
                  lever_env={"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, no_pure_decode=True, continue_after_failed_boot=True,
                  i8=True, plain_order=True, e28_sched=True,
                  top_rung_seqs=24)   # user 2026-11-01: 24 workers / --max-num-seqs 24 on the TOP rung only
E28_PP2_MICROBATCH_TOKENS = -(-E28_LEVERS["top_rung_seqs"] * (1 + 3) // 2)   # 48 tokens per PP2 micro-batch (FULL graphs)
assert max(K100_CAPS) >= E28_PP2_MICROBATCH_TOKENS and E28_PP2_MICROBATCH_TOKENS in K100_CAPS, (K100_CAPS, E28_PP2_MICROBATCH_TOKENS)
# ---- E31 (user 2026-11-01, Exp 31): Flash-Next DIRECT on 30 eval tasks = the proven ORDER2 direct method (one prompt per
# TASK, xhigh thinking, forced close, parse_grid / make_attempts) on THIS boot; the solver is e31_direct.py (CELL 7b rebinds
# E12A.solve_all_r6). E31_LEVERS = E28_LEVERS without e28_sched (Duck's arc2_duck.py stays the original payload file).
E31_GROUPS = {"main": ["2d0172a1", "35ab12c3", "409aa875", "446ef5d2", "4c3d4a41", "4c416de3", "5545f144", "7491f3cf", "7b5033c1", "800d221b", "89565ca0", "9385bd28", "a47bf94d", "a6f40cea", "b6f77b65", "cbebaa4b"], "partial": ["142ca369", "247ef758", "36a08778", "581f7754", "71e489b6", "88e364bc", "a32d8b75", "f931b4a8"], "control": ["135a2760", "136b0064", "1818057f", "0934a4d8", "13e47133", "16b78196"]}
E31_TASKS = [t for g in ("main", "partial", "control") for t in E31_GROUPS[g]]
E31_WINDOW_SEC = int(E31_CFG["window_min"]) * 60                # the solve-window cap (the solver returns when every task is done)
E31_LEVERS = {k: v for k, v in E28_LEVERS.items() if k != "e28_sched"}
assert len(E31_TASKS) == len(set(E31_TASKS)) == 30 and "e28_sched" not in E31_LEVERS and E31_LEVERS["top_rung_seqs"] == 24
assert 1 <= int(E31_CFG["workers"]) <= E31_LEVERS["top_rung_seqs"], E31_CFG
assert E11_HARNESS or int(E31_CFG["think"]) >= 60000, ("user floor: thinking >= 60k (approval needed below)", E31_CFG)
assert 1 <= float(E31_CFG["drain_min"]) < int(E31_CFG["window_min"]), E31_CFG
PROFILE_TABLE = {
    # E28 K100 (size): the unused rows NB5EVAL120 / NB5EVAL30 / NB5EVAL30MTP / NB5QUALITY / NB5SMOKEDIAG / NB5BUDGET are cut (the guard admits E28EVAL120 / RERUN_PROD only)
    "RERUN_ALL": dict(_NB5_T, source="test", n_tasks=None, nb5_phases=[["EVAL", "all", None, "on", "C"]]),
    # E28 K100 (size): the unused rows NB5QUALITY_R2 are cut (the guard admits E28EVAL120 / RERUN_PROD only)
    # NB6 E2: the hidden rerun (every hidden test task, ORDER2, one EVAL phase); E18 R2 RERUN_ALL budget 12 h - 3 min
    "RERUN_PROD": dict(_NB5_T, time_budget_sec=12.0 * 3600 - 3 * 60, source="test", n_tasks=None, **NB6_LEVERS),
    # E28 K100 (size): the unused rows NB6COMMIT are cut (the guard admits E28EVAL120 / RERUN_PROD only)
    # E31: the 30 eval tasks (task_list; ORDER2 inside the direct solver), budget E31_BUDGET_MIN (HARD_STOP = budget - 5 min,
    # the deadman at the budget), ONE phase with the fixed window E31_WINDOW_SEC (a cap; shortened, never skipped, on a late boot)
    "E31DIRECT30": dict(_NB5_T, time_budget_sec=E31_BUDGET_MIN * 60, hard_margin_sec=5 * 60, source="eval", task_list=E31_TASKS,
                        quality_ab=True, equal_windows=True, fixed_window_sec=E31_WINDOW_SEC,
                        nb5_phases=[["Q_E31_FN_DIRECT30", "all", None, "on", "A", "fp8"]], **E31_LEVERS),
}
PROFILE_TABLE["RERUN_PROD"] = dict(PROFILE_TABLE["RERUN_PROD"], **E31_LEVERS)   # E31: the hidden rerun = NB6's (every task, 12 h - 3 min, E18 ladder) + klee100 + the direct solver
if E11_HARNESS and os.environ.get("NB5H_TASKS"):                 # local harness only: a few tasks, short windows
    PROFILE_TABLE["NB5HARNESS"] = dict(_NB5_T, source="eval", task_list=os.environ["NB5H_TASKS"].split(","),
                                       nb5_phases=json.loads(os.environ["NB5H_PHASES"]))
if PROFILE not in PROFILE_TABLE:
    raise ValueError(f"unknown profile {PROFILE}; choose {list(PROFILE_TABLE)}")
P = PROFILE_TABLE[PROFILE]
assert PROFILE in K100_PROFILES or (E11_HARNESS and PROFILE == "NB5HARNESS"), (
    f"E26 K100: this notebook boots klee100 only ({K100_PROFILES}); got {PROFILE}")
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
if P.get("k100_caps"):                                      # E26 K100: the same rule at the largest launch seqs 21 (FULL covers 21 x 4)
    NB5_EXTRA_ARGS = NB5_EXTRA_ARGS + ["--compilation-config", json.dumps(
        {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": K100_CAPS}, separators=(",", ":"))]
NB5_EXTRA_ARGS = NB5_EXTRA_ARGS + list(K100_EXTRA_FLAGS)     # E26 K100: klee100 LOAD K3 flags (text only + worker extension)
NB5_BASE_ENV = json.loads(os.environ.get("NB5_BASE_ENV_JSON", "{}") or "{}")       # Merlin knobs, e.g. {"DRAFT_HEAD": "int8"}
NB5_BASE_ENV.update(K100_ENV)                                 # E26 K100: dev mode for /collective_rpc (after build(), NB5Q-ENV-1)
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
SGL_MODEL_PATH = INP / "models/foysalemonshanto/qwen38-flash-next-klee100-3bpw-mtp/pytorch/autoround-3bpw-mtp-nople/1"   # E26 K100:
# the klee100 mount (tokenizer.json: K7-digest-checked; the W4 p1 Model is not attached; Duck B3 counts reasoning with it)
SGL_DRAFT_PATH = Path("/nonexistent/nb5-no-separate-drafter")  # (read by the rev 6 preflight; MTP is built in)
SGL_MAX_RUNNING = 8                                           # Duck env placeholder; the phase sets the real streams
SGL_RELAUNCH_TIMEOUT_SEC = NB5_READY_TIMEOUT
SGL_URL = None
CONFIG = {"PROFILE": PROFILE, "RERUN": RERUN, "profile_row": P}
os.environ["E12_RAW_REASONING_HISTORY"] = "1"                 # E14-P (E15B CELL 1b): Duck keeps the raw reasoning
E12_BRANCHES = int(os.environ.get("E12_BRANCHES", 2))                 # independent branches per task -> attempt_1/2
_E12_REV5 = True                                # rev 5 build: every profile (incl. the hidden rerun) runs the fixes
_E12_REV6 = True                                # rev 6 build (user 2026-10-21): code-first view + coverage scheduler
_E12_REV4 = PROFILE == "EVAL6" or _E12_REV5                            # rev 4 defaults (pre-run capacity check, 2026-10-20)
E12_STREAMS = int(os.environ.get("E12_STREAMS", 16 if _E12_REV4 else 0))   # Duck ARC3_MAX_ACTIVE_STREAMS; 0 = engine concurrency
E12_MAX_GEN_PER_BRANCH = int(os.environ.get("E12_MAX_GEN", NB5_ATTEMPT["b0"])) or None   # NB5: Flash-Next attempt 1   # E15B rev 3f: attempt 1 = 50,000 (attempt 2: E15B block)   # Duck max_generated_tokens_per_game
E12_LIVE_BRANCHES = int(os.environ.get("E12_LIVE", 12 if _E12_REV4 else 0)) or None   # 6 x 131k = 786k <= KV 867k; S12 (user): 12 x rev 5 context p50 58k = 697k
E12_DUCK_MAX_OUTPUT = int(os.environ.get("E12_DUCK_MAX_OUTPUT", (NB5_ATTEMPT["b0"] - NB5_RESERVE)
                                         if _E12_REV4 else 0)) or None   # rev 6: the first-request cap; later caps are per request
E12_DUCK_TEMPERATURE = float(os.environ.get("E12_DUCK_TEMPERATURE", 1.0 if _E12_REV5 else 0)) or None
E12_REASONING_EFFORT = os.environ.get("E12_REASONING_EFFORT", "xhigh").strip().lower()   # NB5: forced xhigh   # "" = model default (xhigh)
if E12_REASONING_EFFORT:   # user 2026-10-21: low-effort variant of rev 5 (Qwen3.8 template: xhigh / medium / low)
    assert E12_REASONING_EFFORT in ("xhigh", "medium", "low"), E12_REASONING_EFFORT
    os.environ["E12_REASONING_EFFORT"] = E12_REASONING_EFFORT
if _E12_REV5:   # B1-B4, B6-B8 in arc2_duck (E12_ARC2_REV5); B3 counts reasoning with the served model's tokenizer
    os.environ["E12_ARC2_REV5"] = "1"
    os.environ["E12_TOKENIZER_JSON"] = str(SGL_MODEL_PATH / "tokenizer.json")
E12_R6 = {"E12_R6_CAP_FIRST": NB5_CAP, "E12_R6_CAP_PROGRESS": NB5_CAP, "E12_R6_CAP_RESCUE": NB5_CAP,   # NB5: no 12k/24k/32k
          "E12_R6_FIRST_EXPOSURE": 0, "E12_R6_B1_VIEW": "facts", "E12_R6_MIN_ADMIT_SEC": 300,   # E15B #3, #1, #49
          "E12_CHECK_VERIFY_CAP": 24, "E12_AUTOCHECK_CAP": 24, "E12_AUTOCHECK_PER_CALL": 3, "E12_DUCK_DRAIN_TOKENS": 16384,
          "E12_R6_FACTS_MAX_CHARS": 3600, "E12_R6_SHOW_MAX_CELLS": 900, "E12_R6_KV_FRACTION": 1.0,
          "E12_R6_P90_CUTOFF": 0,
          "E12_R6_FACTS_MAX_TOKENS": 800, "E12_R6_FAIR_SHARE": 0,   # E15B #45: no chunks / pauses
          "E12_R6_KV_ADMIT": NB5_GATE, "E12_R6_KV_POLL": 5.0, "E12_R6_CAP_FIRST_B1": NB5_CAP, "E12_R6_SHOW_CALL_CELLS": 1000,
          "E12_R6_NUDGE_TOKENS": 24000, "E12_R6_B1_GRID_MAX_TOKENS": 12000, "E12_R6_MIN_CHUNK": NB5_CAP,
          "E12_R6_CACHE_DIAG": 0}   # NB5: the SGLang cache diagnostic is not run on vLLM
if _E12_REV6:
    os.environ["E12_ARC2_REV6"] = "1"
    for _k, _v in E12_R6.items():
        E12_R6[_k] = type(_v)(os.environ.get(_k, _v))
        os.environ[_k] = str(E12_R6[_k])
E12_DUCK_REQUEST_TIMEOUT = int(os.environ.get("E12_DUCK_REQUEST_TIMEOUT", (7200 if P.get("e28_sched") else 3600) if _E12_REV4 else 0)) or None   # E15B #47
for _k, _v in (("E12_DUCK_MAX_OUTPUT", E12_DUCK_MAX_OUTPUT), ("E12_DUCK_REQUEST_TIMEOUT", E12_DUCK_REQUEST_TIMEOUT),
               ("E12_DUCK_TEMPERATURE", E12_DUCK_TEMPERATURE)):
    if _v:
        os.environ[_k] = str(_v)                                         # read by arc2_duck (CELL 3 / CELL 6)
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
# ---- E26 K100: ONE sha-pinned file of the tf-ref dataset (the commit notebook stays < 400 kB): klee100 LOAD rev 3's K100_EMB
# blobs {name: [sha256, lzma + b85]} (k100load_lib.py, the kit v2 core + tables, the digest manifest) + its worker extension +
# NB5 QUALITY's nb5_lib.py (the same bytes as NB5's packed literal: sha 18efcbc1, checked here as NB5 checks it)
K100Q_PAYLOAD_NAME = "k100q_payload_emb.json"
K100Q_PAYLOAD_SHA = "fd92a814040aea6dda12855badf801675580dcd78b2f2135eb0b86121dd80880"
K100Q_PAYLOAD_USED = {}


def k100q_payload():
    """E26 K100 (klee100 LOAD R1-18 pattern): the payload file from the tf-ref dataset mount; its sha256 is pinned (each blob's
    sha256 is checked again by klee100 LOAD CELL 2b's own decode loop in the engine cell). Missing / changed -> raise."""
    f = next((d / K100Q_PAYLOAD_NAME for d in (INP / "datasets/foysalemonshanto/q27-e25-nb0-tf-ref", INP / "q27-e25-nb0-tf-ref",
                                               Path(os.environ.get("K100_PAYLOAD_DIR") or "/nonexistent"))
              if (d / K100Q_PAYLOAD_NAME).is_file()), None)
    raw = f.read_bytes() if f is not None else b""
    if f is None or hashlib.sha256(raw).hexdigest() != K100Q_PAYLOAD_SHA:
        raise RuntimeError(f"K100Q_PAYLOAD missing or sha mismatch: {f} (dataset q27-e25-nb0-tf-ref, {K100Q_PAYLOAD_NAME})")
    K100Q_PAYLOAD_USED.update(file=str(f), sha256=K100Q_PAYLOAD_SHA, bytes=len(raw))
    return json.loads(raw.decode("utf-8"))


K100_EMB = k100q_payload()


def k100q_ds_text(name, want):
    """E26 K100: one more sha-pinned file of the same tf-ref dataset version (k100q_report.py, nb4_ppack.py). Missing /
    changed -> raise (the caller decides: the report lib is required, the Q-defer lib optional)."""
    f = K100Q_PAYLOAD_USED["file"] and Path(K100Q_PAYLOAD_USED["file"]).with_name(name)
    raw = f.read_bytes() if f and f.is_file() else b""
    if hashlib.sha256(raw).hexdigest() != want:
        raise RuntimeError(f"K100Q dataset file {name} missing or sha mismatch: {f}")
    K100Q_PAYLOAD_USED.setdefault("files", {})[name] = want
    return raw.decode("utf-8")
NB5_LIB_SRC = (lambda _t: _t if hashlib.sha256(_t.encode()).hexdigest() == "18efcbc1dde9c590a14ac1937b45dec563ff76ed4d92b16632d273dec2c88500" else (_ for _ in ()).throw(
    RuntimeError("nb5_lib: payload source sha256 mismatch")))(
    __import__("lzma").decompress(__import__("base64").b85decode(K100_EMB["nb5_lib.py"][1])).decode("utf-8"))   # E26 K100
(TMP / "nb5_lib.py").write_text(NB5_LIB_SRC, encoding="utf-8")
sys.path.insert(0, str(TMP))
import nb5_lib as NB5L
# ---- E26 K100: the k100 report lib + the NB5 FP8 reference (sha256 checked at load)
K100Q_REPORT_SRC = k100q_ds_text("k100q_report.py", "359629ec695cf9b7f1a0002257f1544ffa34c3ab1192e6ca06781cc3e051db6d")   # the tf-ref version
(TMP / "k100q_report.py").write_text(K100Q_REPORT_SRC, encoding="utf-8")
import k100q_report as K100Q                                  # E26 K100: the quality report vs the NB5 FP8 phase
K100Q_REF = K100Q_REF27 = None   # E28 K100: the W4 / 27B reference tables are not used (no W4 reference on 120 tasks)
# ---- E31: ONE sha-pinned lzma + base85 literal: e28k100_lib.py (E28's, unchanged: I8 read-back / static check, the run
# report) + e31_direct.py (the ORDER2 direct solver, Exp 31 Build); E28's scheduler module is not embedded (e28_sched off)
E28_SRC = json.loads((lambda _t: _t if __import__("hashlib").sha256(_t.encode()).hexdigest() == "56149838b59c6ea4703c9aca861e863f339a2c7013177d016f8614bfd9120895" else (_ for _ in ()).throw(RuntimeError("e31_modules: embedded source sha256 mismatch")))(__import__("lzma").decompress(__import__("base64").b85decode("{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;d3Zt6<q*5h=`+9?NszXdT1T%a?()AO}%h2aS+_@aiqFd_aP}eM?%&aGUvx(V5V&KbpaE4<#>FHoDLb((v&alKO_VVH<?-1gNwT+a2wIy$RIGw;D6@c7IT5wS)Gb<{VyG_t<2?^0%n|tM_ZK`GT$C#KlP&~VF!E5k-Cj401XbHYZaQK0Z!&&$Q9qDy|8RVhH2HYN+)>UJuRFAgn&UFh7CAO2&@R?Io_AA_@rGRiZaOD8RN}QcVRU)v!z&0;yK@4y{N8)DR;a?vGATDR_mMnykgP!Rp&+Rus}a3ISjmu;dy|%2?(S~UbSzn{`r&=l2W|y_ic4EB%Ap`tyirL4<el=5OfXwVB4ZC+je^nmOY0csBvZI!4elZXB&ihj-}Q@6CE2trH^$FQkvBtB`G=Cr9Q)Mv7M}l;_3Md-ZJjTF0=S7Mz?fYxXv~9fY4S@L8;-&6S)>Qh4<s<arB70>~s)H0pa#Qexl{TpEy+dFA8wJDcqNtvD4;$JLnrpDLq!J(tn<g-01Z?WTjuKj9llF40<Rso&r9GE+Z%zR+R-&EjL~%Ww!1KX^te#=q3e)dcZn?I=r3ie@9|KV!c#8mop~jr>~uo%QlDLDG8T%L?AbH#H!&~E?s=IkUAl}ij{?qiERX#98;GVv{rc<Gn^r4ABD15rV@rkl&fcsT=}|fC(&u2PL}ovk{j{LSO(7V<nAXn(}@zSH)xf8$Ew0b?*gG;;9;CLsT%G`3aNa5@yyTUXM2+I@o{61l+e(2Ce?)^K{?|(;>t@jMJKp-jW%-&^Qo=G>sFuRU&EU1`~Y8)4)Q`Qux+M_(4ps{h=TOQU-NRBSI6E{!S5^50j#}*_z<X)Flmb2ttgwgT{n5<0$HK){&`GMwB04rIrMvLUb95jm~|o14kPyMCVNgTrB*hgp&)cd;HWHuM^{MKDwVqw6>_e94_WCZ#yXO)m<P_wG@js~W?ITp&EGIcGsE63YrGVkIOl2Rp|D*Swyvc}eM0=F;aHx{G+tXkRrhFRygdGtEu7_yPdJpA`S0JPy1llk0?*9ZSYRM@1u_=w{ggdAmW_oZ82EZ>R>iw-1tJ{1Vcy6_6;0K1z)zcRM3QJp#+bGO9m?sMqost?wL(xT8Wx**LvEI6H5oA%iH8VyQjxpqnU}PLKDtIQFM*c}zgzr&{UWW!fZy|ZznRz^!O|f5l2(8bu0%V6xQF3amv?pz_)pdedi__llpe&LMC<nV8F@kb1_cGEfr<|kyMw-{_SMTW$Y)iJZljFi7p9z1Y_eBaVA#ltKRx$?NLOM2cR%D2nJ4d4McfBz#+(BniKy9g@W&L~7Rrl+tG5_U&ConZ{7NlC_}my)vgX>XsmGUSJpxzN_wl<#f!s4953@A(AE;1&g2+OEV`76bIAr7zXER!P;&oJ@ntBHoFGlZJPalr~2n{E$+!mf<{*|-F)XHx=f=4WPDwgmu9GXkTv@_Ll>LVoHO|*aIFB%rH7=$lRE9?trEMxi#e$e}*(<uhe?BHa|$y;1D2x-&f)%sP#)3VY=giegqU!cDC9RIabai;I#)dwk#>Lg_%#jl|K3+EGkq)n}IpO_aOSNsab1in}URmKB75BWaT;-~rQv5vp76!IoR%a&gK;mNBPrnr$OTFMZc7Rx-(n>V3@W>;CObLO+tUdGL46l39ccorliFn9xeV>i~3m*fJnu+Lu04S<fUvYabvna8b_D`(^X;&NQwXVBC^zuRRDv{T~cAM4ta#!u9!<4=1tXIvvgCE6opaDnAU0|v@_C7A;}4Skb-=lY-<9RLWe8z}aHMR<ChC9z^f4muK|o&%My)U(k@9SVVVuQNe2Kz8j*01#&?#}U-A(;Of2al2!m<di?la1-HR+`buxH3gLqz+F{ZO3<MvNj6rzl1Qcsx#(}t<6Q^|Z*5qG*x%XKBgw=D$H-W{c1HjXCKeqh<T)-6A8tO?o-bm7X%qZDM&BicZ0`^f3Eqs)ecb4vV7&uUjr7D-aZl-GS9Wb2IUTZH!cr?v<m$@xouX{Q^2^t(0{z-azPS9Rk1)HxI;GwV?D3wvffnY(<IF(^Oh&Ai!$Mh5#`B9@F-##hr9g1Cb6*cPhf=r{jeci=1^@W~!l`_Yk(+)$u7_r-GA+?WRWa#Ov}^-i33T(3KC5@;kqnxsP$8mztWT+&A&$o+ye-VBuECej)-dK&>GdnUH*A#-n5e^=vPIdDj{XrHQ!^)I<h+(bj#t$a5<4|P=joHsO*`9FAWhz1?u-_>^12GPG3?M*V0_HXNxm~j4&a~qP2wKk$;seEorohFj!zppW^1KX$d2BSXqdX;*K!(zz(f&*GM)K`9Y@@N-AkB>?vYj6PD_ZsP6-j>Xp0(Lq0$IS6^5!CgY|kmd1F$5FbZ9^=GsO0`tZ@VOy$fbFRvjB)Fh`~4?%O%XzJBNemt@+Wh#h;lOmjJWsT=*6rS&n5B&0ZqNKCrApQA6hi{EiLzw(JD$<gmx6@qQKVWT1;OglG(%^zDoa|4$0GpGrm7ilFDv}Dk)Su`!TCMdtmc#7I9Et0NyL7RNR-n?iozAWx`4|swa@oO|3cyoe7rls9r$2x8jMAt&YqLUv2pG@AP9IPfl=4Io>mYWaWbN~h4^uuVgVm|&@-wkd2hgE~QJ0o~5>J>!tdT^GIj{wM`-8red?Kp3jPB=kvyEFUyT-^03!}_o?l5Hw^w0<};Qq;Si6*@Mt4|5<(Qr>sKb9r@ye>+KOwD@r)4STt+csTm_B1-lz^k+k5m~_?;I#zwX!f%a5Q8k{fUPVH^Hg70Uu!GHvPM(xnSOmS%GjPeo?o_uofv?~HjK+1<ZACrm^Q?GQS4o)0gwm>OvcnB%XDSMhhf|KiJ@C=2b+ml<@BDZ6g!qFkvD&go9ORjjWRGzlX2z=j-sy)ReDW{$Da!DlU>Q^L}fNm;h{woxC(pb^a{YYb_=&%sM+Oa#BNK7hTxea`&q|rpPjEfao07r`7Vgn64}5RcnsW{VIE1ZV&Y(|`A4&7jsvgAn5C=ZH)~_gG=Yu=M7yu_V(y%TNf#F3)#=w`fQL$p=n&-)BP|`KKXi$Q?gi)VPz!vI$|cJ*Evq~z-VF6=lg}Cq^IQ4!t+}Pa9a=T9Xd<7E<TEI}G>uxvDJfrzit~Om+^e`7=G`C}c9g^Y4*rCfzb(JGkUF6Bk4h$M9gCjK!kNr`#_yj5kJsU(ZrA4Aa>Xx|DK)+fp7}dj7xpl%fDarE-CTuV_ID*^3^baeSrnE7q~^acSfg&eR^<8V<qvDt!K4CQW#*-eetH;#8eoMhg8YwE8ma4ejX&mOM0gL}%KV&PW1O~20S9uF^(?y!`0#%WPg_4ID5qOXz0=@p?HXtfH%^Z&-IZHr?y&fk;-hp^U*PtVM++7su1sI<^A*?b$qj(1KVYjJNR}tQ63G`B(LXaQ=>PatI|L6kbsw(US&aSq{Xf%SxTlD*(Z);Hm3aw)9yomYgU<2bqG8?2EO@c){*MIHpf?^Ogj9THJa$2)k-t2GSwrSAk@)LM0CDy$mkX&6=oAg|9{_9{urj_rtvPMk_vl7p04=B8t#T`~jHu0hQjPaFna5=~T^UkTmFT87{1t^k6r)|}M}XAz3{4e3FmiBA_@+Bm9?4#AJl~D<sWk3|fT%Y>${Z>r>Ajn9(JgPh(j#DyS94^TN_qH1sJqr3Sq9wSJqTbDGUGY^Rov|?tP@Oz1y2e$vX*#fOcztiRXM`p83syc{yTkHc0blTRuxH1okO-j!G-oH?b5%w@mTNM_PG|h%?xceF2W1V`2XS?kaF9*rO=2##Cvl~p)@U`lMAj=VPSZOVW_U$lWdNyuAh@&(p|~gm5G<pt$Ml>{?ty0k#ha5W+!5)*{omUUlahxff2W*`w_CTkT7n5e580A9GJ(d0!Ogl9+j@<x{s<aLP|O~>OCft>g4L2dw%#|52ndr0MOH~_*fT~dHuZVsmR-<&uR<1CV}xy=d7+24xd#B+Xo6NsX5<mUmOYF1ZmK9dAWN~O@yQH#g=1dyt<L8g|0jWi$DH$iZ`2)tmGDp;TGpx{h<|&w9k1?YA2)fA4?IOHwGGeCgwuIMp@#!I4NEco8#0G{6B|#;WRr>BLIU|g-6W9$q%xDQ%O)<<87uXrwuAkXo5*-FKYbL`QM@-iii*uTq5QiA4rC-R0dKN8$(8O3P>jBE#ZBkPXu5q?SV}!=`-sBl%v9jZTKVA>d<jemhY=w*E5D${TrD)GdF7d0(TYnUDWZ3Lq-#6OMQwk)wViMg|7*B!;k_t-Pk!&KV>*BQB-&YNQObQ23upXg&?p&1jsBU12c&hwwGt+;3xaI@%^f;?C{gfU;E)HsD0^78<7y<HGrSb)ObK6JN+S;p^1n(Z9?F8_f^+1LAUkCWOTy(>j}>blt2lKM2$$hiWKj%cD&`|5Qa5xdbfZs;2NToj$W>@V)&ae%Arw@W+F!1iC`pwSRbr{|0o7R3IX(ZA-?2>yIn#sL31VdjVBM<aCaD&t%A0V^kuB%8!MX^rYKb+`SD6@E2}`4>8Rwtw;%D(ULzoHV*q$-#@3jlxx^fA?2IV!s4ZZu*>uvJmj#drNLjr?rWtPTPeyxD@RHW2F2jkTB&SccGJr2sWRG>$%ga^N_bH5-=$+W2;#Rcj(4FNImgaFKzU5wJ_EvyyLY@!gia?H}xX)6H;-Pk*K$g-tZ52=SQJBLH<%xF}P{&nBjs)T_s$6n%L0*sx;ZpxlWpFr>C{Z^8ZMMx;09EnjpKDl~wndTc>;kW-JlrP1nRFqk8A)T9ouU`~lniSeQn`z%&Dd+PQi7cJp4DYw4VDuF28c{PoES$CD3VMwiX8|4e!kwEceiu_vgWpAf2e~B*`YG&qA|cBR#jz970g*VTgY8U!~NJyw~MUqKY*5u7sy^7q>wSjHbVsGBZS%czbr(KY136Q&;IC?d2JgcUefoH(DVVl{HN{osG7WoD3kNyIiAE}SUFe|_=atl3{47()#_FvZM3n8+|FX*IPf(sT+S+N*U~R9H8tH(bNA%_O^|tH1(YIS_N4<`KMD@*7XcJ;c#bfsASyKy*Wj4YQBsJz92d#jAFcyP`!{Q+NY!1fD5kyO9s|gh$zj;AKj|q2iLwC&gx5zJ$&<R&>`PVp05x<snI$x#SD%eP&6&ytRFQ8vW|5G&uekfMzV7b$$so-e{+Zs7c-RJf@1TI|W6-Pr9#85FhWW&FTBgvF7IlITZ_1}dQneTsh{=bos6)k#2{2>)cgKav_bnw||4Zh5C4Tc>cnwhIgFZW>079<`{4*z8|BY37g-znb2-YUliF<l%d5j~*la=>z_?M^v`kKGVF|0-ANiHuef$VKzCX9sLs>f4+>~^*;f8=712CWd4u<@@|aEGz7=uz6MJC^EQ$XV$-otcgGD*K-1j0ax)iH7Zl)%xNv+R3AX$(M%+Te-N<*^3ze5pUd5Y&elK7o?l5`o-{b#cu%}c{zaH%CdK>ZJ=&r{n6#p_peSmUTH8yvb>FjMNau;82-AC7M=5-u!-la8*dfy`i)d}*#rUtpP`oC;KWF*r)zl|p3%R0D_My}n#3KAMB-nmFWQFDYG=8)WvGe|A?FaEe3u*J)U`lesZ@vLshf6I(EzW}6nFzVIO1~rM<%TzPJkKjreYFnn4(I$H5K0_(y%8po)ZUC-LGeuS+D6IGT7c$txx$Rqz~5h-1}CzB9i-{_JA@K5*@I9Ec-d%jrOVNb$13}thL0Qs=G*$?DPl5A>5VKgcEp%`bx=91|c<M;9BsC1M}`LE~oaDfaIY<8%65Z(o4@;#e4SRe<+%-{#JO%5<9S^uQ{&%0VI}iHS9Sf098%Z!eTLsiCjh_wi!{b-qT%XX=*1T7(&V+tbN6(vymc#!=@1xfWX^6K9fJTi>{8>rsF^aKj9Lyr6WS9Vf0e<1Dl?D8pALJbh)-rGlzw{WQO3vC4f+1!S|?B2>vI%kc(n!?I9(LWBO5OBb~n2HLZf^YyC&pnU0Lh;Qr(k_Ryu)8>R2QvZFMn`s2BEy2v;1(Cj+~%&6)WTHt<0?z}>9k6J{>sCvu!_Me=oA3v%{lCIs5wafw;e9Oz-wWiJUpan|5GaP(gg9ojP6uAsuGH75<?Ixl|V?5~KGKw6a<uL4mWgBSl+Z2+fA^csoTZBDc9PRj6BL&X_MYS77e4Susx6~AZRD`JH8Bsn4Ow!yHE(#~`QHPXNA#CaoKhbi8!|K%9G~m{lO|<rCwJ8xr;Q=ioVX9O6<Q~Lw)uxIAqdoyAW09$VuYP&pIFW}Z&wlJc{&#W}*pzR9m63s<jk)Q}F|vb>l|@frq72!`x$iPj>KbwW7el;3V{m&FJJLkWq$f>Zr!-?U%C(VCzQ;7+0UWdi9$W~Be{NKQ8&Yh_EJ=CcQ{ElBk+@01-V;T&aZq-sL<kUNotnFuJPREkP50R`GtvEwK6N!9!*-u7>Vk`dtxbx&az9_8OqXKvJ0NHmcIW_5n*U3Vpb)s6v1Bou+=zS<#ee1DPL8WPJyfT=*5*#=tCLp&Hv)SFu%_N>I1@l5hPV>bX$U}pU=&c<yb(7#;0nB`m92EbzYNeN^Cy7|v%mwH<?+834oIM3a+@hYFms0Mqk>d2*z^zK4};CZ1wH(or1?)xdG6|zpo-p}ssx`ldxxNn|5_O|AK6(;vRg4-i!je{`AfyXQ{4w~4T&+!K<G>X`URM3w+rbDKI+2PE9ytaKA{xBf59Y|Y<*$v<5kz;`Ni$sEX#~ik8KlExtytg4D>CS&KNVp`?hS`++mIOW1(pB*rPmnli1)O0sitNN`jk~`16ngFSylo99?zHT4(81xd<Ru1mNvk>#}}nU9i1`1x!->a;%At`-drY&|qtq2H$A>Fz#J+m*pIcfjloE6b`B?zdRLL%&QrcGM#U6b;7&_T|56?*2XtAKv}oP>jDf2F+)j4nTFpY-U@96gn5{*O`dg)A;R9z<ZPgynSbxvc<{?vB&a_fPZ@`+`o(+#(OBVV3chNh5(k+oDoqJ^RfB`mz@;0^(ZSZkzd?a;XnPdO)tSl<-U=GliJ-;k)-q)KDv>?DhC$SyLGn&0#WM!hGIrGehu|>))%#pT=LsnAZ6;zkYKzq67j*3#gnVM~2Sa{$B}OX+jkey2KS1^2)-gJ;32)T%iAIY+leWG<lX_u6u8+ePFbyabQUd+X`<}^J+OZC1T2_-DPL4;<as_?~kY%{EE1G4{?NM^!qaOhr0kc?cq2Q0R>3<ih9tyYzpT)=SLbzLrm=o>HOYTVtB-=XG#MIo6%7d&o2+!IKfTe7NUE}IEiFoPt5=6t}4~D@8MK$(!*fD#7NFK4fJLozZ4q*956Nhxd%>*i1I@5A;{K%KB-|3$~%NiRwTv^~29gyR6=}ubt;&**ibkK6E!8WhC6}r_-$`rs?*(YdmI%62S%3OJ(^kp7JV-!kybfQ_w>xZCY^p)+8@V!J8v?+5fv(m-{cV$xv;{|KgXTA!;P>$NkKRe}atrP?7g%hh3J+<s&d>q02r0O2Z4~qeSx36IiqycAQYk`Y)f<(5SiRD1Ed?37gGX#mCr%-gTaU=9PgCd{2kr%qxB!J6RU%peYzuR7KVlJV-@@mVPi}HMBy_;CmT+h}?y;V3_*Yf=StS&^mr`%@CH=-RbB%@~XMdI|}EhK%vREX*vjV>aUA?|=iDV||t&UVe6nlT)`N<t{R*jym{hr1k9y~o^Oe>Vz3ls+#eT!2=v_v6VPT{^3DoMeMqAlS7g{zFgCNy6`Yuy_CuzM@9D9V?_Dg8_CP`6z(%>Im-9KA2lPU)|Fp$EfH37J%$WQHYH!F40lRxmD!<67BJ@(f0#ZlF}4CEmNLu7C>ld{WC92kW`G61>tCh;Yo^bIlL#T$_~>f3~X6e0aL|&zsmY}St|L&B?LZQ>c$9L7$b1wC`mD|ntd9e=l_;s-w`iUeYchdn^z3{-PLmbtO$gV<3Zs5FBt?brwdRohk){(Npfw0A#N7I{3Zby!ev#!F9EVe9<6nc`3J94L0FW_QAb_@87t2G%|k<N%h8s7<0W4Mqp%zcf<vpz$cp)FVnONjjcRP_>-duc5iR&)j1VkK6PvGDgM{}E)K4*0S(9CKrF@E3!NRsvJb2GxPHuk5?FCdl3IoA2#9+#pn|Vtfpz6*BAS<*5)<UZe`5h^s@|L;&IS>9vV#C-^&VPt({botGpsx4E?PkcY&ncvueLN>*5!opmez~vhLo612K(YT|4-4<-SZbtG;_|z0a|b`1KAAI{8E4}a@hTQiS`;MrVa2-rwL5YhMovUQp*sVM@==@DB0PXqkV=0hEy81KM9jIN(!jGT;zX}fpIphkCZ$LpXJ&I>qXqYQG`?R=biL6rF?tr^#b_|scpgGMgiXB}uENed>atIQ@}hb07Dm?=(9vL;GaYS1|1r0$3X&JzC71VQrndXds8K;IB}+7UF!*OMTvWq?f)2b)%=8t&k1P{<R^G~y3H!d#EqXK4++N>8s3+yXApWD6X=z;r^5d9*VWL*}*~)n4E!%ujo`8Z%+7+oS;<ad<s0TrXkoZl;s(a2O)usA6w;h?kO~h=-$VZ-hIv*g<FKriPC7^7heARD7&sWWL{+!R|gSUd`>py#;sltQB>b%KYJ~z#Qj0~-h5+(k+1*LPONn#?5@xLzs7zgDZ$j48x5L$!0g!~nXrgM%5oxIFp3N<FlFv~lcz14U6UJfX2Beoh+6RGpzC+%=$MYVA}d~*DToALP^A()6rYVb;5<KIS=#?<WsnZCgIT{fZBu2c}D@do;z?KcqYxIRJ_5sc^%H<Da!`Zf9{-##~{_TqM#9f)7NpAk;)FV-d-60DPyFA2G-xm0R{kQoW~4g9<N{qDH>im!%$aEx_@?<f#L;+y_(iBPa*8k~~At9ik2QzN2HA52@?l>L(wpiad8?ELA?J?-6aNex3kJn*tIIR4F~qeS()6=bDGBpK62(%|qfWm7hX;!b_P;s`G!9mh>~V%Tv$g@9{de5@t`cPWOL{~T<fU0^Ind&Rc1|7F>gnd}+orP>2}TUoqs7F4EmQxL;EvHR?PlP8C{QhfOuWX@{Sx<5E^`Voc!T(I$FuyL69z7AvdH|=w<rx62fg!xtnopcp-#`21(MmKA@Jc^6e!gy0e-}u++-ykA7r}DXTTh4fu3Sw+$#)=ym(0v~R0D`Wj3LER-Aa5S*J8*ps39}shOiG}$X$D$h`E#@m`}33k8P+P}{17lqn*y1WE;b)qpnoP`4bff)9Lih2bgEi%5JgOpGS``eL&I;3^U2ORXJ@;ULKGibOH?u2UdVk0b-0+`@9S#<IQVJZD3o?8GZBo@qkr6YIdD<FlS<bp51PNxC(6_nY$P3-*vml;DvnLwSb$knUe3fG6)c;t%EU6oSl>4$8sab`(Xi+5Nfxv}exPoIY$Qd=JMLA?^^bO7A>3hW-7E0IC`wXUaI@_=W|-(Zm#FTMcq<Nwq^BNp8(epk9N;iE@<pT=cxD646EnnEZl?B^4&aY-Hp~)EBvqXK81e1w@$mjh{o!O;Whg%f5tAKABGXG!K6HP~gF0fr)y3HWWd614CxU)_h*ecgm8#88<9q<g%$f2GNB<q#AnBZ163*ENPf&ObdJ%MyFZ;su<JxXYrukUNpvodVd$p~UcXMnkBOqSYiH$tIcfzaGl6vyz&Nm6BecRu%`ey36_4YISB5PbfoIJ!tIBT59h13<=SY~}cqA~>4)yJoH#-#s;8s2Z#8Oa1DF|N3|p}(9q-`=KimWumTA^7KOoj5Qo1*I6)bi*{)ngb9_BsP*naS{JNCB^q#2T&|D$q)g1XhQEDpFk02rV{yBM0l7#k~`vf82AZ*#E65a<;b!T3#A4)$dh>!^2C+;lg`lk1B}_o$_+h)6FfM<-4&_g?UJwNuH+AeXWR6c7;BPJWnndMZMnXb&oe>}O`jAxavk!~V7c`1g@7Hux=pMWudOo?pNLN=Tj~2fd$+|GW-PV$lb7k(swbYE8^z1lI}L==TT?2W;fHCoE=AWIsJ2Eh3#z$n(LC^C{6x{d(pD@YZ_E^avHIs~Z@pobyUfb%Qx-|*+<ytG3ETSoH6b0_k@LL}1|(~M$(x+Bm4O4`uM*@-<`I5dV~EF)qd#4PfLFc{yv6LV{o044ciA|JKay2E2QH6xQp#?upYU1OUFwO{MEO95YwD7-iCe0^SQz@?d_1v^vMlT%Q-S<b395qKlrFj^h9KG2U?{Jo{|sl<T3paZnKRA3kJ!)<aaYT0i~^8;boYO(w%;{fBL1Ab!N85ixqDbFC#93amCySF5>Dax=?RLKB}$I*i#p0fD26#w-%l&a%H&)gj3v;Q2)-2>YCccLnmHu5c^Xbbfv!)GIdnUR@AiCyESODu9Ut+n=6iL;nlaG0Rn^zNCOf*-!~V~5om7ut-E+I*d|9_aQoj>nn$Ne3N=smgiaj1WrlFOC#OgrFB$VJYA<7MG&*qdueQn$|^{CKN?1Pd2d5XtF$(h#(85Z?qTqAfo90e&VXhdVz7>&#WW}K0JU9)tv20+Kxy);U0^NBmC&PHVYuOG1yKM6Z~vh?^KtWjDDbY8mi=zuzl)99VY3+OQbn&hLbkt{X6=Li5kT8~WYGjsO9g9BnluvYVmFLBS~IX$NEH+b^-RTnr^f_hfA#Nq$!*(CLhj!2UoS2PxGl2$jCb`qTNHlD<5_6~9JmH4iI7ncsl_1UL66NJy(jPf!BfI$vT{F#y7a%K8T>-*j(B2<wRvZYyjk@zr~x^%UhJ;?<_-^4pLHniatU92kK60niop(=duZQ%<cubFnBT|q<?hN_I|Un(2hSqw`Q%%~K=L+h*n*PKTq5C)^vE(;3+1d8#kzEtHt7?uMWCTi}#%S3Ya!l7(&)2x;yTRG3@W%}^U&sbg#E#zBI#>nXeUsO$k!gy_)gfT?tu2yY?twu_R3rN_lq+}T6s-4BwX#}O0cjtYAbvk{J7o1SQA}sYe{m6oraldGWwg7B6Dz+oW#a{A+cG^PKW<Zs8kWJ|Nr426g0*jn_@ZAtsJd-20i6{sbR2t!5R@A%yA9rh%)Zmn-GrYm%;PSchG$|o^W-bH>nV#<6{Kw5On)p_1-3nrUP&TM^0Cg0GICQ&Dp>Vlbs{VF5cRe;275$ys`_=&*e$B!lvk$`9rT8eTL}77luTSpaKhRJr4$XW5MKQr0uKO%#>mfU1yKn-bhJ-+Hvze-0<@a*E?o>6VMEL@D_KP6YyM6bGFT2<qitVfEKW)=9xg1iaNP4NWp-Wx8=KSby2KX0w=azmzPZJ<I`^5f#53-<^p9nI$3C(F1pxPl(kJo-luaC(#<p0%Lo03Iy6&#VWKo&D<?!=q9z#q7E#4_4xAz}hqU4ln#{JGkt7Ycz1^p&HN8q%^Gv5DO+223Q0tjo~Y0HoXk?GlIe3k>^;H-i1xvq`5$Y_eY4>Ri$atA^__EGiWH@5%h^CYXg)s<$lE(=dtrCKW+&!S3;eHW~aV>?qv6^e8PrL-CGVn(+bZWkV@<YR^>E2L$-f+IxWjIO$($OKf@}5I5EMsOHB{qsR~>h;P=Vas@g6o)cP6SYtf4Cw&iBbxMiz|8wn7vrViS46p9ME8zOlNAIZC5db&IU89?vUg9??Q5`ZW?JA(k6><T|H3y5IMfe~xj1R%sr>4xXuBOx`wLYU(=?DV!9^OYuamqY{ePx*)A(sY2?1dn5Ksp28Oxs$e9hz=G8kzWtqONUZ-dMuk`OK<;Fviw6$OkDuc56PW3JfR7%;}~dgadL$yiz}c&AspZ!y&wy7ntL~k;9F3Ji{<^kC3u%YVV~ibPc8>Xy%5Bp_#Ju=;Qk;A0f^%`pNQp{%CTFLAoH<;q8>x6!<^wa%LI+LP?icS{0*o7?L4)->-42;B18fQHY8RMCR&wsbX!{md8S5RY9$+0?|GsRl)eH?8NlwXpY<ABFYD3N8%vp(jV#k<*$}%CXB+}8btS`X;*moxN+X87w!-MhMEJz@wekQt!3pKbQSP?*b<Bz^j0L*i$j(JiLp4cDTfTj`&E!C?4|$?@=Gx@`iD#)=~pH6hFQGKO$S1K_iPUvy5r&BhEa`$xS}XZIGw|B3!<Yzdq3~noIuVDUKgMyChBcNkb@wP%=O%V*ahY3SAY;pZulVMOXCZL<VDW>b;ke01*ORBa3wowC28(Z#9aogpNYx3L_c+3F)v!z6y-)m%G~ks`N>MHcjB+{scumiOZX#}GgWh1H~l3onNB%DBHIlnbT?=P=KB7?27QPt<99-TFx!y*jT7X^1FVpQ9LfN@s2r)rr&FWqbu_Tr-Ky=FGhxS985$N%P@7clfA-o*k)DXXz^X!rsufxsT0u&tMjYnfA|ksu2iwSa*P02#*#r_7_D%wC)QCtpJ1a3HU>8cKNe%rEq6t+;-Kd$e5`k&(4`h#Jt@<a0CBDgUpeE0X!Jiyu&dU5;*Iow8g*_FA;c*Z3sJ5>bZ7a%3P4{?h`Vt=wVl_Mobg1z2^jf{{!i;t1oHT@RQ8S!0wp0VJnjHVuw+Nlp_J!1#l;J=z`OWSJ*bLw<&E`-ba0+8M2p>a$J{Lh6?&e4EVpb}nz!YY#)E*d>`@z(W)Z#ENBTWdM^XzC1wj+a-KKofEx)&1as*>fVx*cQBl1G<`=!Tyx%GQ#CF9;B)2<n2w;%$(}{YzS_uOwgqk>*FQ3drWg6PfEyID%pMs}hT@nm8fo6=h<3lFxM`G3s6ZNmQuh4LGffpp0N8Qrg9PiK?Xi=lHX0uL>%zoc@B6H1lT$S;;^&dm{;eH!qvN&3&mFl*B3td2<HI(Vm`GyevC711?FACatf5Q$ar4b73$EaG>KCd<J?alFm=8!DVVB+;`|^Wuk?mty|gbvV0tgt15<xyrWKl;F&~5Ky=qB5JM<B_pHk=&f}T7{K@Nu&(0<(((;L$ZtPqzQI28y-<djNRDZXTQQ&l1TFi?G)uiiwHFHGw5LxBrC$xwi-v|;qF*;jR%=bIseh@so14;I7(a<@5l41K<9(~)zUgo!@H8J;0AhN@<xD=3=k;%n)^V903y<A9aLFV^6Ka0$TX+-0c9P`+7dBF8M(e`Fk`pY62-<hiyeM}g?Nil#>263+rNm_d)1GV;pg-<JhdZn#>4~&&iKO-&`_AwT?riX4~E@ba;OyD~TL;{E6W=CxYRW{fb-?L-p^zCotlywh&3D%mcz<7&L#Nxt<7}`7l-?dD&KaR-HNhWDGphaS=S=JfJO3J+}qL=nj_-`qmS(Ym6T?7*_43OfiX%Lp1cC&njbo@^t>9iV%Vye*X-KTgye#u1fz$A;3`?MBpuu%JKVI;yR^}cP8@v)6TS?*~#Wxpmn(<$<|ERi-Mj^CfzCj(fBbKSYigQ4!aCN6(x4KXg=pxpb}QL`ZJ23B?B+yDO(bm=+Zd4X{gFh+J#ZR%ynYj^=AqPDwaEb#O(*?^<;d&MD~fRF;ML3R<`XCZ1>+=77Qi64auzH9#UzG6tmj>fF~dRRg4OmW~iO<}VnFr;TGv1dleCrIAAVYr+^3~At96VQxGs^s8^ciI2fE@S(9DiCCVj^QCw|2lDHqL6}(1uC`a4B|$N7ckj1Ck)1wI`PYm^7>9!TKC<*;vZ)ae5W@uU3=SgiaZtK4zm5o!xEn8{oOoP<P3d;U#ZH#wD@hy@iunJq8ZPB7L&iT3WWZYs^5NiQ5oG7E!~U-so9b1=*vW?zb`k#f+lbSwb8s-q0o_Cis-jhSz8<|xp0h*uz{cP@hs~dbqr1iB>0y1F}Hhs2v<owrNkm68wrS1pi=haU^pz2P7t1~4wQW1JN42h|EJ_uuHX|canOz*bg*7c6{|4V<%Ly#ciQtlAz7GhGF`|&I3r_o^BiR?98|gf4gb>Ul$gjR%BQq?cXemxQwaG>O^C>G<@2$)i7-KKwUmTMv#|KfJm4i<1nWti0vzBS(hg;|-bTQ4DIaq*bX)t(TNSeatnxW(Dlq{ac(wxS7i`>@|9d42fUz{p#y0511Ho@<6vjrMGu&19TVF-^l|XHxNd0OQVtbN>P_FvZB;pZh8RPkN2O)M#6YZPKcwG<K(34C%MIHPMRy4f~M1p_1ra-@DVHOEo>1u>}g#Qwxe-udS6=Gw6J8F(eDn$y*79SrGoh{)JJi-j<%X`7=k&NTsu9IVB|D?ke%aC^vUuoX8%Zy{)A3;CSZNVz5AGx`CF^WqyMJO2k37g%f!xI9N56+kUt@>K65QngJgj&#n&p)nhI#dZv{yA3B7D@r9M~66(c0q5RhOwUXgGHSlh&G;LeJ)ikn!>4cl))}c%nV<Dxu%I=M3KUsM%SQIM)C~F)T_(L<oz6{;_`C}h&;r-X=5EZ`g~M$?cyfi`g0oYl~K&Zeapegy!c&96)3lX1{*_dvJUBBmH9wZu_|Iic%o+60jU>Ax|};-7a5UnF^-7(CB3$dc4;YAju`O}^o}}+<Y9{7wHj4}<c(G~c0n!vM$4o_+_+S+1R*o!yh0(BkjfwV7A{+8DT5I&g*jw90jy(^hvlUOOE5ycbA45-)j=V3S=1hzN73<eD0Ry+(k-4Ue|&jLp{w-uiPDE-S4iCEMX;3AgbP|rdwxZOKgbS#sN|%wdg2{f?5GtYkhlul9p=4wjcFTP8I+8m;)75oO|yD(McBUGQR`8omKHOYP&g(O+(&(kN(}JRDX=509Hux<wRr_mON{t+`%)KRr8I-P`LN(TS(V$p`r*2UuLczV#~`wUwiW4)-bjcP^1qm~luDih`51HJlC2XZJp){$ofS2R2^&%YKhT1&`Vw5*@q%d=v-8V>C(wzVB-3@hK2?Q7n{I$-o_l!Kl#@zP7ABdMlEiG!YjKN#FT-0+E->|_0jx<ceinMlbLXHb=7<7@gR_dwd4f5yoh#B9<BstQUivnZ@oM|Y0TEKg{WSXR{qoz5P)knf`MSq>1i7~iavbX0vl%Qf3IL#JFA^HF2dW4O1r`fE9=32gqlQUixlok=ckHs=LE${gKQWPe`@d;3;3L*_D6%OQXEU^foaeb6B%i0Hb`|(*svrDg1$L3OF}|Gz&9!9rqjZh8>>c+FXpDv_;Qz!l;i5U}_NimD#XDs0wV4-vB^P(*&e-dur|+6mrH)jX;BUH)gKtisY42{LEKqm75+ZTG5krkQqUs5Snx&P`;)A5F+FBM#yx<xZIFL#WEn-e|6q(%JzAQu<%`$gsjTBb#XvEB`On*V!8<Hiv&4y4($m)&O9)nf8Df2&k=yK$y8cR)C5b2vB@ku2vTY>9hkjQKx6<ro)#DN$^y(N8a$=sZSus^$zZIfAgyK_q3@>b8P>XMuZ^ofadVKlzY(7qB*;uFf4A1liJ?f}Ofr4-v9{a#Nt%iXlxp^YMWR}92jA!7e>RINnN$RN1oMV%+&KU0DBirjpZ2mJ?$Lkm9g@K3t1wfGa#zs6LQjP1^-V^+3aezM;8;X8G0Li@F($vZugl*eWZ$JwtypY3D29F%7l*RzRlEjdZwrV~TkXjN727u3d1he1KV#-Mjm70~YrSc{>Io1<U>fjUX!*-&&hr6fT|-FDFSZg7;)G@}~3f@@L^;*p<!-?`d%ClT@r(O7-;2>SJtv^;}OSQYgy3Q>p$Z*Yf#bjZtpkQu;FvCI!Na2poVf5C=sX+6Om9Lw-KQipy>Zf(~z*{is%t;`qrkmL6H-?K6>jYV!t#RGf+Q^w-^TmdQ|E~gF2HMR5G?So%xJm}4lU9`GQA!A>?tMM8H;|?)Zd3#u^#{nAgHZIl@cg-Z^yxK|V@zs;<&K7^r{y}-~cDZ+nDn8Qxu(ep^W#x21&38y`mEEN8scVvs%Qn<fsX4#Iq`G_=$z+ip$zeAq&ZkiYl|g7H+g<px0k!88>k3>v3%-*tQViahayHDfq5DDH-R(sM(#svrMweBR_Jc#jZjcHB#CENJ)W{dYiyVF@OEpw?h5u|4YFXr!D}^9x+rT=#3h*KCA|L>)y<dcbq#F*~pBhfz_+AfD&Uioh@zW#!6U&(IU$zutL)0LkkWzg;Soh8OiAGoCN=3v568S{C>$ve^tjy51*)?+|VV1Ixz0Ewe3@(jLDSKd<Las}jns``Y$LDCm7hd{AEj>C%>FSKCrGsnuBd9+%ac{d*8)K2(#uE8n7W8p-^#DxUje1YYJm+x%n_{Kr7^@wW7d(5*n@x5!f*M0*P{o=eG7|(wYV$K~$c-gSvv5ccyRwi-GnXUL*7tDNW{O4I@aOo{trm!6*`Y$b#qe7=%#nIHLe^e`qXQBiJpX^^Z;?9$tcKCgCJfCNP@t3C9gUG>w9EO00+-R{UP4lhD9ZyAi2ix$z1cfBS-e^)!q;ado)caI-HhJ&Wa*Q|#l|P9AsZxh*3*#hK93J8reM);LSpd`dOl)LpmYy7uP^>I6o|(E1fx9$2OHq$aB}9@ZPbGU?pI5Kzt*bUbZ!5BzhhC|!4a6Si}PQaYItmjJVn`w(p59~a}n++FCB|dC@5=UQi(K1S;=QPzv6yLd$j6-J0^B+dJ?l$(0+460{zXpxC^2ojj?3CL>j*{2-79^dx7R2XH$@Ns8X6mg^9O%kc)vq?o$d`TEq`La+A8XpHNTFCXezn(hJmfmEc=;q-<|wRQb1V(rZ*Cth1TE`la7fPEb=H<0Pf_6a6B}RBW>`i;}dGoh}BnlL@;5uU#6H$3(tfH_8PruNz;h0<9Kck7s{5lXN%1=ZVtF5IS^s4Gcomf6F-&uc~li{YD~_QCgK1s_%uxKmWjddl3AAVVh`mQ_`BcRj!Ubh2%dnQNb+{nC%p(Rv1^Jgz@+c<mVI@zE2nH+5p#>;8DGkI0w^^2H^3<Wa%mBBh+r%i9M2nez5M)lo>j9=u$b2v+x<|$hAf7ZLp%H2cK>D8N|PN<D>ElT{#V0_B9U~wLRAOkZEC*-0Z`Vk36&2Z#FFfd|S3(&B(AGiV*<RGhFP=)Q$$4FADo^^;0Bix*!QnWq3St(=uWq55~>Ec9g*|A%n<VphUPonk*Xm%PxH~%XnqT<%76eWwpnY%zt+3m>iZt^F}P4Khmf#VmUarZgN=T*~>&4nYMf{#Q*{%!+V<WXMxNe#J@Cqt`}963dmrTQPr$7<Zzkf+9`b!$hi*Jlvb>cI*E$W0gpsm!c_s@9}}RSxYvH;ITbJ|#6Bd3y!zSLe2_i@{tz7p{TOiA{G7{TLYXPm`xzRQNxSQX%@%9e7x#e~|9XA(mmKr>b6aMPp$B)LkvCrIRU-D!f&pT#E<eaOYu4!v{zEegbT4aMV)q3*z&R!$$<-$K^A<0E6Kc+K-%rfM0j<{S=O?{)Og5#WkalP?%kwVt5X#{`+NCw`XL0~wf98oB<$8ijt9c8k4(j-nU1|t(l<MM5rN${~CYMZMLC<~}MMlH32C>Jvv<pv{59@LD-JKtv#^z5c#6o{nXlG<+m&$lO5uPiQhH_L>c+`-t&oDX>FwmXR>*jVN{*vVQy6-8-$J)^nt-60H7$NW;?<FKwVJfya^6Zu?Hu-5lzTWX8!4YyXT`KyPnGuR&)-hUD5ZPWzQJD-4#h+Y}pl;%}x|v$CPk<}W9t)p2Utr>Jcge^Yf*zJz&PIi?#=4(%%|zyTlx@(ok(;(dV9@>;$o;<nFxomYfEg`SVmzfyowuGa9!CclODtW!TXy2ykA;GLY`nle<$<)?>=%FwGNkEZ_8w=NmZZQEghL8Ig?J>lXPU)wrR#WTqpqBqfSV8^QXN?G#0B917KN>b=hhvNTK4vfSbK>R^s%FRKyZ0oQCj1o_^(*)oFohpT&@>TNV)>G-cQ)A5^&Gcizyvw<%O3Bff8=0LK_4s=im6@fQ(EFJHiCAX%i`#$uTZl3gA#2?SL~wT8uk}_-!S7cI)5-=Z#A$od!TF)@NmqY8qC;5l||ZU+Nsr|5OA&p2x%3e>OsK-iu0BLbtO(ChaPSwb;&I_FRg|scL~Rs3$>~I*%OG6OTaSTr#)M#RM+dpyS;+|JwTb>Eow08Fnn{!sTuN;NC7V3fzYvehq;PT6qgA2RHIw<m#=>toFOB@SkY$i^HTx7_bc1X)T;+U7KO~kG=!@Brp$aYu{$qMvJFPpVjurB<qC^#Sft?8Gta>^nSU}^~&YvU)v?xur@@IqTi^YD$!5Pn^2aG_+D3a39=qRSGdFu7|W24@@bVU%x?-2QBnFGTBx~A>*1W)crXedW_{)Ul%xeM9;lhBOvjZ4WS^ATV;XJ*kTU5$qYbJ0>i+9t<LFVKhb#NcJJWYr20S0M(3>5_H8i$J*!ikC?Q%@9bu=AA$i<^2rD{We!L?uWEfZ@R&1TOO%Ac0F`sZOo^#q^PnZ@yt7FRG`MS&+oj{Rr)ssE;!rBg-YBSfMN5^c>}6$cB?gkrFx-(^gazsA?H%u>TA?qsr^3C790pez>gY;>PAJrfGKC|W5z<W3$<g<ZC3Cz_p+RG0ZB1wQMKZ2TrlVVevHvj}p12#x02%aeH|T=c2W*NoQ*n2S)o$*Uns^3Q#<8ymg6{c?B;U;+>A&z{PHeGfpeE!{Su-F1wdSprL;KR$?e{hQK~f?`Qi2%d?(ou4{%0>exXXT~Zdd-K=}7ZKh1U0zqzL(iSDo!x}Z0<WZ+3XP3orlyN=J*#dGzxd{<{?f@V>u&I|A|}oT&X-VKF(W0beSYYWB1g-gD6BL(I!Yam44u~76e@;^L`n9(i&H#-pDz0`MWK>D934c({+~*bZX1MR@qCHJu2EV^*06jVbbpdZ%(KI-_x+Op*)kcBk2q;dxc@ZqY>(>q?UNB|Y!!8+ts4SQyO4KSQ(%<~LiQs(g(#7(M@9ri=C4YePgbiO`WLU*o4aVy1E%gY?k7|?7cF~&dV_(fLnx<-*?%zs(X#ngU&Zc?PERs}KhyHlDZoyq2T$D@-9n%WBUYC2B6X~zl6|PnIH5w0vv1kyGW&>-9s{E^>p7XB?juo80;+db<4D9IPf6E<lsO#hgfM4NAE=vdtwerbMty>W?PN-8IH9p&UQq0gaLE+dk<t7Jxba*NX7m%L9eElEg8}Z$st*hB=96Go_iE|0r)GRKGoQT&uMv|q<KcK9`!ShQ%=*M6Jh2Q=_l5oTJ%<+tQA^0!zGSh;e52K{wFO2*b2A0^?28@-;I@-%4CWHq=!aEx$w@5Ips}vvyv8iZU-T`Y0nvb@ST$>=d8VGHCH|;ZIsFhrbB^c*Z<&}2s5W~x5l5_A+Pq%C#Hz2Oj3bgL#xLCgzF7%f`@r122pH<b#jcu+zyHZl0x*0w&<|7y)$~6YQyj7fKCUY);&IYPODh+ejcV}9xskvH7{J-UaXA`EUDrLlkux8-wd?W=g(W#CN;2991%RwOIZs>9qKA&Vv~Zv~BYiB$Eyf@CD`7?^GudAHn~}jf3EpPe?-~iLYPj~;^oHn7oNP@Zo4TNBoM{#EfWn-7uaGhJc%vPj4RQrdwSpC^Zx7x%>R{&$xm{&CSU}&Sv<nUc6fwR2LX|h~Zl`J3<FY20maQ5ZaT6AJI}8rg=%0RyJmX~)W-ky_=p8Y9U@F#Bz6GT$wVk#UbR;hePLZ7?DQxL0s+pHj*)(lm3skU_9|~VZx(c6wpnQ~)AJLlk%q+i(2&^to!3)a}BAq#7Ttt~uO&|@Tx3DghE5vDTfWZq4Ed}ke>GmjM9C-t0QStOlHGj^8xicnz36(`?nx}(Cy#=0Hb)}xrl!z@!m&C)8S!^Wz@fh^;{``?@qM6W)Bo#c1hL|K+he2E(O)mi@?9YPmFZ{Ib8k5glQU+ZzzI(`o@p*)yE4P3TUa+=S*>fvf7#@3!n=;!WIH_w)CNm3PP{Q9#b0v1c`W!E|YdWmk2Z1<Q58UH<=>|FOI4!S5v<$4;z?Y`BR&vwbPV;3prI=@QXH-q-1bOjlSt+DOA`c^%jrw5yb1K^CHE_Nmo88NbdEBuh*p~HmOLDmB3*gESQiUqdoI0&vF}jeT$j^Tv`qyR$B&e;Z0bMwQ`MDPW3w$-?qS#|!ud@ZgQd_!?`DHjzH}CHy>A_K(v!{oW{Y`FFBJ+~x8Z~L-VTck;ZRSe@Q+uKBqPalSFbSH;-YAw8`CIBnL5z-$J(wWvW`nma5Pu7&%ZfK)ZJk1fVfPb@HC?#HaskIQ<$!Z$Uz*3Dzj80?D)hLPsZy9K-;z<fw^jcqKnDCh+q^(x{C7_yf1I_Y$!%Ak3#K~Mc{85p^0sZit1B+n-2OZ5fEtU|^9W44CS%@YCkwJ#rF3#Bn~*}+EDF-dE|MHKeYr$8sb&d4f<ET7F!dV`3ig!_e{EsmaI=r;X(!!z0nWLEUg2|c?s@bEM2v3&qsexE1x-h%qK0cSAGc28=gnaXmzt9A@l^{)aYYlz!HK_DK~?8h^FM)YTj`$cp|wNX7e(;*!W|E`*z6%sWa6~UjYAkX;b*b@fh$&?gGNZ`klM=D8;dN&%iLW-=F(&XPL(_swwMgnYgC~>7{vxVpp}g*`!2nXfbNGdBzz7tqP6pZLbO_vx(D=o+oUkNIpNX*7)6PLoXm=cRx}kx4!fTBBi$FH)i*_lP;#_V`XHySqv?)&JniFq7V@z%<tpxyc?L~9p}NEURd4Q{<pt=W!svLe<~mL0Dr|iURSeUhry=45QW}6bzZ`Q9%O{aYt&9igPfhEADG;&bW|J01UG+O$|Aam~B#Qp@C*+nkP;(qT{@A2|Ep3ES(}+eN(s6Dc)IW*fa0S7L2vMhHcP-NVdU3};<NK3m*MR<FNh2q=+zogf%M+uBW6sHR$zd_|>tS+>2&uhMhHIpx=$h%~!#df9_kYc>NIuwve3o7>WckV}DTyxb{cxl)oi*xk;Z5%g+Loh*GQ$L`5;F8>s$ak=-E_e?$93H>Ne1rEP#{ZpICOidZ#az{XD&VH6c}Gmzc@Msyd2zAHEll{7SJbAp7I`30XrhI?z`S4j4)!<%<9Faho|I|=F3r}EIy055SgthI=j>h=jdI3AVT_yc6@8PH5rb_Jjm0~8gnCTP?Lq8SKrthy7Yh>ZpYJ$)dZ@WROWe-59v9nYI%NU%%n@X*^Nbjs1L7oUs)(A>{$N!jQF3jSu2_P@b`fUa2$OPcyRsIEj3Rz=_NI+v?)=BmaPKnm1ki6ZA391>WSXndTD1ZLse28WO}io?hNQK0(HJ6Gso0Ra`khJA@JPcmTt*S_1fj)cD+l5GhJjisjUl>DDn1<Wo?6sNK7kAcHKnVymy2bMXJx6TqVFY#p@4PUJj-y88-H}1Af@nM%3Iou)Q?V)j=MjuKKG=M+7KVkA5C~c0y8H_z15?TjuvYrW?bW+EO>ryGW56G%3gvaHA=Ib8UKOtmu%*2>X)aiz891jBW4vhcSf(^8;fTpL)=))nB=JEZYILhO#3aobsIo2JYWMxZrtb2DRm}wYJr%%aFyumzGFM{A@r<^s7fvCq379>Pl}krBSZd-14sPknu~U3U(&DX&B3t6}6s)dzP=CVU_UNv&Bhl0Cj&JUOL~%Ax8IDn}*(VvNFS7ED3bb7ol;VV~PcqzHQPxNPjs$cqAFLz9#9SIRH2glTBJXpP}=$K=n8w)nzP|9*M&CZ%dKpL8mZb2rgk7lT}#*;|%h_C+eTLK4|#_q}JaT@8}$|-zI3+lpaE)SKc%-lprsPer|g3li++K>|_5?rxSc}<y#&ngQNAFyq#JgG;RDwvFU`L2V+ruFrwWd`vsOBfL)k7xpe~O9pXT+qrC$pT7R$CTHq=Jtk@te);|>lj1f!wNkR#&zm=u~vK^O#_KA*k5P1%<Mz4T9nVKOr@ww~F^vB9&D6_wSuiY+chIe>fPbwH`XC|H4{;HvG)kC(gT}L3(gpCb;b6y5&n7o`ZlC{%Y>w2GGddE8^v=6SAplOkWlknfgei2T#FWft)`ZT^r(BB$H2Y<Rqn?#oI7yNiEy5p`FJ4E_-Le?A}r}>NVn>J~6;gHa&Z{@@&K4DZJKrff1dQC#Jk;&MwWtSv-Awg=n<g2-&Q{M4i9UYjM%`U7N+H!zWrVK3s!3B-`42brBI=pSLK-BXf=iEZUr`R?1g+~brP^FpDsN+xu{&rI3<Ka7s@ko2o(o!QQx@Fj)pJ@9One{*R?10nloD_9mrgU?ih`&qSA=)SU|F8_+GhoM-AQOeorFjJ16bfZ8;btx((yU`}YGR%%d2iO{`+b9Q(#yk!`h^pD;r(PUlDPbK^!bDUmA7u^$3oF6AgD}kx&)ddH{vd869gQlg?4!mOV+9}Lf??si(-tpVD?tRx1T((FZ!R|Qkd5of^PNjEC}bzOa}~%L16&$!WVw%*Xb1HwG_8?fs~b_06|@*8iJ(XzQ<!+su72AEhQ2}*E@K9LgBBeN*X^QfLVDj>N>~_u$~wLi!ViCt4Fg5H?6*m<9WPxYfR>vsWn4R5%%4bN351uP6KWx@<O`YB5%t`hDfOnQp<%Eq;JK=0BH2kz69p?mGag%7SKPuu5!oqg<yFQ0q!X3q^AQ=FU2X*IwLO~x8w?n7ONTFWe~gzo(>t5^qnwPH<si1w6NSN3h6nDRG#mwl}nHfmzi3t;(*xpTl>e2iw8zwa>&%sx8{aIs4LKh`=|~|VIAhtr}Yo}A`k-WI7kex=5OzVsthN*|JoYH$VvlsiV~Pa2G;4#{HA0WU9l`9GTHFI$%L(s6(!VUI|u`Q;2qEFH7-E*L^7x~!pYIS$^oM3Swt2tc{a}THxfEfA%AC85J3vYLoW^P166KH#^zH;zjXLFKBg-!u|XZ5_48C;uR5STv%@;3q!xWddg<->_L(YDj*+TdmC0WqOss)fCMDs^W$-tVEpfUR{bF|#P6z?-El-w!n$4P1C0-Mi!`oK5TG70KEM6&&s-uT1Kiuf`yKgUumD?RHuEPf5V2=(7Vn_jO!X-({3GvPg_{T;!YTP!|v$qQVD|u746`N8!v2!1^-p=~A8FdIR(Gu%@)`<ag%#S)+keJPuA{(4n1qUZ(mRqQ<r5Cv#z|6|9@;-j`XQcUq3W;4d(XUjTT~9Cx&fsqF=zjyZN#*UY**y;tU#Q;oqu|}&NZ=|`p8)4QE#f6;w`Q!G@5I9yqI7{9_D`&?)6Q-Lxeph3?)~RA?rZOaZh!El3;>~-@A4zbZLX3euDYJ)<sO86o<Xc~hq$y|zn7N>Su9&i0Cca_2^gUhk~>j;6Qo#2$Q+0L<$ft)*kS?W^MAFY8?{v=@2o|^CtwjEG7bAS%??2jk|KJO0-3Avr!AH_Y`vZkQH4g$siiDe0H97tTa;sMOrh-k?|j@;;H;17ys~89vF3BMwusuG!YMadhC&Qy@%@^FDb~5?t(=Y`A?*q~mTNo~SEyfnRkUypO<AB*;^{yDCeOsW*4{UAZ-Q@p;Pxy#pIvYO%+vMydfBNk;@ssnACcib9>=Can62U_o3}A8<Y4|3z8=uypHD~IWX^YQzY~n5@R71fcQRiPurpT4N(zx)CnxTqwc5PTOoW3>o#H~_sP}VQi7XkM#l@4EuTk?(Tn;kbMKCC?$d$J~<)9aS<g}{Sw?@m>ii8xpuYR~v<fVcPB)XwwBOyk1y06W%*VS7%GKl)cW~iDsm@sV7y~@Lzp!0?&L5}eM4VyyqIe8ACxK1HbcP`H|nbPe|Y|*p~Zdb3r-i!xypGGP2CXwGI)?YPU@C{z^hQ^tYz;*v-*}+*|*s#bLuH*|}$LJ`Rnt(CT&3cC_P8eNhh3!UU2=>OlHr|-b+t=xElJ!UH<exWPd34pY=)^Uih!|PzCLzg=eu8tPL%PkyDMES#*AiP!#>8dw++Zg#<M2NW<qvky`%0blmkS%5=pp8OKOAh+*(9uY*?PZVO{EAdBEX@u&@_{kM7-)v#mNK5w_fviQ5gCX=N)g}qn0~QoSRyvnsB}3ne~~syyAYBqh6MJwpFVI-E1vCXRq5fp=-Faz^P0;OnU@Lul>;gNqd~N731FF*$lCWkE|<-CwH6ug_SNL2sy5XlZ**<?Q+)uNEINsDyW5xd-5^9nk|?Ef;l(TmOr>oFaHzQ%Fzvvo1NB?ZLRJdgwiU$-lxX!+<rx%jp~nd(F<f{LRhN6H=7&joAP?1S3OW6k~}1qfw<@23a1D*;6B?pw=1k%T{Xwp-7!=&wj=h*-c=nGU*ucC1fKizpia9`<cMrzp@CE)N3$+$ZeyKJW!gUU?1+VvWg9i?|H#5veGj6YgRU^=qkH4>ouzcyX4)kGuzEpYyr*lqD2>q4HbRH>-t+Xra;3{K!v2Yh`_}qoA>TV2kKHCBu<)c=we#O{XeFN>bD?=9zC>04eY8a(tDM`$j*lYhm}Je<3ehu0O_n++CrxCLttdzt-bDmVG0f@1LrPfmi#iWym}%xCa;Z6-883ZJ;XA?!uM@=JQdOiYwS_d;Xjle&!ifJ6R0yJ+maAu&vmrzR8)7HWDP!8N_8?aFcmxW_BOxN3a#cTullO-h`)f24;)vHQ9NQ8{*Vkmg>GE#T#?^-9fq1bSaQ+;Obd=CoVY1hGUfhTjn2GY5X6L|B_f~7CAOYTZ-euNzYl_UR!37LF-t!yg=R2~E=4BsNiXrKe=(45D+&k=1+z6#3$l~I|?_%a_vc&wMhDuaiq~WeDjQ`ms_#M&q@DgydL><XMoGU3#<mM7tM_<Qdw=iJr--P$`=+-AW8Rg28Q3HsN0w-J#+}G*&I|A3?HbG*;ee>zSkL85Y!^m+2w<*-NEnRvi=Hm02INq{p$H7!P<skbRaDbXty4`BWc;+CV&^mN%O`8nL5L{ex&&>LD<!nqk5hbJTyVrdONmB%8^(TATW9-u)+TqlZ77Mm)=<~&#1ObB9GqsCP*g^+rA9`9~_}PYutn;<>BQhvq24-{J3eEdp<OTt>?J&{K-Cs=M?cHA6X;3ZG5I@C-?-XBw&|xmLC1?H~pbqrd#flaGz=U~+f)1IA>``}+f9yYFk<T_brIlIoC-e-Pd6_fR>qN8K9g=qf0X*|+bAxfPEB7pI;;Qa)-``E$y@96rXi{QBK;Bq=SsJ?=wLwMu{&j&L`jf@r$>rO18lor>%@**D+*<bBu34q<FAx#XqJF)Cq9l4r`Mu#V==LI>ke?t~X^lJ~e<Hu%hvvV5viiUuFAu>%0e1i|QY9M9dmElHL1~BB6iCdQE&?X7=7#49cH8TIpmBvfZJvimHaW2Nwz#T#<)bdo<1}j0M7bv`7nrA#VqJ1Ax0I9*d%IC*8W!4Z(JRBPN{u<=?djpk|A8K&%Lsl@j=prQBt$FscbH-M1_(Vv4Xo<cUoS9tGkVS~oG^ntmUXQ-DD=y*K+TY}tYL8`5T{^tJmEghxLp@&dTHXo#-4&oN*uMEboTzB(cn(q{TbCzwb<lyCj{>2f77_noo<fey9YF+9YJ^#O>_9s-OVpg7h@ltZef@kMND?zB0v<e>_wt2E3Q(o5vPST_|yuBJUTIjB;p@!@6(+hTJRgJP-QtlJ^T|Iep#n;O&A6`2)e!%a|J9S!aLbVKCd?Y&@qVO=~Zw|)o>g^6+H=XiK}rMZ91fWfpp-DL_vx(y=in^oZFDV6o79^u12PbDtw<!bK@x`A53lBgbe6}CF=aN1XGq1R2{3x%C>YvpD^+tO=s_qfyi(U*6C$$ArbKsJVKHhB2OX=2=~kErr6I0?u+ME>5z~WK7qyB*HsH>Yu70>lzIRCG=q@`8(PZDw=iu#kHk-i!0^I1IVu4n%RgG8WFCYjloYKL=H_rQv_K?Zb(o}J>GBwTH^EXt2WAx~A0RD4z8?CR*1<+>N>!JG7GI2(F>+!dL3&nWoaXp1TQX8SwXgXgv+P5C0P))_k4_Q#$RcKx{Wy?Fl0hof<wao2;)}?RnBC56$EV0cnQ(Xeo24Mbg4yPe$+;23JwI&%stsv*F3QbD){L>eVYRPn0w1+$v-`0Tf;iGIvmERnOY9x?yV@3Tw0GKso3A}%SEo~XiF^yCUPk=CaV?lIm6j{<5Tw)WgnKn9^LRo!sf<4ufTF256A1Vjywd;5-uB0y=<o*nQkx%aJSH)Nz`KL>hnR`}WXUq#AkZI#*<}mh-jM5^Pzi|mdt(V^{ia2piy%K3d-6+EkcsS}z4I2DdaG<;V7dIh2=!U6R=y@~g2OXfjyKi1RfGoCAZfhIUoOUo&JJ&V;w+t-p*qb1PG$V<0v6#n0Yu^)HTa*hc&+8m`0(h#B)#++$Nl7K8;b0RDRMCt@|xUjwoE-IvU>DEXIXI6{s$#;+Pidu=-n1qFgWnVA~Sf8!q0qF9Iv!qiZk*kCALwY2k`$Y_QjjYCV+<rsP~3XYbJx`LzDEO@^uC^6eDf(CV`nb?%r0JZJ`TzjknE%3d)mCE6O?f7-XWHJwjd>rZ@3V862uOU48W<Ju+|c8lYT}(n<^Z9%psI;CLr6v>bf2QCOr5RqGQUq3HEe$3am3i%*ujRo8(z<m$lK35Zm@zmg&RA2L@?ygL<uOn)20|7|4t8wu8D48HEA1P%;Bzd@X5v)|WhJI)nlEp^?zv*i(Rj)8s}3p|SxuhAh5`JEq7vP#!QHoVf|#|d{Hv&t?wFD(LjRc?n(6lUg=Q`7M>%N9<w5g0!Jt~vj9eDW7f<rv%eb5^Imu87yc$jHFp|9^E?56Pf`?awW!L>lDrRxH1d<e=_t#ExHSL}MOWxAW54rQOc(?|@qh)t7V(xls}C0T3VZXR*H2H7-4XQdu3o&#N<-aHvA-4RT(NuIg8o?ovRTkkb6ctG~cc&2yCnsY#gx1+|aG<{ORs#33GLMRAgbH@|9F{-%BS_!p<Fxbl)A(P?mrTa?`ay3*6oZTvmamq+?}NBg0+`Ma59TU<g_<qjjg)d&%E!Kdgo$OLj!j`OhqPc;_rL1DFxgDZiefBG=;4PlBbG&Y!!=MP$8n7C<p57_Gl?8r(c|D-OIBQ=6QBhaHUq|cTir;i|q;eBZvw@Ta}gm~u#Ia55ddJGTaR5q~<+#E4l6E0vZ{s)_fKc#P|i1uGvnholF4WX3RR?t{dN?xTMEhR4({wkQV*Zk5i+x+)d>IQX%(kO;e9Jp#2-@A|0LtCM|_ZtpK2XNM)SU^Lp)+aOHY`i=BY()iEZP;un2CN6jG9Ur0zssvHZ=meWWI*G<cURNtoP1ZD5}mM86T=$g2Xec!Ni6c-C>o#&N}|CUi?Yy4u6&8HZ=%R*)enpX?6yVwOOPl5Vnf>1D^|C)V{0S-PS{<?78w90fYTy$ap<K9{G`<S{d0p#_?WY$nHly1oc6IqQur@rbZG_5Kv=)%dNK0^Geq*g*5rS`D)gfYS_jc{I2M8djcJn-q`)1R9ai$Ii}XW2dBS%5mc+UZfy)jW=vgjAtPOdG4tHU(LP-;+_a_6v%<vo{+}hDgC(UqX*gp`7T0#U2IYF`LEef1Fy`g8Loi+gcUKSaZH4+zB69~1CxKdIAZkjtz9p1cZSID-G9D07=C(Cv&t1L_r0Sx@It)6b<maaPYAm2t!EXB*C&o)QJ#oZ?9N>pc7g7tLyBNX|7SRL&bPQua&spYPf-;&*t`^#FUc+{7G&EbCR^URSAPB7i?A<8NaYI6<T3PT8<q`=<FRS%E+XEojErxmR&(4)~Y6U*go;W?}Q7p!vF)!XR#XcP78NdI{n*jukHTnhCiV1B|AlXtho78Z^ZF&VfA8xZ{wC8(t{OJN{eP!b=#hab=wSitb1pGH14ocePn4dqfCzj>wm`ojS*3W{d!_X~nOWu#A-T3Z5*kS`Kyd@#%G+8i+{vMb368VOwcT0$ac^tw_kAfTn$8ZoaF(n8Mxb!)hLF}i#~l_z4EOF3$fM!48m7ysLtVH^g4q=$q#x$3Dwm>`gx^nbkQepc)~f>&=v=WH3u`>Tb~HgOmH81qS<8N8rf#O#lpYqP~TCsdfECU{mcT;83FF#*WE=F3a*f83XTzRR}zR$#^te}msDX#->($TXw4z`Oqwjq4^^FpAiq>$mUVvKas4Ts_8w9qy<IbT+%L_P74~N1)cz0K;f%mlGjRdKC)ucb9$1S_$PS=TZ}hqC4zThfO8_WM8a)?4JdRFrtMuh;QZJrDR;sC_yAl13Yov0xTyQi=zS)y4FBv;zE$H*LX@*B*J)AYI4-JfzFYsFLpKX;h!5Xf%i%O^l<7IV#(czwS1T#O7koR&9^l6Uf!(WNaZS4zS$0~-Ct$T!zd${b^e9jeZ!CyMbkUjes|<f&;<sbUM;~cF!4dDI0oM|-z!Qav!v)Zg=&Plv7LCtDdBUPqLQ&6#`Se-#{JKg9aM6Y71132{yK%(sO0P*93S(a+Tp1{XCbkcC8B$Y9bB4Q+Rru$k(?ZRW#G_}vuAFqd|T{>U^4qDVHSFz@42H=2QQy)a?S}k{x?>?8gmd>@LI0t$@%>2IY*(I^UlV-G6&PlFs#)13`STMe2{c<&WgGVu|R&0l!ty=yKCzas42U(XxX2gzijLJmqCVAegBX4d$ge&JUE-v?_3v@ip1!!5{_?nb?_vas9pnj=5__-Z_!Ep|25*+o6yFcytUJf(SYL@jpLmp-_Qj3a3&nmUh}-*5ECm?s#=Z4ylYBzL2O(#roZ~i=D==TY{vxGaFV}!UbcF(8WfP)l;l=O)1Dz)jMURytIgDf3gdn5crM{gc;1<70z~rwm{DIQQ^@1OaoUZJEHf!tzDdwihk#_&BGW&2J{HZC@31~l4}Pm=Emf$I0v<*RAR_`2u40;l-0d+OPwPAYF*i4E(^+DlvOrpQqst17<trbnXCRJV!`ygmJq_dkzdY@{C(#T_=7*J@-pRFOBRj!B3UWMNZs2gF9fNicy<bHv^s)&~xu=s82pvjc@6<}yns=jA({#26tCkMrl|#OqFGY9mUc3*t7SRAP$@?sG4J1#_lh&txDbY~k#8Y*cU>L<9NGB|WGLYO103)UfZ=~2&H7amD2lwyYq}2bjyNid=&Jq;+3hwkw=$c}2ts>IC1w{u`7_VVs?EB57rD{`XX~0*NPX}YA8t46c^fl2Yw{~ca5RH>FY@j^9_*kk-hf4A?-V_<~#T|Gf+mvop=9y<V1c(PH6muQPQzUPXhlKc%;F0$Q3d!#+-QLmDrPHWhH-mvC_>KzN9{G;RCJw1)$t_zf^NC&vk6yTbjKU_j7R@Wm`ZbjKyXS=a8nn%@L`@^Ndr4U_P$Rdi)J$o+7hxr8XaYumU!2DCBAjJ1^tx)U;e0x)Sh$Xw0z==IE92EGB+u>WQQ9DR0ugP>J#_oBne;cgP{O>YM<F`iR#Ag+Etjs&%VVj=ZJDzca%=GV;MuQcMrky7CIQoa<;#d#*CFb<&;$;$fWo=%YYl)nmYsJYrfmws0O)Kd-%_#VDTBKf5vnC*qx%5m(&=-hzE8r=2Yd8i)_PUP{5o&4FtzGT)RJ*3BL$&xeG2<;g;*8T5h1nSl#D-!N6+2<-5vSa)GhZ{md;YSO~aQ>EfNSFp5*DT)t_3CRT9VYvN8BM98HsWIV?q|?r9l800#chn={}<2HQ`!*yG^P;~>gMx-gMY%?dGMF&y8YA?N(bd_@@$6eA{wgadkp3qw(F%rt$IcLW}l9JHo8Tfj3B1ZErHSJNta{>eP~2<J}Dk2T?1Ctd3FU*bGW_)B>$LfX21;xxxx*FSZaq|1IG0Wi~~iU45P1J*NUT#^DPbiHpz<*H#^zx7t4zn8>uQ6VR{sEy}E(^bWq?TR<ht(OU+<r6;!L3F{W=F=4IT-ZVDAHRZCAy?}>6(odP!$qX+f{vx-;lcW55-u>{B)fSQuZ2dIV9T-kqa=L_vFtra%<P92&FS_{axZ6XQJHvuF%z#jQlG6<Tp9Z)@)&2mR`Gr(%G}T_@Sr}>aW8UyIc7OD0mq=Y7h9cwN0D+xGRzTiu_EMfOOwFPUqyFArg-oBNxS3umYBjY>v*F@A<t`anpIAQrY61>IE*vqQr;Gym){S;bC0|0#%jKjP01DEA}VQO9aB>m&FVWan^7o*17&SNQdn$~oTZY{%r1PF0f=c8$OQbAwuY>=4Wck!78H>RL>jrjR|ELLcBu@2Ue(n79dn)UArX9{D+Q^0Yh+joxMhwgd)-Guhqs>dv2MWQ52|8dlpak=E_gslNPPm#eHJ}!E!>UQd^6J0gf`L#6v~#K)jeSnHK5iV8Kt%BZny(ezrbNj9n8BiH!6#nP}t?SOk^{sWrP&YCx}qOZPGK$z6flSArD}u=B}*H)4;noJ>0*)B~ASfBZjp~t@lRiHGWVZ1Nvmv>A%tFN<fNZm&pyFCPJGseyLIS!1K!iJT>J;y^%FdYy92=6u3OQ;9KUJ=QVj8I^i9FUa$^KU|iyK+`U6ou|$Xukj9~NnJlrnb(~|@Tw$!FI5_@_`MEOHie<MYuG1i+iWo=&G9Dz7sG*3j9jGJu)e9zXXmle&8a4vs8=gTVp92@dKV3i4xH$$BFW6`!jm+bjS8gRNz#3$TM<r{I$_<-ATaQM>t^Q|#VffLg1n|rA4;6+!M@0bx#D-_tTf6^~{OMHMNwfh(e}9cRS@R!`PYiLA&VENXZl-<e?-AmL!ahQT^uxLAdIh?JoOe#<d6VcC@hsccJW*MHs+(l!<_d;iPy~*ct~391C`9l|Phcx~hd37q&eL}0BvZ!wqW*|~W>!QM7LKO(Msbiv3)%2px#UH|R#q)bmrc&F7+)KhQ>d<JyQP5WhlPI41VwlT1Ayv;Gqg7V9+6_zMJy3OY%l)0$?wR=^`sD*gescs`NASUOHDm_KWKm35?2$Dpp5&vPm(+q5Fpi#YmjVJ2WQ1yRuUU;mjS~4T?Vzh#EavTo|8$~8f}w*iS+k%uHHobw0Eo^Aqq)sHpVf?(Ibhdm!t|x1IfoymOm~J#JY+(+1b$?h#?ajEGUH4S&icJ*k49))rknUGy;5<DEaJqghSDbE(a88S2qyD0*7ATb<J~<Rn?^qM@ASWR<@=j*2qOHleNcN)%+Y<?*o5NV$nqHU=p8P5^<*1t%H&}=wwh;xK^Li7$84Jkg|P~${aD@Mb+BzvrGU+6LsrW;6Pa112DkzB7@51&0^~wU{Tvm#;|%NIRumy;h>r{L%6ZR;(u=n=4~M1bQe1;?9**A8Fy-g5*D(C0fJ1fD8Dd^7R;ZkI`(ZLS6qK<#Bi-Br0Y<Z6b==4UaaS5O}ZVNYfp)^j0@e+!V2l+{$}-}PkjsJT`yc>x32o-LQKEvqQ$TEcVM~oWmU4wyWaZkbCNBKf*w!^<t^4~KcKd3au<%inKW;H7j|(UEA2=%Z{Iz`FVNtlQ9_f}1um8HN-=#de_yW<gzRZK2;5gmciWOyDYPQyPe!lM?z?JN+^m4{a@E0;1Gn_q!;irxbEO{kHtU`XVRsC=qt)oEn&e#@$UVULQxA+n2D@!Llk3QW)em=VO=K48D*t@<&f6t}FlU}IVHtK6JB%!B-Rjo5_YPer<nGr0zP%Om!17ah^i{SUXII1PmzxS2iA+}$<VTjRqCojk&P%q2c7TOtQ-_q)9PjVHfe7YiYCiNkKQ*!%V!Ny#m47`ewcH!)P8dUFnd!BW#L`qf%n_F4o*dI3m`lyt4#LlGJeft*xo%BWZdD|?1<|hs#~$)ZshrZcng!P;8#}(i@o38e3#(v|I)6+)&+Z)!>W?w7{HzTU3c(#1dNllA$8xay?V<TCQY%P@T(IhV>}WXjC2>9XwJbu@O`*?t0|C1jr~>Yy!E}ZOhr86Vx{QtN9mum)ZWq{z)n-Qa5ZMFX8$cP{3YUNZy9JH`xReZSBMUTu{mrWuwObu_8$5@W+;nZ+GOy$oDi=!S5%YAahGHa$`;!kbVHYJSFlD2pj_A&tHzG4C{+>CL4y^wwvlAo|LvAwNw^xz=XxPBkhZn!#0ZIr&T+5Mny%VYN$>t~5?)xhaX5?=G8kzJ)PfdTR&i=AJQ+$D}qu%d!PLZ_$A|lW{SR5O;|F0`9XA@JTZ+f5ES-L2_vBW38ui^mbPJkg&a-QkZLv`NN<pwWf0uJLl+EYSVEZ8Z*KvH^)L_)>VwS0k80YNJGXjmaeI-1lY?#Jr3#-{_I<xAdalqGsnlA+frX|Y|;&lUu}8kI=d$W68h$hs=vYr;XtN0FV)%<%MZW2Ol`x?7i-TTHe~kv+%%2~Selh!`hz{(!Ib*p|#2Mt3eX`g2*LS=alGW$JT}O}XSvi%7LOre09?x9gq5YPXm~NIgP3xC+3i&+H5dKT>kq&J10G-5nai&wynca=hZ_o?W7I5oH0Ah=L~D(Ko<FY-zD|*dcAD3YF9D-L%WWACd$t`e{=YgwjpLla!RfxDWe{=<&e8Fo$P3=5W|{VGbqmwJBLx+bb)rYH+LAtR9K0ymreI%W6(AN9zH3Dd14YlI~R+T!AxX>KHALYW(B>J^QeZVjURgyXfcKD&q-wDbm66E@Ku?F!1js_>&+=LXj_CZ2>x02UnEXPhtjKg@?^@l2<fro^_4YXg9}^>1Xfk{PF$W|AKI6HCg-2s=VY@Nlfs#k!8Hp+cS~8%dzf1L3#o=83V#CNTOlzdqDi(E2yK6X#}<~3tlp1rRw?XiG-6$Y2*@SGmR;l`y_p|*M|n=FD`?Jlm3K0pI(H#znF75;MFa%p~Ffvn1f0QoGFG-yV=n}fZl+Fr_B!?ci(iEx9OS5U5p(SL>-QPFHT>GirttKoM_4-sfvloI`9BJ3x}M(6KM_(Cr#K)ol9>XYzwa)ndM<pLjCzj)-P%XH=O(m8T7)tfQ4UxXi=v9$m6v?v&q-zRG!bM{e0%9b`~OJLgqS;cQ!{V`$w~|fEdNo0Z;>S(Q(BubCAzCc8BoYGnlC~8R9m*P&RLYaag%j@AuCsXq8&G6gUdBJKg#qe&OuM(rV-7^2B0QtQh|v&oyULyJ@JViTt?ce|Xw^8=gqCFX`8Fzte?200000O$^Hi`|$a*00FVc0jcH%cozNLvBYQl0ssI200dcD")).decode("utf-8")))
for _n, _s in E28_SRC.items():
    (TMP / _n).write_text(_s, encoding="utf-8")
import e28k100_lib as E28L
import e31_direct as E31D   # E31
E28S, E28S_ERROR = None, None
if "e28_sched_patch.py" in E28_SRC:
    try:
        import e28_sched_patch as E28S
    except Exception as _e:
        E28S_ERROR = f"{type(_e).__name__}: {_e}"[:400]
        print("!!!!!!!!!! E28 SCHED MODULE NOT LOADED:", E28S_ERROR, flush=True)
else:
    E28S_ERROR = "E31: the A / B / C scheduler is not embedded (e28_sched off: Duck is not run)"
PPK, PPK_ERROR = None, None   # E26 K100 Q-defer: NB4S's nb4_ppack.py rev 2 (sha 8dadf076); a miss = no Q-defer, the run goes on
try:
    (TMP / "nb4_ppack.py").write_text(k100q_ds_text("nb4_ppack.py", "8dadf0763d43d22966603745198ad327ed6ed36001cc60a79925ea18818f2dea"), encoding="utf-8")
    import nb4_ppack as PPK
except Exception as _e:
    PPK_ERROR = f"{type(_e).__name__}: {_e}"[:400]
    print("K100 Q-DEFER LIB NOT LOADED (the run continues without Q-defer):", PPK_ERROR, flush=True)
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
CONFIG["nb5"]["k100"] = {"pins": K100_PINS, "pin_steps_gib": K100_PIN_STEPS_GIB, "seqs_table": K100_SEQS_TABLE, "captures": K100_CAPS,
                         "flags": K100_EXTRA_FLAGS, "env": K100_ENV, "window_sec": K100_WINDOW_SEC, "expected": K100_EXPECTED,
                         "report_lib_sha256": hashlib.sha256(K100Q_REPORT_SRC.encode()).hexdigest() if K100Q_REPORT_SRC else None,
                         "boot_levers": K100_BOOT_LEVERS, "tasks": K100_TASKS, "new_groups": K100_NEW_GROUPS,
                         "ref27_sha256": hashlib.sha256(json.dumps(K100Q_REF27, sort_keys=True).encode()).hexdigest() if K100Q_REF27 else None}   # E26 K100
CONFIG["nb5"]["e28"] = {"t1": E28_T1, "b2": E28_B2, "server_max_len": 131072, "b2_timeout": E28_B2_TIMEOUT,
                        "b2_window": E28_B2_WINDOW, "window_sec": E28_WINDOW_SEC, "i8_env": E28_I8_ENV, "levers": E28_LEVERS,
                        "modules": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in E28_SRC.items()},
                        "sched_sha": getattr(E28S, "E28_SCHED_SHA", None), "sched_error": E28S_ERROR,
                        "lib": E28L.LIB_VERSION}   # E28 K100
CONFIG["nb5"]["e31"] = {"cfg": dict(E31_CFG), "budget_min": E31_BUDGET_MIN, "window_sec": E31_WINDOW_SEC, "tasks": E31_TASKS,
                        "groups": E31_GROUPS, "deadman_sec": K100_DEADMAN_SEC, "lib": E31D.LIB_VERSION,
                        "levers": E31_LEVERS}   # E31
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
# ---- E28 K100: the A / B / C scheduler (e28_sched_patch.py, Build/E28_SCHED_API.md section 2) on arc2_duck.py: after the
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
    _e28p = getattr(self, "_e28_plan", None)         # E28 SCHED N1: the attempt's own plan budget (B2 = 120k)
    pk = ((NB5_PROMPT_ALLOW + int(_e28p["budget"]) + NB5_TOOL_ALLOW) if _e28p
          else NB5_BLOCKS.get("peak_b0" if int(getattr(self, "arc2_branch", 0) or 0) == 0 else "peak_b1"))
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
    _e28p = getattr(self, "e28_plan", None)          # E28 SCHED N2: an E28 attempt reserves its own projected peak
    need = (int(NB5L.cdiv(min(MAX_LEN, NB5_PROMPT_ALLOW + int(_e28p["budget"]) + NB5_TOOL_ALLOW), NB5_BLOCKS["block"])
                + int(NB5_BLOCKS["f"] or 0)) if (_e28p and NB5_ADMIT_MODE != "live" and NB5_BLOCKS.get("f") is not None)
            else int(NB5_BLOCKS["b0" if self.branch == 0 else "b1"] or 1) if NB5_ADMIT_MODE != "live"
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


def _e28_try_reserve(key, plan):
    """E28 SCHED N4: the scheduler's non-blocking fit check = a reservation taken NOW or refused NOW (never waits), so a
    free worker can backfill a smaller attempt when the head (a B2) does not fit. The blocks are N2's; the
    attempt's own acquire in _nb5_session_run then finds its key held. Same grant rules as NB5L.Reservation.acquire
    (idle grant; live hold; C-7 = a need above the capacity is left to _nb5_session_run, which denies it)."""
    res = NB5_RES
    if res is None:
        return True
    if NB5_ADMIT_MODE == "slots":
        need = 1
    elif NB5_ADMIT_MODE == "live":
        need = int(NB5_BLOCKS["start"] or 1)
    elif NB5_BLOCKS.get("f") is not None:
        need = int(NB5L.cdiv(min(MAX_LEN, NB5_PROMPT_ALLOW + int(plan["budget"]) + NB5_TOOL_ALLOW), NB5_BLOCKS["block"])
                   + int(NB5_BLOCKS["f"] or 0))
    else:
        need = int(NB5_BLOCKS["b0" if int(plan.get("branch") or 0) == 0 else "b1"] or 1)
    with res.cond:
        if key in res.held:
            return True
        n = need + int(res.f_extra)
        if n > res.cap:
            return True
        if res.held and (res.live_hold or res.used + n > res.cap):
            return False
        if not res.held:
            res.stats["idle_grants"] += 1
        res.held[key] = n
        res.used += n
        res.stats["grants"] += 1
        res.stats["e28_try_grants"] = res.stats.get("e28_try_grants", 0) + 1
        res.stats["max_used"] = max(res.stats["max_used"], res.used)
        res.stats["max_held"] = max(res.stats["max_held"], len(res.held))
        return True


def _e28_release(key):
    """E28 SCHED N4: frees a reservation taken by _e28_try_reserve (idempotent; the attempt's own finally also frees)."""
    if NB5_RES is not None:
        NB5_RES.release(key)


E12A.E28_TRY_RESERVE, E12A.E28_RELEASE = _e28_try_reserve, _e28_release   # read by the E28 scheduler (E28_SCHED=1)
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
# E26 K100: klee100 LOAD R1-18 lab loader (the same sha ef393fbb as the packed literal of NB5)
K100_LAB_SHA = "ef393fbb2ff7669ee4c2c99ee841b60da66514a3b86b7c0ce81fd5f64689ab5e"     # E25 klee100 R1-18: the E23 lab = a sha-checked file of the NB0 reference dataset (< 400 kB)
K100_LAB_FILE = next((d / "k100load_lab.py" for d in (INP / "datasets/foysalemonshanto/q27-e25-nb0-tf-ref", INP / "q27-e25-nb0-tf-ref",
                                                      Path(os.environ.get("K100_PAYLOAD_DIR") or "/nonexistent"))
                      if (d / "k100load_lab.py").is_file()), None)
if K100_LAB_FILE is None or hashlib.sha256(K100_LAB_FILE.read_bytes()).hexdigest() != K100_LAB_SHA:
    raise RuntimeError(f"K100_PAYLOAD_LAB missing or sha mismatch: {K100_LAB_FILE} (dataset q27-e25-nb0-tf-ref, k100load_lab.py)")
LAB_SRC = K100_LAB_FILE.read_bytes().decode("utf-8")
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



# ==================================================================== E26 K100: klee100 LOAD rev 3 model path (sha 7773e63f)
import struct                                    # klee100 LOAD CELL 1 ("import struct   # E25 klee100"): st_header below
R["setup"]["k100q_payload"] = K100Q_PAYLOAD_USED  # E26 K100: the dataset file CELL 1 loaded (k100q_payload)


# ---- klee100 LOAD rev 3 CELL 2b (verbatim; K100_EMB was loaded in CELL 1 from the dataset file instead of the inline literal)
# ==================================================================== CELL 2b — E25 klee100: fixkeys overlay, model dir, K6, digests (NEW)
import base64 as _b64
import lzma as _lzma
K1 = R.setdefault("k100", {})
K100_EMB = dict(K100_EMB)   # E26 K100: CELL 1's k100q_payload() (klee100 LOAD rev 3's blobs, tf-ref dataset)
K100_PIN = json.loads("{\"klee100_config_sha256\": \"fb17316df83cc1cf6cb4f5336fefb33566102cfeda16d1e9fc89ae1a4c61b9ce\", \"klee100_index_sha256\": \"873191b509904c7007ee9a791ecc18a939afbf11797e1549c0e3279bf31cefd4\", \"klee100_tokenizer_sha256\": \"06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523\", \"overlay_sha256\": {\"config.json\": \"c78d48f36421925a0fdb1c75aafe667c8d4e554d4431faa8f87f8924f09b9e27\", \"quantization_config.json\": \"e55659bc3c7cf29c668fafa45804dec879eb24ff5ab1e25eaa587ca462f2ed35\", \"model.safetensors.index.json\": \"c21d8c98821febdbab08331a4e55cb34c8e56bf4b902f5f45f1a047536213872\", \"FIXKEYS_DIFF.json\": \"70d06119bbea99b1db466fc026b0f4fb584a50b196f05fe2bc9e48595d6d8ab1\"}, \"digests_sha256\": \"b50690a074a884298771cc408238dad9fd881bddbbac89372616114a65e1ae50\", \"ref_sha256\": {\"A_2k_fresh.npz\": \"ea5c2a5a68d50e2e4abe7bd42595599775b867e817bb37b045ee0e0d3a24fe55\", \"A_2k_repeat.npz\": \"ea5c2a5a68d50e2e4abe7bd42595599775b867e817bb37b045ee0e0d3a24fe55\", \"A_40k_fresh.npz\": \"0254ad67ffd27d7e5952c894e9ba6a4b098ae91785b550d008bddd31e9ada1b9\", \"A_40k_repeat.npz\": \"0254ad67ffd27d7e5952c894e9ba6a4b098ae91785b550d008bddd31e9ada1b9\", \"A_8k_fresh.npz\": \"e3332ec70a40ab4377c6815d384fcfbade455a451404d5293c4a5e9d6c6cfe21\", \"A_8k_repeat.npz\": \"e3332ec70a40ab4377c6815d384fcfbade455a451404d5293c4a5e9d6c6cfe21\", \"A_90k_fresh.npz\": \"30c5800f6fa6e1d8cd3848f21ed1ebfa3b24e2b0207ca1553e9dfbdd0e520b35\", \"A_90k_repeat.npz\": \"30c5800f6fa6e1d8cd3848f21ed1ebfa3b24e2b0207ca1553e9dfbdd0e520b35\", \"A_dec2k_cached.npz\": \"da9098138876abc1e321d51eeaff6ffae270fca0d0d79db9dff4fb96e9e81463\", \"A_dec2k_cold.npz\": \"da9098138876abc1e321d51eeaff6ffae270fca0d0d79db9dff4fb96e9e81463\", \"A_dec40k_cached.npz\": \"062fcaebb2e31f489f97e057639d62195414e7a64660f5243abb275eaff15e76\", \"A_dec40k_cold.npz\": \"062fcaebb2e31f489f97e057639d62195414e7a64660f5243abb275eaff15e76\", \"A_dec8k_cached.npz\": \"bf9fbfb9904fe44b8dda56b6fba3b0c5f2ebb8f0e1ca463b4012f5bfac91edff\", \"A_dec8k_cold.npz\": \"bf9fbfb9904fe44b8dda56b6fba3b0c5f2ebb8f0e1ca463b4012f5bfac91edff\", \"A_sc2k_tf.npz\": \"4c0885ce1deeb48da3178843d4a29bf44944be71e4c097246b4d6eaf5d666902\", \"A_sc40k_tf.npz\": \"1e6f07a49cee93c57470c177834633a614c7ccaa8e2a6a22d42459b1cfc45994\", \"A_sc8k_tf.npz\": \"446eec9e4554172bde0fd73f96812ad2e06b779b61adeaca6eb5fb79246319da\", \"C_2k_fresh.npz\": \"ea5c2a5a68d50e2e4abe7bd42595599775b867e817bb37b045ee0e0d3a24fe55\", \"C_2k_repeat.npz\": \"ea5c2a5a68d50e2e4abe7bd42595599775b867e817bb37b045ee0e0d3a24fe55\", \"C_40k_fresh.npz\": \"0254ad67ffd27d7e5952c894e9ba6a4b098ae91785b550d008bddd31e9ada1b9\", \"C_40k_repeat.npz\": \"0254ad67ffd27d7e5952c894e9ba6a4b098ae91785b550d008bddd31e9ada1b9\", \"C_8k_fresh.npz\": \"e3332ec70a40ab4377c6815d384fcfbade455a451404d5293c4a5e9d6c6cfe21\", \"C_8k_repeat.npz\": \"e3332ec70a40ab4377c6815d384fcfbade455a451404d5293c4a5e9d6c6cfe21\", \"k100load_lab.py\": \"ef393fbb2ff7669ee4c2c99ee841b60da66514a3b86b7c0ce81fd5f64689ab5e\", \"nb0_reference.json\": \"e1e7429678abd83315313bd2f095ce1da5e5aab333013f31a801c60e7ece51a7\", \"qsa_kv_scales_262k.json\": \"5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67\"}, \"kit_v2_notebook_sha256\": \"b39db932e2227ab697b7fded01354440d80bb907c52cc99ef5a84a98a6df407b\", \"kit_core_sha256\": \"db19969dcd6be4da263271904f4bdb598cef237a307a9b31960ce2723bcb63ef\", \"lib_sha256\": \"0ba5f7c6328cdaa24276ac92eb22d4b2015b578675e9c36c711d1ad1a7f7f1b9\", \"mtp_tensor_bytes\": 1425540096, \"lab_payload_sha256\": \"ef393fbb2ff7669ee4c2c99ee841b60da66514a3b86b7c0ce81fd5f64689ab5e\", \"scales_sha256\": \"5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67\", \"nb0_builder_sha256\": \"166b5afd7f6d2a052c9496885b358fbc86f397816ca829ad3081cf21f7de65de\", \"nb1_builder_sha256\": \"15a926ad5e23055602a625009dee6a07cce3426972b8e92017e9f0e7c7bc288d\"}")       # build-time facts: v2 overlay shas, klee100 input shas, the NB0 reference SHA256SUMS
EMB = TMP / "k100_emb"
_emb_ok = {}
for _n, (_want, _blob) in K100_EMB.items():
    _raw = _lzma.decompress(_b64.b85decode(_blob))
    _emb_ok[_n] = hashlib.sha256(_raw).hexdigest() == _want
    (EMB / _n).parent.mkdir(parents=True, exist_ok=True)
    (EMB / _n).write_bytes(_raw)
K1["embedded_sha_ok"] = _emb_ok
if not all(_emb_ok.values()):
    raise RuntimeError("K100_EMBEDDED_SHA_MISMATCH: " + json.dumps(_emb_ok))
sys.path.insert(0, str(EMB))
import k100load_lib as KL                # E25 klee100 pure helpers (unit-tested: merge / K6 / digests / pin ladder / labels)
import k100kit_core as KC                # the klee100 CPU kit v2's own core (fixkeys / graft / write_overlay)
CKPT = TMP / "fn_klee100"                # E25 klee100: rebinds RC2's merged-checkpoint path (CELL 2)
OVERLAY = TMP / "k100_fixkeys_overlay"
DIGESTS = json.loads((EMB / "klee100_shard_digests.json").read_text())
K100_SHA_THREADS = int(os.environ.get("K100_SHA_THREADS") or min(8, max(2, os.cpu_count() or 4)))


def st_header(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n))


def k100_overlay():
    """The fixkeys overlay regenerated by the v2 kit's own code (Kit.fixkeys / graft / write_overlay) from the MOUNTED
    klee100 config.json (sha pinned) and the kit's compact tensor tables; the three files must equal the v2 kit artifact
    byte for byte. Cross-checks: the mounted index (sha pinned) weight_map == the klee100 table; the mounted FP8 PLE
    shard headers == Merlin's PLE entries the graft writes into the index."""
    rec = {}
    raw = (K100_DIR / "config.json").read_bytes()
    rec["config_sha256"] = hashlib.sha256(raw).hexdigest()
    rec["config_sha_ok"] = rec["config_sha256"] == K100_PIN["klee100_config_sha256"]
    iraw = (K100_DIR / "model.safetensors.index.json").read_bytes()
    rec["index_sha256"] = hashlib.sha256(iraw).hexdigest()
    rec["index_sha_ok"] = rec["index_sha256"] == K100_PIN["klee100_index_sha256"]
    tabs, enc = {}, {}
    for who in ("klee100", "merlin"):
        enc[who] = json.loads((EMB / f"tables/{who}.json").read_text())
        tabs[who] = KC.expand_table(enc[who])
        rec[f"table_{who}_ok"] = KC.canon_sha(tabs[who]) == enc[who]["canon_sha256"] and len(tabs[who]) == enc[who]["n_tensors"]
    rec["mounted_index_equals_table"] = json.loads(iraw)["weight_map"] == {n: v[2] for n, v in tabs["klee100"].items()}
    ple = {n: v for n, v in tabs["merlin"].items() if n.startswith(KC.PLE_PFX + "ngram_embedding.")}
    bad = []
    for f in sorted({v[2] for v in ple.values()}):
        try:
            h = st_header(PLE_DIR / f)
            got = {n: (v["dtype"], tuple(v["shape"]), f) for n, v in h.items() if n != "__metadata__"}
            if got != {n: v for n, v in ple.items() if v[2] == f}:
                bad.append(f)
        except Exception as e:
            bad.append(f"{f}: {e!r}"[:120])
    rec["ple_headers_equal_graft"], rec["ple_header_mismatch"] = not bad, bad
    kit = KC.Kit.__new__(KC.Kit)
    kit.R, kit.log, kit.tabs, kit.enc_klee100 = {"artifacts": {}}, log, tabs, enc["klee100"]
    kit.kcfg_raw, kit.kcfg = raw, json.loads(raw)
    cfg_fix, diff = kit.fixkeys()
    idx, _tab, chk = kit.graft()
    files = kit.write_overlay(OVERLAY, cfg_fix, diff, idx)
    rec.update(graft={k: v for k, v in chk.items() if k != "files"}, diff_problems=diff["problems"], files=files,
               equal_v2_artifact={n: files[n]["sha256"] == K100_PIN["overlay_sha256"][n] for n in KL.OVERLAY_FILES})
    rec["ok"] = bool(rec["config_sha_ok"] and rec["index_sha_ok"] and rec["table_klee100_ok"] and rec["table_merlin_ok"]
                     and rec["mounted_index_equals_table"] and rec["ple_headers_equal_graft"] and chk["ok"] and not diff["problems"]
                     and all(rec["equal_v2_artifact"].values()))
    shutil.copyfile(OVERLAY / "FIXKEYS_DIFF.json", OUT / "FIXKEYS_DIFF.json")
    return rec


def mirror_compare(plan):
    """KAGGLE_MIRROR_MANIFEST.json (the uploader's streaming hashes) vs the trusted digest manifest, for every planned file
    it lists (recorded; the streaming sha256 below is the gate)."""
    try:
        raw = (K100_DIR / "KAGGLE_MIRROR_MANIFEST.json").read_bytes()
        mm = json.loads(raw)
    except Exception as e:
        return {"error": repr(e)[:200]}
    want, found = {n: s for n, _, _, s, _ in plan}, {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in want and isinstance(v, dict) and isinstance(v.get("sha256"), str):
                    found[k] = v["sha256"]
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(mm)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "listed": len(found), "agree": sum(found[k] == want[k] for k in found),
            "disagree": sorted(k for k in found if found[k] != want[k])}


K100_SCALES_SHA = "5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67"      # R3-1: qsa_kv_scales_262k.json (12 QSA layers) = the W4 Merlin p1 checkpoint's sidecar
SCALES_CALIBRATION = "EXPERIMENTAL_REUSED_CALIBRATION (the W4-calibrated QSA sidecar reused on klee100)"   # V-7


def k100_scales(tree=None):
    """R3-1: the FP8 E4M3 QSA static K/V scales sidecar for K3: <MERLIN_TREE>/models/flash-next/scales/qsa_kv_scales_262k.json
    when the install has it, else qsa_kv_scales_262k.json of the reference dataset; sha-checked (K100_SCALES_SHA) and copied
    into the klee100 model dir, where Merlin's FP8 plan reads it (SCALES / VLLM_QSA_KV_SCALES). Missing / changed -> ok
    False: K3 is skipped (SKIPPED_NO_SCALES); the BF16 boots do not need it."""
    cands = ([Path(tree) / "models/flash-next/scales/qsa_kv_scales_262k.json"] if tree else []) + \
            ([REF_DIR / "qsa_kv_scales_262k.json"] if REF_DIR is not None else [])
    looked, src, raw = [], None, None
    for c in cands:                                         # V-5: the first SHA-VERIFIED candidate, every one recorded
        if not c.is_file():
            looked.append({"path": str(c), "exists": False})
            continue
        b_ = c.read_bytes()
        h_ = hashlib.sha256(b_).hexdigest()
        looked.append({"path": str(c), "exists": True, "sha256": h_, "ok": h_ == K100_SCALES_SHA})
        if h_ == K100_SCALES_SHA and src is None:
            src, raw = c, b_
    if src is None:
        return {"ok": False, "why": "no sha-verified qsa_kv_scales_262k.json (Merlin tree, then the reference dataset)", "looked": looked}
    (CKPT / "qsa_kv_scales_262k.json").write_bytes(raw)
    return {"ok": True, "sha256": K100_SCALES_SHA, "layers": len(json.loads(raw)), "src": str(src), "looked": looked,
            "file": str(CKPT / "qsa_kv_scales_262k.json"), "calibration": SCALES_CALIBRATION}


def merge_ckpt():
    """E25 klee100 (rebinds RC2's merge; same contract): the fixkeys overlay (regenerated, sha-checked), one model dir of
    symlinks (every klee100 file + the 10 FP8 PLE shards) + the overlay files, K6 (every indexed shard resolves), then the
    streaming sha256 of every used file vs the trusted HF LFS digest manifest in a background thread (sha_gate waits for
    it before the first launch). Fail closed: any miss -> ok False, no boot."""
    SHA.update(thread=None, ok=None, bad=[], sec=None)
    rec = {"dir": str(CKPT)}
    try:
        rec["overlay"] = k100_overlay()
    except Exception as e:
        rec["overlay"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-1500:]}
    if not rec["overlay"].get("ok"):
        return dict(rec, ok=False, bad=["FIXKEYS_OVERLAY"],
                    why="FIXKEYS_OVERLAY: " + json.dumps({k: v for k, v in rec["overlay"].items() if k not in ("files", "tb")}, default=str)[:600])
    md = KL.build_model_dir(CKPT, K100_DIR, PLE_DIR, OVERLAY, sorted(PLE_EXPECTED))
    rec["model_dir"] = {k: (len(v) if k in ("klee100_links", "ple_links") else v) for k, v in md.items()}
    rec["k6"] = k6 = KL.k6_check(CKPT)
    plan = KL.digest_plan(DIGESTS, K100_DIR, PLE_DIR)
    rec["digest_plan"] = {"files": len(plan), "gb": round(sum(x[2] for x in plan) / 1e9, 2), "threads": K100_SHA_THREADS,
                          "names": [x[0] for x in plan]}
    ple_d = {f: [v["size"], v["sha256"]] for f, v in DIGESTS["repos"]["merlin_fp8_ple"]["shards"].items()}
    rec["ple_expected_equals_digests"] = ple_d == {k: list(v) for k, v in PLE_EXPECTED.items()}
    rec["digest_manifest_sha_ok"] = hashlib.sha256((EMB / "klee100_shard_digests.json").read_bytes()).hexdigest() == K100_PIN["digests_sha256"]
    rec["mirror_manifest"] = mirror_compare(plan)
    bad = ([] if md["ok"] else ["MODEL_DIR: " + "; ".join(md["errors"][:3])]) + ([] if k6["ok"] else [k6["why"]])
    if not rec["ple_expected_equals_digests"]:
        bad.append("PLE_EXPECTED (RC2) != the digest manifest's FP8 PLE shards")
    if not rec["digest_manifest_sha_ok"]:
        bad.append("digest manifest sha")
    rec.update(ok=not bad, bad=bad, why="; ".join(bad) or None)

    def hash_all():
        res = KL.verify_digests(plan, threads=K100_SHA_THREADS, log=log)
        with _SAVE_LOCK:                   # R1-13: never mutate R while another thread's save() dumps it
            SHA.update(ok=res["ok"], bad=res["bad"], sec=res["sec"])
            K1["digests"] = res
            R["setup"]["sha256"] = {k: res[k] for k in ("ok", "bad", "n_planned", "n_checked", "gb", "sec", "gb_per_s", "threads")}
        save()
    if rec["ok"]:
        SHA["thread"] = threading.Thread(target=hash_all, daemon=True)
        SHA["thread"].start()
    return rec
save()

# ---- NB1 CELL 6 rpc() (verbatim, as picked by klee100 LOAD)
def rpc(port, method, *args, timeout=120):
    """POST /collective_rpc (dev mode): a worker method by name with string args; the per-rank results or the error."""
    body = {"method": method, "args": [str(a) for a in args], "timeout": timeout}
    try:
        r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/collective_rpc", data=json.dumps(body).encode(),
                                               headers={"Content-Type": "application/json"}), timeout=timeout + 60).read())
        return {"results": r.get("results")}
    except Exception as e:
        return {"error": repr(e)[:300]}


def k100_worker_ext():
    """E26 K100 (= NB1 / klee100 LOAD apply_60490's extension part): the read-only worker extension (K-7 method map, V-4
    scales) next to the installed tree's vllm/ (on the server's PYTHONPATH), py_compile with the Merlin venv, and both
    vllm and nb1_worker_ext resolve inside the tree. Any miss raises (no boot)."""
    if FAKE:
        return {"ok": None, "fake": True, "why": "local harness: no Merlin tree"}
    eng = merlin_install()
    tree, py = Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"
    ext = tree / "nb1_worker_ext.py"
    ext.write_bytes((EMB / "nb1_worker_ext.py").read_bytes())
    rec = {"file": str(ext), "sha256": hashlib.sha256(ext.read_bytes()).hexdigest(), "sha256_expected": K100_EMB["nb1_worker_ext.py"][0]}
    cp = subprocess.run([py, "-c", "import py_compile, sys\nfor f in sys.argv[1:]: py_compile.compile(f, doraise=True)", str(ext)],
                        capture_output=True, text=True, timeout=300)
    rec["py_compile"] = {"rc": cp.returncode, "err": cp.stderr[-400:]}
    envp = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    envp["PYTHONPATH"] = str(tree)            # what Merlin's build() exports for the server
    cp2 = subprocess.run([py, "-c", "import importlib.util as u; print(u.find_spec('vllm').origin); print(u.find_spec('nb1_worker_ext').origin)"],
                         capture_output=True, text=True, timeout=300, env=envp)
    rec["origins"] = cp2.stdout.strip()[-400:]
    origin_ok = all(l.startswith(str(tree)) for l in cp2.stdout.strip().splitlines()[-2:]) and len(cp2.stdout.strip().splitlines()) >= 2
    rec["ok"] = bool(rec["sha256"] == rec["sha256_expected"] and not cp.returncode and origin_ok)
    if not rec["ok"]:
        raise RuntimeError("K100_WORKER_EXT_FAILED: " + json.dumps(rec)[:600])
    return rec


def k100_setup():
    """E26 K100, before the first boot (klee100 LOAD rev 3 CELL 8): the OVERLAY_MERGE_K6 gate (fixkeys overlay + model dir + K6
    at merge time; the K7 / K3 digests are the sha gate), the worker extension, the FP8 QSA static scales sidecar (the
    installed Merlin tree first, then the tf-ref dataset; sha 5cbe6ae8) copied into the klee100 dir. Any miss raises: no boot,
    CELL 0's fallback submission stays."""
    m = (R["sections"].get("A0") or {}).get("merge") or {}
    out = {"overlay_merge_k6": {"ok": m.get("ok") is True, "why": m.get("why") or m.get("bad")}}
    NB5_REPORT["k100_setup"] = out
    if not out["overlay_merge_k6"]["ok"]:
        raise RuntimeError("K100_OVERLAY_MERGE_K6_FAILED: " + json.dumps(out["overlay_merge_k6"], default=str)[:600])
    out["worker_ext"] = k100_worker_ext()
    try:                                                     # Q-defer: the apply never raises (k100_qdefer_apply is fail-safe)
        out["qdefer"] = k100_qdefer_apply()
    except Exception as e:
        K100_QD.update(applied=False, env={}, status="APPLY_FAILED", error=repr(e)[:600])
        out["qdefer"] = dict(K100_QD)
    out["qdefer_tree"] = K100_QD.get("tree_check")
    try:                                                     # audit I: the tree check, OUTSIDE the try above
        out["qdefer_tree"] = k100_qd_gate()
    finally:
        out["qdefer_tree"] = K100_QD.get("tree_check")
        nb5_write_report()
    K1["scales"] = out["scales"] = k100_scales(((MERLIN_ENG or {}).get("kv") or {}).get("MERLIN_TREE"))
    nb5_write_report()
    if out["scales"].get("ok") is not True:
        raise RuntimeError("K100_NO_SCALES: " + json.dumps(out["scales"], default=str)[:600])
    log("K100 setup", json.dumps({k: (v.get("ok") if isinstance(v, dict) else v) for k, v in out.items()}))
    return out


K100_QD = {"applied": False, "env": {}, "status": None}   # E26 K100 Q-defer: the patch state + the server env it allows


K100_QD_CHAIN = ("vllm/v1/worker/gpu/pp_utils.py", "vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu_worker.py")
K100_QD_SHAS = {"vllm/v1/worker/gpu/pp_utils.py": ["6be931deaf8bee6dc9571c3c84d7e92c9f26c618c09dcb58a8b384bff487a662", "8b7fa4a6e4598ff4d4a56eb6e595efb1e494276b12b187b099973c262e90c69a"], "vllm/v1/worker/gpu/model_runner.py": ["05e889afba5dd60b6995b78aaa1f69051770ab02b638680097eccbf4e592d64c", "2c25fda60dbd5f2506ff989c29ccfe8fc0ff082fbad457fc3ae9c89ed9a185d6"], "vllm/v1/worker/gpu_worker.py": ["a97d08ef042a53fcd139bae041fff407fce75796650d936c200c699f0d63a489", "5642b89d918b30e5b70b1067d53776e379613035b7b6e91e93cf0ee4279cfef5"]}   # {rel: [Merlin ad42e37 pre-sha, nb4_ppack rev 2 post-sha]} (= nb4_ppack.PATCHES)


def k100_qd_tree_check(tree=None):
    """Audit I: the Merlin tree before any boot is trusted only when it is the verified patch (status APPLIED, every file at
    its post-sha) or Merlin's original (every file at its pre-sha, no failed restore / py_compile). Anything else
    (an incomplete or inconsistent restore, a restore / py_compile error, a mixed or unknown tree) is unverified."""
    tree = tree or K100_QD.get("tree")
    if tree is None:
        if FAKE:
            return {"verified": True, "state": "FAKE_NO_TREE", "why": "local fake server: no Merlin tree"}
        eng = merlin_install()
        tree = eng["kv"]["MERLIN_TREE"]
    sh = {}
    for rel in K100_QD_CHAIN:
        try:
            sh[rel] = hashlib.sha256((Path(tree) / rel).read_bytes()).hexdigest()
        except Exception as e:
            sh[rel] = f"unreadable: {type(e).__name__}"
    state = ("applied" if all(sh[r] == K100_QD_SHAS[r][1] for r in K100_QD_CHAIN) else
             "not_applied" if all(sh[r] == K100_QD_SHAS[r][0] for r in K100_QD_CHAIN) else "unverified")
    rpc = ((K100_QD.get("restore") or {}).get("py_compile") or {}).get("rc", 0) if "restore" in K100_QD else 0
    if state == "applied":
        ok = bool(K100_QD.get("applied") and K100_QD.get("status") == "APPLIED")
    else:
        ok = state == "not_applied" and not K100_QD.get("applied") and rpc == 0
    return {"verified": ok, "state": state, "status": K100_QD.get("status"), "restore_py_compile_rc": rpc, "shas": sh, "tree": str(tree)}


def k100_qd_gate(tree=None):
    """Audit I, the k100 gates' rule: an unverified tree is NO_GO_K100_QDEFER_TREE_UNVERIFIED outside the rerun (raised: no
    boot, CELL 0's fallback submission stays); in the rerun it is recorded loudly and the run goes on without NB4_PPDEFER."""
    try:
        chk = k100_qd_tree_check(tree)
    except Exception as e:
        chk = {"verified": False, "state": "CHECK_ERROR", "why": f"{type(e).__name__}: {e}"[:300]}
    K100_QD["tree_check"] = chk
    if chk["verified"]:
        return chk
    K100_QD.update(applied=False, env={})
    msg = "NO_GO_K100_QDEFER_TREE_UNVERIFIED: " + json.dumps({k: chk.get(k) for k in ("state", "status", "restore_py_compile_rc", "why")})
    log("!!!!!!!!!! K100 " + msg + (" -> RERUN: recorded, the run continues WITHOUT NB4_PPDEFER" if RERUN else " -> commit: no boot"))
    if not RERUN:
        raise RuntimeError(msg)
    return chk


def k100_qd_consistent(tree):
    """Audit G: the patched files must be a prefix of [pp_utils, model_runner, gpu_worker] (patched model_runner calls
    pp_handler.flush_deferred_receive, which only patched pp_utils has; patched gpu_worker is guarded) and every file is at
    Merlin's pre- or the patch's post-sha. With NB4_PPDEFER unset, patched files run the original code paths."""
    st = []
    for rel in K100_QD_CHAIN:
        pre, post = PPK.PATCHES[rel][0], PPK.PATCHES[rel][1]
        try:
            h = hashlib.sha256((Path(tree) / rel).read_bytes()).hexdigest()
        except Exception:
            h = None
        st.append("patched" if h == post else ("original" if h == pre else "other"))
    n = st.index("original") if "original" in st else len(st)
    return {"states": dict(zip(K100_QD_CHAIN, st)), "consistent": "other" not in st and all(x == "original" for x in st[n:])}


def k100_qd_restore(tree, snap, py):
    """Audit G (NB6Q's nb6q_restore, ordered): the three files back to their snapshot bytes in REVERSE dependency order
    (gpu_worker, model_runner, pp_utils), atomic replace, stale pyc dropped, sha == snapshot; the first failing file stops
    the restore (so the patched set stays a prefix); then py_compile with the Merlin venv. Never raises."""
    out, ok = {}, True
    for rel in reversed(K100_QD_CHAIN):
        f = Path(tree) / rel
        try:
            if f.read_bytes() != snap[rel]:
                tmp = f.with_name(f.name + ".k100qtmp")
                tmp.write_bytes(snap[rel])
                os.replace(tmp, f)
                out[rel] = "restored"
            else:
                out[rel] = "unchanged"
            for pyc in (f.parent / "__pycache__").glob(f.stem + ".*.pyc"):
                pyc.unlink()
            if hashlib.sha256(f.read_bytes()).hexdigest() != hashlib.sha256(snap[rel]).hexdigest():
                raise RuntimeError("sha != snapshot after the restore")
        except Exception as e:
            out[rel], ok = f"error: {type(e).__name__}: {e}"[:300], False
            break
    try:
        pc = PPK._py_compile(py, [Path(tree) / rel for rel in K100_QD_CHAIN])
    except Exception as e:
        pc = {"rc": None, "err": repr(e)[:300]}
    return {"ok": ok and pc.get("rc") == 0, "files": out, "py_compile": pc, **k100_qd_consistent(tree)}


def k100_qdefer_apply(tree=None, py=None):
    """E26 K100 Q-defer (user 2026-11-01; NB4S: +21 % at 8k@8, +20 % at 40k@8, oracle EXACT): NB4S's path,
    nb4_ppack.apply_ppack_patch on the installed Merlin tree (pre / post sha of pp_utils.py, model_runner.py, gpu_worker.py,
    py_compile) after the install, before any boot. Fail-safe (audit G): the three files are snapshotted first; any miss ->
    k100_qd_restore (reverse order), a loud log, no env (NB4_PPDEFER unset = the original code paths). Never raises."""
    K100_QD.update(applied=False, env={}, status=None, error=None)
    snap = None
    if PPK is None:
        K100_QD.update(status="NO_LIB", error=globals().get("PPK_ERROR"))
    elif tree is None and FAKE:
        K100_QD["status"] = "FAKE_SKIPPED"
    else:
        try:
            if tree is None:
                eng = merlin_install()
                tree, py = Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"
            K100_QD.update(tree=str(tree), py=py)
            snap = {rel: (Path(tree) / rel).read_bytes() for rel in K100_QD_CHAIN}
            K100_QD["apply"] = PPK.apply_ppack_patch(tree, py)
            v = PPK.verify_ppack_patch(tree)
            if v["state"] != "applied" or any(v["files"][r]["sha"] != PPK.PATCHES[r][1] for r in K100_QD_CHAIN):
                raise RuntimeError("K100_QD_POST_SHA_VERIFY_FAILED: " + json.dumps(v)[:600])
            K100_QD.update(applied=True, status="APPLIED", env={PPK.ENV_DEFER: "1"})
        except Exception as e:
            K100_QD.update(status="APPLY_FAILED", error=f"{type(e).__name__}: {e}"[:900])
            if snap is None:
                K100_QD["status"] = "APPLY_FAILED_NOT_TOUCHED"
            else:
                try:
                    K100_QD["restore"] = r = k100_qd_restore(tree, snap, py)
                    K100_QD["status"] = "APPLY_FAILED_RESTORED" if r["ok"] else (
                        "APPLY_FAILED_RESTORE_INCOMPLETE" + ("_CONSISTENT" if r["consistent"] else "_INCONSISTENT"))
                except Exception as e2:
                    K100_QD.update(status="APPLY_FAILED_RESTORE_ERROR", restore_error=repr(e2)[:400])
    log("K100 Q-DEFER", "APPLIED: NB4_PPDEFER=1 on every launch" if K100_QD["applied"] else
        f"!!!!!!!!!! NOT APPLIED ({K100_QD['status']}: {K100_QD.get('error')}): the run continues WITHOUT the patch and the env !!!!!!!!!!")
    return {k: v for k, v in K100_QD.items() if k != "apply"}


def k100_qd_readback(s):
    """E26 K100: nb4_ppack.readback_label (Q-defer, TP2 x PP2) on the server's whole log: ENGAGED / ENGAGED_FALLBACK /
    NOT_ENGAGED / NOT_INTEGRATED + the per-rank exec share. Recorded only, never fatal (commit and rerun)."""
    if not K100_QD.get("applied"):
        return {"label": "NOT_APPLIED", "status": K100_QD.get("status")}
    try:
        lab, det = PPK.readback_label(s.log() or "", "Q-defer")
    except Exception as e:
        return {"label": "READBACK_ERROR", "error": f"{type(e).__name__}: {e}"[:300]}
    ex = [v.get("exec_share") for v in (det.get("defer_per_rank") or {}).values() if v.get("exec_share") is not None]
    return {"label": lab, "exec_share_min": min(ex) if ex else None, "detail": det}


_merlin_fn_server_nb5 = merlin_fn_server       # RC2 CELL 2 + NB5's edits (launch.build + host_phase + the NB5 levers)


def merlin_fn_server(name, kv, port, kv_bytes):
    """E26 K100 (= klee100 LOAD rev 3's wrapper on the NB5 boot path): K6 before every launch (one missing indexed shard -> no
    server is started), the NB5 plan, VLLM_QSA_KV_SCALES for the FP8 boot (Merlin also sets it from SCALES), K-9: vLLM's own
    `serve --help=all` against THIS argv (a missing flag -> no launch, FAIL_CLI_FLAGS)."""
    k6 = KL.k6_check(CKPT)
    K1.setdefault("k6_before_launch", {})[name] = {k: k6.get(k) for k in ("ok", "why", "index_files", "missing")}
    if not k6["ok"]:
        save()
        raise RuntimeError("K6_INDEXED_SHARD_MISSING before launch: " + str(k6["why"]))
    s = _merlin_fn_server_nb5(name, kv, port, kv_bytes)
    if kv == "fp8":                                          # R3-1: K3's static K/V scales (Merlin also sets it from SCALES)
        s.env["VLLM_QSA_KV_SCALES"] = str(CKPT / "qsa_kv_scales_262k.json")
    for k_ in ("NB4_PPDEFER", "NB4_PPACK"):                  # Q-defer: only on a verified patch (never inherited)
        s.env.pop(k_, None)
    s.env.update(K100_QD.get("env") or {})
    K100_QD.setdefault("launch_env", {})[name] = s.env.get("NB4_PPDEFER")
    cli = K1.setdefault("cli_flags", {})[name] = KL.cli_flags_check(HELP.get(name))
    if cli["ok"] is False:
        save()
        raise RuntimeError("FAIL_CLI_FLAGS: " + str(cli["why"]))
    save()
    return s


def k100_gate_verdict(mm, sc, kv, fake=False, fake_gates=False):
    """E26 K100: klee100 LOAD rev 3's K3 config checks on the NB5 boot: GPU_METHOD_MAP_PASS = the method map's gate_ok
    (GPU_METHOD_MAP_PASS or METHOD_CLASS_ONLY) and, on FP8 KV, STATIC_SCALES_APPLIED (every rank applied its stage's QSA
    scales). fatal = the first failing check (None = pass). A fake server is recorded, never gated (unless fake_gates)."""
    checks = {"GPU_METHOD_MAP_PASS": (mm or {}).get("gate_ok") is True}
    if kv == "fp8":
        checks["STATIC_SCALES_APPLIED"] = (sc or {}).get("ok") is True
    fatal = next((k for k, v in checks.items() if not v), None)
    gated = not fake or fake_gates
    return {"checks": checks, "ok": fatal is None, "fatal": fatal if gated else None, "gated": gated,
            "method_map_verdict": (mm or {}).get("verdict"), "why": (mm or {}).get("why") if fatal == "GPU_METHOD_MAP_PASS" else
            ((sc or {}).get("why") if fatal else None)}


def k100_boot_gates(s, mtp_on):
    """E26 K100 (klee100 LOAD rev 3 k100_facts, the K3 gate part): the worker extension's per-rank method map (raw rows kept)
    -> KL.method_map_check on this boot's 25/23 partition; the per-rank static-scales lines -> KL.scales_check vs the sidecar;
    the modules' own k / v scales (recorded); then k100_gate_verdict."""
    t = s.log() or ""
    part = KL.parse_partition((getattr(s, "env", None) or {}).get("VLLM_PP_LAYER_PARTITION"))
    o = {"partition": list(part)}
    o["method_map_raw"] = raw_mm = rpc(s.port, "k100_method_map")
    o["method_map"] = mm = KL.method_map_check((raw_mm or {}).get("results"), need_mtp=bool(mtp_on), partition=part)
    sc = None
    if NB5_BOOT_KV == "fp8":
        try:
            side = json.loads((CKPT / "qsa_kv_scales_262k.json").read_text())
        except Exception as e:
            side = {}
            o["scales_sidecar_error"] = repr(e)[:200]
        o["scales"] = sc = KL.scales_check(t, side, part)
        sc["calibration"] = SCALES_CALIBRATION
        o["scales_rpc"] = rpc(s.port, "k100_kv_scales")
    o.update(k100_gate_verdict(mm, sc, NB5_BOOT_KV, fake=bool(FAKE), fake_gates=os.environ.get("K100_FAKE_GATES") == "1"))
    log("K100 boot gates", json.dumps({k: o.get(k) for k in ("checks", "ok", "fatal", "gated", "method_map_verdict", "why")}, default=str)[:900])
    return o


def k100_finish(tasks, window, phase_dir=None, window_start_epoch=None, task_solve=None):
    """E26 K100: per task finished / solved + the finish time in the window (each attempt's last request; k100q_report)."""
    return K100Q.finish_table(tasks, window, phase_dir, window_start_epoch, task_solve, P.get("task_list"))



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


# ---- RC2 CELL 1 input block, wrapped
def nb5_rc2_inputs():
    # RC2 CELL 1 input discovery (exact Model paths), GPU preflight, watchdog
    global PLE_EXPECTED, MD, PART_DIRS, PARTS, k, d, mf, PLE_DIR, MERLIN_PAY, INTEL, WHEELHOUSE, ARC, MERLIN_OK, ABORT, NVSMI_OK
    global K100_DIR, REF_DIR, _g                                     # E26 K100
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
    PARTS = {}                                     # E25 klee100: no W4 parts; klee100 + the FP8 PLE overlay + the Merlin env
    K100_DIR = MD / "qwen38-flash-next-klee100-3bpw-mtp/pytorch/autoround-3bpw-mtp-nople/1"            # E25 klee100
    if not (K100_DIR / "KAGGLE_MIRROR_MANIFEST.json").is_file():                                          # E25 klee100
        _g = sorted((MD / "qwen38-flash-next-klee100-3bpw-mtp").glob("*/*/*/KAGGLE_MIRROR_MANIFEST.json"))  # E25 klee100
        K100_DIR = _g[0].parent if len(_g) == 1 else None                                                 # E25 klee100
    REF_DIR = next((d for d in (INP / "datasets/foysalemonshanto/q27-e25-nb0-tf-ref", INP / "q27-e25-nb0-tf-ref")   # E25 klee100
                    if (d / "nb0_reference.json").is_file()), None)                                       # E25 klee100
    MERLIN_OK = K100_DIR is not None and PLE_DIR is not None and MERLIN_PAY is not None                   # E25 klee100
    R["inputs"] = {"klee100_dir": str(K100_DIR), "ref_dir": str(REF_DIR), "ple_dir": str(PLE_DIR),   # E25 klee100
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
    _src = CKPT if MERLIN_PROMPTS else (K100_DIR or INTEL)   # E25 klee100: the mount's own tokenizer when the merge failed
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


    TOK_SHA = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"   # E25 klee100: klee100@600715b3 tokenizer.json (HF LFS; ids checked vs NB0 in CELL 7)
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
    if NB5_ADMIT_MODE == "slots" or P.get("e28_workers_rule"):   # R2 (user decision): workers = max_num_seqs = the 64k-context rule
        # E28 K100: the R2 rule (21 workers) also when a profile is not "slots" (E28 runs "slots", as K100Q)
        _wr = NB5L.workers_rule(pool, block, plan["f_eff"])
        _seq = caps.get("max_num_seqs")
        _w = min(_wr, int(_seq)) if _seq else _wr
        if P.get("top_rung_seqs") and _seq and int(_seq) == int(P["top_rung_seqs"]):   # E28 K100 (user 2026-11-01): the top
            _w = int(_seq)                   # rung launched 24 -> 24 workers (lower rungs: K100Q's R2 value, the min above)
        plan["r2_workers"] = {"rule": _wr, "rule_text": "max(1, floor(pool / (ceil(65536 / B) + F)))", "pool": pool,
                              "f_eff": plan["f_eff"], "launch_max_num_seqs": _seq, "workers": _w,
                              "rule_vs_launch": None if not _seq else ("EQUAL" if _wr == int(_seq) else
                                                                       ("RULE_BELOW_LAUNCH" if _wr < int(_seq) else "RULE_ABOVE_LAUNCH")),
                              "projected_120k_workers": plan["workers"],
                              "captures_cover_workers": NB5L.captures_cover(NB5_CAPS16 if P.get("caps16") else (K100_CAPS if P.get("k100_caps") else [32]), _w),   # E26 K100
                              "captures_cover_pp2_microbatch": max(K100_CAPS) >= -(-_w * 4 // 2), "top_rung_seqs": P.get("top_rung_seqs"),
                              "rung_workers": _w}   # E28 K100: this rung's workers
        if plan["workers"] >= 1:                   # C-8 stays: no attempt kind fits the pool -> NO_GO (workers 0)
            plan["workers"] = _w
    if caps.get("max_num_seqs") is None:
        plan["warning"] = "max_num_seqs not found in the server log: workers = pool streams"
    plan["e28"] = E28L.capacity_note(pool, block, plan["f_eff"], NB5_GATE, E28_T1, E28_B2, MAX_LEN, MAX_LEN,
                                     plan["workers"])   # E28 K100: T1 / B2 blocks (recorded; "slots" admits by workers)
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
    tasks = NB5L.qab_tasks(E12_RECORDS, SOLUTIONS, rep.get("eval_unit_records"), NB5_QAB_GROUP_OF, P.get("task_list") or list(TASK_IDS),   # E28 K100
                           {"b0": E15B["E12_B_ATTEMPT_TOKENS"], "b1": E15B["E12_B_ATTEMPT_TOKENS_B1"]}, ws_ep, we_ep,
                           (NB5_TRACE.summary() or {}).get("by_attempt"))
    return {"window": win, "order": rep.get("order"),
            "fp8_status": NB5_QAB_FP8_LABEL if NB5_BOOT_KV == "fp8" else "BF16_REFERENCE",
            "capacity_levers": {"pool_blocks": cap.get("pool_blocks"), "pool_source": cap.get("pool_source"),
                                "capacity_blocks": cap.get("capacity_blocks"), "f_eff": cap.get("f_eff"), "f_log": cap.get("f_log"),
                                "f_source": cap.get("f_source"), "pin_gib": tr.get("pin_gib"), "kv_bytes": tr.get("kv_bytes"),
                                "lever_env": tr.get("levers"), "max_num_seqs_launch": tr.get("max_num_seqs_launch"),
                                "workers": int(CONCURRENCY), "r2_workers": cap.get("r2_workers"),
                                "captures": NB5_CAPS16 if P.get("caps16") else (K100_CAPS if P.get("k100_caps") else None),   # E26 K100
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
        global NB5_BOOT_EXTRA, NB5_BOOT_LEVERS
        _ab = dict(NB5_ACTIVE_BOOT)                       # E26 K100: the exact booted config (kv bytes, seqs, captures, levers)
        if _ab:
            NB5_BOOT_EXTRA, NB5_BOOT_LEVERS = list(_ab["extra"]), dict(_ab["levers"])
        _kvb = int(_ab["kv_bytes"]) if _ab else NB5_KV_BYTES[NB5_BOOT_KV]
        srv = merlin_fn_server(old.name, _ab.get("kv") or NB5_BOOT_KV, old.port, _kvb)
        srv.env.update(_ab.get("env_keep") or {})           # E26 K100 Q-defer: the relaunch keeps NB4_PPDEFER
        NB5_REPORT.setdefault("relaunch_boot", []).append({"kv_bytes": _kvb, "max_num_seqs": _ab.get("max_num_seqs"),
                                                       "pin_gib": _ab.get("pin_gib"), "from": _ab.get("name")})
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


NB5_ACTIVE_BOOT = {}   # E26 K100: the booted try (kv_bytes, extra = seqs + captures, levers, name); e21_relaunch re-uses it
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
        NB5_ACTIVE_BOOT.clear()                               # E26 K100: this try booted: its exact config for a relaunch
        NB5_ACTIVE_BOOT.update(name=name, kv=NB5_BOOT_KV, kv_bytes=int(kv_bytes), extra=list(NB5_BOOT_EXTRA),
                               levers=dict(NB5_BOOT_LEVERS), max_num_seqs=trec.get("max_num_seqs_launch"),
                               pin_gib=trec.get("pin_gib"), env_keep={k_: v_ for k_, v_ in (getattr(s, "env", None) or {}).items()
                                                                       if k_ == "NB4_PPDEFER"})
        trec["config_readback"] = NB5L.config_readback(s.log(), s.argv, NB5_BOOT_KV, mtp_on)    # C-11
        trec["i8_readback"] = ({"label": "FAKE_SERVER", "want": NB5_BOOT_LEVERS.get("DRAFT_HEAD")} if FAKE else
                               E28L.i8_readback(s.log(), s.argv, NB5_BOOT_LEVERS.get("DRAFT_HEAD")))   # E28 K100 I8 (NB4T, recorded)
        E28_I8["readback"] = trec["i8_readback"].get("label")
        if e28_i8_gate(trec):                                # E28 K100 I8 (audit 2026-11-01): INT8 requested -> I8_ENGAGED required
            return {}                                        # this server stops (with_servers); the ladder retries the same pin, BF16
        if not trec["config_readback"]["ok"] and not RERUN:
            trec["status"] = "INCONCLUSIVE_CONFIG_MISMATCH"
            for sp in phases:
                NB5_REPORT["phases"].setdefault(sp[0], {"spec": sp, "boot": name}).update(
                    status="INCONCLUSIVE_CONFIG_MISMATCH", config_readback=trec["config_readback"]["checks"])
            return {}
        trec["qdefer_readback_boot"] = k100_qd_readback(s)   # E26 K100 Q-defer: init lines (recorded, never fatal)
        trec["k100_gates"] = k100_boot_gates(s, mtp_on)      # E26 K100: klee100 LOAD rev 3 K3 checks (method map, static scales)
        if trec["k100_gates"].get("fatal") and not RERUN:    # fatal outside the rerun (NB5 rule for probe defects); recorded in it
            trec["status"] = "NO_GO_K100_" + str(trec["k100_gates"]["fatal"])
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
        trec["qdefer_readback"] = k100_qd_readback(E11_SERVER or s)   # E26 K100 Q-defer: after the phases (never fatal)
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
        tries = (NB5L.pin_tries(NB5_KV_BYTES[NB5_BOOT_KV], start, steps=K100_PIN_STEPS_GIB) if P.get("k100")   # E26 K100: the K3 ladder
                 else NB5L.pin_tries(NB5_KV_BYTES[NB5_BOOT_KV], start))
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
        NB5_BOOT_LEVERS = dict(levers, **(E28_I8_ENV if E28_I8["on"] else {}))   # E28 K100 I8: DRAFT_HEAD=int8 into build()
        seqs = NB5L.seqs_for(NB5_BOOT_KV, tr["gib"]) if NB5_SEQS_RULE == "table" else (K100_SEQS_TABLE[round(float(tr["gib"]), 2)] if NB5_SEQS_RULE == "k100" else None)   # E26 K100
        if P.get("top_rung_seqs") and abs(float(tr["gib"]) - K100_PIN_STEPS_GIB[0]) < 1e-9:   # E28 K100 (user 2026-11-01):
            seqs = int(P["top_rung_seqs"])            # 24 on the TOP rung only (an I8 -> BF16 retry there keeps 24)
        NB5_BOOT_EXTRA = base_extra + (["--max-num-seqs", str(seqs)] if seqs else [])
        name = (tag if not P.get("pin_ladder") else f"{tag}_pin{int(round(tr['gib'] * 100))}" + ("" if levers or not NB5_LEVER_ENV else "_nolever")
                + ("_i8" if E28_I8["on"] else ""))   # E28 K100
        trec = {"name": name, "pin_gib": tr["gib"], "kv_bytes": tr["kv_bytes"], "max_num_seqs_launch": seqs, "levers": dict(levers),
                "e28_i8": bool(E28_I8["on"])}   # E28 K100
        brec["tries"].append(trec)
        globals()["NB6_READY_SEC"] = ((30 if len(brec["tries"]) == 1 else 15) * 60 if P.get("e18_ladder")
                                      else NB5_READY_TIMEOUT)          # NB6 P3: E18 R2 cell 6 L485
        nb5_boot_try(trec, name, tr["kv_bytes"], phases, mtp_on)
        _e28_log = trec.pop("log_text_for_ladder", "")   # E28 K100: kept for the I8 drop record
        fail = NB5L.boot_failure(trec.get("status"), _e28_log, levers, trec.get("phases_started", False))
        if P.get("e18_ladder"):   # NB6 P3: E18 R2 retries on ANY failed attempt (not ready, stall, smoke) before solving
            fail["failed"] = trec.get("status") != "DONE" and not trec.get("phases_started", False)
        trec["ladder"] = fail
        log("NB5 boot try", name, trec.get("status"), "| ladder", json.dumps({k: fail[k] for k in ("failed", "lever_suspect")}))
        nb5_write_report()
        if (E28_I8["on"] and not ABORT.get("why") and not trec.get("phases_started") and not (P.get("e18_ladder") and len(brec["tries"]) >= 3)
                and (fail["failed"] or str(trec.get("status")) in ("INCONCLUSIVE_CONFIG_MISMATCH", "I8_READBACK_UNVERIFIED"))):
            # E28 K100 I8 (user 2026-11-01): a try that failed with the INT8 draft head, or booted without a verified INT8
            # head (audit 2026-11-01: I8_READBACK_UNVERIFIED) -> the SAME pin with the BF16 head (I8_UNSUPPORTED_ON_K100, a
            # lever drop: the proven pin and the 21 workers stay)
            _e28_dr = (dict(label=E28L.I8_DROP, kind="READBACK_UNVERIFIED", status=trec.get("status"), readback=trec.get("i8_failure"))
                       if str(trec.get("status")) == "I8_READBACK_UNVERIFIED" else E28L.i8_failure_reason(trec.get("status"), _e28_log))
            E28_I8.update(on=False, dropped=dict(_e28_dr, try_name=name, pin_gib=tr["gib"]))
            brec.setdefault("levers_dropped", []).append({"keys": ["DRAFT_HEAD"], "why": E28L.I8_DROP + ": " + E28_I8["dropped"]["kind"],
                                                          "try": name})
            log("!!!!!!!!!! E28 K100", E28L.I8_DROP, json.dumps(E28_I8["dropped"], default=str)[:600], "-> the same pin, BF16 draft head")
            continue
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


    E12_ORDER = (list(TASK_IDS) if os.environ.get("E28_SCHED", "").strip() == "1" or P.get("plain_order")   # E28 SCHED N3: no ORDER2
                 else sorted(TASK_IDS, key=lambda t: (-_e12_order2_score(t), _e12_total_cells(t), _E12_INDEX[t])))
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


# ==================================================================== CELL 7b — E31: the DIRECT solver replaces Duck (E12A.solve_all_r6)
# User 2026-11-01 (Exp 31): e21_cell6 looks E12A.solve_all_r6 up at call time, so this rebind swaps Duck for the proven ORDER2
# direct method (e31_direct.py, sha-pinned in CELL 1's E28_SRC literal) with Duck's signature, return and record contract:
# e21_cell6 (window, preflight, watchdog, the E15B writer callback), e21_cell7 (score, bundle), the nb5 phase record and the
# E28 report run unchanged. E12A.build_submission -> e31_build_submission, DIRECT-only (audit E31-2): the direct records' own
# ORDER2 make_attempts (attempt_1 = the model grid, attempt_2 = the test input / its mirror), the identity for a task without a
# record; no Duck candidate / CPU rule. The solver returns as soon as every task is done (E31_CFG window_min is a cap). Full
# thinking of every request: phase dir duck/<task>_b0/ (transcript.txt + token_ids.json).
E31_RUN = {"calls": [], "tokenizer": None}


def e31_tok():
    """The checkpoint's own tokenizer + chat template (CELL 8's NB5_LOCAL_TOK: the pinned klee100 tokenizer / template hashes
    verified), ORDER2's render (enable_thinking True, reasoning_effort xhigh) and decode; EOS ids from generation_config.json."""
    if not NB5_LOCAL_TOK.get("verified"):
        raise RuntimeError("E31: the Flash-Next tokenizer / template are not the pinned ones: " + str(NB5_LOCAL_TOK.get("why")))
    _hf, _tp = NB5_LOCAL_TOK["hf"], NB5_LOCAL_TOK["template"]
    _gp = next((p_ for p_ in (CKPT / "generation_config.json", Path(str(globals().get("K100_DIR") or "/nonexistent")) / "generation_config.json")
                if p_.is_file()), None)
    _eos = (json.loads(_gp.read_text(encoding="utf-8")).get("eos_token_id") if _gp is not None else None) or []
    t_ = {"render": lambda m_: _hf.apply_chat_template(m_, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                                      reasoning_effort="xhigh", chat_template=_tp),
          "encode": lambda s_: TOK.encode(s_, add_special_tokens=False).ids,
          "decode": lambda i_, k_: TOK.decode(list(i_), skip_special_tokens=bool(k_)),
          "eos_ids": _eos if isinstance(_eos, list) else [_eos]}
    E31_RUN["tokenizer"] = {"generation_config": str(_gp), "eos_ids": t_["eos_ids"],
                            "stop_ids_equal_spec": t_["encode"](E31D.STOP_THINKING_TEXT) == E31D.FN_STOP_IDS,
                            "think_close_ids": t_["encode"]("</think>")}
    return t_


def e31_solve_all_r6(tasks, order, *, out_dir, deadline, base_url, model_id, on_branch_done=None, workers=10,
                     capacity_probe=None, **_rest):
    """E12A.solve_all_r6's signature: Duck-only keyword arguments are recorded (ignored_args), never used."""
    _w = min(int(E31_CFG["workers"]), int(E12_RUN["streams"]))   # merge list 3: the engine's streams (DIRECT runs no sandbox:
    # the isolation-gated Duck `workers` argument is recorded, not used)
    # audit E31-6: the client talks to THIS notebook's booted server only (E28's SGL_URL, port NB5_PORT; never E18's 8011);
    # e21_relaunch keeps the port, so the URL holds across a relaunch; the E15B watchdog is the one lifecycle owner
    _srv_url = str(getattr(E11_SERVER, "url", "") or "").rstrip("/")
    assert SGL_URL and SGL_URL.rstrip("/") == _srv_url and SGL_URL.rstrip("/").endswith(f":{NB5_PORT}") \
        and str(base_url).rstrip("/") == SGL_URL.rstrip("/") + "/v1", ("E31: base_url != the booted server", SGL_URL, _srv_url, base_url, NB5_PORT)
    E31_RUN["calls"].append({"tasks": len(tasks), "workers_arg": int(workers), "workers": _w, "base_url": base_url,
                             "model_id": model_id, "deadline_in_sec": round(deadline - time.monotonic(), 1), "ignored": sorted(_rest)})
    return E31D.solve_all_direct(tasks, order, out_dir=out_dir, deadline=deadline, base_url=SGL_URL, model_id=model_id,
                                 on_branch_done=on_branch_done, workers=_w, capacity_probe=capacity_probe, cfg=E31_CFG,
                                 tok=e31_tok(), solutions=SOLUTIONS, report_dir=BASE_OUT, server_down=E12A.SERVER_DOWN,
                                 metrics=e21_counters, sched_last=E12A.SCHED_LAST, sampler_path=OUT / "nb5_sampler.jsonl",
                                 t0_pc=NOTEBOOK_T0, harness=E11_HARNESS, ignored=sorted(_rest))


if not hasattr(E12A, "_e31_orig_solve_all_r6"):      # once (a re-run of this cell never wraps twice)
    E12A._e31_orig_solve_all_r6, E12A._e31_orig_build_submission = E12A.solve_all_r6, E12A.build_submission


def e31_build_submission(tasks, records):
    """audit E31-2: DIRECT-only (no Duck candidate, no harvested program, no CPU train-exact rule): the direct record's ORDER2
    make_attempts per test input; a task without a direct record = the identity (ORDER2 write_submission)."""
    return E31D.build_submission_direct(tasks, records)


E12A.solve_all_r6, E12A.build_submission = e31_solve_all_r6, e31_build_submission
CONFIG["e31_solver"] = {"lib": E31D.LIB_VERSION, "solve_all_r6": "e31_solve_all_r6", "build_submission": "e31_build_submission"}
print("E31: E12A.solve_all_r6 -> the direct solver (", E31D.LIB_VERSION, ") | E31_CFG", json.dumps(E31_CFG), "| budget",
      E31_BUDGET_MIN, "min | deadman", K100_DEADMAN_SEC, "s", flush=True)


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
    k100_setup()   # E26 K100: OVERLAY_MERGE_K6 gate, worker extension, static scales sidecar (any miss: no boot)
    E28_I8.update(want=bool(P.get("i8")), on=bool(P.get("i8")))   # E28 K100 I8: DRAFT_HEAD=int8 from the first boot try
    if E28_I8["want"]:
        E28_I8["static"] = e28_i8_static()
        if E28_I8["static"].get("verdict") == "UNSUPPORTED_STATIC":
            E28_I8.update(on=False, dropped={"label": E28L.I8_DROP, "kind": "STATIC", "why": E28_I8["static"].get("why")})
            log("!!!!!!!!!! E28 K100", E28L.I8_DROP, "(static):", json.dumps(E28_I8["static"], default=str)[:600], "-> BF16 draft head")
    NB5_REPORT["e28_i8"] = E28_I8
    NB5_REPORT["e28_sched"] = E28_SCHED_REC
    log("E28 K100 I8", json.dumps({k: E28_I8.get(k) for k in ("want", "on", "dropped")}, default=str)[:600], "| static",
        (E28_I8.get("static") or {}).get("verdict"), "| scheduler applied", E28_SCHED_REC.get("applied"))
    if P.get("fixed_window_sec"):    # E26 K100: NB5 QUALITY's solve window, fixed (not split: one phase)
        NB5_EQ.update(window_sec=float(P["fixed_window_sec"]), plan={"window_sec": float(P["fixed_window_sec"]), "fixed": True,
                      "source": "E28 K100: the fixed 6-h EVAL120 window (README section 1)"})
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
                           for n, p in _q.items()} for t in (P.get("task_list") or NB5_ALL_IDS)}}   # E28 K100
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
        if not P.get("fixed_window_sec"):              # E26 K100: NB5's two-phase line only (Q_FP8_K3 / Q_BF16_K3)
            print("deployment_comparison:", _qa["deployment_comparison"]["status"], json.dumps(_qa["windows_check"]), "| matrix:",
              json.dumps({n: (p["outcomes"], p["tasks_missing"]) for n, p in _qa["deployment_comparison"]
                          ["missing_incomplete_failed"]["phases"].items()}), "| precision_comparison: UNVERIFIED | phase_diff:",
              json.dumps(_qa["phase_diff"]["differences"], default=str)[:2500], flush=True)
        else:
            print("deployment_comparison: not applicable (E28 K100: one phase; e28_k100_report.json is its report)", flush=True)
    nb5_write_report()
    print("NB5 REPORT:", json.dumps({"throughput": NB5_REPORT["throughput"], "phases": {n: {k: p.get(k) for k in (
        "status", "score", "workers", "running_check", "commit_mode")} for n, p in _ph.items()}}, default=str)[:6000], flush=True)
except BaseException:
    print("NB5 report error:", traceback.format_exc()[-2000:], flush=True)
# ---- E28 K100: the run report (e28k100_lib.build_e28_report): score, tok/s, preemptions, prefix hit, the per-attempt table
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
try:
    _pack = WORKING / "Q27X_NB5_BUNDLE.zip"
    with zipfile.ZipFile(_pack, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in [BASE_OUT / "nb5_report.json", BASE_OUT / "e28_k100_report.json", SUBMISSION_PATH,   # E28 K100
                   ENG_OUT / "nb5_engine.json", ENG_OUT / "telemetry.jsonl"] \
                + sorted(LOGS.glob("*.log")) + sorted(TRACES.glob("*")) + sorted(BASE_OUT.glob("phase_*/e12_run_report.json")) \
                + sorted(BASE_OUT.glob("phase_*/nb5_sampler.jsonl")):
            if fp.exists() and fp.is_file():
                z.write(fp, arcname=fp.relative_to(WORKING).as_posix() if WORKING in fp.parents else fp.name)
    print("NB5 bundle:", _pack, round(_pack.stat().st_size / 1e6, 1), "MB", flush=True)
except BaseException:
    print("NB5 bundle error:", traceback.format_exc()[-1500:], flush=True)
