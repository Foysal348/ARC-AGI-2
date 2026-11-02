"""E33 Poetiq solver module builder (Experiment 33, user 2026-11-02, fixed spec DESIGN_E33_ATTEMPT_CHAIN_2026-11-02.md).

python e33_poetiq_build.py          -> writes Build/e33_poetiq.py (LF, no CR) and prints its sha256 + the source shas
python e33_poetiq_build.py --check  -> rebuilds in memory and compares with Build/e33_poetiq.py (exit 1 on a difference)

The module = (pattern of Experiment 29 Build/e29_direct_build.py make_direct(): pinned sources by sha256, exact-once edits)
  1. E18 R2 CELL 1 (notebook cell index 2): the POETIQ_* settings block, VERBATIM (two slices; the E10_KV gate lines between
     them are left out: E33's KV admission is the E28 reservation hook of the scheduler);
  2. E29 DIRECT (= E31) transport constants, VERBATIM (SEED .. FN_EOS_IDS, drain_need_min, phase1_cap);
  3. E18 R2 CELL 4 (notebook cell index 5) = the Poetiq core, VERBATIM except ONE marked "# E33" edit: the import-time
     sandbox self-test block (cell lines 673-679) becomes an on-demand cached call (sandbox_ready());
  4. E33: the sandbox concurrency bound (E18's runner unchanged behind a process-wide semaphore);
  5. E29 DIRECT transport classes, VERBATIM (_Completion, _Output, request_body, StreamEngine, _health);
  6. E33 driver (DRIVER below): E18 cell 7's per-unit logic (render, seeds, admission, phase 2, answer text, verify,
     commit, rank) for ONE (task, attempt, iteration) on the E29 transport; every deviation from E18 is marked "# E33".
Sources (read-only): the E18 R2 notebook sha 116d2d41 (the scored version 2), e29_direct.py sha 8b42802c.
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "e33_poetiq.py"
E18_NB = Path(r"D:/ARC AGI 2/experiments/Experiment 18 - S1-60K on vLLM 0.32 (planned)/Experiment Notebooks"
              r"/Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb")
E18_NB_SHA = "116d2d4101a53eb4ff833e9c97e5a779f0f2bae5735129a4663ad1b42ad79a1d"
E18_CELL1_SHA = "02bdec93be72f7ceb77be277a424f8ade7ae1f29ff6778b8688c80ad8d1576eb"     # notebook cell index 2 (CELL 1)
E18_CORE_SHA = "21d7295094180efbf1434ba640c365b73b3a1e7b96d1944816fca43d6c369778"      # notebook cell index 5 (Poetiq core)
E18_SOLVER_SHA = "59f619b65d9b2794e76aac698c78a4d3ce716969a35e79dcbc610a931e5fe234"    # notebook cell index 7 (solver; read only)
E29_DIRECT = Path(r"D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Build/e29_direct.py")
E29_DIRECT_SHA = "8b42802ccf076f5541ab62f6ce3a1db58b2a2a750e10cfbba52af687e5feb93c"


def sha(t):
    return hashlib.sha256(t if isinstance(t, bytes) else t.encode("utf-8")).hexdigest()


def once(text, old, new, label):
    assert text.count(old) == 1, (label, text.count(old))
    return text.replace(old, new)


def segment(text, start, end, label, include_end=True):
    """The exact slice of text from the line that starts with `start` (once in text) to the end of the line holding the first
    `end` after it."""
    assert text.count(start) == 1, (label, "start", text.count(start))
    i = text.index(start)
    assert i == 0 or text[i - 1] == "\n", (label, "start not at a line start")
    j = text.find(end, i)                       # the FIRST occurrence of `end` after the start line
    assert j > i, (label, "end not found")
    j = text.index("\n", j + len(end) - 1) + 1 if include_end else text.rfind("\n", 0, j) + 1   # end may span lines
    return text[i:j]


def load_sources(nb_path=E18_NB, direct_path=E29_DIRECT):
    raw = Path(nb_path).read_bytes()
    assert sha(raw) == E18_NB_SHA and b"\r" not in raw, ("the E18 R2 notebook is not the pinned 116d2d41", sha(raw))
    nb = json.loads(raw)
    cell = lambda i: "".join(nb["cells"][i]["source"])
    c1, core, solver = cell(2), cell(5), cell(7)
    assert sha(c1) == E18_CELL1_SHA and sha(core) == E18_CORE_SHA and sha(solver) == E18_SOLVER_SHA
    d = Path(direct_path).read_bytes()
    assert sha(d) == E29_DIRECT_SHA and b"\r" not in d, ("e29_direct.py is not the pinned 8b42802c", sha(d))
    return {"cell1": c1, "core": core, "solver": solver, "direct": d.decode("utf-8")}


# ------------------------------------------------------------------ the one edit of the E18 core (cell index 5)
CORE_SELFTEST_OLD = """# Fail closed without losing the cell-0 competition-format fallback.
try:
    E10_SANDBOX_SELFTEST=poetiq_sandbox_selftest()
except BaseException as _e10_sandbox_error:
    E10_SANDBOX_SELFTEST={'ok':False,'mode':'unavailable',
                         'error':type(_e10_sandbox_error).__name__+': '+str(_e10_sandbox_error)[:1200]}
print('[E10] Sandbox self-test:',E10_SANDBOX_SELFTEST,flush=True)
"""
CORE_SELFTEST_NEW = """# E33: E18 ran the self-test here at import (cell lines 673-679: try poetiq_sandbox_selftest(), except BaseException ->
# {'ok': False, 'mode': 'unavailable', 'error': ...}, print). A module must not spawn processes at import: sandbox_ready()
# (below) runs exactly that block once on demand, caches it, and Poetiq sends no request unless it passed (E18: no engine).
E10_SANDBOX_SELFTEST = None
"""


def core_text(src):
    return once(src["core"], CORE_SELFTEST_OLD, CORE_SELFTEST_NEW, "C1 selftest")


def cell1_poetiq(src):
    a = segment(src["cell1"], "# ---- E10 Poetiq donor configurations", "POETIQ_SANDBOX_WORKERS = int(", "cell1 POETIQ_*")
    b = segment(src["cell1"], "assert POETIQ_NUM_EXPERTS >= 1 and POETIQ_MAX_ITERATIONS >= 1",
                "assert POETIQ_MAX_SOLUTIONS >= 1 and 0 <= POETIQ_SELECTION_PROB <= 1", "cell1 asserts")
    return a, b


def direct_parts(src):
    d = src["direct"]
    consts = segment(d, "SEED = 20261005", "FN_EOS_IDS = (248046, 248044)", "direct consts")
    drain = segment(d, "def drain_need_min(", "return int(max_tests) * int(answer) / float(tok_s) / 60.0", "drain_need_min")
    cap = segment(d, "def phase1_cap(", "return max(0, min(int(think), int(max_model_len) - int(prompt_len) - int(answer) - CONTEXT_SAFETY))",
                  "phase1_cap")
    classes = segment(d, "class _Completion:", "        with self.lock:\n            return len(self.streams)", "StreamEngine")
    health = segment(d, "def _health(", "        return False", "_health")
    return consts, drain, cap, classes, health


HEADER = '''"""E33 Poetiq solver (Experiment 33, user 2026-11-02; GENERATED by Build/e33_poetiq_build.py, do not edit).

Stages C (iteration 1) and D (iteration 2) of the E33 attempt chain (DESIGN_E33_ATTEMPT_CHAIN_2026-11-02.md section 2):
Poetiq S1-60K = the E18 R2 notebook (sha __NB__, the scored version 2, LB-proven) on the E29 Flash-Next engine.
  - VERBATIM from E18 R2: the POETIQ_* settings (CELL 1) and the whole Poetiq core (cell index 5: prompts, problem format,
    diagrams, code parse, sandbox = subprocess + UID drop + rlimits + seccomp + timeout, verify + per-example feedback, solution
    history with scores, voting, two attempts, the sandbox self-test); one marked "# E33" edit (the import-time self-test runs
    on demand: sandbox_ready()).
  - VERBATIM from E29 DIRECT (= the E31 Flash-Next Direct solver, GPU-proven on this engine; e29_direct.py sha __D__): the
    constants (SEED, MAX_MODEL_LEN 131072, THINK_FLOOR 60000, STOP_THINKING_TEXT, FN_STOP_IDS, FN_THINK_CLOSE_ID, ...),
    phase1_cap, request_body, StreamEngine (streaming token-id /v1/completions, abort = close the stream), _health.
  - E33 driver: E18 cell 7's per-unit logic (e10_render_request, e10_seed, e10_admit, e10_start_phase2, e10_answer_text,
    e10_after_decode, e10_commit_iteration, e10_rank_task) for ONE (task, attempt, iteration), on the E29 client with the E29
    rules (fail-closed stop ids, EOS from generation_config, resends keep the streamed ids, health / SERVER_DOWN waits, the
    drain -> forced close, the hard stop, thinking traces). Every deviation from E18 carries a "# E33" comment.
API: poetiq_run(tid, task, *, iteration, prior, deadline, base_url, model_id, out_dir, tok, cfg, task_index, health,
     server_down, harness, log, attempt=None) -> record (the E29 DIRECT record shape + "poetiq" + "poetiq_state").
     can_admit(deadline, cfg, now, harness) -> (bool, info): the same admission rule, for a scheduler hook.
     sandbox_ready(force=False) -> the cached E18 self-test; set_sandbox_workers(n).
"""
import json
import os
import queue as _queue
import re
import socket
import threading
import time
import urllib.request as _urlreq
from collections import deque
from pathlib import Path

'''

E18_BUDGETS = '''
# ==================================================================== E18 R2 CELL 1: the generation budget (S1-60K)
# E33: literals (E18 read Q27X_* env overrides; E33's switches are CELL 1 literals passed in cfg): E18 cell 1 lines 40-43.
REASONING_EFFORT = "xhigh"            # E18 line 40; E33: carried by tok["render"] (E29 e29d_tok: enable_thinking, xhigh)
THINK_BUDGET = 60000                  # E18 line 41 (S1-60K)
ANSWER_MAX_TOKENS = 8192              # E18 line 42
DRAIN_ANSWER_MAX_TOKENS = 8192        # E18 line 43
# E18 line 56 SAMPLING top_p 0.95 / top_k 20 = E29's TOP_P / TOP_K below (asserted at the end of this module); the
# temperature is POETIQ_SOLVER_TEMPERATURE (1.0), as E18 e10_make_params sends it.

# ==================================================================== E18 R2 CELL 1 (cell index 2, lines 109-125 + 129-131), VERBATIM
'''

DIRECT_HDR = '''
# ==================================================================== E29 DIRECT (e29_direct.py) constants, VERBATIM
# E33: SEED = E18 cell 1 line 31 (same value); MAX_MODEL_LEN = the E29 Merlin server's --max-model-len (E18: 106,496 on its
# own server); STOP_THINKING_TEXT = E18 cell 1 line 44 (byte-identical, asserted by the builder).
'''

CORE_HDR = '''
# ==================================================================== E18 R2 Poetiq core (notebook cell index 5), VERBATIM + one E33 edit
'''

SANDBOX_BOUND = '''
# ==================================================================== E33: the sandbox CPU bound
# Kaggle has 4 vCPUs shared with the server process. E18 bounded its verifications with a 4-thread pool
# (POETIQ_SANDBOX_WORKERS); E33 calls Poetiq from up to 24 scheduler worker threads at once, so E18's runner (unchanged,
# timeouts and rlimits included) sits behind ONE process-wide semaphore: at most _SANDBOX["workers"] (default 2) sandbox
# children run at any time. poetiq_verify_code / poetiq_sandbox_selftest look the name up at call time and get this wrapper.
_E18_POETIQ_SANDBOX_RUN = poetiq_sandbox_run      # E33: E18's runner, byte-identical above
_SANDBOX = {"workers": 2, "sem": threading.BoundedSemaphore(2), "lock": threading.Lock(), "st_lock": threading.Lock(),
            "selftest": None, "live": 0, "max_live": 0, "runs": 0}


def set_sandbox_workers(n):
    """E33: the number of concurrent sandbox children (>= 1); returns the value in force."""
    n = max(1, int(n))
    with _SANDBOX["lock"]:
        if n != _SANDBOX["workers"]:
            _SANDBOX["workers"], _SANDBOX["sem"] = n, threading.BoundedSemaphore(n)
        return _SANDBOX["workers"]


def poetiq_sandbox_run(code, grid, timeout_s=5.0):    # E33: E18's runner behind the process-wide bound
    sem = _SANDBOX["sem"]
    with sem:
        with _SANDBOX["lock"]:
            _SANDBOX["live"] += 1
            _SANDBOX["runs"] += 1
            _SANDBOX["max_live"] = max(_SANDBOX["max_live"], _SANDBOX["live"])
        try:
            return _E18_POETIQ_SANDBOX_RUN(code, grid, timeout_s)
        finally:
            with _SANDBOX["lock"]:
                _SANDBOX["live"] -= 1


def sandbox_ready(force=False):
    """E33: E18 cell 5 lines 673-679 (the self-test block) run once on demand and cached -> {'ok': bool, ...}."""
    global E10_SANDBOX_SELFTEST
    with _SANDBOX["st_lock"]:
        if _SANDBOX["selftest"] is not None and not force:
            return _SANDBOX["selftest"]
        try:
            res = poetiq_sandbox_selftest()
        except BaseException as _e10_sandbox_error:
            res = {'ok': False, 'mode': 'unavailable',
                   'error': type(_e10_sandbox_error).__name__ + ': ' + str(_e10_sandbox_error)[:1200]}
        _SANDBOX["selftest"] = E10_SANDBOX_SELFTEST = res
        return res
'''

CLASSES_HDR = '''
# ==================================================================== E29 DIRECT transport (e29_direct.py), VERBATIM
'''

DRIVER = r'''
# ==================================================================== E33 driver: ONE Poetiq unit (task x attempt x iteration)
# E18 cell 7 ran every (task, expert) through iterations 1..10 in one scheduler loop on an in-process engine. E33 runs ONE
# iteration per call (stage C = iteration 1, stage D = iteration 2; the E33 scheduler owns the order and the slots), on the
# E29 DIRECT client. Attempt 2 of a task = Poetiq expert 1 (E18's own seed / message formula with expert_idx = attempt - 1),
# so attempt 2 never repeats attempt 1's seeds.
LIB_VERSION = "e33_poetiq 1.0 (2026-11-02) = E18 R2 Poetiq core (116d2d41 cell 5) + E29 DIRECT transport (8b42802c)"
POETIQ_STATE_VERSION = 1
CFG_DEFAULT = {"think": THINK_BUDGET, "answer": ANSWER_MAX_TOKENS, "drain_answer": DRAIN_ANSWER_MAX_TOKENS,
               "temperature": POETIQ_SOLVER_TEMPERATURE,
               "drain_min": 9.0,            # E33: covers one 8,192-token answer at 18 tok/s per stream (7.6 min) + margin
               "tok_s": PER_STREAM_TOK_S,   # E33 admission: configured per-stream decode rate (E28 K100 measured ~18)
               "tok_s_measured": None,      # E33 admission: a measured per-stream rate the scheduler may pass (lower wins)
               "admit_margin_sec": 60.0,    # E33 admission: prefill + first token allowance (E29D_CAN_START's + 60 s)
               "sandbox_workers": 2, "sandbox_timeout": POETIQ_SANDBOX_TIMEOUT, "max_error_retries": 2,
               "max_model_len": MAX_MODEL_LEN, "attempt": 1, "branch": None}
RATE_MIN_TOKENS = 1000                     # E33: a phase-1 request with fewer tokens is not a rate sample
_RATES = deque(maxlen=16)                  # E33: recent measured per-stream phase-1 decode rates (tok/s), all calls
_RATES_LOCK = threading.Lock()


def _norm_base(url):
    """E33: the StreamEngine appends /v1/completions; E29's hook passes SGL_URL + '/v1' -> accept both forms."""
    u = str(url or "").rstrip("/")
    return u[:-3] if u.endswith("/v1") else u


def _note_rate(rs):
    try:
        n, a, b = int(rs.get("n") or 0), rs.get("t_first"), rs.get("t_end")
        if n >= RATE_MIN_TOKENS and a and b and b > a:
            with _RATES_LOCK:
                _RATES.append(n / (b - a))
    except Exception:
        pass


def measured_tok_s():
    """E33: the median of the recent measured per-stream phase-1 rates (>= 3 samples), else None."""
    with _RATES_LOCK:
        v = sorted(_RATES)
    return v[len(v) // 2] if len(v) >= 3 else None


def admission_rate(cfg=None):
    """E33: the per-stream tok/s used by the admission rule = the LOWEST of cfg tok_s, cfg tok_s_measured, measured_tok_s()."""
    cfg = dict(CFG_DEFAULT, **(cfg or {}))
    c = [float(x) for x in (cfg.get("tok_s"), cfg.get("tok_s_measured"), measured_tok_s()) if isinstance(x, (int, float)) and x > 0]
    return min(c) if c else PER_STREAM_TOK_S


def _floor(cfg, harness):
    """The thinking floor: THINK_FLOOR (60,000, user rule) always; only the unit-test harness may set a small one."""
    return int(cfg.get("harness_floor") or cfg["think"]) if harness else THINK_FLOOR


def can_admit(deadline, cfg=None, now=None, harness=False):
    """E33 admission rule (user rule: never admit with less than the 60,000 thinking floor):
    admit iff now < ADMIT and now + admit_margin_sec + floor / rate <= DRAIN, with T = deadline (time.monotonic() of the solve
    end), DRAIN = T - drain_min * 60 (this solver's forced close), ADMIT = min(T - ADMIT_STOP_BEFORE_END_SEC, DRAIN) (E29) and
    rate = admission_rate(cfg). -> (ok, info). The context check (phase1_cap >= floor) is in poetiq_run (it needs the prompt)."""
    cfg = dict(CFG_DEFAULT, **(cfg or {}))
    floor = _floor(cfg, harness)
    t_end = float(deadline)
    drain = t_end - float(cfg["drain_min"]) * 60.0
    admit = min(t_end - ADMIT_STOP_BEFORE_END_SEC, drain)
    rate = admission_rate(cfg)
    need = float(cfg["admit_margin_sec"]) + floor / rate
    now = time.monotonic() if now is None else float(now)
    ok = now < admit and now + need <= drain
    return ok, {"floor": floor, "rate_tok_s": round(rate, 3), "need_sec": round(need, 1), "sec_to_drain": round(drain - now, 1),
                "sec_to_admit_stop": round(admit - now, 1), "measured_tok_s": measured_tok_s(), "ok": ok}


def _json_copy(x):
    return json.loads(json.dumps(x, default=str))


def _state_from_prior(prior, tid, expert, tix):
    """E33: E18's per-(task, expert) state E10_STATES[key] (history = [{code, feedback, score}], candidate_history = the full
    verification results, best, best_score, iteration = iterations done) from the iteration-1 record or its poetiq_state."""
    st = {"version": POETIQ_STATE_VERSION, "task_id": tid, "expert": expert, "task_index": tix, "iteration": 0, "history": [],
          "candidate_history": [], "best_index": None, "best_score": -1.0, "status": "queued", "last_error": None,
          "iterations": [], "prior_mismatch": None}
    src = None
    if isinstance(prior, dict):
        src = prior["poetiq_state"] if isinstance(prior.get("poetiq_state"), dict) else (prior if "history" in prior else None)
    if src:
        src = _json_copy(src)
        for k in ("iteration", "history", "candidate_history", "best_index", "best_score", "status", "last_error", "iterations"):
            if k in src:
                st[k] = src[k]
        mm = {k: [src.get(k), v] for k, v in (("task_id", tid), ("expert", expert), ("task_index", tix)) if src.get(k) != v}
        st["prior_mismatch"] = mm or None
    return st


def poetiq_rank_state(st):
    """E18 cell 7 e10_rank_task for ONE expert: Poetiq's vote over the expert best, then the vote over the other executed
    programs as extra distinct candidates. # E33: the best is deduplicated by its index (the state is JSON, not id())."""
    cands = st.get("candidate_history") or []
    bi = st.get("best_index")
    bests = [cands[bi]] if isinstance(bi, int) and 0 <= bi < len(cands) else []
    voted = poetiq_vote(bests, use_new_voting=POETIQ_USE_NEW_VOTING,
                        count_failed_matches=POETIQ_COUNT_FAILED_MATCHES,
                        iters_tiebreak=POETIQ_ITERS_TIEBREAK,
                        low_to_high_iters=POETIQ_LOW_TO_HIGH_ITERS)
    history = [r for i, r in enumerate(cands) if not (bests and i == bi)]
    if history:
        voted.extend(poetiq_vote(history, use_new_voting=POETIQ_USE_NEW_VOTING,
                                 count_failed_matches=POETIQ_COUNT_FAILED_MATCHES,
                                 iters_tiebreak=POETIQ_ITERS_TIEBREAK,
                                 low_to_high_iters=POETIQ_LOW_TO_HIGH_ITERS))
    return voted


def poetiq_seed(task_index, expert, it, phase):
    """E18 cell 7 e10_seed: SEED + task_index * 1000 + expert * 100 + iteration * 2 + phase (iteration 0-based)."""
    return SEED + int(task_index) * 1000 + int(expert) * 100 + int(it) * 2 + int(phase)


def poetiq_message_seed(task_index, expert):
    """E18 cell 7 e10_render_request: seed = POETIQ_SEED + task_index * 1000 + expert * 100."""
    return POETIQ_SEED + int(task_index) * 1000 + int(expert) * 100


def poetiq_message(task, it, expert, history, task_index, max_feedback=None):
    """E18 cell 7 e10_render_request without the chat template (the user message)."""
    idx = POETIQ_SOLVER_VARIANTS[expert % len(POETIQ_SOLVER_VARIANTS)]
    limit = POETIQ_MAX_SOLUTIONS if max_feedback is None else max_feedback
    return poetiq_build_message(task, it, expert, history,
                                variant=idx, selection_probability=POETIQ_SELECTION_PROB,
                                max_solutions=limit, improving_order=POETIQ_IMPROVING_ORDER,
                                shuffle=POETIQ_SHUFFLE_EXAMPLES, seed=poetiq_message_seed(task_index, expert))


def poetiq_run(tid, task, *, iteration, prior=None, deadline, base_url, model_id, out_dir, tok, cfg=None, task_index=None,
               health=None, server_down=None, harness=False, log=print, attempt=None):
    """ONE Poetiq iteration of one task (stage C: iteration 1; stage D: iteration 2 with prior = the iteration-1 record or
    its poetiq_state) on the booted server -> record (never None; status "done" | "not_admitted" | "not_run").
    deadline = time.monotonic() of the solve end T: no admission unless 60,000 thinking tokens fit before the drain point
    (can_admit), drain at T - drain_min (phase 1 aborted, ids kept -> forced-close phase 2), every request aborted at T - 60 s
    (partial answer parsed). tok = E29's {"render", "encode", "decode", "eos_ids"}. task_index = int or {task: index} (the
    sorted full task set, E18's seed index). attempt (or cfg["attempt"]) = 1 | 2 -> Poetiq expert 0 | 1."""
    t_call_pc, t_call_ep = time.perf_counter(), time.time()
    cfg = dict(CFG_DEFAULT, **(cfg or {}))
    attempt = int(attempt if attempt is not None else (cfg.get("attempt") or 1))
    iteration = int(iteration)
    assert iteration >= 1 and attempt >= 1, (iteration, attempt)
    expert, it = attempt - 1, iteration - 1                    # E18: expert index, 0-based iteration (st['iteration'])
    tix = int((task_index or {}).get(tid, 0)) if isinstance(task_index, dict) else int(task_index or 0)
    think, answer, temp = int(cfg["think"]), int(cfg["answer"]), float(cfg["temperature"])
    drain_answer, mml, max_retries = int(cfg["drain_answer"]), int(cfg["max_model_len"]), int(cfg["max_error_retries"])
    sandbox_timeout = float(cfg["sandbox_timeout"])
    set_sandbox_workers(cfg["sandbox_workers"])
    floor = _floor(cfg, harness)
    base = _norm_base(base_url)
    out_dir = Path(out_dir)
    bdir = out_dir / f"{tid}_a{attempt}_i{iteration}"
    bdir.mkdir(parents=True, exist_ok=True)
    T_END = float(deadline)
    HARD = T_END - HARD_BEFORE_END_SEC
    DRAIN = T_END - float(cfg["drain_min"]) * 60.0
    n_test, n_train = len(task["test"]), len(task["train"])
    st = _state_from_prior(prior if iteration > 1 else None, tid, expert, tix)
    seed1, seed2 = poetiq_seed(tix, expert, it, 1), poetiq_seed(tix, expert, it, 2)
    P = {"lib": LIB_VERSION, "iteration": iteration, "attempt": attempt, "expert": expert, "task_index": tix, "admitted": False,
         "reason": None, "score": None, "train_score": 0.0, "train_pass": False, "test_usable": False, "program": None,
         "parse_status": None, "feedback": None, "mode": None, "end_reason": None, "think_tokens": 0, "answer_tokens": 0,
         "think_closed": False, "prompt_tokens": None, "p1_tokens": 0, "p2_tokens": 0, "p1_finish": None, "p2_finish": None,
         "p1_cap": None, "p2_cap": None, "p2_kept": 0, "drained": False, "feedback_solutions": 0, "feedback_trimmed": False,
         "history_in": len(st["history"]), "seed_message": poetiq_message_seed(tix, expert), "seed_p1": seed1, "seed_p2": None,
         "request_errors": [], "error_retries": 0, "server_down_polls": 0, "health_fail_polls": 0, "admission": None,
         "sec": None, "verify_sec": None, "verify_error": None, "test_errors": None, "e18_two_attempts": None,
         "prior_mismatch": st.get("prior_mismatch"), "floor": floor, "cfg": {k: cfg[k] for k in sorted(cfg)},
         "sandbox": None, "error": None}
    rec_path = out_dir / "e33_poetiq_records.jsonl"

    def _publish(rec):
        try:
            (bdir / "branch_result.json").write_text(json.dumps(rec, default=str), encoding="utf-8")
        except Exception as e:
            P["error"] = (P["error"] or "") + f" branch_result: {type(e).__name__}: {e}"[:200]
        try:
            with open(rec_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"task_id": tid, "status": rec["status"], "accepted": rec["accepted"],
                                    "generated_tokens": rec["generated_tokens"],
                                    **{k: v for k, v in P.items() if k not in ("program", "feedback", "cfg", "e18_two_attempts")}},
                                   default=str) + "\n")
        except Exception:
            pass
        try:
            log(f"[E33 POETIQ] {tid} a{attempt} i{iteration} {rec['status']} {P['reason'] or P['end_reason']} mode={P['mode']} "
                f"think {P['think_tokens']} parse={P['parse_status']} score={P['score']} train_pass={P['train_pass']} "
                f"tests {sum(g is not None for g in (rec['submissions'][0]['test_outputs'] if rec['submissions'] else []))}/{n_test}",
                flush=True)
        except Exception:
            pass
        return rec

    def _not_run(status, reason, **extra):
        P.update(reason=reason, **extra)
        P["sec"] = round(time.perf_counter() - t_call_pc, 2)
        rec = {"task_id": tid, "branch": cfg.get("branch"), "status": status, "accepted": False, "generated_tokens": 0,
               "total_tokens": 0, "t0": t_call_ep, "t1": time.time(), "note": f"poetiq {status}: {reason}", "turns": 0,
               "failures": 0, "b_budget": 0, "view": "e33_poetiq", "usage": [], "submissions": [], "e33_poetiq": True,
               "attempt": attempt, "iteration": iteration, "poetiq": dict(P), "poetiq_state": st}
        return _publish(rec)

    # ---- pre-checks (no request is sent unless every one passes)
    if not harness and think < THINK_FLOOR:                      # user rule (E29 asserts it; E33: this item off, recorded)
        return _not_run("not_run", "cfg_think_below_floor", error=f"think {think} < {THINK_FLOOR}")
    encode, decode, render = tok["encode"], tok["decode"], tok["render"]
    THINK_CLOSE_IDS = encode("</think>")
    STOP_IDS = list(encode(STOP_THINKING_TEXT))                  # tokenized once; continuation = ids only (E18 / E29)
    EOS_IDS = set(int(x) for x in (tok.get("eos_ids") or []))   # E29: from the checkpoint's generation_config.json
    if (len(THINK_CLOSE_IDS) != 1 or THINK_CLOSE_IDS[0] != FN_THINK_CLOSE_ID or STOP_IDS != FN_STOP_IDS or not EOS_IDS):
        return _not_run("not_run", "FAIL_CLOSED_TOKEN_IDS", error=json.dumps(               # E29 fail-closed (before any request)
            {"stop_ids": STOP_IDS, "think_close_ids": list(THINK_CLOSE_IDS), "eos_ids": sorted(EOS_IDS)})[:600])
    THINK_CLOSE_ID = THINK_CLOSE_IDS[0]
    sb = sandbox_ready()
    P["sandbox"] = {k: sb.get(k) for k in ("ok", "mode", "error")}
    if not sb.get("ok"):                                         # E18: no engine without a passing self-test
        return _not_run("not_run", "sandbox_unavailable")

    # ---- E18 e10_admit: render, tokenize, phase-1 cap; trim low-score feedback examples only if the floor needs it
    def render_request(max_feedback=None):
        msg = poetiq_message(task, it, expert, st["history"], tix, max_feedback)
        return msg, render([{"role": "user", "content": msg}])

    msg, rendered = render_request()
    ids = list(encode(rendered))
    cap = phase1_cap(len(ids), think, answer, mml)
    if cap < floor and st["history"]:                            # E33: floor 60,000 (E18: 49,152)
        for keep in range(min(POETIQ_MAX_SOLUTIONS, len(st["history"])) - 1, -1, -1):
            msg, rendered = render_request(max_feedback=keep)
            ids = list(encode(rendered))
            cap = phase1_cap(len(ids), think, answer, mml)
            if cap >= floor:
                P["feedback_trimmed"] = True
                P["feedback_solutions_kept"] = keep
                break
    P.update(prompt_tokens=len(ids), p1_cap=cap,
             feedback_solutions=min(len(st["history"]), POETIQ_MAX_SOLUTIONS) if not P["feedback_trimmed"] else P["feedback_solutions_kept"])
    if cap < floor:
        return _not_run("not_admitted", "insufficient_context", error=f"prompt {len(ids)} tokens leaves {cap} thinking room")

    # ---- E33 admission: the floor must fit before the drain point; wait while the server is down / unhealthy
    health_fn = health or (lambda: _health(base))
    gstate = {"h_t": -1e9, "h_ok": True}

    def server_ok(now_mono):
        """E29 server_ok: False while the watchdog relaunches the server (SERVER_DOWN) or /health fails (polled <= 5 s)."""
        if server_down is not None and server_down.is_set():
            P["server_down_polls"] += 1
            return False
        if now_mono - gstate["h_t"] >= HEALTH_EVERY_SEC:
            try:
                ok_ = bool(health_fn())
            except Exception:
                ok_ = False
            gstate["h_t"], gstate["h_ok"] = now_mono, ok_
            if not ok_:
                P["health_fail_polls"] += 1
        return gstate["h_ok"]

    waited = False
    while True:
        ok_t, adm = can_admit(T_END, cfg, harness=harness)
        P["admission"] = adm
        if not ok_t:
            return _not_run("not_admitted", "server_down_until_late" if waited else "time")
        if server_ok(time.monotonic()):
            break
        waited = True
        time.sleep(0.2)

    # ---- the unit (E18 req dict + E29 st fields)
    eng = StreamEngine(base, model_id, req_log=out_dir / "e33_poetiq_requests.jsonl")
    u = {"phase": 1, "prompt_ids": ids, "p1_ids": [], "p2_ids": [], "p2_prefix": [], "p1_finish": None, "p2_finish": None,
         "mode": None, "p2_cap": None, "pending": None, "errors": [], "retries": 0, "p1_rid": None, "p2_rid": None,
         "t_admit": time.perf_counter(), "t_admit_ep": time.time(), "p1_rids": [], "p2_rids": []}

    def rid(phase, n):
        return f"{tid}~a{attempt}~i{iteration}|{phase}|{n}"

    def start_phase2(resend=False, answer_cap=answer):
        """E18 e10_start_phase2 (closed -> 'continue', else trailing EOS stripped + STOP_IDS -> 'forced_close'; cap = min(answer,
        room), < 64 -> no room) + E29 N1 (a phase-2 resend keeps the streamed phase-2 ids, cap = the remaining answer)."""
        if not resend:
            closed = THINK_CLOSE_ID in u["p1_ids"]
            if not closed:
                while u["p1_ids"] and u["p1_ids"][-1] in EOS_IDS:
                    u["p1_ids"].pop()
            u["p2_prefix"] = [] if closed else list(STOP_IDS)
            u["mode"] = "continue" if closed else "forced_close"
            u["answer_cap"] = int(answer_cap)
        kept = list(u["p2_ids"]) if resend else []
        while kept and kept[-1] in EOS_IDS:
            kept.pop()
        prefix = list(u["prompt_ids"]) + list(u["p1_ids"]) + list(u["p2_prefix"]) + kept
        cap_ = min(int(u["answer_cap"]) - len(kept), mml - len(prefix) - 8)
        if cap_ < 64:
            return False
        u["phase"], u["p2_cap"], u["p2_ids"], u["p2_prefix_len"] = 2, cap_, kept, len(prefix)
        P["p2_kept"] += len(kept)
        u["p2_rid"] = rid(2, u["retries"])
        u["p2_rids"].append(u["p2_rid"])
        P["seed_p2"] = seed2
        eng.add_request(u["p2_rid"], prefix, cap_, temp, seed2)
        return True

    def resend_phase1():
        """E29 resend_phase1: phase 1 again, token-exact (prompt + the kept phase-1 ids, trailing EOS stripped), cap = the phase-1
        cap - the kept ids, the same seed; < 64 left = a length end -> the normal phase 2."""
        while u["p1_ids"] and u["p1_ids"][-1] in EOS_IDS:
            u["p1_ids"].pop()
        cap_ = int(cap) - len(u["p1_ids"])
        if cap_ < 64:
            u["p1_finish"] = "length"
            return start_phase2()
        u["p1_rid"] = rid(1, u["retries"])
        u["p1_rids"].append(u["p1_rid"])
        P["phase1_resent"] = P.get("phase1_resent", 0) + 1
        eng.add_request(u["p1_rid"], list(u["prompt_ids"]) + list(u["p1_ids"]), cap_, temp, seed1)
        return True

    end, drained, err_tb = None, False, None
    try:
        u["p1_rid"] = rid(1, 0)
        u["p1_rids"].append(u["p1_rid"])
        eng.add_request(u["p1_rid"], ids, cap, temp, seed1)
        P["admitted"] = True
        while end is None:
            now = time.monotonic()
            if now >= HARD:                                      # hard stop: abort, parse what exists
                if not u["pending"]:
                    eng.abort_request([u["p1_rid"] if u["phase"] == 1 else u["p2_rid"]])
                end = "hard_stop"
                break
            if now >= DRAIN and not drained:                     # drain: phase 1 -> forced-close phase 2 (ids kept)
                drained = True
                if u["phase"] == 1:
                    if u["pending"]:
                        P["drain_pending"] = True                # a unit waiting for the server keeps waiting
                    else:
                        eng.abort_request([u["p1_rid"]])
                        u["p1_finish"] = "drained"
                        P["drained"] = True
                        if not start_phase2(answer_cap=drain_answer):
                            end = "drain_no_room"
                            break
            if u["pending"]:                                     # E29: wait for the server, then continue (never a finish)
                if server_ok(now):
                    kind, u["pending"] = u["pending"], None
                    u["retries"] += 1
                    if kind == "p1" and now < DRAIN and not drained:
                        ok_ = resend_phase1()
                    else:
                        ok_ = start_phase2(resend=(kind == "p2"), answer_cap=drain_answer if drained else answer)
                    if not ok_:
                        end = "no_room_for_code"
                        break
                else:
                    time.sleep(0.05)
                continue
            for o in eng.step():
                cur = u["p1_rid"] if u["phase"] == 1 else u["p2_rid"]
                if o.request_id != cur or u["pending"] or end is not None:
                    continue                                     # stale output after an abort / resend
                ph = u["phase"]
                comp = o.outputs[0] if o.outputs else None
                if comp is not None:
                    (u["p1_ids"] if ph == 1 else u["p2_ids"]).extend(list(comp.token_ids))
                if not o.finished:
                    continue
                fr = comp.finish_reason if comp is not None else None
                if fr == "error":                                # E29: never a finish (bounded resends)
                    u["errors"].append({"phase": ph, "rid": o.request_id, "t": round(time.time(), 1)})
                    u["p1_finish" if ph == 1 else "p2_finish"] = "error"
                    if u["retries"] >= max_retries:
                        end = "error"
                    else:
                        u["pending"] = "p1" if ph == 1 else "p2"
                        gstate["h_t"] = -1e9
                    continue
                if ph == 1:
                    u["p1_finish"] = fr
                    if THINK_CLOSE_ID in u["p1_ids"] and fr != "length":
                        end = "natural"
                    elif not start_phase2(answer_cap=drain_answer if drained else answer):
                        end = "no_room_for_code"
                else:
                    u["p2_finish"] = fr
                    end = u["mode"]
    except BaseException as e:                                   # E18 C1 / E29 F2: the unit ends, what exists is parsed
        import traceback
        err_tb = traceback.format_exc()[-3000:]
        end = "exception"
        if not isinstance(e, Exception):
            try:
                eng.abort_request([x for x in (u["p1_rid"], u["p2_rid"]) if x])
            except BaseException:
                pass
            raise
    finally:
        try:
            live = [x for x in (u["p1_rid"], u["p2_rid"]) if x and x in eng.streams]
            if live:
                eng.abort_request(live)
        except Exception:
            pass
    t_done = time.perf_counter()
    for r_ in u["p1_rids"]:
        _note_rate(eng.req_stats.get(r_) or {})

    # ---- E18 e10_answer_text (the text after the FIRST </think>, skip_special_tokens) + e10_after_decode
    all_ids = list(u["p1_ids"]) + list(u["p2_prefix"]) + list(u["p2_ids"])
    if THINK_CLOSE_ID in all_ids:
        k = all_ids.index(THINK_CLOSE_ID)
        text, closed = decode(all_ids[k + 1:], True), True
    else:
        k, text, closed = len(u["p1_ids"]), "", False
    code = poetiq_parse_code(text)
    P.update(mode=u["mode"] or "natural", end_reason=end, think_tokens=int(k), think_closed=closed,
             answer_tokens=int(len(all_ids) - k - 1) if closed else 0, p1_tokens=len(u["p1_ids"]), p2_tokens=len(u["p2_ids"]),
             p1_finish=u["p1_finish"], p2_finish=u["p2_finish"], p2_cap=u["p2_cap"], request_errors=list(u["errors"]),
             error_retries=u["retries"], parse_status="ok" if code else ("no_think_close" if not closed else "no_code_block"),
             program=code, error=err_tb, sec=round(t_done - u["t_admit"], 2), answer_chars=len(text))
    try:      # user rule: the FULL thinking of every request is saved (E18 poetiq_thinking + E31 transcript / token ids)
        head = (f"# task={tid} expert={expert} iteration={iteration} mode={P['mode']} finish={end} prompt_tokens={len(ids)} "
                f"p1_tokens={len(u['p1_ids'])} p2_tokens={len(u['p2_ids'])} attempt={attempt} p1_finish={u['p1_finish']} "
                f"p2_finish={u['p2_finish']} errors={u['errors']}\n")
        (bdir / "transcript.txt").write_text("\n".join([
            head, "### PROMPT", rendered, "### GENERATED (phase 1 + forced-close ids + phase 2, special tokens kept)",
            decode(all_ids, False), "### ANSWER (after the first </think>)", text, ""]), encoding="utf-8")
        (bdir / "token_ids.json").write_text(json.dumps({"prompt": ids, "p1": u["p1_ids"], "p2_prefix": u["p2_prefix"],
                                                         "p2": u["p2_ids"]}), encoding="utf-8")
        if code:
            (bdir / "program.py").write_text(code, encoding="utf-8")
    except Exception as e:
        P["error"] = (P["error"] or "") + f" trace save: {type(e).__name__}: {e}"[:200]

    # ---- E18 verification (poetiq_verify_code in the sandbox) + e10_commit_iteration
    verification = None
    if code and time.monotonic() >= T_END:                      # E33: no sandbox start after the solve end (the writer's time)
        P["verify_error"] = "not verified: past the solve end"
    elif code:
        tv = time.perf_counter()
        try:
            verification = poetiq_verify_code(code, task, sandbox_timeout)
        except Exception as exc:
            P["verify_error"] = repr(exc)[:1000]
        P["verify_sec"] = round(time.perf_counter() - tv, 2)
    else:
        st["last_error"] = "missing code block"
    result = None
    if verification:
        result = dict(verification)
        result.update({"task_id": tid, "expert": expert, "iteration": iteration, "code": code})
        feedback, score = result["feedback"], result["score"]
        st["history"].append({"code": code, "feedback": feedback, "score": score})
        st["candidate_history"].append(result)
        usable = bool(result.get("test_usable", False))
        if (usable or result["passed"]) and score >= st["best_score"]:
            st["best_score"] = score
            st["best_index"] = len(st["candidate_history"]) - 1      # E33: index (E18: st['best'] = result)
        if result["passed"] and usable:
            st["status"] = "train_verified"
        elif result["passed"] and not usable:
            st["status"] = "train_pass_test_unusable"
        else:
            st["status"] = "queued"
        try:
            (bdir / "verification.json").write_text(json.dumps(result, default=str), encoding="utf-8")
        except Exception:
            pass
    elif code:
        st["last_error"] = "verification failed"
    st["iteration"] = int(st.get("iteration") or 0) + 1
    passed = bool(result and result["passed"])
    grids = [r.get("grid") for r in result["test_results"]] if result else [None] * n_test
    train_exact = [bool(r["success"]) for r in result["train_results"]] if result else [False] * n_train
    P.update(score=result["score"] if result else None, train_score=float(result["score"]) if result else 0.0,
             train_pass=passed, test_usable=bool(result and result["test_usable"]), feedback=result["feedback"] if result else None,
             test_errors=[r.get("error") for r in result["test_results"]] if result else None)
    try:
        P["e18_two_attempts"] = poetiq_two_attempts(task, poetiq_rank_state(st))
    except Exception as e:
        P["error"] = (P["error"] or "") + f" rank: {type(e).__name__}: {e}"[:200]
    st["iterations"].append({k: P[k] for k in ("iteration", "mode", "end_reason", "think_tokens", "parse_status", "score",
                                               "train_pass", "seed_p1", "seed_p2")})
    t1 = time.time()
    gen = len(u["p1_ids"]) + len(u["p2_ids"])
    rec = {"task_id": tid, "branch": cfg.get("branch"), "status": "done", "accepted": passed,
           "generated_tokens": gen, "total_tokens": len(ids) + len(u["p2_prefix"]) + gen, "t0": u["t_admit_ep"], "t1": t1,
           "note": "", "turns": 1 + int(bool(u["p2_rids"])), "requests": len(u["p1_rids"]) + len(u["p2_rids"]), "failures": len(u["errors"]), "b_budget": int(cap) + answer,
           "view": "e33_poetiq",
           "usage": [{"prompt": len(ids), "cached": None, "completion": len(u["p1_ids"])}]
                    + ([{"prompt": int(u.get("p2_prefix_len") or 0), "cached": None, "completion": len(u["p2_ids"])}]
                       if u["p2_rids"] else []),
           "submissions": [{"test_outputs": grids, "accepted": passed, "train_exact": train_exact, "t": t1,
                            "branch": cfg.get("branch"), "origin": "e33_poetiq", "name": f"e33_poetiq_a{attempt}_i{iteration}"}],
           "e33_poetiq": True, "attempt": attempt, "iteration": iteration, "poetiq": dict(P), "poetiq_state": st}
    return _publish(rec)


assert (TOP_P, TOP_K) == (0.95, 20)        # = E18 cell 1 line 56 SAMPLING top_p / top_k
'''


def make_module(src=None):
    src = src or load_sources()
    c1a, c1b = cell1_poetiq(src)
    consts, drain, cap, classes, health = direct_parts(src)
    # E18 cell 1 STOP_THINKING_TEXT / SEED == E29's (byte-identical)
    assert ('STOP_THINKING_TEXT = "\\n\\nConsidering the limited time by the user, I have to give the solution based on the thinking '
            'directly now.\\n</think>\\n\\n"') in src["cell1"]
    ns = {}
    exec(compile(consts, "e29_consts", "exec"), ns)
    assert ns["STOP_THINKING_TEXT"] == ("\n\nConsidering the limited time by the user, I have to give the solution based on the "
                                        "thinking directly now.\n</think>\n\n") and ns["SEED"] == 20261005
    assert "\nSEED = 20261005\n" in src["cell1"] and "THINK_BUDGET = int(os.environ.get(\"Q27X_THINK_BUDGET\", 60000))" in src["cell1"]
    assert '"top_p": 0.95, "top_k": 20}' in src["cell1"]
    core = core_text(src)
    t = (HEADER.replace("__NB__", E18_NB_SHA[:8]).replace("__D__", E29_DIRECT_SHA[:8])
         + E18_BUDGETS + c1a + c1b + DIRECT_HDR + consts + "\n\n" + drain + "\n\n" + cap + CORE_HDR + core + SANDBOX_BOUND
         + CLASSES_HDR + classes + "\n\n" + health + "\n" + DRIVER)
    assert "\r" not in t
    compile(t, "e33_poetiq.py", "exec")
    return t


def verbatim_regions(src=None):
    """The source regions that must appear in the module byte for byte (used by the tests)."""
    src = src or load_sources()
    c1a, c1b = cell1_poetiq(src)
    consts, drain, cap, classes, health = direct_parts(src)
    core = src["core"]
    a, b = core.split(CORE_SELFTEST_OLD)
    assert b == "", "the self-test block is the last part of the E18 core cell"
    return {"e18_cell1_poetiq_a": c1a, "e18_cell1_poetiq_b": c1b, "e18_core_before_selftest": a,
            "e29_consts": consts, "e29_drain_need_min": drain, "e29_phase1_cap": cap, "e29_stream_engine": classes,
            "e29_health": health}


if __name__ == "__main__":
    text = make_module()
    if "--check" in sys.argv:
        cur = OUT.read_bytes() if OUT.is_file() else b""
        ok = cur == text.encode("utf-8")
        print(json.dumps({"check": ok, "file_sha256": sha(cur), "built_sha256": sha(text)}, indent=1))
        sys.exit(0 if ok else 1)
    OUT.write_bytes(text.encode("utf-8"))
    assert b"\r" not in OUT.read_bytes()
    print(json.dumps({"e33_poetiq.py": sha(text), "bytes": len(text.encode("utf-8")), "e18_notebook": E18_NB_SHA,
                      "e18_core_cell5": E18_CORE_SHA, "e29_direct": E29_DIRECT_SHA}, indent=1))
