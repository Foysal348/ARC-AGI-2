"""E28 SCHED: the A -> B -> C attempt scheduler for the Duck ARC-AGI-2 harness (Experiment 28, user rule 2026-11-01).

The two E28 notebooks (NB-W4 from NB6Q, NB-K100 from K100Q) run Duck config B (E15B R3K2 on Merlin Flash-Next):
`E12A.solve_all_r6` in arc2_duck.py (payload q27x_e15b_duck_9e4b89285136.bin, manifest sha b9f67010). The scheduler,
the per-attempt budget / context window (Arc2ToolAgent.__init__) and the answer selection (build_submission) all live in
that file, so this module patches arc2_duck.py itself: the notebook calls apply_e28_sched(E12_DIR) in CELL 3 right
after the payload manifest check and before `import arc2_duck`. (The notebook cell text has no scheduler of its own; a
cell-text patch would have to replace solve_all_r6 wholesale. The three notebook-side lines that need the per-attempt
plan, the NB5 guard's projected peak, the reservation size and the ORDER2 line, are offered as NOTEBOOK_EDITS.)

User rule (Experiment 28 README section 3 / 4), as implemented by the patch:
  Phase A  attempt 1 (branch 0) of every task, plain task order (no ORDER2: the notebook passes it, NOTEBOOK_EDITS N3),
           E28_T1 = 104,000 generated tokens per attempt (reserve E12_B_RESERVE 4,096 -> longest request 99,904).
  Phase B  a fresh, independent re-run (branch 2: new agent, new conversation, a new random seed per request when
           LOCAL_ANALYZER_SEED is unset) of every task whose attempt 1 did not end with a full host-verified train match:
           B1 = ended on its own with best train < 100 % (and failed attempts: a failure note / verdict) -> E28_T1;
           B2 = cut by the cap (budget spent, CONTEXT_EXHAUSTED, INCONCLUSIVE_CONTEXT_LENGTH, BUDGET_LIMITED, commit
           done with a tiny rest, a forced commit at the budget) -> E28_B2_TOKENS = 120,000 (user decision
           2026-11-01, from data: none of the 5 attempts above 95k tokens was right) and the Duck window
           E28_B2_WINDOW = 131,072 - 512 = 130,560 (the normal window: the server stays at max-model-len 131,072).
           Cut by the solve window (time) = neither (unfinished phase-A work, no re-run).
  Phase C  attempt 2 (branch 1), E28_T1: first the tasks still without a full train match after B (task order), then
           the tasks with a full train match by the generated tokens of that attempt, ascending.
  Workers  no idling while something fits (user 2026-11-01): at every admission check the candidates are tried in
           priority order (B2 items that already waited first, then A, B, C1, C2 ascending); the first that fits the
           projected KV (the notebook's non-blocking E28_TRY_RESERVE hook, NOTEBOOK_EDITS N4; unset = everything fits)
           starts; a head that does not fit stays at the head and is re-checked every time, lower items backfill.
           The KV admission gate (waits, never preempts), min_admit_sec, the per-request deadline clamp, the join
           grace and the final freeze are the rev 6 scheduler's.
  Slots    attempt_1 = the best-train candidate of A and B (a full B replaces A; A's answer becomes the attempt_2
           fallback; neither full -> the higher train match, a tie keeps A); attempt_2 = C's answer when it exists and
           differs, else the displaced attempt-1 answer, else the next distinct select_attempts_r5 candidate. The
           submission format (build_submission -> the E15B / E18 R2 writer) is unchanged.
  Record   every attempt: phase (A / B1 / B2 / C), budget, window, best train match, tokens, end_reason, kind, worker
           slot, answer slots -> rec["e28"] and arc2_duck.SCHED_LAST["e28"] (-> e12_run_report.json).
E28_SCHED unset (or not "1"): the patched file takes the original paths everywhere (scheduler, budgets, selection).

Patch-module pattern = nb4_ppack.py / dflash_seam.py: exact pre / post sha256, a strict no-fuzz unified diff, atomic
write, py_compile, three states (ORIGINAL / FULLY_PATCHED / INVALID_MIXED), revert.
"""
import hashlib
import json
import os
import py_compile
import re
import sys
import tempfile
from pathlib import Path

DUCK_REL = "arc2_duck.py"
DUCK_PRE = "b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128"   # payload 9e4b8928 manifest
DUCK_POST = "35b8d8577ab05be60e4e66f80400e57ccec3005dfad9b5290d9fa6f53adee57f"
DUCK_MARKER = "# E28 SCHED: an env-gated (E28_SCHED=1) A -> B -> C attempt scheduler and its answer slots."
ENV_SCHED = "E28_SCHED"
ENV_T1 = "E28_T1"
ENV_B2_TOKENS = "E28_B2_TOKENS"
ENV_B2_WINDOW = "E28_B2_WINDOW"
ENV_B2_TIMEOUT = "E28_B2_REQUEST_TIMEOUT"      # optional; default = the base request timeout x B2 / T1 (B2 only)
ENV_NAMES = (ENV_SCHED, ENV_T1, ENV_B2_TOKENS, ENV_B2_WINDOW, ENV_B2_TIMEOUT)
T1 = 104000
B2_TOKENS = 120000
MAX_MODEL_LEN = 131072
B2_WINDOW = MAX_MODEL_LEN - 512                 # 130,560

DUCK_PATCH = r'''# E28 SCHED onto arc2_duck.py (Duck payload q27x_e15b_duck_9e4b89285136.bin, Experiment 15 src/duck_b).
# pre-patch sha256  b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128
# post-patch sha256 35b8d8577ab05be60e4e66f80400e57ccec3005dfad9b5290d9fa6f53adee57f
--- a/arc2_duck.py
+++ b/arc2_duck.py
@@ -1444,10 +1444,11 @@
             self._arc2_rev5 = _rev5()
             self._arc2_rev6 = _rev6()
             self.arc2_view = view if self._arc2_rev6 else "grids"
+            self._e28_plan = _e28_plan_for(task_id, branch)   # E28 SCHED: this attempt's plan (None = off)
             # Rev 1 #1: the module-level prompt builder already ran in ToolAgent.__init__; replace its result
             if self._arc2_rev6:
                 self._system_prompt = arc2_system_prompt_r6(task, DP, self._tool_output_tokens, view=self.arc2_view,
-                                                            budget=b_attempt_budget(branch))   # rev 3f
+                                                            budget=_e28_budget(self._e28_plan, branch))   # rev 3f; E28 SCHED
                 self._arc2_prelude = task_prelude_r6(task)
             else:
                 self._system_prompt = arc2_system_prompt(task, DP, self._tool_output_tokens, rev5=self._arc2_rev5)
@@ -1456,6 +1457,8 @@
             # rev 6 state: deadline (R6-S6), request caps (R6-Q1), progress (R6-S4), verification cache (R6-K3), metrics
             self._arc2_deadline: float | None = None
             self._arc2_window = _env_int("E12_DUCK_CONTEXT_WINDOW", TA._LOCAL_ANALYZER_CONTEXT_WINDOW)
+            if self._e28_plan and self._e28_plan.get("window"):   # E28 SCHED: the B2 re-run's Duck window
+                self._arc2_window = int(self._e28_plan["window"])
             self._arc2_cap = {"first": _env_int("E12_R6_CAP_FIRST", 12288), "progress": _env_int("E12_R6_CAP_PROGRESS", 24576),
                               "rescue": _env_int("E12_R6_CAP_RESCUE", 32768),
                               # G4-2: branch 1 (a grids view) makes its first call at 12.4-18.0k (G4B), so its first cap
@@ -1464,7 +1467,7 @@
             # E15B (Notes/02 #3-#5, #48): one attempt = this agent; a fixed generated-token budget, the last part reserved
             # for one commit turn (the request cap = remaining - reserve, then the remaining tokens)
             self._arc2_b = _b_mode()
-            self._arc2_b_budget = b_attempt_budget(branch)      # rev 3f: per attempt (45k / 30k)
+            self._arc2_b_budget = _e28_budget(self._e28_plan, branch)      # rev 3f: per attempt; E28 SCHED: the plan's
             self._arc2_b_reserve = _env_int("E12_B_RESERVE", 4096)
             self._arc2_b_commit_sent = False
             self._arc2_b_cut_code: list[str] = []     # rev 3h: code of tool calls cut at the output limit
@@ -2769,7 +2772,7 @@
 _RULE_CACHE: dict = {}      # id(task) -> (task, rule outputs); the task object is kept so the id stays its own
 
 
-def select_attempts_r5(task: dict, branch_records: list[dict]) -> list[dict]:
+def select_attempts_r5(task: dict, branch_records: list[dict], _e28_lists: list | None = None) -> list[dict]:
     """rev 5 B2: candidates = host-accepted submissions + host-accepted exact checks (B1). Each branch contributes its
     best = latest accepted submission, else its latest accepted check (EVAL30 247ef758: the latest exact version was the
     correct one). Per test input: branch bests grouped by identical output (more branches first, then an explicit
@@ -2829,6 +2832,8 @@
                     grids.append(g)
         if not grids:
             grids = [q["input"]]
+        if _e28_lists is not None:        # E28 SCHED: the ordered candidates per test input
+            _e28_lists.append(list(grids))
         a1 = grids[0]
         a2 = next((g for g in grids[1:] + [q["input"]] if g != a1), a1)
         result.append({"attempt_1": a1, "attempt_2": a2})
@@ -2839,6 +2844,8 @@
     """Kaggle submission.json: every task, one {attempt_1, attempt_2} per test input (fallback = the test input)."""
     sub = {}
     select = select_attempts_r5 if _rev5() else select_attempts
+    if _e28_on():
+        select = select_attempts_e28      # E28 SCHED: the A / B / C answer slots (same output format)
     for tid, task in tasks.items():
         try:
             sub[tid] = select(task, records.get(tid) or [])
@@ -3247,6 +3254,20 @@
     stage2_p90_cutoff (user-approved option, off by default): no stage-2 item starts when the time left is below the p90
     of the stage-2 run durations seen so far (>= 5 runs, capped at 900 s); stage-1 starts keep min_admit_sec only, so
     every task still gets its first exposure. Every request is deadline-clamped either way (R6-S6)."""
+    if _e28_on():     # E28 SCHED: the A -> B -> C scheduler (Experiment 28); unset = the rev 6 scheduler below
+        return _e28_solve_all(tasks, order, out_dir=out_dir, deadline=deadline, base_url=base_url, model_id=model_id,
+                              save_request_logs=save_request_logs, on_branch_done=on_branch_done, workers=workers,
+                              b1_view=b1_view, min_admit_sec=min_admit_sec, capacity_probe=capacity_probe,
+                              agent_cls=agent_cls, analyzer_timeout=analyzer_timeout, join_grace_sec=join_grace_sec,
+                              kv_admit_limit=kv_admit_limit, kv_poll_sec=kv_poll_sec, kv_stale_sec=kv_stale_sec,
+                              weak_commit=weak_commit,
+                              ignored=dict(branches=branches, max_generated_tokens=max_generated_tokens,
+                                           max_generated_tokens_b1=max_generated_tokens_b1,
+                                           first_exposure_tokens=first_exposure_tokens, min_chunk_tokens=min_chunk_tokens,
+                                           fair_share=fair_share, share_warmup_sec=share_warmup_sec,
+                                           stage2_p90_cutoff=stage2_p90_cutoff, skip_b1_if_accepted=skip_b1_if_accepted,
+                                           extra_attempts=extra_attempts, max_attempts=max_attempts,
+                                           b1_accepted_mode=b1_accepted_mode))
     import heapq
     agent_cls = agent_cls or make_agent_class()
     if skip_b1_if_accepted and not b1_accepted_mode:
@@ -3659,3 +3680,723 @@
 SERVER_DOWN = threading.Event()
 SERVER_WAITS = {"waits": 0, "sec": 0.0}
 SERVER_LAST: dict = {}     # G4-15: the latest server gauges (set by the scheduler's probe), read at each request start
+
+
+# ==================================================================== E28 SCHED (Experiment 28, user rule 2026-11-01)
+# E28 SCHED: an env-gated (E28_SCHED=1) A -> B -> C attempt scheduler and its answer slots. Unset = the code above runs
+# exactly as before (solve_all_r6, the agent budgets and build_submission take their original paths).
+#   Phase A: attempt 1 (branch 0) of every task in the given order (the notebook passes the plain order: no ORDER2),
+#            E28_T1 generated tokens per attempt (the E15B reserve E12_B_RESERVE stays).
+#   Phase B: a fresh, independent re-run (branch 2: new agent, new conversation) of every task whose attempt 1 did not
+#            end with a full host-verified train match. B1 = it ended on its own (or failed) with the best train match
+#            < 100 % -> E28_T1. B2 = it was cut by the cap (budget end, CONTEXT_EXHAUSTED, INCONCLUSIVE_CONTEXT_LENGTH,
+#            BUDGET_LIMITED, the commit turn done with a tiny rest, a forced commit at the budget) -> E28_B2_TOKENS
+#            with the Duck window
+#            E28_B2_WINDOW. An attempt cut by the solve window (time) is neither: it is unfinished phase-A work.
+#   Phase C: attempt 2 (branch 1): (1) the tasks still without a full train match after phase B, task order, E28_T1;
+#            (2) the tasks with a full train match, by the generated tokens of that attempt, ascending.
+# Admission (user 2026-11-01: no worker may idle while something fits): at every admission check (a worker frees, KV
+# frees) the candidates are tried in priority order: B2 items that already waited (did not fit before) first, then A
+# (task order), then B (task order), then C1 (task order), then C2 (tokens ascending). The first candidate that fits the
+# projected KV starts at once; a head that does not fit stays at the head and is re-checked at every admission check,
+# while a lower-priority item that fits starts in its place (backfill). Fit = E28_TRY_RESERVE(key, plan), a
+# non-blocking reservation set by the notebook (NOTEBOOK_EDITS N4); unset = everything fits. A worker waits only when
+# nothing fits at all (or the KV gate holds); with nothing running the head is admitted anyway (never a deadlock). A B /
+# C item is queued as soon as its evidence exists. The rev 6 safeties are kept: the KV admission gate (waits, never
+# preempts), min_admit_sec, the per-request deadline clamp (BranchSession), the join grace, the final freeze. Every
+# attempt records its phase, budget, window, best train match, tokens, end reason, worker slot, admission wait, backfill
+# starts while it waited and answer slots (SCHED_LAST["e28"], rec["e28"]); idle worker seconds are counted by reason.
+E28_ENV = "E28_SCHED"
+E28_CAP_VERDICTS = ("CONTEXT_EXHAUSTED", "INCONCLUSIVE_CONTEXT_LENGTH", "BUDGET_LIMITED")   # -> B2
+E28_PLANS: dict = {}          # (task, branch) -> the attempt's plan; read by Arc2ToolAgent.__init__ (budget, window)
+E28_BRANCH = {"A": 0, "C": 1, "B1": 2, "B2": 2}
+E28_TRY_RESERVE = None       # notebook hook (N4): (key, plan) -> bool, a non-blocking projected-KV reservation
+E28_RELEASE = None           # notebook hook (N4): (key) -> None, frees a reservation (idempotent)
+
+
+def _e28_on() -> bool:
+    return os.environ.get(E28_ENV, "").strip() == "1"
+
+
+def _e28_plan_for(task_id, branch):
+    """The plan of this attempt when E28 is on and the E28 scheduler created it, else None (original behaviour)."""
+    if not _e28_on():
+        return None
+    p = E28_PLANS.get((str(task_id), int(branch)))
+    return dict(p) if p else None
+
+
+def _e28_budget(plan, branch) -> int:
+    return int(plan["budget"]) if plan and plan.get("budget") else b_attempt_budget(branch)
+
+
+def e28_config() -> dict:
+    t1 = _env_int("E28_T1", 104000)       # user decision 2026-11-01: T1 104k, B2 120k, the normal 131,072 - 512 window
+    return {"t1": t1, "b2_tokens": _env_int("E28_B2_TOKENS", 120000), "b2_window": _env_int("E28_B2_WINDOW", 131072 - 512),
+            "b2_request_timeout": _env_float("E28_B2_REQUEST_TIMEOUT", 0.0) or None}
+
+
+def e28_train(rec: dict) -> dict:
+    """Best train match of one attempt record. full = a host-verified train-exact candidate exists (an accepted
+    submission or an accepted check: the host re-ran it on every train pair). k / n = the best host-checked candidate's
+    exact train pairs, else the sandbox's own progress line (untrusted: source 'progress'), else 0 / 0."""
+    cands = [c for c in (rec.get("submissions") if isinstance(rec.get("submissions"), list) else []) +
+             (rec.get("checked") if isinstance(rec.get("checked"), list) else []) if isinstance(c, dict)]
+    full = bool(rec.get("accepted")) or any(c.get("accepted") for c in cands)
+    best, src = None, None
+    for c in cands:
+        te = c.get("train_exact")
+        if isinstance(te, list) and te:
+            k, n = (len(te) if c.get("accepted") else sum(bool(x) for x in te)), len(te)
+            if best is None or (k / n, k) > (best[0] / best[1], best[0]):
+                best, src = (k, n), "host"
+    pb = (rec.get("progress") or {}).get("best") or {}
+    if best is None and pb.get("n"):
+        best, src = (int(pb.get("k") or 0), int(pb.get("n") or 0)), "progress"
+    k, n = best if best else (0, 0)
+    if full:
+        k = n = n or int(pb.get("n") or 0)
+        src = src or "accepted"
+    return {"full": full, "k": k, "n": n, "frac": 1.0 if full else (round(k / n, 4) if n else 0.0),
+            "acc": pb.get("acc"), "source": src}
+
+
+def e28_end_epoch(rec: dict):
+    """When the attempt's last run ended (epoch): t0 + the run durations; the record's t1 is rewritten by the final
+    freeze, so it is only the fallback."""
+    runs = [r for r in rec.get("runs") or [] if isinstance(r, dict)]
+    if rec.get("t0") and runs:
+        return float(rec["t0"]) + sum(float(r.get("sec") or 0) for r in runs)
+    return float(rec["t1"]) if rec.get("t1") else None
+
+
+def e28_classify(rec: dict, budget=None, at_deadline=None, window_end_epoch=None, margin_sec: float = 90.0) -> dict:
+    """How one attempt ended (E28 rule). kind: TRAIN_PASS (a host-verified full train match, however it ended) | B2 (cut
+    by the cap) | B1 (ended on its own, train < 100 %) | FAIL (a failure note / verdict: re-run like B1) | TIME (cut by
+    the solve window: no re-run). at_deadline = the live check (the scheduler); else the run end vs window_end_epoch.
+    Priority (GPT audits 2026-11-01): 1 full train; 2 cap verdicts; 3 admission-window notes; 4 explicit budget evidence
+    (tiny rest skipped, a commit turn sent), even with a note; 5 time cut (paused, at the deadline / window end);
+    6 any other verdict / note; 7 the near-budget heuristic (tokens >= budget - 1,024); 8 no tokens; else B1."""
+    tr = e28_train(rec)
+    tok = int(rec.get("generated_tokens") or 0)
+    v = rec.get("nb5_verdict")
+    note = str(rec.get("note") or "").strip()
+    be = rec.get("b_events") or {}
+    bud = int(rec.get("b_budget") or budget or 0)
+    out = {"train": tr, "tokens": tok, "budget": bud or None}
+
+    def _r(kind, why):
+        out.update(kind=kind, end_reason=why)
+        return out
+    if tr["full"]:
+        return _r("TRAIN_PASS", "accepted_submission" if rec.get("accepted") else "train_pass_check")
+    if v in E28_CAP_VERDICTS:
+        return _r("B2", str(v))
+    if note.startswith(("nb5: no KV reservation before the admit deadline", "nb5: SMOKE early exit")):
+        return _r("TIME", "admission_window")
+    if be.get("tiny_request_skipped"):
+        return _r("B2", "commit_done_tiny_rest")
+    if be.get("commit_at_tokens") is not None:
+        return _r("B2", "forced_commit")
+    if rec.get("status") == "paused":
+        return _r("TIME", "paused_at_window_end")
+    if at_deadline is None and window_end_epoch:
+        end = e28_end_epoch(rec)
+        at_deadline = end is not None and end >= float(window_end_epoch) - float(margin_sec)
+    if at_deadline:
+        return _r("TIME", "running_at_window_end")
+    if v or note:
+        return _r("FAIL", ((f"verdict {v}: " if v else "") + note)[:200])
+    if bud and tok >= bud - 1024:
+        return _r("B2", "budget_spent")
+    if tok <= 0:
+        return _r("FAIL", "no_tokens")
+    return _r("B1", "self_ended")
+
+
+def e28_b_phase(kind: str):
+    """The phase-B re-run of an attempt-1 kind: B1 / FAIL -> 'B1' (same T1), B2 -> 'B2' (E28_B2_TOKENS), else None."""
+    return {"B1": "B1", "FAIL": "B1", "B2": "B2"}.get(kind)
+
+
+def _e28_phase_of(rec: dict) -> str:
+    p = (rec.get("e28") or {}).get("phase")
+    if p:
+        return "B" if str(p).startswith("B") else str(p)
+    return {0: "A", 1: "C", 2: "B"}.get(int(rec.get("branch") or 0), "X")
+
+
+def _e28_cand(rec: dict, qi: int):
+    """This attempt's candidate for test input qi: its latest accepted submission, else its latest accepted check (=
+    select_attempts_r5's branch best), else the non-accepted candidate with the most exact train pairs (then latest)."""
+    t = lambda s: s.get("t", 0)
+    subs = [s for s in (rec.get("submissions") or []) if isinstance(s, dict)]
+    chks = [s for s in (rec.get("checked") or []) if isinstance(s, dict)]
+
+    def _g(s):
+        outs = s.get("test_outputs") or []
+        g = outs[qi] if qi < len(outs) else None
+        return g if valid_grid(g) else None
+    for s in sorted([s for s in subs if s.get("accepted")], key=t, reverse=True) + \
+            sorted([s for s in chks if s.get("accepted")], key=t, reverse=True):
+        g = _g(s)
+        if g is not None:
+            n = len(s.get("train_exact") or [])
+            return {"grid": g, "full": True, "k": n, "n": n, "frac": 1.0}
+    part = sorted([s for s in subs + chks if not s.get("accepted")],
+                  key=lambda s: (sum(bool(e) for e in s.get("train_exact") or []), t(s)), reverse=True)
+    for s in part:
+        g = _g(s)
+        if g is not None:
+            te = s.get("train_exact") or []
+            k, n = sum(bool(e) for e in te), len(te)
+            return {"grid": g, "full": False, "k": k, "n": n, "frac": (k / n) if n else 0.0}
+    return None
+
+
+def select_attempts_e28(task: dict, branch_records: list[dict], _slots_out: list | None = None) -> list[dict]:
+    """E28 answer slots (user decision 2026-11-01), the same output format as select_attempts_r5. attempt_1 = the
+    best-train candidate of attempt 1 (A) and its phase-B re-run (B): a full B replaces A (A's answer becomes the
+    attempt_2 fallback); neither full -> the higher train match (a tie keeps A). attempt_2 = phase C's answer when it
+    exists and differs, else the displaced attempt-1 answer, else the next distinct select_attempts_r5 candidate, else
+    the test input."""
+    lists: list = []
+    select_attempts_r5(task, branch_records, _e28_lists=lists)
+    by = {}
+    for r in branch_records:
+        by.setdefault(_e28_phase_of(r), r)
+    out = []
+    for qi, q in enumerate(task["test"]):
+        grids = lists[qi] if qi < len(lists) else [q["input"]]
+        ca, cb, cc = (_e28_cand(by[p], qi) if p in by else None for p in ("A", "B", "C"))
+        if cb is not None and (ca is None or (cb["full"] and not ca["full"])
+                               or (not ca["full"] and not cb["full"] and cb["frac"] > ca["frac"])):
+            p1, s1, disp, sd = cb, "B", ca, "A"
+        elif ca is not None:
+            p1, s1, disp, sd = ca, "A", cb, "B"
+        else:
+            p1, s1, disp, sd = None, "r5", None, None
+        a1 = p1["grid"] if p1 is not None else grids[0]
+        pref = ([(cc["grid"], "C")] if cc is not None else []) + ([(disp["grid"], sd)] if disp is not None else [])
+        a2, s2 = next(((g, s) for g, s in pref + [(g, "r5") for g in grids] + [(q["input"], "input")] if g != a1),
+                      (a1, "same"))
+        out.append({"attempt_1": a1, "attempt_2": a2})
+        if _slots_out is not None:
+            _slots_out.append({"qi": qi, "attempt_1": s1, "attempt_2": s2,
+                               "a1_train": None if p1 is None else {k: p1[k] for k in ("full", "k", "n")},
+                               "a_full": None if ca is None else ca["full"], "b_full": None if cb is None else cb["full"],
+                               "c_present": cc is not None})
+    return out
+
+
+def e28_slot_table(tasks: dict, records: dict) -> dict:
+    """task -> the per-test-input answer-slot sources (A / B / C / r5 / input / same) for the run report."""
+    tab = {}
+    for tid, task in tasks.items():
+        rows: list = []
+        try:
+            select_attempts_e28(task, records.get(tid) or [], _slots_out=rows)
+        except Exception as exc:
+            rows = [{"error": f"{type(exc).__name__}: {exc}"[:200]}]
+        tab[tid] = rows
+    return tab
+
+
+def e28_c_key(kind_full: bool, tokens: int, idx: int) -> tuple:
+    """Phase-C priority: (1, task order) for a task still without a full train match after phase B; (2, generated tokens
+    of the full-match attempt, task order) for a task with one (ascending tokens)."""
+    return (2, int(tokens), int(idx)) if kind_full else (1, 0, int(idx))
+
+
+def _e28_solve_all(tasks: dict, order: list[str], *, out_dir: Path, deadline: float, base_url: str, model_id: str,
+                   save_request_logs: bool = False, on_branch_done: Callable[[dict], None] | None = None,
+                   workers: int = 10, b1_view: str = "facts", min_admit_sec: float = 240.0,
+                   capacity_probe: Callable[[], dict | None] | None = None, agent_cls=None,
+                   analyzer_timeout: float | None = None, join_grace_sec: float = 60.0, kv_admit_limit: float = 0.95,
+                   kv_poll_sec: float = 5.0, kv_stale_sec: float = 30.0, weak_commit: bool = False,
+                   ignored: dict | None = None) -> dict[str, list[dict]]:
+    """The E28 scheduler (see the block comment above). Returns {task: [latest record per branch]} like solve_all_r6."""
+    import heapq
+    agent_cls = agent_cls or make_agent_class()
+    cfg = e28_config()
+    timeout = float(analyzer_timeout or os.environ.get("E12_DUCK_REQUEST_TIMEOUT") or 900.0)
+    b2_timeout = float(cfg["b2_request_timeout"] or timeout * max(1.0, cfg["b2_tokens"] / max(1, cfg["t1"])))
+    started = time.monotonic()
+    epoch0 = time.time()
+    order = [t for t in order if t in tasks]
+    idx = {t: i for i, t in enumerate(order)}
+    E28_PLANS.clear()
+    sessions: dict[tuple[str, int], BranchSession] = {}
+    cond = threading.Condition()
+    qa: list = list(order)              # phase A: FIFO in the given order
+    qb: list = []                       # phase B: heap (task order)
+    qc: list = []                       # phase C: heap (e28_c_key)
+    seq = [0]
+    warnings: list = []
+    if _env_int("E12_B_ATTEMPT_TOKENS", cfg["t1"]) != cfg["t1"]:
+        warnings.append(f"E12_B_ATTEMPT_TOKENS={_env_int('E12_B_ATTEMPT_TOKENS', 0)} != E28_T1={cfg['t1']} (the "
+                        "notebook's reservation blocks follow NB5_ATTEMPT; E28 attempts use E28_T1)")
+    try:
+        if int(getattr(_import_duck()[0], "_LOCAL_ANALYZER_SEED", -1)) >= 0:
+            warnings.append("LOCAL_ANALYZER_SEED is fixed: a phase-B re-run is not an independent seed")
+    except Exception as exc:
+        warnings.append(f"seed check failed: {type(exc).__name__}")
+    for w_ in warnings:
+        log.warning("E28: %s", w_)
+    state = {"running": 0, "stop": False, "events": [], "durations": [],
+             "kv": None, "kv_t": None, "kv_samples": 0, "kv_missing": 0, "kv_holds": 0, "kv_hold_waits": 0,
+             "kv_hold_sec": 0.0, "holding_since": None, "kv_idle_overrides": 0, "gauges": [], "kv_stale_checks": 0,
+             "publish_seq": 0, "task_sigs": {}, "last_free_t": None, "kv_superseded": 0}
+    e28 = {"started": {"A": 0, "B1": 0, "B2": 0, "C": 0}, "done": {"A": 0, "B1": 0, "B2": 0, "C": 0},
+           "queued": {"A": len(qa), "B1": 0, "B2": 0, "C1": 0, "C2": 0},
+           "first_start_sec": {}, "last_start_sec": {}, "queue_empty_sec": {}, "a_class": {}, "b_class": {},
+           "rows": {}, "setup_failed": [], "c_started": 0, "pop_seq": 0, "backfill_starts": 0, "forced_admits": 0,
+           "fit_checks": 0, "fit_fails": 0, "fit_errors": 0, "admission_checks": 0, "backfill_log": [],
+           "idle": {"nothing_fits": 0.0, "kv_gate": 0.0, "no_runnable": 0.0}}
+    meta: dict = {}                     # (task, branch) -> admission wait bookkeeping of a queued B / C item
+    active: set = set()
+    records: dict[str, list[dict]] = {t: [] for t in order}
+
+    def _sec() -> float:
+        return round(time.monotonic() - started, 1)
+
+    def plan(tid: str, phase: str, **kw) -> dict:
+        p = {"task": tid, "branch": E28_BRANCH[phase], "phase": phase,
+             "budget": cfg["b2_tokens"] if phase == "B2" else cfg["t1"],
+             "window": cfg["b2_window"] if phase == "B2" else None,
+             "timeout": b2_timeout if phase == "B2" else timeout}
+        p.update(kw)
+        return p
+
+    def push_b(tid: str, phase: str, a_cls: dict):
+        seq[0] += 1
+        heapq.heappush(qb, (idx[tid], seq[0], tid, plan(tid, phase, b_class=a_cls.get("kind"),
+                                                         b_reason=a_cls.get("end_reason"))))
+        e28["queued"][phase] += 1
+        meta[(tid, E28_BRANCH[phase])] = _meta()
+
+    def push_c(tid: str, full: bool, tokens: int, src: str):
+        key = e28_c_key(full, tokens, idx[tid])
+        seq[0] += 1
+        heapq.heappush(qc, (key, seq[0], tid, plan(tid, "C", c_priority=key[0], c_key_tokens=int(tokens) if full else None,
+                                                    c_from=src)))
+        e28["queued"]["C2" if full else "C1"] += 1
+        meta[(tid, E28_BRANCH["C"])] = _meta()
+
+    def _meta() -> dict:
+        return {"queued_sec": _sec(), "first_tried_sec": None, "fit_fails": 0, "backfill": 0, "waiting": False}
+
+    def _fits(tid: str, p: dict) -> bool:
+        hook = E28_TRY_RESERVE
+        if hook is None:
+            return True
+        e28["fit_checks"] += 1
+        try:
+            ok = bool(hook((tid, int(p["branch"])), dict(p)))
+        except Exception as exc:          # a broken hook never blocks admission (fail-open, counted)
+            e28["fit_errors"] += 1
+            state["events"].append({"t": round(time.monotonic() - started), "kind": "e28_fit_error",
+                                    "error": f"{type(exc).__name__}: {exc}"[:200]})
+            return True
+        e28["fit_fails"] += 0 if ok else 1
+        return ok
+
+    def candidates() -> list:
+        """Under the lock: every queued item in priority order as (where, entry, tid, plan): B2 items that already
+        waited, then A (task order), then B, then C1, then C2 (ascending tokens)."""
+        bq = sorted(qb)
+        wb2 = lambda e: e[3]["phase"] == "B2" and meta.get((e[2], 2), {}).get("waiting")
+        return ([("B", e, e[2], e[3]) for e in bq if wb2(e)] + [("A", tid, tid, None) for tid in qa] +
+                [("B", e, e[2], e[3]) for e in bq if not wb2(e)] + [("C", e, e[2], e[3]) for e in sorted(qc)])
+
+    def take(where, entry, tid):
+        if where == "A":
+            qa.remove(tid)
+            if not qa:
+                e28["queue_empty_sec"].setdefault("A", _sec())
+            return tid, plan(tid, "A"), idx[tid]
+        if where == "B":
+            qb.remove(entry)
+            heapq.heapify(qb)
+            if not qb:
+                e28["queue_empty_sec"]["B"] = _sec()
+            return tid, entry[3], 100000 + idx[tid]
+        qc.remove(entry)
+        heapq.heapify(qc)
+        e28["c_started"] += 1
+        if not qc:
+            e28["queue_empty_sec"]["C"] = _sec()
+        return tid, entry[3], 200000 + e28["c_started"]
+
+    def pick(force: bool):
+        """Under the lock: the first candidate (priority order) that fits; the skipped ones keep their place (a B2 that
+        did not fit is checked first next time). Nothing fits and force (nothing runs) = the head anyway (the
+        attempt's own blocking acquire then grants it: never a deadlock)."""
+        e28["admission_checks"] += 1
+        failed_budgets, skipped = set(), []
+        cands = candidates()
+        for where, entry, tid, p in cands:
+            pp = p if p is not None else plan(tid, "A")
+            k = (tid, int(pp["branch"]))
+            m = meta.get(k)
+            if m is not None and m["first_tried_sec"] is None:
+                m["first_tried_sec"] = _sec()
+            ok = pp["budget"] not in failed_budgets and _fits(tid, pp)
+            if not ok:
+                failed_budgets.add(pp["budget"])       # the same budget = the same need: not tried twice per check
+                if m is not None:
+                    m["fit_fails"] += 1
+                    m["waiting"] = True
+                skipped.append(k)
+                continue
+            if skipped:                   # a backfill start: counted for every skipped item that waits
+                e28["backfill_starts"] += 1
+                for sk in skipped:
+                    if sk in meta:
+                        meta[sk]["backfill"] += 1
+                if len(e28["backfill_log"]) < 500:
+                    e28["backfill_log"].append({"t": _sec(), "started": [pp["phase"], tid],
+                                                "skipped": [list(x) for x in skipped]})
+            return take(where, entry, tid)
+        if force and cands:
+            e28["forced_admits"] += 1
+            where, entry, tid, _ = cands[0]
+            return take(where, entry, tid)
+        return None
+
+    def sess(tid: str, p: dict, qpos: int) -> BranchSession:
+        b = int(p["branch"])
+        if (tid, b) not in sessions:
+            E28_PLANS[(tid, b)] = dict(p)
+            s_ = BranchSession(agent_cls, task_id=tid, task=tasks[tid], branch=b, view=("facts" if b == 0 else b1_view),
+                               queue_position=qpos, out_dir=out_dir, deadline=deadline, base_url=base_url,
+                               model_id=model_id, save_request_logs=save_request_logs, analyzer_timeout=p["timeout"],
+                               started_at=started, max_generated_tokens=int(p["budget"]))
+            s_.e28_plan = dict(p)
+            sessions[(tid, b)] = s_
+        return sessions[(tid, b)]
+
+    def row_of(s_: BranchSession, rec: dict, cls: dict | None, worker_i, t_start, t_end, qlen=None, start_seq=None) -> dict:
+        p = getattr(s_, "e28_plan", None) or {}
+        ag = s_.agent
+        r = {"task": s_.task_id, "branch": s_.branch, "phase": p.get("phase"), "b_class": p.get("b_class"),
+             "b_reason": p.get("b_reason"), "c_priority": p.get("c_priority"), "c_key_tokens": p.get("c_key_tokens"),
+             "budget": p.get("budget"), "window_plan": p.get("window"),
+             "window": getattr(ag, "_arc2_window", None) if ag is not None else None,
+             "agent_budget": getattr(ag, "_arc2_b_budget", None) if ag is not None else None,
+             "session_cap": s_.max_generated_tokens, "request_timeout": p.get("timeout"),
+             "tokens": int(rec.get("generated_tokens") or 0), "worker_slot": worker_i,
+             "start_sec": t_start, "end_sec": t_end, "status": rec.get("status"),
+             "queues_at_start": qlen,           # [A, B, C] queue lengths right after this item was taken
+             "start_seq": start_seq}            # the admission order over the whole run (1, 2, ...)
+        m = meta.get((s_.task_id, s_.branch))
+        if m is not None:                  # B / C items: admission wait from runnable (first tried) to start
+            r.update(queued_sec=m["queued_sec"], first_tried_sec=m["first_tried_sec"], fit_fails=m["fit_fails"],
+                     backfill_starts_while_waiting=m["backfill"],
+                     wait_sec=(round(t_start - m["first_tried_sec"], 1) if t_start is not None
+                               and m["first_tried_sec"] is not None else None))
+        if cls is not None:
+            r.update(kind=cls.get("kind"), end_reason=cls.get("end_reason"), best_train=cls.get("train"))
+        return r
+
+    def publish(rec: dict, final: bool = False):
+        with cond:
+            if state.get("closed") and not final:
+                return
+            rec = copy.deepcopy(rec)
+            state["publish_seq"] += 1
+            rec["publish_seq"] = state["publish_seq"]
+            rows = records.setdefault(rec["task_id"], [])
+            rows[:] = [r for r in rows if r.get("branch") != rec.get("branch")] + [rec]
+        if on_branch_done is not None:
+            try:
+                on_branch_done(copy.deepcopy(rec))
+            except Exception as exc:
+                log.warning("on_branch_done failed: %s", exc)
+
+    pstate = {"miss_since": None, "warned": False, "last_g": -1e9}
+
+    def probe_once():
+        try:
+            snap = capacity_probe() or {}
+        except Exception:
+            snap = {}
+        kv = _finite01(snap.get("kv_usage"))
+        now = time.monotonic()
+        with cond:
+            state["kv"], state["kv_t"] = kv, now
+            state["kv_samples"] += 1
+            if kv is None:
+                state["kv_missing"] += 1
+                if pstate["miss_since"] is None:
+                    pstate["miss_since"] = now
+                if not pstate["warned"] and now - pstate["miss_since"] >= kv_stale_sec:
+                    pstate["warned"] = True
+                    log.warning("KV usage metric unavailable; admitting without the KV gate")
+                    state["events"].append({"t": round(now - started), "kind": "kv_metric_missing"})
+            else:
+                pstate["miss_since"], pstate["warned"] = None, False
+            SERVER_LAST.clear()
+            SERVER_LAST.update({k: snap.get(k) for k in ("kv_usage", "mamba_usage", "running", "queue")}, t=now)
+            if now - pstate["last_g"] >= 30 and len(state["gauges"]) < 3000:
+                pstate["last_g"] = now
+                state["gauges"].append([round(now - started), kv, _finite01(snap.get("mamba_usage")),
+                                        snap.get("running"), snap.get("queue"), state["running"],
+                                        len(qa) + len(qb) + len(qc)])
+            cond.notify_all()
+
+    def probe_loop():
+        while not state["stop"] and capacity_probe is not None:
+            t_end = time.monotonic() + max(0.02, float(kv_poll_sec))
+            while not state["stop"] and time.monotonic() < t_end:
+                time.sleep(min(0.5, max(0.01, float(kv_poll_sec))))
+            if not state["stop"]:
+                probe_once()
+
+    def kv_blocked() -> bool:
+        kv, t = state["kv"], state["kv_t"]
+        if kv is None or t is None or time.monotonic() - t > kv_stale_sec:
+            if kv is not None and t is not None:
+                state["kv_stale_checks"] += 1
+            return False
+        if state["last_free_t"] is not None and t < state["last_free_t"]:
+            # E28 no-idle: the sample predates the latest attempt end (its KV / slot was freed since): it no longer
+            # describes the pool, so it never holds a free worker (the fit check decides; slots mode: a free slot)
+            state["kv_superseded"] += 1
+            return False
+        return kv >= kv_admit_limit
+
+    def hold_end(now: float):
+        if state["holding_since"] is not None:
+            state["kv_hold_sec"] += now - state["holding_since"]
+            state["events"].append({"t": round(now - started), "kind": "kv_release", "kv": state["kv"]})
+            state["holding_since"] = None
+
+    def route(tid: str, p: dict, cls: dict):
+        """Under the lock: queue the next phase of a task from the attempt that just ended."""
+        ph = p["phase"]
+        full = cls["kind"] == "TRAIN_PASS"
+        if ph == "A":
+            e28["a_class"][tid] = {k: cls.get(k) for k in ("kind", "end_reason", "tokens")}
+            if full:
+                push_c(tid, True, cls["tokens"], "A")
+            elif e28_b_phase(cls["kind"]):
+                push_b(tid, e28_b_phase(cls["kind"]), cls)
+            # TIME: unfinished phase-A work (the solve window ended); no re-run
+        elif ph in ("B1", "B2"):
+            e28["b_class"][tid] = {k: cls.get(k) for k in ("kind", "end_reason", "tokens")}
+            push_c(tid, full, cls["tokens"], ph)
+
+    def worker(wi: int):
+        while True:
+            try:
+                return _worker_body(wi)
+            except Exception as exc:
+                with cond:
+                    state["worker_errors"] = state.get("worker_errors", 0) + 1
+                    state["events"].append({"t": round(time.monotonic() - started), "kind": "worker_error",
+                                            "error": f"{type(exc).__name__}: {exc}"[:200]})
+                log.warning("E28 scheduler worker error (restarting): %s", exc)
+                time.sleep(1.0)
+
+    def _worker_body(wi: int):
+        while True:
+            with cond:
+                while True:
+                    left = deadline - time.monotonic()
+                    if left < min_admit_sec or state["stop"]:
+                        return
+                    has = bool(qa or qb or qc)
+                    if has and kv_blocked():
+                        if state["running"] == 0:     # nothing runs, so nothing can free KV: never deadlock on the gate
+                            state["kv_idle_overrides"] += 1
+                        else:
+                            if state["holding_since"] is None:
+                                state["holding_since"] = time.monotonic()
+                                state["kv_holds"] += 1
+                                state["events"].append({"t": round(time.monotonic() - started), "kind": "kv_hold",
+                                                        "kv": state["kv"], "running": state["running"]})
+                            state["kv_hold_waits"] += 1
+                            _w0 = time.monotonic()
+                            cond.wait(timeout=min(5.0, max(0.5, float(kv_poll_sec))))
+                            e28["idle"]["kv_gate"] += time.monotonic() - _w0
+                            continue
+                    if has:
+                        got = pick(force=state["running"] == 0)
+                        if got is None:               # nothing fits at all: wait for the next admission event
+                            _w0 = time.monotonic()
+                            cond.wait(timeout=1.0)
+                            e28["idle"]["nothing_fits"] += time.monotonic() - _w0
+                            continue
+                        hold_end(time.monotonic())
+                        tid, p, qpos = got
+                        qlen = [len(qa), len(qb), len(qc)]
+                        e28["pop_seq"] += 1
+                        start_seq = e28["pop_seq"]
+                        try:
+                            s = sess(tid, p, qpos)
+                        except Exception as exc:
+                            state["setup_failed"] = state.get("setup_failed", 0) + 1
+                            e28["setup_failed"].append({"task": tid, "phase": p["phase"],
+                                                        "error": f"{type(exc).__name__}: {exc}"[:200]})
+                            records.setdefault(tid, []).append({"task_id": tid, "branch": p["branch"], "status": "done",
+                                                                "accepted": False, "submissions": [], "checked": [],
+                                                                "note": f"session setup {type(exc).__name__}: {exc}"[:500],
+                                                                "e28": {"phase": p["phase"], "kind": "FAIL"}})
+                            log.warning("%s %s: session setup failed: %s", tid, p["phase"], exc)
+                            if E28_RELEASE is not None:
+                                try:
+                                    E28_RELEASE((tid, int(p["branch"])))
+                                except Exception:
+                                    pass
+                            route(tid, p, {"kind": "FAIL", "end_reason": "session_setup", "tokens": 0})
+                            cond.notify_all()
+                            continue
+                        ph = p["phase"]
+                        e28["started"][ph] += 1
+                        e28["first_start_sec"].setdefault(ph, _sec())
+                        e28["last_start_sec"][ph] = _sec()
+                        state["running"] += 1
+                        active.add((tid, s.branch))
+                        t_start = _sec()
+                        s.record["e28"] = row_of(s, s.record, None, wi, t_start, None, qlen, start_seq)
+                        break
+                    if state["running"] == 0:
+                        state["stop"] = True
+                        cond.notify_all()
+                        return
+                    _w0 = time.monotonic()            # no queued item: B / C items appear when running attempts end
+                    cond.wait(timeout=5)
+                    e28["idle"]["no_runnable"] += time.monotonic() - _w0
+            t0 = time.monotonic()
+            tag = {"stage": {"A": 1, "C": 2}.get(ph, 3), "e28_phase": ph}
+            try:
+                if s.status != "done":
+                    s.run(pause_at_tokens=None, tag=tag)
+            except Exception as exc:
+                s.record["note"] = f"setup {type(exc).__name__}: {exc}"[:500]
+                s.status = "done"
+                log.warning("%s b%d: %s", tid, s.branch, s.record["note"])
+            if E28_RELEASE is not None:        # the notebook's wrapper releases in its finally; this covers early returns
+                try:
+                    E28_RELEASE((tid, s.branch))
+                except Exception:
+                    pass
+            at_dl = time.monotonic() >= deadline - 5.0
+            with cond:
+                seen = state["task_sigs"].setdefault(tid, set())
+                sigs = s.verified_signatures()
+                if s.record["runs"] and not s._frozen:
+                    s.record["runs"][-1]["new_for_task"] = len(sigs - seen)
+                seen |= sigs
+            try:
+                rec = s.write()
+                cls = e28_classify(rec, budget=p["budget"], at_deadline=at_dl)
+                if not s._frozen:
+                    s.record["e28"] = row_of(s, rec, cls, wi, t_start, _sec(), qlen, start_seq)
+                    e28["rows"][f"{tid}_b{s.branch}"] = dict(s.record["e28"])
+                publish(s.write())
+            except Exception as exc:
+                cls = {"kind": "FAIL", "end_reason": f"classify/publish {type(exc).__name__}", "tokens": 0,
+                       "train": None}
+                log.warning("%s b%d: publish failed: %s", tid, s.branch, exc)
+            with cond:
+                try:
+                    route(tid, p, cls)
+                except Exception as exc:
+                    state["stage2_errors"] = state.get("stage2_errors", 0) + 1
+                    state["events"].append({"t": round(time.monotonic() - started), "kind": "e28_route_error",
+                                            "task": tid, "error": f"{type(exc).__name__}: {exc}"[:200]})
+                e28["done"][ph] += 1
+                state["running"] -= 1
+                state["last_free_t"] = time.monotonic()
+                active.discard((tid, s.branch))
+                state["durations"].append(round(time.monotonic() - t0, 1))
+                cond.notify_all()
+
+    if capacity_probe is not None:
+        probe_once()
+    probe = threading.Thread(target=probe_loop, daemon=True)
+    probe.start()
+    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(max(1, int(workers)))]
+    for t in threads:
+        t.start()
+    join_until = deadline + float(join_grace_sec)
+    for t in threads:
+        t.join(timeout=max(0.0, join_until - time.monotonic()))
+    state["stop"] = True
+    with cond:
+        hold_end(time.monotonic())
+        cond.notify_all()
+        state["closed"] = True
+        final = list(sessions.values())
+        state["stragglers"] = sum(1 for t in threads if t.is_alive())
+    win_end = epoch0 + (deadline - started)
+    for s in final:
+        if s.agent is not None:
+            fr = s.freeze()
+            row = dict(fr.get("e28") or {})
+            if row.get("kind") is None:          # a straggler or an attempt still running at the join: classify now
+                cls = e28_classify(fr, budget=(s.e28_plan or {}).get("budget"), window_end_epoch=win_end)
+                row.update(kind=cls["kind"], end_reason=cls["end_reason"], best_train=cls["train"],
+                           tokens=int(fr.get("generated_tokens") or 0), final_classified=True)
+                fr = dict(fr, e28=row)
+            e28["rows"][f"{s.task_id}_b{s.branch}"] = row
+            publish(fr, final=True)
+    try:
+        slots = e28_slot_table(tasks, records)
+    except Exception as exc:
+        slots = {"error": f"{type(exc).__name__}: {exc}"[:200]}
+    rows_ = []
+    for key, row in sorted(e28["rows"].items(), key=lambda kv: (idx.get(kv[1].get("task"), 1 << 30), kv[1].get("branch", 9))):
+        r = dict(row)
+        sl = slots.get(r.get("task")) if isinstance(slots, dict) else None
+        if isinstance(sl, list):
+            src = {"A": 0, "C": 1, "B": 2}
+            r["answer_slots"] = [("attempt_1" if src.get(x.get("attempt_1")) == r.get("branch") else
+                                  "attempt_2" if src.get(x.get("attempt_2")) == r.get("branch") else None)
+                                 for x in sl if isinstance(x, dict)]
+        rows_.append(r)
+    SCHED_LAST.clear()
+    SCHED_LAST.update(events=state["events"], allowed=max(1, int(workers)), durations=state["durations"][-500:],
+                      sessions={f"{s.task_id}_b{s.branch}": s.status for s in final},
+                      stage1_left=len(qa), heap_left=len(qb) + len(qc), shares=[], p90_skipped=0,
+                      stage2_p90_cutoff=False, stragglers=state.get("stragglers", 0),
+                      worker_errors=state.get("worker_errors", 0), setup_failed=state.get("setup_failed", 0),
+                      tasks_exposed=len({t for (t, b), s_ in sessions.items() if b == 0 and s_.agent is not None}),
+                      tasks_initialized=len({t for (t, b), s_ in sessions.items() if b == 0 and s_.agent is not None}),
+                      tasks_with_request_sent=len({t for (t, b), s_ in sessions.items() if b == 0 and s_.agent is not None
+                                                   and ((getattr(s_.agent, "arc2_rev6_events", {}) or {}).get("requests_started")
+                                                        or s_.agent.generated_tokens > 0)}),
+                      workers=max(1, int(workers)), kv_admit_limit=kv_admit_limit, kv_samples=state["kv_samples"],
+                      kv_missing=state["kv_missing"], kv_holds=state["kv_holds"], kv_hold_waits=state["kv_hold_waits"],
+                      kv_hold_sec=round(state["kv_hold_sec"], 1), kv_idle_overrides=state["kv_idle_overrides"],
+                      kv_stale_checks=state["kv_stale_checks"], kv_superseded=state["kv_superseded"], b1_deferred_at_pop=0, b1_skipped=0, extra_started=0,
+                      extra_by_attempt={}, extra_waits=0, skip_b1_if_accepted=False, extra_attempts=False,
+                      b1_accepted_mode="", b1_deferred=0, b1_deferred_started=0, weak_commit=bool(weak_commit),
+                      **_weak_stats(sessions, order, weak_commit),
+                      stage2_errors=state.get("stage2_errors", 0), max_attempts=3,
+                      gauges=state["gauges"][-3000:],
+                      gauge_fields=["t", "kv", "mamba", "server_running", "server_queue", "workers_running", "queued"],
+                      e28={"enabled": True, "config": dict(cfg, request_timeout=timeout, b2_request_timeout=b2_timeout,
+                                                          workers=max(1, int(workers)), min_admit_sec=min_admit_sec,
+                                                          tasks=len(order)),
+                           "ignored_args": dict(ignored or {}), "warnings": warnings,
+                           "started": e28["started"], "done": e28["done"], "queued": e28["queued"],
+                           "left": {"A": len(qa), "B": len(qb), "C": len(qc)},
+                           "first_start_sec": e28["first_start_sec"], "last_start_sec": e28["last_start_sec"],
+                           "queue_empty_sec": e28["queue_empty_sec"],
+                           "idle_worker_seconds": round(e28["idle"]["nothing_fits"] + e28["idle"]["kv_gate"], 3),
+                           "idle_sec_by_reason": {k: round(v, 3) for k, v in e28["idle"].items()},
+                           "backfill_starts": e28["backfill_starts"], "forced_admits": e28["forced_admits"],
+                           "admission_checks": e28["admission_checks"], "fit_checks": e28["fit_checks"],
+                           "fit_fails": e28["fit_fails"], "fit_errors": e28["fit_errors"],
+                           "fit_hook": E28_TRY_RESERVE is not None, "backfill_log": e28["backfill_log"],
+                           "a_class": e28["a_class"],
+                           "b_class": e28["b_class"], "setup_failed": e28["setup_failed"], "attempts": rows_,
+                           "slots": slots,
+                           "row_fields": ["task", "branch", "phase", "b_class", "b_reason", "c_priority", "c_key_tokens",
+                                          "budget", "window", "tokens", "best_train", "end_reason", "kind",
+                                          "worker_slot", "start_seq", "queues_at_start", "wait_sec",
+                                          "backfill_starts_while_waiting", "answer_slots"]})
+    return records
'''

PATCHES = {DUCK_REL: (DUCK_PRE, DUCK_POST, DUCK_MARKER, DUCK_PATCH)}

APPLIER_SRC = r'''
def apply_unified(text, patch):
    """Strict unified-diff applier (no fuzz, no offset): every context / removed line must match at its stated line,
    else ValueError. Lines before the first hunk (comments, ---/+++ headers) are ignored. (= NB1 / NB5 / NB4 P5.)"""
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


def reverse_unified(patch):
    """The reverse patch: hunk headers swapped (-a,b +c,d -> -c,d +a,b), '+' and '-' lines swapped."""
    out = []
    started = False
    for ln in patch.splitlines(True):
        if ln.startswith("@@"):
            started = True
            m = re.match(r"@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@(.*)", ln, re.S)
            if not m:
                raise ValueError("bad hunk header: " + ln[:80])
            out.append(f"@@ -{m.group(2)} +{m.group(1)} @@{m.group(3)}")
        elif started and ln[:1] == "+":
            out.append("-" + ln[1:])
        elif started and ln[:1] == "-":
            out.append("+" + ln[1:])
        else:
            out.append(ln)
    return "".join(out)
'''
exec(APPLIER_SRC)  # noqa: S102 - defines apply_unified / reverse_unified (= the nb4_ppack / dflash_seam applier text)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(t):
    return sha256_bytes(t.encode("utf-8"))


PATCH_SHA = {rel: sha256_text(p[3]) for rel, p in PATCHES.items()}
E28_SCHED_SHA = {DUCK_REL: {"pre": DUCK_PRE, "post": DUCK_POST}, "patch": PATCH_SHA[DUCK_REL],
                 "applier": sha256_text(APPLIER_SRC)}

# ------------------------------------------------------------------ notebook-side text edits (NB5-family cell text)
# Verified (count == 1) against E25 nb6q_code_export.py (NB6Q) and e26_k100q_code_export.py (K100Q). The builders apply
# them with their own once(); the module never edits a builder or a notebook.
_N1_OLD = '    pk = NB5_BLOCKS.get("peak_b0" if int(getattr(self, "arc2_branch", 0) or 0) == 0 else "peak_b1")\n'
_N1_NEW = ('    _e28p = getattr(self, "_e28_plan", None)         # E28 SCHED N1: the attempt\'s own plan budget (B2 = 120k)\n'
           '    pk = ((NB5_PROMPT_ALLOW + int(_e28p["budget"]) + NB5_TOOL_ALLOW) if _e28p\n'
           '          else NB5_BLOCKS.get("peak_b0" if int(getattr(self, "arc2_branch", 0) or 0) == 0 else "peak_b1"))\n')
_N2_OLD = ('    need = (int(NB5_BLOCKS["b0" if self.branch == 0 else "b1"] or 1) if NB5_ADMIT_MODE != "live"\n'
           '            else int(NB5_BLOCKS["start"] or 1))\n')
_N2_NEW = ('    _e28p = getattr(self, "e28_plan", None)          # E28 SCHED N2: an E28 attempt reserves its own projected peak\n'
           '    need = (int(NB5L.cdiv(min(MAX_LEN, NB5_PROMPT_ALLOW + int(_e28p["budget"]) + NB5_TOOL_ALLOW), NB5_BLOCKS["block"])\n'
           '                + int(NB5_BLOCKS["f"] or 0)) if (_e28p and NB5_ADMIT_MODE != "live" and NB5_BLOCKS.get("f") is not None)\n'
           '            else int(NB5_BLOCKS["b0" if self.branch == 0 else "b1"] or 1) if NB5_ADMIT_MODE != "live"\n'
           '            else int(NB5_BLOCKS["start"] or 1))\n')
_N3_OLD = "    E12_ORDER = sorted(TASK_IDS, key=lambda t: (-_e12_order2_score(t), _e12_total_cells(t), _E12_INDEX[t]))\n"
_N3_NEW = ('    E12_ORDER = (list(TASK_IDS) if os.environ.get("E28_SCHED", "").strip() == "1"   # E28 SCHED N3: no ORDER2\n'
           '                 else sorted(TASK_IDS, key=lambda t: (-_e12_order2_score(t), _e12_total_cells(t), _E12_INDEX[t])))\n')
_N4_OLD = "E12A.BranchSession.run = _nb5_session_run\n"
_N4_NEW = _N4_OLD + r'''

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
'''
NOTEBOOK_EDITS = [("N1 _nb5_peak_len: the E28 plan's projected peak", _N1_OLD, _N1_NEW),
                  ("N2 _nb5_session_run: the E28 plan's reservation blocks", _N2_OLD, _N2_NEW),
                  ("N3 e21_cell6: no ORDER2 under E28_SCHED=1 (plain TASK_IDS order)", _N3_OLD, _N3_NEW),
                  ("N4 CELL 3: the scheduler's non-blocking fit check + release hooks (backfill)", _N4_OLD, _N4_NEW)]


def apply_notebook_edits(text, labels=None):
    """Apply NOTEBOOK_EDITS (all, or the given labels' prefixes N1 / N2 / N3) to one cell / export text; each must
    occur exactly once. Returns (new_text, applied_labels)."""
    done = []
    for lab, old, new in NOTEBOOK_EDITS:
        if labels is not None and lab.split()[0] not in labels:
            continue
        if text.count(old) != 1:
            raise ValueError(f"E28 notebook edit {lab}: {text.count(old)} matches (expected 1)")
        text = text.replace(old, new)
        done.append(lab)
    return text, done


# ------------------------------------------------------------------ env
def server_env():
    """The env of the NOTEBOOK process that turns E28 on (Duck reads it at call time; nothing goes to the vLLM server).
    E28_B2_REQUEST_TIMEOUT stays unset (default: base timeout x B2 / T1 for B2 attempts only)."""
    return {ENV_SCHED: "1", ENV_T1: str(T1), ENV_B2_TOKENS: str(B2_TOKENS), ENV_B2_WINDOW: str(B2_WINDOW)}


# ------------------------------------------------------------------ states
def _is_text(target):
    return isinstance(target, str) and "\n" in target


def _state_of(data):
    h = sha256_bytes(data)
    if h == DUCK_POST and DUCK_MARKER.encode() in data:
        return "FULLY_PATCHED", h
    if h == DUCK_PRE:
        return "ORIGINAL", h
    return "INVALID_MIXED", h


def tree_state(target):
    """ORIGINAL (pre sha) | FULLY_PATCHED (post sha + marker) | INVALID_MIXED (anything else, a missing file too).
    target = a directory holding arc2_duck.py, the path of that file, or its source text."""
    if _is_text(target):
        return _state_of(target.encode("utf-8"))[0]
    f = _file(target)
    if not f.is_file():
        return "INVALID_MIXED"
    return _state_of(f.read_bytes())[0]


def _file(target):
    p = Path(target)
    return p / DUCK_REL if p.is_dir() else p


def verify_e28_sched(target):
    """(ok, why): ok only when FULLY_PATCHED; why = 'STATE sha256=... (pre ..., post ...)'."""
    if _is_text(target):
        st, h = _state_of(target.encode("utf-8"))
    else:
        f = _file(target)
        if not f.is_file():
            return False, f"INVALID_MIXED missing {f}"
        st, h = _state_of(f.read_bytes())
    return st == "FULLY_PATCHED", f"{st} sha256={h} (pre {DUCK_PRE[:12]}, post {DUCK_POST[:12]})"


def _write_atomic(f, data):
    fd, tmp = tempfile.mkstemp(prefix=f.name + ".", suffix=".e28tmp", dir=str(f.parent))
    try:
        with os.fdopen(fd, "wb") as h:
            h.write(data)
        os.replace(tmp, f)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _drop_pyc(f):
    d = f.parent / "__pycache__"
    if d.is_dir():
        for pyc in d.glob(f.stem + ".*.pyc"):
            try:
                pyc.unlink()
            except OSError:
                pass


def _py_compile(f):
    fd, cf = tempfile.mkstemp(suffix=".pyc")
    os.close(fd)
    try:
        py_compile.compile(str(f), cfile=cf, doraise=True)
        return {"rc": 0, "err": ""}
    except py_compile.PyCompileError as exc:
        return {"rc": 1, "err": str(exc)[-600:]}
    finally:
        try:
            os.unlink(cf)
        except OSError:
            pass


def _transform(data, forward):
    """The patched (forward) or original (reverse) bytes, sha-checked; raises on any mismatch."""
    text = data.decode("utf-8")
    if forward:
        out = apply_unified(text, DUCK_PATCH).encode("utf-8")  # noqa: F821 - from APPLIER_SRC
        if sha256_bytes(out) != DUCK_POST:
            raise RuntimeError(f"E28_SCHED_POST_SHA_MISMATCH {DUCK_REL}: {sha256_bytes(out)}")
    else:
        out = apply_unified(text, reverse_unified(DUCK_PATCH)).encode("utf-8")  # noqa: F821
        if sha256_bytes(out) != DUCK_PRE:
            raise RuntimeError(f"E28_SCHED_REVERT_RESULT_MISMATCH {DUCK_REL}: {sha256_bytes(out)}")
    return out


def apply_e28_sched(target):
    """Patch arc2_duck.py. target = the extracted Duck package directory (E12_DIR) or the file path -> the file is
    rewritten atomically, the post sha + marker + py_compile verified; target = the source text -> {"text": patched}.
    ORIGINAL -> patched; FULLY_PATCHED -> no-op ('already_applied'); INVALID_MIXED -> RuntimeError (nothing written)."""
    if _is_text(target):
        st, h = _state_of(target.encode("utf-8"))
        rec = {"mode": "text", "before": st, "sha_before": h, "e28_sched_sha": E28_SCHED_SHA}
        if st == "FULLY_PATCHED":
            rec.update(applied="already_applied", text=target)
        elif st == "ORIGINAL":
            out = _transform(target.encode("utf-8"), True)
            compile(out.decode("utf-8"), DUCK_REL, "exec")
            rec.update(applied="applied", text=out.decode("utf-8"))
        else:
            raise RuntimeError(f"E28_SCHED_PRE_SHA_MISMATCH {DUCK_REL}: {h}")
        rec["after"], rec["sha_after"] = _state_of(rec["text"].encode("utf-8"))
        rec["ok"] = rec["after"] == "FULLY_PATCHED"
        return rec
    f = _file(target)
    data = f.read_bytes() if f.is_file() else b""
    st, h = _state_of(data)
    rec = {"mode": "file", "file": str(f), "before": st, "sha_before": h, "e28_sched_sha": E28_SCHED_SHA}
    if st == "FULLY_PATCHED":
        rec["applied"] = "already_applied"
    elif st == "ORIGINAL":
        _write_atomic(f, _transform(data, True))
        rec["applied"] = "applied"
    else:
        raise RuntimeError(f"E28_SCHED_PRE_SHA_MISMATCH {f}: {h}")
    _drop_pyc(f)
    rec["after"], rec["sha_after"] = _state_of(f.read_bytes())
    rec["py_compile"] = _py_compile(f)
    if rec["after"] != "FULLY_PATCHED" or rec["py_compile"]["rc"]:
        raise RuntimeError("E28_SCHED_VERIFY_FAILED: " + json.dumps({k: rec.get(k) for k in ("after", "sha_after", "py_compile")}))
    rec["ok"] = True
    return rec


def revert_e28_sched(target):
    """Back to the exact original arc2_duck.py (reverse patch, sha == pre). FULLY_PATCHED -> reverted; ORIGINAL ->
    no-op ('not_applied'); INVALID_MIXED -> RuntimeError (nothing written). Text in -> {"text": original}."""
    if _is_text(target):
        st, h = _state_of(target.encode("utf-8"))
        rec = {"mode": "text", "before": st, "sha_before": h}
        if st == "ORIGINAL":
            rec.update(reverted="not_applied", text=target)
        elif st == "FULLY_PATCHED":
            rec.update(reverted="reverted", text=_transform(target.encode("utf-8"), False).decode("utf-8"))
        else:
            raise RuntimeError(f"E28_SCHED_REVERT_SHA_MISMATCH {DUCK_REL}: {h}")
        rec["after"], rec["sha_after"] = _state_of(rec["text"].encode("utf-8"))
        rec["ok"] = rec["after"] == "ORIGINAL"
        return rec
    f = _file(target)
    data = f.read_bytes() if f.is_file() else b""
    st, h = _state_of(data)
    rec = {"mode": "file", "file": str(f), "before": st, "sha_before": h}
    if st == "ORIGINAL":
        rec["reverted"] = "not_applied"
    elif st == "FULLY_PATCHED":
        _write_atomic(f, _transform(data, False))
        rec["reverted"] = "reverted"
    else:
        raise RuntimeError(f"E28_SCHED_REVERT_SHA_MISMATCH {f}: {h}")
    _drop_pyc(f)
    rec["after"], rec["sha_after"] = _state_of(f.read_bytes())
    rec["py_compile"] = _py_compile(f)
    if rec["after"] != "ORIGINAL" or rec["py_compile"]["rc"]:
        raise RuntimeError("E28_SCHED_REVERT_VERIFY_FAILED: " + json.dumps({k: rec.get(k) for k in ("after", "sha_after", "py_compile")}))
    rec["ok"] = True
    return rec


def patched_text(original_text):
    """The patched arc2_duck.py text from the original text (build-time helper, e.g. to embed or inspect)."""
    return apply_e28_sched(original_text)["text"]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("apply", "verify", "revert", "state"):
        fn = {"apply": apply_e28_sched, "verify": verify_e28_sched, "revert": revert_e28_sched, "state": tree_state}[sys.argv[1]]
        print(json.dumps(fn(sys.argv[2]), default=str, indent=1))
    else:
        print(json.dumps({"E28_SCHED_SHA": E28_SCHED_SHA, "ENV_NAMES": ENV_NAMES, "server_env": server_env()}, indent=1))
