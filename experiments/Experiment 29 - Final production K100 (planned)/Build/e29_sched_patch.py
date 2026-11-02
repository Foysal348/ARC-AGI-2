"""E29 SCHED: "attempts supply" (scheduler S-1) on top of the proven E28 A -> B -> C attempt scheduler (Experiment 29).

API (what the notebook calls; names and semantics fixed 2026-11-01)
--------------------------------------------------------------------
    import e29_sched_patch as E29S       # e28_sched_patch.py sits in the same directory (imported here as E28S)

    # CELL 3, right after the payload manifest check, before `import arc2_duck` (replaces the E28 calls):
    rec = E29S.apply_e29_sched(E12_DIR)            # ORIGINAL -> E28 diff + E29 diff; E28_PATCHED -> E29 diff;
                                                   # FULLY_PATCHED -> no-op (rec["applied"] == "already_applied");
                                                   # anything else -> RuntimeError("E29_SCHED_PRE_SHA_MISMATCH ..."),
                                                   # nothing written; result not the post sha -> E29_SCHED_VERIFY_FAILED
    ok, why = E29S.verify_e29_sched(E12_DIR)       # ok only when FULLY_PATCHED (E29 post sha + marker)
    os.environ.update(E29S.server_env(supply=True))   # E28_SCHED=1, E28_T1, E28_B2_TOKENS, E28_B2_WINDOW, E29_SUPPLY=1
    import arc2_duck as E12A
    # rerun (all test tasks, the original scheduler):
    rec = E29S.revert_e29_sched(E12_DIR)           # FULLY_PATCHED or E28_PATCHED -> the exact ORIGINAL bytes
    E29S.tree_state(E12_DIR)                       # "ORIGINAL" | "E28_PATCHED" | "FULLY_PATCHED" | "INVALID_MIXED"
    E29S.ENV_NAMES                                 # E28S.ENV_NAMES + ("E29_SUPPLY", "E29_SUPPLY_MAX_ATTEMPTS")
    E29S.E29_SCHED_SHA                             # pre / e28_post / post / patch / e28_patch / applier shas
    E29S.NOTEBOOK_EDITS, E29S.apply_notebook_edits # = E28S's N1-N4, unchanged
    E29S.T1, E29S.B2_TOKENS, E29S.MAX_MODEL_LEN, E29S.B2_WINDOW   # = E28S (104000, 120000, 131072, 130560)
    target = a directory holding arc2_duck.py (E12_DIR), the file path, or the source text (text mode -> rec["text"]).
    rec keys: mode, file, before, sha_before, applied, e28_step, after, sha_after, py_compile, ok, e28_sched_sha,
    e29_sched_sha (+ text in text mode). e28_step = "applied" (from ORIGINAL) | "already_applied".

What it does
------------
The module carries a SECOND strict unified diff (E29_PATCH) from the E28 post text of arc2_duck.py (sha 35b8d857...)
to the E29 post text; the E28 diff (E28S.DUCK_PATCH) and the strict applier are reused from e28_sched_patch.py.
Runtime, only when E28_SCHED=1 AND E29_SUPPLY=1 (read at the solve call / at every submission build):
  - _e28_solve_all: when the A, B and C queues are ALL empty and left >= min_admit_sec (the existing guard), a free
    worker takes a supply item instead of waiting:
      S1 = a task whose phase-A attempt ended (classified, not TIME), no TRAIN_PASS among its ended attempts, not all
           of its ended attempts cap-cut (kind B2) -> a fresh independent attempt, budget E28_T1, the normal window;
      S2 = only when no S1 candidate exists: a task without TRAIN_PASS whose ended attempts were ALL cap-cut ->
           E28_B2_TOKENS, E28_B2_WINDOW, the B2 request timeout (as phase B2).
    Priority: fewest attempts started for the task (running ones count), then task order; one task may run several
    supply attempts at once (only when every other eligible task has at least as many started). A task stops getting
    supply at its first TRAIN_PASS or at E29_SUPPLY_MAX_ATTEMPTS (default 16) attempts (A + B + C + supply: the
    attempts started, running ones too, plus its phase-C attempt while that is still to come).
    Branches 3, 4, 5, ... per task; plan = phase S1 / S2, branch, budget, window, timeout, s_index (1, 2, ... per
    task), s_reason, view. The fit hook (E28_TRY_RESERVE) is used as for A / B / C (slots mode: always fits); the KV
    gate, the forced head admit with nothing running and the deadline guard are unchanged. A / B / C items keep
    priority: every admission check takes them first. route() records an S attempt's class and queues nothing.
    Waits with supply on but nothing eligible are counted as idle_sec_by_reason["no_supply"] (not "no_runnable").
  - Answers (select_attempts_e28 -> select_attempts_e29 with E29_SUPPLY=1; the E15B writer is unchanged):
    attempt_1 = the best-train candidate among A, B, then S attempts in branch order (replace only when full vs not
    full, or neither full and a strictly higher train fraction; ties keep the earlier) - identical to the E28 rule
    without S records; displaced fallback = the best of the remaining A / B / S candidates by (full, fraction,
    earlier); attempt_2 = C's candidate when it exists and differs, EXCEPT when C is not full and the displaced is
    full (then the displaced first); then the select_attempts_r5 lists; then the test input. Slot sources "S<branch>".
  - Report: SCHED_LAST["e28"]["supply"] = {enabled, max_attempts, started {S1, S2}, done {S1, S2}, first_start_sec,
    by_task {task: supply attempts}, attempts_by_task {task: all attempts started}, train_pass_from_supply [tasks],
    slots_from_supply, s_class}; rows carry phase S1 / S2, s_index, s_reason, view_plan; answer_slots map S rows.
E29_SUPPLY unset: every code path runs as the E28 post file (tested: same records / slots / SCHED_LAST["e28"] minus
timing fields on E28 post vs E29 post).

Duck-code findings (arc2_duck.py b9f67010) for branches >= 3
-------------------------------------------------------------
  - Budget / window / prompt budget: Arc2ToolAgent.__init__ reads the plan via _e28_plan_for(task_id, branch) from
    E28_PLANS[(task, branch)] (any branch number); without a plan b_attempt_budget(branch >= 1) would give the
    attempt-2 budget, so supply attempts always get their plan (S1 = T1, S2 = 120,000 + window 130,560).
  - The only other branch-dependent agent behaviour is the view: the first request cap uses "first_b1" (24,576) for
    any view other than "facts" (view-based, not branch-based); run dirs are "<task>_b<branch>"; record replacement
    is per (task, branch); select_attempts_r5 takes each branch's best and breaks ties on the lowest branch. No
    `branch == 1` special case affects a branch >= 3 attempt.
  - View choice: odd supply index -> b1_view, even -> "facts" (the notebooks pass b1_view "facts", E15B #3, so in
    production every supply attempt uses "facts"; the alternation only matters if b1_view changes).
  - Dispatch priority: supply attempts get queue positions 300000 + n (the engine priority gate ranks them after A /
    B / C, whose positions are 0+, 100000+, 200000+).
  - Notebook side (not patched here): N1 / N2 / N4 size the reservation from plan["budget"] (S2 = the B2 size); the
    CELL 7 report counts branch >= 2 as "extra" attempts (report only); use SCHED_LAST["e28"]["attempts"].
"""
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    _HERE = str(Path(__file__).resolve().parent)
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
except NameError:          # exec'd without __file__: e28_sched_patch must already be importable
    pass
import e28_sched_patch as E28S  # noqa: E402

DUCK_REL = E28S.DUCK_REL
DUCK_PRE = E28S.DUCK_PRE                       # b9f67010... (payload 9e4b8928 manifest)
E28_POST = E28S.DUCK_POST                      # 35b8d857...
E28_MARKER = E28S.DUCK_MARKER
E29_POST = "d8603109f9530896e9e4a56e52806202fc2cd4c58bd664d9a25222fe6f7592c2"
E29_MARKER = "# E29 SUPPLY: an env-gated (E28_SCHED=1 + E29_SUPPLY=1) attempts supply (S1 / S2) on the E28 scheduler."
DUCK_POST, DUCK_MARKER = E29_POST, E29_MARKER
ENV_SUPPLY = "E29_SUPPLY"
ENV_SUPPLY_MAX = "E29_SUPPLY_MAX_ATTEMPTS"     # default 16 attempts per task in total (A + B + C + supply)
SUPPLY_MAX_ATTEMPTS = 16
ENV_NAMES = tuple(E28S.ENV_NAMES) + (ENV_SUPPLY, ENV_SUPPLY_MAX)
T1, B2_TOKENS, MAX_MODEL_LEN, B2_WINDOW = E28S.T1, E28S.B2_TOKENS, E28S.MAX_MODEL_LEN, E28S.B2_WINDOW
NOTEBOOK_EDITS = E28S.NOTEBOOK_EDITS
apply_notebook_edits = E28S.apply_notebook_edits
apply_unified = E28S.apply_unified
reverse_unified = E28S.reverse_unified
E28_PATCH = E28S.DUCK_PATCH

E29_PATCH = r'''# E29 SCHED onto arc2_duck.py (the E28 SCHED post text; Duck payload q27x_e15b_duck_9e4b89285136.bin).
# pre-patch sha256  35b8d8577ab05be60e4e66f80400e57ccec3005dfad9b5290d9fa6f53adee57f (E28 post)
# post-patch sha256 d8603109f9530896e9e4a56e52806202fc2cd4c58bd664d9a25222fe6f7592c2
--- a/arc2_duck.py
+++ b/arc2_duck.py
@@ -3860,6 +3860,8 @@
     attempt_2 fallback); neither full -> the higher train match (a tie keeps A). attempt_2 = phase C's answer when it
     exists and differs, else the displaced attempt-1 answer, else the next distinct select_attempts_r5 candidate, else
     the test input."""
+    if _e29_on():      # E29 SUPPLY: the A / B / S best-train rule (S attempts = branches >= 3)
+        return select_attempts_e29(task, branch_records, _slots_out)
     lists: list = []
     select_attempts_r5(task, branch_records, _e28_lists=lists)
     by = {}
@@ -3956,15 +3958,17 @@
     meta: dict = {}                     # (task, branch) -> admission wait bookkeeping of a queued B / C item
     active: set = set()
     records: dict[str, list[dict]] = {t: [] for t in order}
+    sup = _e29_supply_state(e28) if _e29_on() else None   # E29 SUPPLY: attempts supply (None = off, = E28)
 
     def _sec() -> float:
         return round(time.monotonic() - started, 1)
 
     def plan(tid: str, phase: str, **kw) -> dict:
-        p = {"task": tid, "branch": E28_BRANCH[phase], "phase": phase,
-             "budget": cfg["b2_tokens"] if phase == "B2" else cfg["t1"],
-             "window": cfg["b2_window"] if phase == "B2" else None,
-             "timeout": b2_timeout if phase == "B2" else timeout}
+        b2x = phase in ("B2", "S2")      # E29 SUPPLY: an S2 attempt is planned like B2; an S branch comes in kw
+        p = {"task": tid, "branch": E28_BRANCH.get(phase), "phase": phase,
+             "budget": cfg["b2_tokens"] if b2x else cfg["t1"],
+             "window": cfg["b2_window"] if b2x else None,
+             "timeout": b2_timeout if b2x else timeout}
         p.update(kw)
         return p
 
@@ -4064,11 +4068,36 @@
             return take(where, entry, tid)
         return None
 
+    def supply_plan(tid: str, ph: str, why: str) -> dict:
+        """E29 SUPPLY: the plan of this task's next supply attempt (branch 3, 4, ...; not committed until taken)."""
+        k = sup["s_n"].get(tid, 0) + 1
+        return plan(tid, ph, branch=sup["next_b"].get(tid, 3), s_index=k, s_reason=why,
+                    view=(b1_view if k % 2 == 1 else "facts"))
+
+    def supply_pick(force: bool):
+        """E29 SUPPLY, under the lock, only when the A / B / C queues are empty: the first eligible supply attempt
+        (S1 before S2; fewest attempts started, then task order) that fits (the same fit hook as pick); nothing fits
+        and force (nothing runs) = the head anyway."""
+        e28["admission_checks"] += 1
+        cands = _e29_candidates(sup, order, e28["a_class"])
+        failed_budgets = set()
+        for ph_, tid, why in cands:
+            pp = supply_plan(tid, ph_, why)
+            if pp["budget"] not in failed_budgets and _fits(tid, pp):
+                return _e29_take(sup, e28, tid, pp)
+            failed_budgets.add(pp["budget"])
+        if force and cands:
+            e28["forced_admits"] += 1
+            ph_, tid, why = cands[0]
+            return _e29_take(sup, e28, tid, supply_plan(tid, ph_, why))
+        return None
+
     def sess(tid: str, p: dict, qpos: int) -> BranchSession:
         b = int(p["branch"])
         if (tid, b) not in sessions:
             E28_PLANS[(tid, b)] = dict(p)
-            s_ = BranchSession(agent_cls, task_id=tid, task=tasks[tid], branch=b, view=("facts" if b == 0 else b1_view),
+            s_ = BranchSession(agent_cls, task_id=tid, task=tasks[tid], branch=b,
+                               view=(p.get("view") or ("facts" if b == 0 else b1_view)),   # E29 SUPPLY: S view
                                queue_position=qpos, out_dir=out_dir, deadline=deadline, base_url=base_url,
                                model_id=model_id, save_request_logs=save_request_logs, analyzer_timeout=p["timeout"],
                                started_at=started, max_generated_tokens=int(p["budget"]))
@@ -4089,6 +4118,8 @@
              "start_sec": t_start, "end_sec": t_end, "status": rec.get("status"),
              "queues_at_start": qlen,           # [A, B, C] queue lengths right after this item was taken
              "start_seq": start_seq}            # the admission order over the whole run (1, 2, ...)
+        if p.get("phase") in ("S1", "S2"):     # E29 SUPPLY
+            r.update(s_index=p.get("s_index"), s_reason=p.get("s_reason"), view_plan=p.get("view"))
         m = meta.get((s_.task_id, s_.branch))
         if m is not None:                  # B / C items: admission wait from runnable (first tried) to start
             r.update(queued_sec=m["queued_sec"], first_tried_sec=m["first_tried_sec"], fit_fails=m["fit_fails"],
@@ -4176,6 +4207,8 @@
         """Under the lock: queue the next phase of a task from the attempt that just ended."""
         ph = p["phase"]
         full = cls["kind"] == "TRAIN_PASS"
+        if sup is not None:                    # E29 SUPPLY: the task's ended attempt kinds (TRAIN_PASS stops supply)
+            _e29_note_end(sup, tid, ph, cls)
         if ph == "A":
             e28["a_class"][tid] = {k: cls.get(k) for k in ("kind", "end_reason", "tokens")}
             if full:
@@ -4186,6 +4219,8 @@
         elif ph in ("B1", "B2"):
             e28["b_class"][tid] = {k: cls.get(k) for k in ("kind", "end_reason", "tokens")}
             push_c(tid, full, cls["tokens"], ph)
+        elif ph in ("S1", "S2") and sup is not None:   # E29 SUPPLY: recorded, nothing queued
+            sup["s_class"].setdefault(tid, []).append({k: cls.get(k) for k in ("kind", "end_reason", "tokens")})
 
     def worker(wi: int):
         while True:
@@ -4207,7 +4242,9 @@
                     if left < min_admit_sec or state["stop"]:
                         return
                     has = bool(qa or qb or qc)
-                    if has and kv_blocked():
+                    sup_has = (not has and sup is not None          # E29 SUPPLY: queues empty, supply eligible
+                               and bool(_e29_candidates(sup, order, e28["a_class"])))
+                    if (has or sup_has) and kv_blocked():
                         if state["running"] == 0:     # nothing runs, so nothing can free KV: never deadlock on the gate
                             state["kv_idle_overrides"] += 1
                         else:
@@ -4221,8 +4258,9 @@
                             cond.wait(timeout=min(5.0, max(0.5, float(kv_poll_sec))))
                             e28["idle"]["kv_gate"] += time.monotonic() - _w0
                             continue
-                    if has:
-                        got = pick(force=state["running"] == 0)
+                    if has or sup_has:
+                        got = (pick(force=state["running"] == 0) if has
+                               else supply_pick(force=state["running"] == 0))      # E29 SUPPLY
                         if got is None:               # nothing fits at all: wait for the next admission event
                             _w0 = time.monotonic()
                             cond.wait(timeout=1.0)
@@ -4230,6 +4268,8 @@
                             continue
                         hold_end(time.monotonic())
                         tid, p, qpos = got
+                        if sup is not None:            # E29 SUPPLY: attempts started per task (running ones count)
+                            _e29_note_take(sup, tid, p)
                         qlen = [len(qa), len(qb), len(qc)]
                         e28["pop_seq"] += 1
                         start_seq = e28["pop_seq"]
@@ -4267,7 +4307,7 @@
                         return
                     _w0 = time.monotonic()            # no queued item: B / C items appear when running attempts end
                     cond.wait(timeout=5)
-                    e28["idle"]["no_runnable"] += time.monotonic() - _w0
+                    e28["idle"]["no_supply" if sup is not None else "no_runnable"] += time.monotonic() - _w0   # E29
             t0 = time.monotonic()
             tag = {"stage": {"A": 1, "C": 2}.get(ph, 3), "e28_phase": ph}
             try:
@@ -4353,8 +4393,8 @@
         sl = slots.get(r.get("task")) if isinstance(slots, dict) else None
         if isinstance(sl, list):
             src = {"A": 0, "C": 1, "B": 2}
-            r["answer_slots"] = [("attempt_1" if src.get(x.get("attempt_1")) == r.get("branch") else
-                                  "attempt_2" if src.get(x.get("attempt_2")) == r.get("branch") else None)
+            r["answer_slots"] = [("attempt_1" if _e29_src_branch(src, x.get("attempt_1")) == r.get("branch") else
+                                  "attempt_2" if _e29_src_branch(src, x.get("attempt_2")) == r.get("branch") else None)
                                  for x in sl if isinstance(x, dict)]
         rows_.append(r)
     SCHED_LAST.clear()
@@ -4395,8 +4435,171 @@
                            "a_class": e28["a_class"],
                            "b_class": e28["b_class"], "setup_failed": e28["setup_failed"], "attempts": rows_,
                            "slots": slots,
+                           **({"supply": _e29_report(sup, e28, rows_, slots, order)} if sup is not None else {}),
                            "row_fields": ["task", "branch", "phase", "b_class", "b_reason", "c_priority", "c_key_tokens",
                                           "budget", "window", "tokens", "best_train", "end_reason", "kind",
                                           "worker_slot", "start_seq", "queues_at_start", "wait_sec",
                                           "backfill_starts_while_waiting", "answer_slots"]})
     return records
+
+
+# ==================================================================== E29 SUPPLY (Experiment 29, scheduler S-1)
+# E29 SUPPLY: an env-gated (E28_SCHED=1 + E29_SUPPLY=1) attempts supply (S1 / S2) on the E28 scheduler.
+# Off (E29_SUPPLY unset / not "1", or E28_SCHED off) = the E28 code above runs exactly as before. On: when the A, B and
+# C queues are all empty and the admission window is open (left >= min_admit_sec), a free worker starts a supply
+# attempt instead of idling (K100Q: 93 % of the throughput gap was work exhaustion at the end):
+#   S1 = a task whose attempt 1 ended (classified, not TIME), no TRAIN_PASS among its ended attempts, not all of them
+#        cap-cut (B2) -> a fresh independent attempt, E28_T1, the normal window;
+#   S2 = (only when no S1 candidate exists) a task without TRAIN_PASS whose ended attempts were ALL cap-cut ->
+#        E28_B2_TOKENS, E28_B2_WINDOW, the B2 request timeout.
+# Priority: fewest attempts started for the task (running ones count), then task order; the same task may run several
+# supply attempts at once when no other eligible task has fewer. A task stops getting supply at its first TRAIN_PASS or
+# at E29_SUPPLY_MAX_ATTEMPTS attempts (A + B + C + supply; a C still to come counts). Branches 3, 4, ... (next unused per task); view:
+# odd supply index -> b1_view, even -> "facts". A / B / C items always come first (checked at every admission).
+# Answers (select_attempts_e29): attempt_1 = best train among A, B, then S by branch (full beats non-full; else a
+# strictly higher train fraction; ties keep the earlier); attempt_2 = C (E28 rule) unless C is not full and the
+# displaced best of the rest is full (then the displaced first); then the r5 lists; then the test input.
+E29_ENV = "E29_SUPPLY"
+E29_ENV_MAX = "E29_SUPPLY_MAX_ATTEMPTS"
+E29_S_QPOS = 300000          # dispatch index base of supply attempts (after A 0+, B 100000+, C 200000+)
+
+
+def _e29_on() -> bool:
+    return _e28_on() and os.environ.get(E29_ENV, "").strip() == "1"
+
+
+def _e29_supply_state(e28: dict) -> dict:
+    """The supply bookkeeping; adds the S1 / S2 counters (started / done / queued) and the no_supply idle reason."""
+    for k in ("started", "done", "queued"):
+        e28[k].update(S1=0, S2=0)
+    e28["idle"]["no_supply"] = 0.0
+    return {"max": max(1, _env_int(E29_ENV_MAX, 16)), "n": {}, "ended": {}, "pass": set(), "s_n": {}, "next_b": {},
+            "s_class": {}, "seq": 0, "c": set()}
+
+
+def _e29_candidates(sup: dict, order: list, a_class: dict) -> list:
+    """[(phase, task, reason)] in priority order: every S1 candidate, or (none) every S2 candidate. The cap counts the
+    attempts started (running ones too) plus the task's phase-C attempt while it is still to come (E28 always queues
+    it), so A + B + C + supply never exceeds E29_SUPPLY_MAX_ATTEMPTS (for a cap >= 3)."""
+    s1, s2 = [], []
+    for i, t in enumerate(order):
+        n = sup["n"].get(t, 0)
+        if t in sup["pass"] or n + (0 if t in sup["c"] else 1) >= sup["max"]:
+            continue
+        a = a_class.get(t)
+        ended = sup["ended"].get(t) or []
+        if not a or a.get("kind") in (None, "TIME") or not ended:
+            continue                    # attempt 1 not ended yet, or cut by the solve window (no re-run)
+        (s2 if all(k == "B2" for k in ended) else s1).append((n, i, t))
+    if s1:
+        return [("S1", t, "no_train_pass") for _, _, t in sorted(s1)]
+    return [("S2", t, "all_cap_cut") for _, _, t in sorted(s2)]
+
+
+def _e29_take(sup: dict, e28: dict, tid: str, p: dict):
+    """Commit a supply attempt chosen by supply_pick: its branch, its supply index; -> (task, plan, queue position)."""
+    sup["next_b"][tid] = int(p["branch"]) + 1
+    sup["s_n"][tid] = int(p["s_index"])
+    sup["seq"] += 1
+    e28["queued"][p["phase"]] += 1
+    return tid, p, E29_S_QPOS + sup["seq"]
+
+
+def _e29_note_take(sup: dict, tid: str, p: dict):
+    sup["n"][tid] = sup["n"].get(tid, 0) + 1
+    if p.get("phase") == "C":
+        sup["c"].add(tid)
+
+
+def _e29_note_end(sup: dict, tid: str, ph: str, cls: dict):
+    kind = cls.get("kind")
+    sup["ended"].setdefault(tid, []).append(kind)
+    if kind == "TRAIN_PASS":
+        sup["pass"].add(tid)
+
+
+def _e29_src_branch(src: dict, s):
+    """The branch of an answer-slot source: A / C / B (E28) or S<branch> (supply)."""
+    if s in src:
+        return src[s]
+    if isinstance(s, str) and s[:1] == "S" and s[1:].isdigit():
+        return int(s[1:])
+    return None
+
+
+def _e29_report(sup: dict, e28: dict, rows: list, slots, order: list) -> dict:
+    firsts = [e28["first_start_sec"][k] for k in ("S1", "S2") if k in e28["first_start_sec"]]
+    tp = {r.get("task") for r in rows if str(r.get("phase") or "").startswith("S") and r.get("kind") == "TRAIN_PASS"}
+    n_slots = 0
+    if isinstance(slots, dict):
+        for v in slots.values():
+            for x in (v if isinstance(v, list) else []):
+                if isinstance(x, dict):
+                    n_slots += sum(1 for k in ("attempt_1", "attempt_2") if _e29_src_branch({}, x.get(k)) is not None)
+    return {"enabled": True, "max_attempts": sup["max"],
+            "started": {"S1": e28["started"]["S1"], "S2": e28["started"]["S2"]},
+            "done": {"S1": e28["done"]["S1"], "S2": e28["done"]["S2"]},
+            "first_start_sec": min(firsts) if firsts else None,
+            "by_task": {t: sup["s_n"][t] for t in order if t in sup["s_n"]},
+            "attempts_by_task": {t: sup["n"][t] for t in order if t in sup["n"]},
+            "train_pass_from_supply": [t for t in order if t in tp],
+            "slots_from_supply": n_slots, "s_class": sup["s_class"]}
+
+
+def _e29_phase_of(rec: dict) -> str:
+    p = str((rec.get("e28") or {}).get("phase") or "")
+    if p[:1] == "S":
+        return "S"
+    if p:
+        return "B" if p.startswith("B") else p
+    b = int(rec.get("branch") or 0)
+    return {0: "A", 1: "C", 2: "B"}.get(b, "S" if b >= 3 else "X")
+
+
+def select_attempts_e29(task: dict, branch_records: list[dict], _slots_out: list | None = None) -> list[dict]:
+    """E29 answer slots (same output format as select_attempts_r5 / select_attempts_e28). attempt_1 = the best-train
+    candidate among A, B, then the S attempts in branch order: a candidate replaces the current best only if it is full
+    and the best is not, or neither is full and its train fraction is strictly higher (ties keep the earlier). The
+    displaced fallback = the best of the remaining A / B / S candidates by (full, fraction, earlier). attempt_2 = phase
+    C's candidate when it exists and differs, except when C is not full and the displaced fallback is full (then the
+    displaced first); then the select_attempts_r5 candidates; then the test input. Without S records = the E28 rule."""
+    lists: list = []
+    select_attempts_r5(task, branch_records, _e28_lists=lists)
+    by, srec = {}, []
+    for r in branch_records:
+        ph = _e29_phase_of(r)
+        if ph == "S":
+            srec.append(r)
+        else:
+            by.setdefault(ph, r)
+    srec.sort(key=lambda r: int(r.get("branch") or 0))
+    out = []
+    for qi, q in enumerate(task["test"]):
+        grids = lists[qi] if qi < len(lists) else [q["input"]]
+        ca, cb, cc = (_e28_cand(by[p], qi) if p in by else None for p in ("A", "B", "C"))
+        cands = [(s, c) for s, c in [("A", ca), ("B", cb)] + [(f"S{int(r.get('branch') or 0)}", _e28_cand(r, qi))
+                                                               for r in srec] if c is not None]
+        best = None
+        for i, (s, c) in enumerate(cands):
+            if best is None or (c["full"] and not best[2]["full"]) or (
+                    not c["full"] and not best[2]["full"] and c["frac"] > best[2]["frac"]):
+                best = (i, s, c)
+        rest = [(i, s, c) for i, (s, c) in enumerate(cands) if best is not None and i != best[0]]
+        disp = max(rest, key=lambda x: (x[2]["full"], x[2]["frac"], -x[0])) if rest else None
+        p1, s1 = (best[2], best[1]) if best is not None else (None, "r5")
+        a1 = p1["grid"] if p1 is not None else grids[0]
+        cl = [(cc["grid"], "C")] if cc is not None else []
+        dl = [(disp[2]["grid"], disp[1])] if disp is not None else []
+        pref = dl + cl if (cc is not None and not cc["full"] and disp is not None and disp[2]["full"]) else cl + dl
+        a2, s2 = next(((g, s) for g, s in pref + [(g, "r5") for g in grids] + [(q["input"], "input")] if g != a1),
+                      (a1, "same"))
+        out.append({"attempt_1": a1, "attempt_2": a2})
+        if _slots_out is not None:
+            row = {"qi": qi, "attempt_1": s1, "attempt_2": s2,
+                   "a1_train": None if p1 is None else {k: p1[k] for k in ("full", "k", "n")},
+                   "a_full": None if ca is None else ca["full"], "b_full": None if cb is None else cb["full"],
+                   "c_present": cc is not None}
+            if srec:
+                row["s_full"] = {s: c["full"] for s, c in cands if s[:1] == "S"}
+            _slots_out.append(row)
+    return out
'''

PATCHES = {DUCK_REL: (E28_POST, E29_POST, E29_MARKER, E29_PATCH)}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(t):
    return sha256_bytes(t.encode("utf-8"))


E29_SCHED_SHA = {"pre": DUCK_PRE, "e28_post": E28_POST, "post": E29_POST, "patch": sha256_text(E29_PATCH),
                 "e28_patch": sha256_text(E28_PATCH), "applier": E28S.E28_SCHED_SHA["applier"],
                 DUCK_REL: {"pre": DUCK_PRE, "e28_post": E28_POST, "post": E29_POST}}


# ------------------------------------------------------------------ env
def server_env(supply=True):
    """The env of the NOTEBOOK process (Duck reads it at call time; nothing goes to the vLLM server). supply=True =
    E28S.server_env() + E29_SUPPLY=1 (E29_SUPPLY_MAX_ATTEMPTS unset = 16); supply=False = exactly E28S.server_env()."""
    env = dict(E28S.server_env())
    if supply:
        env[ENV_SUPPLY] = "1"
    return env


# ------------------------------------------------------------------ states
def _is_text(target):
    return isinstance(target, str) and "\n" in target


def _file(target):
    p = Path(target)
    return p / DUCK_REL if p.is_dir() else p


def _state_of(data):
    h = sha256_bytes(data)
    if h == E29_POST and E29_MARKER.encode() in data:
        return "FULLY_PATCHED", h
    if h == E28_POST and E28_MARKER.encode() in data:
        return "E28_PATCHED", h
    if h == DUCK_PRE:
        return "ORIGINAL", h
    return "INVALID_MIXED", h


def tree_state(target):
    """ORIGINAL (pre sha) | E28_PATCHED (E28 post sha + E28 marker) | FULLY_PATCHED (E29 post sha + E29 marker) |
    INVALID_MIXED (anything else, a missing file too). target = a directory holding arc2_duck.py, the file, or its text."""
    if _is_text(target):
        return _state_of(target.encode("utf-8"))[0]
    f = _file(target)
    if not f.is_file():
        return "INVALID_MIXED"
    return _state_of(f.read_bytes())[0]


def verify_e29_sched(target):
    """(ok, why): ok only when FULLY_PATCHED; why = 'STATE sha256=... (pre ..., e28 ..., post ...)'."""
    if _is_text(target):
        st, h = _state_of(target.encode("utf-8"))
    else:
        f = _file(target)
        if not f.is_file():
            return False, f"INVALID_MIXED missing {f}"
        st, h = _state_of(f.read_bytes())
    return st == "FULLY_PATCHED", f"{st} sha256={h} (pre {DUCK_PRE[:12]}, e28 {E28_POST[:12]}, post {E29_POST[:12]})"


def _forward(data, st):
    """ORIGINAL / E28_PATCHED bytes -> the E29 post bytes, every step sha-checked (raises on any mismatch)."""
    text = data.decode("utf-8")
    if st == "ORIGINAL":
        text = apply_unified(text, E28_PATCH)
        if sha256_text(text) != E28_POST:
            raise RuntimeError(f"E29_SCHED_E28_STEP_SHA_MISMATCH {DUCK_REL}: {sha256_text(text)}")
    out = apply_unified(text, E29_PATCH).encode("utf-8")
    if sha256_bytes(out) != E29_POST:
        raise RuntimeError(f"E29_SCHED_POST_SHA_MISMATCH {DUCK_REL}: {sha256_bytes(out)}")
    return out


def _backward(data, st):
    """FULLY_PATCHED / E28_PATCHED bytes -> the exact ORIGINAL bytes, every step sha-checked."""
    text = data.decode("utf-8")
    if st == "FULLY_PATCHED":
        text = apply_unified(text, reverse_unified(E29_PATCH))
        if sha256_text(text) != E28_POST:
            raise RuntimeError(f"E29_SCHED_REVERT_E28_STEP_MISMATCH {DUCK_REL}: {sha256_text(text)}")
    out = apply_unified(text, reverse_unified(E28_PATCH)).encode("utf-8")
    if sha256_bytes(out) != DUCK_PRE:
        raise RuntimeError(f"E29_SCHED_REVERT_RESULT_MISMATCH {DUCK_REL}: {sha256_bytes(out)}")
    return out


def apply_e29_sched(target):
    """Patch arc2_duck.py to the E29 post text. target = E12_DIR or the file path -> atomic rewrite, pyc dropped,
    py_compile + post sha + marker verified (E29_SCHED_VERIFY_FAILED); target = the source text -> {"text": ...}.
    ORIGINAL -> E28 diff then E29 diff; E28_PATCHED -> E29 diff; FULLY_PATCHED -> no-op ('already_applied');
    INVALID_MIXED -> RuntimeError('E29_SCHED_PRE_SHA_MISMATCH ...'), nothing written."""
    if _is_text(target):
        data = target.encode("utf-8")
        st, h = _state_of(data)
        rec = {"mode": "text", "before": st, "sha_before": h, "e28_sched_sha": E28S.E28_SCHED_SHA,
               "e29_sched_sha": E29_SCHED_SHA}
        if st == "FULLY_PATCHED":
            rec.update(applied="already_applied", e28_step="already_applied", text=target)
        elif st in ("ORIGINAL", "E28_PATCHED"):
            out = _forward(data, st).decode("utf-8")
            compile(out, DUCK_REL, "exec")
            rec.update(applied="applied", e28_step="applied" if st == "ORIGINAL" else "already_applied", text=out)
        else:
            raise RuntimeError(f"E29_SCHED_PRE_SHA_MISMATCH {DUCK_REL}: {h}")
        rec["after"], rec["sha_after"] = _state_of(rec["text"].encode("utf-8"))
        rec["ok"] = rec["after"] == "FULLY_PATCHED"
        return rec
    f = _file(target)
    data = f.read_bytes() if f.is_file() else b""
    st, h = _state_of(data)
    rec = {"mode": "file", "file": str(f), "before": st, "sha_before": h, "e28_sched_sha": E28S.E28_SCHED_SHA,
           "e29_sched_sha": E29_SCHED_SHA}
    if st == "FULLY_PATCHED":
        rec.update(applied="already_applied", e28_step="already_applied")
    elif st in ("ORIGINAL", "E28_PATCHED"):
        E28S._write_atomic(f, _forward(data, st))
        rec.update(applied="applied", e28_step="applied" if st == "ORIGINAL" else "already_applied")
    else:
        raise RuntimeError(f"E29_SCHED_PRE_SHA_MISMATCH {f}: {h}")
    E28S._drop_pyc(f)
    rec["after"], rec["sha_after"] = _state_of(f.read_bytes())
    rec["py_compile"] = E28S._py_compile(f)
    if rec["after"] != "FULLY_PATCHED" or rec["py_compile"]["rc"]:
        raise RuntimeError("E29_SCHED_VERIFY_FAILED: " + json.dumps({k: rec.get(k) for k in ("after", "sha_after", "py_compile")}))
    rec["ok"] = True
    return rec


def revert_e29_sched(target):
    """Back to the exact ORIGINAL arc2_duck.py (sha == pre) from FULLY_PATCHED or E28_PATCHED; ORIGINAL -> no-op
    ('not_applied'); INVALID_MIXED -> RuntimeError('E29_SCHED_REVERT_SHA_MISMATCH ...'), nothing written."""
    if _is_text(target):
        data = target.encode("utf-8")
        st, h = _state_of(data)
        rec = {"mode": "text", "before": st, "sha_before": h}
        if st == "ORIGINAL":
            rec.update(reverted="not_applied", text=target)
        elif st in ("FULLY_PATCHED", "E28_PATCHED"):
            rec.update(reverted="reverted", text=_backward(data, st).decode("utf-8"))
        else:
            raise RuntimeError(f"E29_SCHED_REVERT_SHA_MISMATCH {DUCK_REL}: {h}")
        rec["after"], rec["sha_after"] = _state_of(rec["text"].encode("utf-8"))
        rec["ok"] = rec["after"] == "ORIGINAL"
        return rec
    f = _file(target)
    data = f.read_bytes() if f.is_file() else b""
    st, h = _state_of(data)
    rec = {"mode": "file", "file": str(f), "before": st, "sha_before": h}
    if st == "ORIGINAL":
        rec["reverted"] = "not_applied"
    elif st in ("FULLY_PATCHED", "E28_PATCHED"):
        E28S._write_atomic(f, _backward(data, st))
        rec["reverted"] = "reverted"
    else:
        raise RuntimeError(f"E29_SCHED_REVERT_SHA_MISMATCH {f}: {h}")
    E28S._drop_pyc(f)
    rec["after"], rec["sha_after"] = _state_of(f.read_bytes())
    rec["py_compile"] = E28S._py_compile(f)
    if rec["after"] != "ORIGINAL" or rec["py_compile"]["rc"]:
        raise RuntimeError("E29_SCHED_REVERT_VERIFY_FAILED: " + json.dumps({k: rec.get(k) for k in ("after", "sha_after", "py_compile")}))
    rec["ok"] = True
    return rec


def patched_text(original_text):
    """The E29 post text from the ORIGINAL (or E28 post) text (build-time helper)."""
    return apply_e29_sched(original_text)["text"]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] in ("apply", "verify", "revert", "state"):
        fn = {"apply": apply_e29_sched, "verify": verify_e29_sched, "revert": revert_e29_sched, "state": tree_state}[sys.argv[1]]
        print(json.dumps(fn(sys.argv[2]), default=str, indent=1))
    else:
        print(json.dumps({"E29_SCHED_SHA": E29_SCHED_SHA, "ENV_NAMES": ENV_NAMES, "server_env": server_env()}, indent=1))
