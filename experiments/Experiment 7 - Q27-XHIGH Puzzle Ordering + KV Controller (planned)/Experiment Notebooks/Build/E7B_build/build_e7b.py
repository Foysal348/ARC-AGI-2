"""Experiment 7 (implementation B) builder, v3. Claude, 2026-10-12.

Source: source/Q27-XHIGH-v1_4_4_E7_CODEX_ORDER_KV_CONTROLLER.ipynb (GPT implementation B, sha256 d10475e4...),
itself v1.4.4 DIRECT (LB 26.94) + exact Codex ordering + guarded KV admission.

Every replacement must hit exactly once.
v2:
  1. PROXY48 evaluation profile (EDA MAIN_DEBUG_48: 48 tasks / 56 test outputs; 3 h budget, same admit/drain fractions
     as EVAL12). EVAL12 stays available.
  2. Per-variant defaults -> one notebook per experiment arm (Kaggle batch runs cannot set env vars).
  3. Per-unit admit/finish minutes in records (t_admit_min, t_done_min).
  4. Final submission guarantee.
v3 (fixes from the two final GPT audits, 2026-10-12):
  a. Shadow mode counts at most ONE decision per telemetry poll (was: once per scheduler loop -> inflated).
  b. start_phase2() is transactional: host state / counters change only after engine.add_request() succeeds
     (success path token- and seed-identical to v1.4.4).
  c. Final submission validation is attempt-wise (a valid attempt is never discarded because the other one is
     invalid) and runs BEFORE run_report.json / Q27X_BUNDLE.zip, so the bundle and the final file agree;
     report["submission_audit"] records it. A last idempotent re-check stays at the end of the cell.
  d. Matched PROXY48 controls: A0 (original order, cap 16, KV off) and A1 (Codex order, cap 16, KV off).
  e. Hidden-rerun guard: in a competition rerun every E7 feature is OFF (original order, KV off, engine cap 16)
     unless E7_HIDDEN_APPROVED (constant False on Kaggle).
  f. SCHED["not_admitted"] is computed even when the scheduler raised.
v4 (user goal, 2026-10-12): no fixed 16-18 ceiling. Engine cap 24 = user maximum (ladder 24 -> 16); every unit beyond the
  v1.4.4 baseline of 16 is admitted when the LIFETIME KV projection (all active units + the candidate at their
  expected final length, self-calibrated by measured KV) stays under 0.90. Preemption -> cooldown + tighter target.
v5 (two audits of v4, 2026-10-13; cap 24 unchanged): length model learns generated length from NATURAL closes only
  (forced / cut / drained / hard-stopped = full budget), p90 after >= 8 per tier; rapid-growth gate; circuit breaker
  (2 preemptions in 30 min -> extras off, baseline 16 continues); fixed mode no longer bypasses the projection;
  block counters once per poll; important events never dropped; records / submission I/O errors non-fatal; a
  validator error rebuilds the submission from the in-memory answers instead of writing identity.
usage: python build_e7b.py   (writes the notebooks to ../../ = Experiment Notebooks/)
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "source" / "Q27-XHIGH-v1_4_4_E7_CODEX_ORDER_KV_CONTROLLER.ipynb"
SRC_SHA = "d10475e4f6b1ae6f"
OUT_DIR = HERE.parent.parent

PROXY48 = ["0934a4d8", "20a9e565", "4c7dc4dd", "89565ca0", "136b0064", "291dc1e1", "16b78196", "271d71e2",
           "28a6681f", "2b83f449", "2c181942", "31f7f899", "35ab12c3", "3dc255db", "409aa875", "4c416de3",
           "64efde09", "6e453dd6", "7491f3cf", "7b0280bc", "7b80bb43", "7c66cb00", "1818057f", "332f06d7",
           "53fb4810", "7666fa5d", "da515329", "8f3a5a89", "db695cfb", "97d7923e", "de809cff", "c7f57c3e",
           "cb2d8a2c", "58490d8a", "dbff022c", "5961cc34", "80a900e0", "269e22fb", "8b9c3697", "e376de54",
           "b99e7126", "c4d067a0", "135a2760", "4a21e3da", "71e489b6", "800d221b", "dd6b8c4b", "36a08778"]
assert len(PROXY48) == len(set(PROXY48)) == 48

VARIANTS = {
    # User decision (2026-10-12): ONE notebook, ordering + KV controller together, ONE GPU run on PROXY48.
    "Q27-XHIGH-v1_4_4_E7_COMBINED.ipynb":
        dict(profile="PROXY48", cap=24, kv="adaptive", ordering=True,
             banner="**Experiment 7 COMBINED:** PROXY48 evaluation (48 tasks / 56 outputs, 3 h budget), Codex puzzle "
                    "ordering ON, engine cap **24** (user maximum; fallback 16). **The KV controller decides how many units run**: "
                    "the first 16 are admitted as in v1.4.4; every further unit is admitted only if the projected KV usage "
                    "with ALL running units and the new one at their expected final length stays under 0.90 (expected "
                    "final = worst case until 8 naturally finished units of the same tier teach it, p90; forced/cut units "
                    "count as the full budget), and only if the current KV growth would not reach the ceiling within 5 min. "
                    "Preemption -> cooldown + tighter target; 2 preemptions in 30 min -> no more extras (16 continue). Reports: KV timeline, admission decisions with reasons, per-unit admit/finish "
                    "minutes and tier."),
}

HIDDEN_NOTE = ("**Hidden rerun:** `E7_HIDDEN_APPROVED = True` (user decision 2026-10-12) -> the competition rerun uses the "
               "SAME Codex ordering + KV controller as this evaluation run. Set it to False for pure v1.4.4 scheduling.")

CELL8_PREFIX = r'''# ==================================================================== E7 v3 final submission validation (runs FIRST)
# Attempt-wise: a valid attempt is never discarded because the other attempt (or another test input) is invalid.
# Runs BEFORE run_report.json / Q27X_BUNDLE.zip so the bundle carries the final file; re-checked at the end.
def ensure_final_submission():
    try:
        _sub = _load_json(SUBMISSION_PATH)
    except Exception:
        _sub = {}
    if not isinstance(_sub, dict):
        _sub = {}
    fixed, repaired, answered, n_out = {}, 0, 0, 0
    for _tid, _task in test_challenges.items():
        _rows = _sub.get(_tid)
        _rows = _rows if isinstance(_rows, list) else []
        out_rows = []
        for _qi, _q in enumerate(_task["test"]):
            _old = _rows[_qi] if _qi < len(_rows) and isinstance(_rows[_qi], dict) else {}
            _row = {}
            for _key in ("attempt_1", "attempt_2"):
                _cand = _old.get(_key)
                if valid_grid(_cand):
                    _row[_key] = _cand
                else:
                    _row[_key] = _q["input"]
                    repaired += 1
            answered += int(_row["attempt_1"] != _q["input"])
            out_rows.append(_row)
            n_out += 1
        fixed[_tid] = out_rows
    unknown = len([k for k in _sub if k not in test_challenges])
    write_json_atomic(SUBMISSION_PATH, fixed)
    audit = {"tasks": len(fixed), "test_outputs": n_out, "answered_by_model": answered,
             "repaired_attempts": repaired, "unknown_keys_dropped": unknown}
    print("SUBMISSION OK:", SUBMISSION_PATH, json.dumps(audit), flush=True)
    return audit

def _submission_fallback():
    # v5 (audit): never overwrite valid answers with identity; rebuild from the in-memory answers (FINAL) first
    try:
        write_submission()
        return "rebuilt_from_final"
    except BaseException:
        print("SUBMISSION REBUILD ERROR:", traceback.format_exc()[-800:])
        write_json_atomic(SUBMISSION_PATH, identity_submission(test_challenges))
        return "identity_written"

try:
    SUBMISSION_AUDIT = ensure_final_submission()
except BaseException:
    print("SUBMISSION VALIDATION ERROR:", traceback.format_exc()[-1500:])
    SUBMISSION_AUDIT = {"error": "validation_failed", "fallback": _submission_fallback()}

'''

CELL8_SUFFIX = r'''
# ==================================================================== E7 v3 final re-check (idempotent)
try:
    _final_audit = ensure_final_submission()
    if isinstance(SUBMISSION_AUDIT, dict) and _final_audit.get("repaired_attempts", 0) not in (0,):
        print("NOTE: the final re-check repaired attempts after the bundle was written:", _final_audit)
except BaseException:
    print("SUBMISSION GUARANTEE ERROR:", traceback.format_exc()[-1500:])
    print("fallback:", _submission_fallback())
'''

OLD_P2 = '''    closed = THINK_CLOSE_ID in st["p1_ids"]
    if not closed:                                  # v1.1 (F8): an EOS before </think> must not precede the stop text
        while st["p1_ids"] and st["p1_ids"][-1] in EOS_IDS:
            st["p1_ids"].pop()
    prefix = list(st["prompt_ids"]) + list(st["p1_ids"]) + ([] if closed else list(STOP_IDS))
    room = MAX_MODEL_LEN - len(prefix) - 8
    cap = min(int(answer_cap), room)
    if cap < 64:
        return False
    st["phase"] = 2; st["p2_prefix"] = [] if closed else list(STOP_IDS); st["mode"] = "continue" if closed else "forced_close"
    SCHED["phase2_continue" if closed else "phase2_forced"] += 1
    engine.add_request(_req(st["uid"], 2), {"prompt_token_ids": prefix}, make_params(cap, SEED + 10 * UNIT_INDEX[st["uid"]] + 2))
'''
NEW_P2 = '''    closed = THINK_CLOSE_ID in st["p1_ids"]
    # E7 v3 (audit): transactional. Build everything locally; mutate host state ONLY after the engine accepted the
    # request. The success path is token- and seed-identical to v1.4.4.
    p1_clean = list(st["p1_ids"])
    if not closed:                                  # v1.1 (F8): an EOS before </think> must not precede the stop text
        while p1_clean and p1_clean[-1] in EOS_IDS:
            p1_clean.pop()
    prefix = list(st["prompt_ids"]) + p1_clean + ([] if closed else list(STOP_IDS))
    room = MAX_MODEL_LEN - len(prefix) - 8
    cap = min(int(answer_cap), room)
    if cap < 64:
        st["p1_ids"] = p1_clean                     # as v1.4.4 on this path (EOS stripped, no phase 2)
        return False
    rid = _req(st["uid"], 2)
    try:
        engine.add_request(rid, {"prompt_token_ids": prefix}, make_params(cap, SEED + 10 * UNIT_INDEX[st["uid"]] + 2))
    except Exception:
        try:
            engine.abort_request([rid])
        except Exception as _cleanup_err:
            SCHED["phase2_cleanup_error"] = repr(_cleanup_err)[:300]
        raise
    st["p1_ids"] = p1_clean
    st["phase"] = 2; st["p2_prefix"] = [] if closed else list(STOP_IDS); st["mode"] = "continue" if closed else "forced_close"
    SCHED["phase2_continue" if closed else "phase2_forced"] += 1
'''


def once(text, old, new, where):
    n = text.count(old)
    if n != 1:
        sys.exit(f"replacement not unique ({n}) in {where}: {old[:80]!r}")
    return text.replace(old, new)


def src_of(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)


def apply_v5_cell7(t):
    """v5 (two audits of v4, 2026-10-13): cap 24 and 'KV decides' unchanged; safer length model and guards."""
    # F1: learn GENERATED length from NATURAL closes only, p90 after >= 8 per tier; forced / cut / drained /
    # hard-stopped units count as the full budget (they never under-state the final length).
    t = once(t, 'E7_MIN_FINISHED_PER_TIER = 4\n',
             'E7_MIN_FINISHED_PER_TIER = 8        # v5: was 4 (p75 from 4 units was optimistic)\n'
             'E7_FINAL_QUANTILE = 0.90            # v5: was 0.75\n'
             'E7_GROWTH_HORIZON_SEC = 300.0       # v5: no extra unit if the current KV growth hits the ceiling within 5 min\n'
             'E7_GROWTH_MIN_SPAN_SEC = 30.0\n'
             'E7_BREAKER_N, E7_BREAKER_WINDOW_SEC = 2, 1800.0   # v5: 2 preemptions within 30 min -> extras off for good\n'
             'E7_KEEP_EVENTS = {"preemption_or_counter_reset", "circuit_breaker_open", "admit"}\n', "v5 constants")
    t = once(t, '    def note_finished(self, tier, final_live):\n'
                '        self.final_lens.setdefault(int(tier), []).append(int(final_live))\n',
             '    def note_finished(self, tier, prompt_len, final_live, natural):\n'
             '        gen = int(final_live) - int(prompt_len) if natural else THINK_BUDGET + ANSWER_MAX_TOKENS + len(STOP_IDS)\n'
             '        self.final_lens.setdefault(int(tier), []).append(max(0, gen))\n', "v5 note_finished")
    t = once(t, '        """expected FINAL live length of a unit: p75 of finished units of the same tier once at least\n'
                '        E7_MIN_FINISHED_PER_TIER finished; before that the worst case (prompt + full 49,152 + 8,192 budget)."""\n'
                '        xs = sorted(self.final_lens.get(int(tier), []))\n'
                '        if len(xs) >= E7_MIN_FINISHED_PER_TIER:\n'
                '            return max(int(prompt_len), xs[min(len(xs) - 1, int(0.75 * len(xs)))])\n',
             '        """expected FINAL live length of a unit: prompt + p90 of the generated length of finished units of the\n'
             '        same tier once at least E7_MIN_FINISHED_PER_TIER finished (forced/cut = full budget); before that the\n'
             '        worst case (prompt + full 49,152 + 8,192 budget)."""\n'
             '        xs = sorted(self.final_lens.get(int(tier), []))\n'
             '        if len(xs) >= E7_MIN_FINISHED_PER_TIER:\n'
             '            return int(prompt_len) + xs[min(len(xs) - 1, int(math.ceil(E7_FINAL_QUANTILE * len(xs))) - 1)]\n',
             "v5 expected_final")
    # F3: circuit breaker; F6: important events never dropped, per-poll metrics only in the telemetry file
    t = once(t, '        self.last_projection = None\n',
             '        self.last_projection = None\n'
             '        self.preemption_times = []      # v5: circuit breaker\n'
             '        self.last_block_key = None      # v5: block counters once per (poll, reason)\n'
             '        self.last_growth = None\n', "v5 init")
    t = once(t, '        self.stats[event] += 1\n        if len(self.events) < self.max_events:\n',
             '        self.stats[event] += 1\n'
             '        if event == "metrics_ok":       # v5: per-poll samples live in kv_telemetry.json, not in events\n'
             '            return\n'
             '        if len(self.events) < self.max_events or event in E7_KEEP_EVENTS:\n', "v5 log")
    t = once(t, '            self.penalty = min(0.12, self.penalty + 0.04)   # v4: cooldown + tighter target, then resume\n'
                '            self.log(now, "preemption_or_counter_reset", old=self.preemption_counter, current=cur)\n',
             '            self.penalty = min(0.12, self.penalty + 0.04)   # v4: cooldown + tighter target, then resume\n'
             '            self.log(now, "preemption_or_counter_reset", old=self.preemption_counter, current=cur)\n'
             '            self.preemption_times = [x for x in self.preemption_times if now - x <= E7_BREAKER_WINDOW_SEC] + [now]\n'
             '            if len(self.preemption_times) >= E7_BREAKER_N and not self.frozen_expansion:\n'
             '                self.frozen_expansion = True   # v5: extras off for the rest of the run (baseline 16 continues)\n'
             '                self.log(now, "circuit_breaker_open", preemptions_in_window=len(self.preemption_times))\n',
             "v5 breaker")
    # F2: rapid-growth gate (window slope, not the noisy max pair)
    t = once(t, '    def extra_gate(self, *, now, active_count, next_prompt_tokens):\n',
             '    def kv_growth_per_sec(self):\n'
             '        """KV slope over the telemetry window (>= E7_GROWTH_MIN_SPAN_SEC), None if too short."""\n'
             '        h = list(self.history)\n'
             '        if len(h) < 2 or h[-1][0] - h[0][0] < E7_GROWTH_MIN_SPAN_SEC:\n'
             '            return None\n'
             '        return max(0.0, (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0]))\n\n'
             '    def note_block(self, now, reason):\n'
             '        key = (self.last_poll_at if self.last_poll_at is not None else int(now // 10), reason)\n'
             '        if key != self.last_block_key:\n'
             '            self.stats["block_" + reason] += 1\n'
             '            self.last_block_key = key\n\n'
             '    def extra_gate(self, *, now, active_count, next_prompt_tokens):\n', "v5 growth + block methods")
    # F4: fixed mode no longer bypasses the projection
    t = once(t, '        if self.mode == "fixed":\n            return True, "guarded_fixed_extra"\n        proj = self.lifetime_projection(d["kv"])\n',
             '        g = self.kv_growth_per_sec()\n'
             '        self.last_growth = None if g is None else round(g * 60, 5)\n'
             '        if g is None:\n'
             '            return False, "growth_window_short"\n'
             '        if d["kv"] + g * E7_GROWTH_HORIZON_SEC >= Q27X_KV_PROJECTED_CEILING - self.penalty:\n'
             '            return False, "kv_rapid_growth"\n'
             '        proj = self.lifetime_projection(d["kv"])\n', "v5 gate")
    t = once(t, '                "kv_penalty": self.penalty, "last_projection": self.last_projection,\n',
             '                "kv_penalty": self.penalty, "last_projection": self.last_projection,\n'
             '                "last_growth_per_min": self.last_growth, "preemption_times_min":\n'
             '                    [round((x - NOTEBOOK_T0) / 60, 2) for x in self.preemption_times],\n'
             '                "final_len_rule": {"min_per_tier": E7_MIN_FINISHED_PER_TIER, "quantile": E7_FINAL_QUANTILE,\n'
             '                                   "learned_from": "natural closes; others = full budget"},\n', "v5 report")
    t = once(t, '                E7_GATE.stats["block_" + _reason] += 1\n',
             '                E7_GATE.note_block(time.perf_counter(), _reason)\n', "v5 block dedupe")
    t = once(t, '            if len(E7_TELEMETRY) < 4000:\n', '            if len(E7_TELEMETRY) < 12000:   # v5: 12 h at 12 s = 3,600 polls\n',
             "v5 telemetry cap")
    # F7/F8: record + file I/O must not stop the scheduler; only natural closes teach the length model
    t = once(t, '    E7_GATE.note_finished(CODEX_PRIORITY_TIERS[UNIT_TASK[uid]],\n'
                '                          len(st["prompt_ids"]) + len(st["p1_ids"]) + len(st["p2_prefix"]) + len(st["p2_ids"]))\n'
                '    with open(RECORDS_PATH, "a", encoding="utf-8") as f:\n'
                '        f.write(json.dumps(rec) + "\\n")\n',
             '    try:\n'
             '        E7_GATE.note_finished(CODEX_PRIORITY_TIERS[UNIT_TASK[uid]], len(st["prompt_ids"]),\n'
             '                              len(st["prompt_ids"]) + len(st["p1_ids"]) + len(st["p2_prefix"]) + len(st["p2_ids"]),\n'
             '                              natural=(st["mode"] is None and st["end_reason"] == "natural"))\n'
             '    except Exception as _e:\n'
             '        print("E7 note_finished failed:", uid, repr(_e)[:200])\n'
             '    try:   # v5 (audit): a records-file error must not stop the scheduler (RECORDS keeps it in memory)\n'
             '        with open(RECORDS_PATH, "a", encoding="utf-8") as f:\n'
             '            f.write(json.dumps(rec) + "\\n")\n'
             '    except Exception as _e:\n'
             '        SCHED["records_io_errors"] = SCHED.get("records_io_errors", 0) + 1\n'
             '        print("records.jsonl write failed:", uid, repr(_e)[:200])\n', "v5 record io")
    t = once(t, '          flush=True)\n    write_submission()\n',
             '          flush=True)\n'
             '    try:\n'
             '        write_submission()\n'
             '    except Exception as _e:   # v5: the final write in the finally block / cell 8 retries\n'
             '        SCHED["submission_io_errors"] = SCHED.get("submission_io_errors", 0) + 1\n'
             '        print("submission write failed:", repr(_e)[:200])\n', "v5 submission io")
    # hidden rerun visibility (no assert: an assert could kill the hidden run)
    t = once(t, 'E7_TELEMETRY = []\n',
             'E7_TELEMETRY = []\n'
             'if RERUN and E7_HIDDEN_APPROVED and (not Q27X_ORDERING or E7_GATE.mode != "adaptive"):\n'
             '    print("WARNING E7 hidden rerun: ordering", Q27X_ORDERING, "controller", E7_GATE.mode,\n'
             '          "(adaptive falls back to shadow = v1.4.4 scheduling when the engine is not the preferred recipe)")\n',
             "v5 hidden visibility")
    return t


def build(name, v):
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    c = nb["cells"]
    # ---- cell 0 (markdown): variant banner
    set_src(c[0], v["banner"] + "\n\n" + HIDDEN_NOTE + "\n\n" + src_of(c[0]))
    # ---- cell 2 (config)
    t = src_of(c[2])
    t = once(t, 'PROFILE = "RERUN_ALL" if RERUN else os.environ.get("Q27X_PROFILE", "EVAL12").strip().upper()',
             f'PROFILE = "RERUN_ALL" if RERUN else os.environ.get("Q27X_PROFILE", "{v["profile"]}").strip().upper()',
             "cell2 profile")
    t = once(t, '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n',
             '    "EVAL12":    {"source": "eval", "n_tasks": 12, "time_budget_sec": 3.0 * 3600, "admit_frac": 0.80, "drain_frac": 0.90},\n'
             '    # E7 (Claude): EDA LB-mirror proxy (MAIN_DEBUG_48), 48 tasks / 56 test outputs; same 3 h budget and\n'
             '    # admit/drain fractions as EVAL12 (whether every unit finishes in 3 h on GPU is unknown -> measured)\n'
             f'    "PROXY48":   {{"source": "eval", "task_list": {json.dumps(PROXY48)}, "time_budget_sec": 3.0 * 3600,\n'
             '                  "admit_frac": 0.80, "drain_frac": 0.90},\n',
             "cell2 profile table")
    t = once(t, 'MAX_NUM_SEQS = int(os.environ.get("Q27X_MAX_NUM_SEQS", 16))',
             f'MAX_NUM_SEQS = int(os.environ.get("Q27X_MAX_NUM_SEQS", {v["cap"]}))', "cell2 cap")
    t = once(t, 'Q27X_KV_MODE = os.environ.get("Q27X_KV_MODE", "shadow").strip().lower()',
             f'Q27X_KV_MODE = os.environ.get("Q27X_KV_MODE", "{v["kv"]}").strip().lower()', "cell2 kv mode")
    t = once(t, 'Q27X_ORDERING = os.environ.get("Q27X_ORDERING", "1") == "1"',
             f'Q27X_ORDERING = os.environ.get("Q27X_ORDERING", "{1 if v["ordering"] else 0}") == "1"', "cell2 ordering")
    t = once(t, 'assert 0 < Q27X_KV_EXTRA_THRESHOLD < Q27X_KV_PROJECTED_CEILING < Q27X_KV_EMERGENCY < 1\n',
             'assert 0 < Q27X_KV_EXTRA_THRESHOLD < Q27X_KV_PROJECTED_CEILING < Q27X_KV_EMERGENCY < 1\n'
             '# E7 v3 hidden-rerun guard (audit): in a competition rerun every experimental E7 feature is OFF (original\n'
             '# order, KV off, engine cap 16 = v1.4.4 scheduling) unless this validated configuration was explicitly\n'
             '# approved for the hidden run. Kaggle cannot set env vars, so on Kaggle this is effectively the constant False.\n'
             'E7_HIDDEN_APPROVED = True   # user decision 2026-10-12: E7 ordering + KV controller also run in the hidden rerun\n'
             'E7_HIDDEN_GUARD_APPLIED = bool(RERUN and not E7_HIDDEN_APPROVED)\n'
             'if E7_HIDDEN_GUARD_APPLIED:\n'
             '    Q27X_ORDERING, Q27X_KV_MODE, MAX_NUM_SEQS = False, "off", 16\n', "cell2 hidden guard")
    t = once(t, '    "request_engine_cap": MAX_NUM_SEQS,\n',
             '    "request_engine_cap": MAX_NUM_SEQS,\n    "hidden_guard_applied": E7_HIDDEN_GUARD_APPLIED,\n',
             "cell2 config guard")
    # v4 (user goal): no fixed 16-18 ceiling; the KV projection decides, up to the engine cap
    t = once(t, 'Q27X_KV_EXTRA_CAP = int(os.environ.get("Q27X_KV_EXTRA_CAP", "18"))\n'
                'if not 16 <= Q27X_KV_EXTRA_CAP <= 20:\n    raise ValueError("Q27X_KV_EXTRA_CAP must be in [16,20]")',
             f'Q27X_KV_EXTRA_CAP = int(os.environ.get("Q27X_KV_EXTRA_CAP", "{v["cap"]}"))   # v4: = engine cap; KV decides\n'
             'if not 16 <= Q27X_KV_EXTRA_CAP <= 64:\n    raise ValueError("Q27X_KV_EXTRA_CAP must be in [16,64]")',
             "cell2 extra cap")
    t = once(t, 'os.environ.get("Q27X_KV_EXTRA_THRESHOLD", "0.62")', 'os.environ.get("Q27X_KV_EXTRA_THRESHOLD", "0.85")',
             "cell2 thr")
    t = once(t, 'os.environ.get("Q27X_KV_PROJECTED_CEILING", "0.82")', 'os.environ.get("Q27X_KV_PROJECTED_CEILING", "0.90")',
             "cell2 ceiling")
    t = once(t, 'os.environ.get("Q27X_KV_EMERGENCY", "0.90")', 'os.environ.get("Q27X_KV_EMERGENCY", "0.93")',
             "cell2 emergency")
    set_src(c[2], t)
    # ---- cell 6 (engine): cap ladder 32 -> 24 -> 16 with the SAME preferred recipe before any degraded recipe
    t = src_of(c[6])
    t = once(t, '    if MAX_NUM_SEQS > 16:\n        _ENGINE_TRIALS.append((16, 0, *_LADDER[0]))\n',
             '    for _cap_fb in (24, 16):          # v4: 32 -> 24 -> 16, same preferred recipe\n'
             '        if MAX_NUM_SEQS > _cap_fb:\n'
             '            _ENGINE_TRIALS.append((_cap_fb, 0, *_LADDER[0]))\n', "cell6 cap ladder")
    set_src(c[6], t)
    # ---- cell 4 (tasks): fixed task-list profiles
    t = src_of(c[4])
    t = once(t, '    TASK_IDS = sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch), P["n_tasks"]))\n',
             '    if P.get("task_list"):   # E7: fixed proxy list (every id must exist in the evaluation file)\n'
             '        _missing = [t for t in P["task_list"] if t not in _ch]\n'
             '        assert not _missing, f"{PROFILE}: tasks missing from evaluation file: {_missing}"\n'
             '        TASK_IDS = sorted(P["task_list"])\n'
             '    else:\n'
             '        TASK_IDS = sorted(random.Random(TASK_SAMPLE_SEED).sample(sorted(_ch), P["n_tasks"]))\n',
             "cell4 task ids")
    set_src(c[4], t)
    # ---- cell 7 (scheduler)
    t = src_of(c[7])
    t = once(t, '"e7_priority_tier": CODEX_PRIORITY_TIERS[UNIT_TASK[uid]]}',
             '"e7_priority_tier": CODEX_PRIORITY_TIERS[UNIT_TASK[uid]],\n'
             '           "t_admit_min": round((st["t_admit"] - NOTEBOOK_T0) / 60, 2),\n'
             '           "t_done_min": round((st["t_done"] - NOTEBOOK_T0) / 60, 2)}', "cell7 record")
    t = once(t, '        self.max_events = 3000\n',
             '        self.max_events = 3000\n'
             '        self.last_shadow_poll = None   # v3: one shadow decision per telemetry poll\n', "cell7 shadow init")
    t = once(t, '            self.stats["shadow_would_admit" if allowed else "shadow_reject_" + reason] += 1\n',
             '            # v3 (audit): count at most ONE shadow decision per telemetry poll (decide() runs after every\n'
             '            # engine step; counting every call inflated the opportunity count).\n'
             '            if self.last_poll_at is not None and self.last_poll_at != self.last_shadow_poll:\n'
             '                self.stats["shadow_would_admit" if allowed else "shadow_reject_" + reason] += 1\n'
             '                self.last_shadow_poll = self.last_poll_at\n', "cell7 shadow count")
    t = once(t, OLD_P2, NEW_P2, "cell7 start_phase2")
    # ---- v4: lifetime KV projection decides how many units run (user goal)
    t = once(t, 'class E7KVAdmission:\n',
             '# v4 (user goal): the controller computes, from KV, how many units can run. Per-sequence state overhead\n'
             '# fitted on run 7 (16 seqs: KV 0.30 at 2 min with ~80k live prompt tokens); the measured/model ratio\n'
             '# re-calibrates the projection at every poll, so the constant only has to be roughly right.\n'
             'E7_KV_PER_SEQ = 0.0116\n'
             'E7_MIN_FINISHED_PER_TIER = 4\n\n'
             'class E7KVAdmission:\n', "cell7 v4 constants")
    t = once(t, '        self.last_shadow_poll = None   # v3: one shadow decision per telemetry poll\n',
             '        self.last_shadow_poll = None   # v3: one shadow decision per telemetry poll\n'
             '        self.penalty = 0.0             # v4: tightened after each preemption (cooldown, never a permanent freeze)\n'
             '        self.final_lens = {}           # v4: tier -> final live lengths of finished units\n'
             '        self.load = None               # v4: (n_active, live_sum, expected_final_sum, candidate_final)\n'
             '        self.last_projection = None\n', "cell7 v4 init")
    t = once(t, '            self.frozen_expansion = True\n',
             '            self.penalty = min(0.12, self.penalty + 0.04)   # v4: cooldown + tighter target, then resume\n',
             "cell7 v4 preemption")
    t = once(t, '    def positive_kv_growth_per_sec(self):\n',
             '    def note_finished(self, tier, final_live):\n'
             '        self.final_lens.setdefault(int(tier), []).append(int(final_live))\n\n'
             '    def expected_final(self, tier, prompt_len):\n'
             '        """expected FINAL live length of a unit: p75 of finished units of the same tier once at least\n'
             '        E7_MIN_FINISHED_PER_TIER finished; before that the worst case (prompt + full 49,152 + 8,192 budget)."""\n'
             '        xs = sorted(self.final_lens.get(int(tier), []))\n'
             '        if len(xs) >= E7_MIN_FINISHED_PER_TIER:\n'
             '            return max(int(prompt_len), xs[min(len(xs) - 1, int(0.75 * len(xs)))])\n'
             '        return int(prompt_len) + THINK_BUDGET + ANSWER_MAX_TOKENS + len(STOP_IDS)\n\n'
             '    def set_load(self, n_active, live_sum, final_sum, candidate_final):\n'
             '        self.load = (int(n_active), int(live_sum), int(final_sum), int(candidate_final))\n\n'
             '    def lifetime_projection(self, kv_now):\n'
             '        """KV usage if every active unit AND the candidate reach their expected final length."""\n'
             '        if not self.load or self.kv_tokens <= 0:\n'
             '            return None\n'
             '        n, live_sum, final_sum, cand_final = self.load\n'
             '        model_now = E7_KV_PER_SEQ * n + live_sum / self.kv_tokens\n'
             '        f = min(1.6, max(0.6, kv_now / model_now)) if model_now > 0 else 1.0\n'
             '        return f * (E7_KV_PER_SEQ * (n + 1) + (final_sum + cand_final) / self.kv_tokens)\n\n'
             '    def positive_kv_growth_per_sec(self):\n', "cell7 v4 methods")
    old_gate = t[t.index('        if d["kv"] >= Q27X_KV_EXTRA_THRESHOLD:\n'):t.index('        return True, "adaptive_extra"\n') + len('        return True, "adaptive_extra"\n')]
    t = once(t, old_gate,
             '        # v4 (user goal): measured-KV guard, then the lifetime projection decides (no fixed 16-18 ceiling)\n'
             '        if d["kv"] >= Q27X_KV_EXTRA_THRESHOLD - self.penalty:\n'
             '            return False, "kv_over_extra_threshold"\n'
             '        if self.mode == "fixed":\n'
             '            return True, "guarded_fixed_extra"\n'
             '        proj = self.lifetime_projection(d["kv"])\n'
             '        self.last_projection = None if proj is None else round(proj, 4)\n'
             '        if proj is None:\n'
             '            return False, "projection_unavailable"\n'
             '        if proj > Q27X_KV_PROJECTED_CEILING - self.penalty:\n'
             '            return False, "kv_lifetime_projection"\n'
             '        return True, "adaptive_extra"\n', "cell7 v4 gate")
    t = once(t, '                "frozen_expansion": self.frozen_expansion,\n',
             '                "frozen_expansion": self.frozen_expansion,\n'
             '                "kv_penalty": self.penalty, "last_projection": self.last_projection,\n'
             '                "finished_final_len_by_tier": {str(k): sorted(v) for k, v in self.final_lens.items()},\n',
             "cell7 v4 report")
    t = once(t, '            _uid = queue[0]\n',
             '            _uid = queue[0]\n'
             '            # v4: exact live tokens + expected final length of every active unit, and of the candidate\n'
             '            _lives = [(len(s["prompt_ids"]) + len(s["p1_ids"]) + len(s["p2_prefix"]) + len(s["p2_ids"]), s)\n'
             '                      for s in active]\n'
             '            E7_GATE.set_load(len(active), sum(_l for _l, _ in _lives),\n'
             '                             sum(max(_l, E7_GATE.expected_final(CODEX_PRIORITY_TIERS[UNIT_TASK[_s["uid"]]],\n'
             '                                                                len(_s["prompt_ids"]))) for _l, _s in _lives),\n'
             '                             E7_GATE.expected_final(CODEX_PRIORITY_TIERS[UNIT_TASK[_uid]], len(PROMPT_IDS[_uid])))\n',
             "cell7 v4 load")
    t = once(t, '    RECORDS.append(rec)\n',
             '    RECORDS.append(rec)\n'
             '    E7_GATE.note_finished(CODEX_PRIORITY_TIERS[UNIT_TASK[uid]],\n'
             '                          len(st["prompt_ids"]) + len(st["p1_ids"]) + len(st["p2_prefix"]) + len(st["p2_ids"]))\n',
             "cell7 v4 note finished")
    t = once(t, 'finally:\n    SCHED.setdefault("wall_sec", round(time.perf_counter() - SCHED["started"], 1))\n',
             'finally:\n'
             '    if SCHED_ERROR is not None:   # v3 (audit): run_scheduler() did not reach its own not_admitted line\n'
             '        SCHED["not_admitted"] = max(0, len(PROMPT_IDS) - SCHED["admitted"])\n'
             '    SCHED.setdefault("wall_sec", round(time.perf_counter() - SCHED["started"], 1))\n', "cell7 not_admitted")
    t = apply_v5_cell7(t)
    set_src(c[7], t)
    # ---- cell 8 (report): validation FIRST, audit in the report, idempotent re-check at the end
    t = src_of(c[8])
    t = once(t, '    write_json_atomic(OUT / "run_report.json", report)\n',
             '    report["submission_audit"] = SUBMISSION_AUDIT\n'
             '    write_json_atomic(OUT / "run_report.json", report)\n', "cell8 audit in report")
    t = CELL8_PREFIX + t.rstrip("\n") + "\n" + CELL8_SUFFIX
    set_src(c[8], t)
    for cell in c:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    out = OUT_DIR / name
    data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    out.write_bytes(data)
    print("BUILT", out.name, len(data), "bytes sha256", hashlib.sha256(data).hexdigest()[:16])


if __name__ == "__main__":
    got = hashlib.sha256(SRC.read_bytes()).hexdigest()[:16]
    if got != SRC_SHA:
        sys.exit(f"source sha {got} != {SRC_SHA}")
    for _name, _v in VARIANTS.items():
        build(_name, _v)
